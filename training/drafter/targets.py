"""Frozen drafter targets: per-(loop, layer) features, last-loop labels and loop-d readouts.

Native targets (the original per-loop-KV models) run one teacher-forced forward and capture features
with decoder-layer hooks. Ouro runs its L decoder layers R times, so a hook fires R*L times per
forward in loop-major order: call k is (loop k // L, layer k % L). Features are keyed by call index;
keying by layer id would silently keep only the last loop.

The replay target (a shared-KV W64 checkpoint) is not computed by a parallel teacher-forced forward;
every document is replayed exactly with training.sharekv.exact_documents.

Dory (DoryTarget) is block-wise looped and has no Transformers port: its features come from the native
PyTorch model in training.hf_models.dory, whose loop-d state is already the output block's readout state.

A feature is the output of 0-based layer l inside loop r (pre-final-norm residual stream). The
loop-r readout is lm_head(norm(output of layer L-1 in loop r)), the model's own per-loop hidden state;
that loop output is captured separately for labels and native readouts, not fed to the drafter.
"""
import warnings

import torch

from ..common import registry
from ..hf_models import load_model, model_classes, model_config, validate_config
from ..hf_models.dory.native import FEATURE_LAYERS as DORY_FEATURE_LAYERS
from ..hf_models.dory.native import LOOPS as DORY_LOOPS
from ..hf_models.dory.native import NativeDory, feature_layer_ids
from ..sharekv import exact_documents, flash_available, loop_count


def default_feature_layers(num_layers):
    """DFlash rule round(1 + j(L-4)/4), j = 0..4, applied inside one loop."""
    return [round(1 + j * (num_layers - 4) / 4) for j in range(5)]


def target_feature_layers(config):
    """Default feature layers of a target config.json: Dory's physical taps, otherwise the DFlash rule."""
    if config.get("model_type") == "dory":
        return list(DORY_FEATURE_LAYERS)
    return default_feature_layers(config["num_hidden_layers"])


class OuroTarget:
    family = "ouro"

    def __init__(self, path, feat_layers, device, dtype=torch.bfloat16):
        _, cls = model_classes(self.family)
        model = cls.from_pretrained(path, config=model_config(path, self.family),
                                    local_files_only=True, dtype=dtype, attn_implementation="sdpa",
                                    device_map={"": str(device)}).eval().requires_grad_(False)
        self._bind(model, feat_layers)

    @classmethod
    def from_model(cls, model, feat_layers):
        """Capture an already loaded native model without loading a second weight copy."""
        target = cls.__new__(cls)
        target._bind(model.eval().requires_grad_(False), feat_layers)
        return target

    def _bind(self, model, feat_layers):
        self.model = model
        cfg = model.config
        if self.family == "nanbeige":
            validate_config(cfg, "nanbeige")
            loops = model.model._get_num_loops()
        else:
            loops = cfg.total_ut_steps
        self.L, self.R, self.H = cfg.num_hidden_layers, loops, cfg.hidden_size
        self.feat_layers = list(feat_layers)
        if (not self.feat_layers or len(set(self.feat_layers)) != len(self.feat_layers)
                or any(not 0 <= layer < self.L for layer in self.feat_layers)):
            raise ValueError("Feature layers must be distinct native decoder-layer indices")
        self._slot = {l: i for i, l in enumerate(self.feat_layers)}
        self._calls, self._buf, self._out = 0, None, None
        for layer in self.model.model.layers:
            layer.register_forward_hook(self._hook)

    @property
    def embed(self):
        return self.model.model.embed_tokens

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def final_norm(self):
        return self.model.model.norm

    def _hook(self, _module, _args, out):
        k = self._calls
        self._calls += 1
        if self._buf is None:
            return
        r, l = divmod(k, self.L)
        h = out[0] if isinstance(out, tuple) else out
        j = self._slot.get(l)
        if j is not None:
            self._buf[r, :, j * self.H:(j + 1) * self.H] = h[0]
        if l == self.L - 1:
            self._out[r] = h[0]

    @torch.no_grad()
    def features(self, input_ids):
        """Document [T] -> features [R, T, F], pre-final-norm loop outputs [R, T, H].

        readout_logits applies final_norm once. ReplayTarget instead returns
        already normalized loop states from exact replay.
        """
        T = input_ids.numel()
        dtype = self.embed.weight.dtype
        self._buf = torch.empty(self.R, T, len(self.feat_layers) * self.H, dtype=dtype,
                                device=input_ids.device)
        self._out = torch.empty(self.R, T, self.H, dtype=dtype, device=input_ids.device)
        self._calls = 0
        try:
            self.model.model(input_ids=input_ids[None], use_cache=False)
            if self._calls != self.R * self.L:
                raise RuntimeError(f"Expected {self.R * self.L} decoder-layer calls, got {self._calls}")
            return self._buf, self._out
        finally:
            self._buf = self._out = None

    @torch.no_grad()
    def readout_logits(self, loop_out):
        return self.lm_head(self.final_norm(loop_out))

    @torch.no_grad()
    def readout_argmax(self, loop_out, chunk=2048):
        """loop_out [N, H] (layer L-1 output of some loop) -> argmax of that loop's readout [N]."""
        out = torch.empty(loop_out.shape[0], dtype=torch.long, device=loop_out.device)
        for s in range(0, loop_out.shape[0], chunk):
            out[s:s + chunk] = self.readout_logits(loop_out[s:s + chunk]).argmax(-1)
        return out


class NanbeigeTarget(OuroTarget):
    """Native two-loop Nanbeige target with the same pre-norm feature contract.

    Each loop applies its final norm before entering the next loop. Decoder
    hooks capture the pre-norm residual; readout_logits applies that norm once.
    Native parallel teacher forcing has ordinary per-loop KV and is distinct
    from ReplayTarget's query-local W64 exact replay.
    """

    family = "nanbeige"


class DoryTarget:
    """Frozen original Dory1.0-4B (per-loop KV, R = 4) by teacher forcing.

    One NativeDory forward runs the recurrent block through loops 1..R and then the output block on
    s_1..s_R as independent sequences. Features are the physical sublayers 3 (input block), 19/35/51
    (recurrent block in loop d) and 67 (output block on s_d); the loop-d state is the output block's
    result on s_d, so a readout is the scaled head on it (there is no separate final norm). Dory shares
    KV without adaptation, so its one drafter trains on these features for both of its targets.

    attention "flash" uses the vLLM fork's FlashAttention; "sdpa" the masked reference (always on CPU,
    and with a warning when vllm.vllm_flash_attn is unavailable).
    """

    family = "dory"

    def __init__(self, path, feat_layers, device, dtype=torch.bfloat16, attention="flash"):
        self._bind(NativeDory.from_pretrained(path, device, dtype), feat_layers, attention)

    @classmethod
    def from_model(cls, model, feat_layers=DORY_FEATURE_LAYERS, attention="flash"):
        target = cls.__new__(cls)
        target._bind(model, feat_layers, attention)
        return target

    def _bind(self, model, feat_layers, attention):
        c = model.config
        feature_layer_ids(vars(c))  # rejects another block layout
        if tuple(feat_layers) != DORY_FEATURE_LAYERS:
            raise ValueError(f"The Dory drafter's feature taps are the physical sublayers {DORY_FEATURE_LAYERS}")
        self.model = model.eval().requires_grad_(False)
        self.feat_layers = list(feat_layers)
        self.L, self.R, self.H = c.num_hidden_layers, DORY_LOOPS, c.hidden_size
        if self.embed.weight.device.type != "cuda":
            attention = "sdpa"
        elif attention == "flash" and not flash_available():
            warnings.warn("vllm.vllm_flash_attn is not installed; Dory features fall back to the SDPA reference")
            attention = "sdpa"
        self.attention = attention

    @property
    def embed(self):
        return self.model.backbone.embeddings

    @property
    def lm_head(self):
        return self.model.lm_head

    @torch.no_grad()
    def features(self, input_ids):
        """Document [T] -> features [R, T, F], output-block readout states [R, T, H]."""
        return self.model.features(input_ids, self.feat_layers, self.R, self.attention)

    @torch.no_grad()
    def readout_logits(self, state):
        return self.lm_head(state)

    @torch.no_grad()
    def readout_argmax(self, state, chunk=128):
        """state [N, H] (output block on some s_d) -> argmax of the depth-d readout [N]; FP32 logits."""
        return torch.cat([self.readout_logits(h).argmax(-1) for h in state.split(chunk)])


def native_target(name, feat_layers, device, dtype=torch.bfloat16, path=None, attention="flash"):
    """Native target of registry model `name`, loaded from path (default: its pinned original).

    attention selects the attention of Dory's teacher-forced forward (DoryTarget); the Transformers
    ports always use SDPA.
    """
    family = registry.family(name)
    path = path or registry.resolve(name)
    if family == "dory":
        return DoryTarget(path, feat_layers, device, dtype, attention)
    cls = {"nanbeige": NanbeigeTarget}.get(family, OuroTarget)
    return cls(path, feat_layers, device, dtype)


class ReplayTarget:
    """Frozen shared-KV W64 checkpoint: features, labels and readouts by exact chunk replay.

    The parallel teacher-forced forward of OuroTarget is not the shared model's computation, so
    every document is replayed with exact_documents (native prompt, response chunks of at most
    W tokens). Loop states come back already normed, so a readout is lm_head(state).
    replay_attention "flash" uses FlashAttention with CUDA-graph chunk steps; "sdpa" the masked reference
    (about 3.3x slower; used on CPU, and with a warning when vllm.vllm_flash_attn is unavailable).
    A replay batch holds at most batch_documents documents, batch_tokens padded tokens and
    prompt_tokens padded prompt tokens (the prompt pass keeps every loop's KV), so short documents
    share a batch without raising the peak memory of 8 full-length documents.
    """

    def __init__(self, name, checkpoint, feat_layers, device, window=64, batch_documents=16,
                 replay_attention="flash", batch_tokens=8 * 16384, prompt_tokens=16384):
        self.model = load_model(name, device, checkpoint, frozen=True)
        cfg = self.model.config
        self.L, self.R, self.H = cfg.num_hidden_layers, loop_count(self.model), cfg.hidden_size
        self.feat_layers = list(feat_layers)
        self.window, self.batch_documents = window, batch_documents
        self.batch_tokens, self.prompt_tokens = batch_tokens, prompt_tokens
        self.replay_attention = replay_attention if torch.device(device).type == "cuda" else "sdpa"
        if self.replay_attention == "flash" and not flash_available():
            warnings.warn("vllm.vllm_flash_attn is not installed; exact replay falls back to the SDPA reference")
            self.replay_attention = "sdpa"

    @property
    def embed(self):
        return self.model.model.embed_tokens

    @property
    def lm_head(self):
        return self.model.lm_head

    @torch.no_grad()
    def features_many(self, docs):
        """docs [{"ids", "resp_start"}] -> per document (features [R, T, F], normed loop states [R, T, H]).

        Documents are replayed longest first in batches (see the class), so a batch pads little.
        """
        order = sorted(range(len(docs)), key=lambda i: -len(docs[i]["ids"]))
        batches = []
        for i in order:
            current = batches[-1] if batches else None
            if (current and len(current) < self.batch_documents
                    and (len(current) + 1) * len(docs[current[0]]["ids"]) <= self.batch_tokens
                    and (len(current) + 1) * max(docs[j]["resp_start"] for j in current + [i]) <= self.prompt_tokens):
                current.append(i)
            else:
                batches.append([i])
        out = [None] * len(docs)
        for rows in batches:
            batch = [{"input_ids": docs[i]["ids"], "prompt_length": docs[i]["resp_start"]} for i in rows]
            for i, r in zip(rows, exact_documents(self.model, batch, self.window, feature_layers=self.feat_layers,
                                                  attention=self.replay_attention,
                                                  loop_states=True)):
                out[i] = (r["features"], r["loops"])
        return out

    @torch.no_grad()
    def readout_logits(self, loop_state):
        return self.lm_head(loop_state)

    @torch.no_grad()
    def readout_argmax(self, loop_state, chunk=2048):
        out = torch.empty(loop_state.shape[0], dtype=torch.long, device=loop_state.device)
        for s in range(0, loop_state.shape[0], chunk):
            out[s:s + chunk] = self.readout_logits(loop_state[s:s + chunk]).argmax(-1)
        return out
