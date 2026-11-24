# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""training.hf_models.inkling2.native on CPU: the reference forward of looped Inkling2.

The model is a tiny checkpoint in the released on-disk format (tests/vllm/inkling2_tiny.py) with Mini's block
structure: blocks of two layers that each end in a full-attention layer, a one-block prelude, a two-block core
run three times and a one-block coda (16 layer visits), sliding window 8, relative extent 16 and log scaling
from position 8, so 28 tokens cross every boundary. Three kinds of evidence:

  - `Reference`, an independent FP64 model written here in the formulation of the transformers port (the sink
    as a softmax column, score / sum routing weights applied after the down projection, folded expert weights,
    grouped cross-correlation convs, softmax attention over normalized blocks) and run once per pass count;
  - properties of the forward itself: the two execution modes, causality, the window and the relative-bias
    rule, per-depth readouts, the controllers, the taps, the router and the conv state;
  - opt-in: the released Mini weights (a local snapshot, as tests/vllm/test_inkling2_weights.py finds it) and the
    model of transformers PR #51050 in an interpreter named by INKLING2_HF_ORACLE_PYTHON.
"""
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.models import inkling2_weights as weights

from training.hf_models.inkling2 import native
from training.hf_models.inkling2.native import Inkling2Cache, NativeInkling2, Precision

ROOT = Path(__file__).resolve().parents[2]
MINI = ("thinkingmachines/Inkling-2-Mini", "1fb4134b9a74f9681054f4ef82e4bf1f458a2337")
TOKENS, DEPTHS = 28, (1, 2, 3)
# BF16 keeps 8 significant bits, so a stored value moves by up to 2 ** -9 of itself. The tiny model's logits
# reach 3.6 and its served logits were measured within 0.11 of the exact ones with routing fixed (about 15 such
# steps adding up over 16 layer visits); the bound leaves a factor of two.
BF16_NOISE = 0.25


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tiny = load_module("inkling2_tiny", ROOT / "tests" / "vllm" / "inkling2_tiny.py")


def mini_like_config(**text_overrides):
    loop = dict(core_start=2, core_end=6, num_passes=3, gate_rank=4, is_varloop=False)
    return tiny.tiny_config(num_hidden_layers=8, local_layer_ids=[0, 2, 4, 6], loop_config=loop, **text_overrides)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return tiny.write_tiny_checkpoint(tmp_path_factory.mktemp("inkling2"), mini_like_config(), mxfp8_layers=(6, 7))


@pytest.fixture(scope="module")
def checkpoint(snapshot):
    return weights.load_inkling2_checkpoint(snapshot)


@pytest.fixture(scope="module")
def exact(checkpoint):
    return NativeInkling2(checkpoint, "exact")


@pytest.fixture(scope="module")
def tokens():
    return torch.randint(0, 100, (TOKENS,), generator=torch.Generator().manual_seed(0))


@pytest.fixture(scope="module")
def full(exact, tokens):
    """One full-sequence forward with every depth and every tap."""
    return exact.forward(tokens, depths=DEPTHS, taps=range(17), return_logits=True)


def edited(snapshot, edit):
    """A fresh checkpoint whose tensors `edit(name, tensor)` replaced (None keeps a tensor)."""
    checkpoint = weights.load_inkling2_checkpoint(snapshot)
    for name, tensor in list(checkpoint.tensors.items()):
        changed = edit(name.removeprefix(weights.LLM_PREFIX), tensor)
        if changed is not None:
            checkpoint.tensors[name] = changed.to(tensor.dtype)
    return checkpoint


def fixed_routing(snapshot):
    """Router biases two apart: sigmoid(logit) + bias then selects the same experts whatever the input, so a
    BF16 rounding step cannot flip an expert and only continuous differences remain."""
    def edit(name, tensor):
        if name.endswith("mlp.gate.bias"):
            layer = int(name.split(".")[1])
            return torch.roll(2.0 * torch.arange(tensor.numel(), dtype=torch.float32), layer)

    return edited(snapshot, edit)


def stepwise(model, tokens, chunks=None, **kwargs):
    """Incremental forward over `chunks` (default: one token at a time): the cache and the logits per depth."""
    cache, outputs = Inkling2Cache(), []
    for part in tokens.split(chunks or 1):
        outputs.append(model.forward(part, depths=DEPTHS, cache=cache, return_logits=True, **kwargs))
    return cache, {depth: torch.cat([out.readouts[depth].logits for out in outputs]) for depth in DEPTHS}


def confident(logits, margin):
    """Rows whose two largest logits are more than `margin` apart."""
    top = logits.topk(2, dim=-1).values
    return top[:, 0] - top[:, 1] > margin


class Reference:
    """Independent FP64 Inkling2 on the checkpoint tensors (the test oracle), sharing no code with the module.

    Whole sequences with explicit loops over tokens and heads, in the formulation of transformers PR #51050
    where it differs from the vLLM kernels the module follows. `run(tokens, passes)` is the model configured
    with that many passes, so a depth is never read out of a longer run.
    """

    def __init__(self, checkpoint):
        self.c = checkpoint.config
        self.w = {name.removeprefix(weights.LLM_PREFIX): tensor.double() for name, tensor in checkpoint.tensors.items()}
        self.experts = []
        for layer in range(self.c.num_hidden_layers):
            dense = checkpoint.experts(layer)
            self.experts.append((dense.w13.double() * dense.w13_scale.double()[:, None, None],
                                 dense.w2.double() * dense.w2_scale.double()[:, None, None]))

    def of(self, layer, name):
        return self.w[f"layers.{layer}.{name}"]

    @staticmethod
    def unit(x, eps=1e-6):
        return x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps)

    def norm(self, x, weight):
        return self.unit(x, self.c.rms_norm_eps) * weight

    def mix(self, blocks, partial, queries):
        """[Q, T, H]: softmax attention of each query over the RMS-normalized sources, scaled by H ** -0.5."""
        hidden = self.c.hidden_size
        sources = self.unit(torch.stack([*blocks, partial], 1))  # [T, N, H]
        scores = sources @ queries.reshape(-1, hidden).T * hidden ** -0.5
        return torch.einsum("tnq,tnh->qth", scores.softmax(1), sources)

    @staticmethod
    def clip(x, threshold):
        return x * (threshold / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)

    @staticmethod
    def conv(x, weight):
        """Static short conv as a grouped cross-correlation over the zero-padded sequence."""
        kernel = weight.shape[-1]
        y = F.conv1d(x.T[None], weight, padding=kernel - 1, groups=x.shape[1])[0, :, :x.shape[0]].T
        return (y + x) / math.sqrt(kernel + 1)

    @staticmethod
    def dynamic_conv(x, static, deltas, heads):
        """Dynamic short conv token by token: tap `lag` weights the input `lag` positions back."""
        channels, kernel = x.shape[1], static.shape[0]
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            delta = deltas[t].view(heads, kernel)
            for lag in range(min(kernel, t + 1)):
                out[t] += (static[lag] + delta[:, lag].repeat_interleave(channels // heads)) * x[t - lag]
        return (out + x) / math.sqrt(2 * kernel + 1)

    def attention(self, layer, x_q, x_k, x_v):
        c = self.c
        heads, kv_heads, dim = c.attention_dims(layer)
        local, tokens = c.is_local(layer), x_q.shape[0]

        def w(name):
            return self.of(layer, "attn." + name)

        tau = torch.ones(tokens, dtype=torch.float64)
        if not local:
            position = torch.arange(tokens, dtype=torch.float64)
            tau = 1 + c.log_scaling_alpha * torch.log(((position + 1) / c.log_scaling_n_floor).clamp(min=1.0))
        q = (x_q @ w("wq_du.weight").T).view(tokens, heads, dim) * w("q_l2_clip.scale")
        r = (x_q @ w("wr_du.weight").T).view(tokens, heads, c.d_rel) * w("rel_l2_clip.scale")
        q = self.clip(q, c.qk_l2_clip_threshold) * tau[:, None, None]
        r = self.clip(r, c.rel_l2_clip_threshold) * tau[:, None, None]
        k, v = x_k @ w("wk_dv.weight").T, x_v @ w("wv_dv.weight").T
        if local:
            k, v = self.conv(k, w("k_sconv.weight")), self.conv(v, w("v_sconv.weight"))
        else:
            k = self.dynamic_conv(k, w("k_sconv.static_w"), x_k @ w("k_sconv.weight_proj").T, kv_heads)
            v = self.dynamic_conv(v, w("v_sconv.static_w"), x_v @ w("v_sconv.weight_proj").T, kv_heads)
        k = self.clip(k.view(tokens, kv_heads, dim), c.qk_l2_clip_threshold)
        v = self.unit(v.view(tokens, kv_heads, dim), c.rms_norm_eps)
        bias = r @ w("rel_logits_proj.proj")  # [T, heads, extent]
        extent = bias.shape[-1]
        sink = x_q @ w("attn_sink_proj.weight").T
        out = torch.zeros(tokens, heads, dim, dtype=torch.float64)
        for t in range(tokens):
            first = max(0, t - c.sliding_window_size + 1) if local else 0
            back = t - torch.arange(first, t + 1)  # distances of the visible keys
            for h in range(heads):
                group = h // (heads // kv_heads)
                scores = k[first:t + 1, group] @ q[t, h] * dim ** -0.5
                scores = scores + torch.where(back < extent, bias[t, h, back.clamp(max=extent - 1)], 0.0)
                # The sink is one more softmax column whose value is zero.
                probs = torch.cat((scores, sink[t, h:h + 1])).softmax(0)[:-1]
                out[t, h] = probs @ v[first:t + 1, group]
        return (out.view(tokens, -1) * torch.sigmoid(x_q @ w("gate_attn.weight").T)) @ w("wo_ud.weight").T

    def moe(self, layer, x):
        c = self.c
        scores = torch.sigmoid(x @ self.of(layer, "mlp.gate.weight").T)
        keys = scores + self.of(layer, "mlp.gate.bias")
        w13, w2 = self.experts[layer]
        out, chosen = torch.zeros_like(x), []
        for t in range(x.shape[0]):
            ranked = sorted(range(c.n_routed_experts), key=lambda e: (-float(keys[t, e]), e))[:c.num_experts_per_tok]
            share = scores[t, ranked] / scores[t, ranked].sum() * c.route_scale
            for expert, weight in zip(ranked, share):
                gate, up = (w13[expert] @ x[t]).view(-1, 2).unbind(1)  # interleaved (gate, up) rows
                hidden = (c.mlp_gate_softcap * torch.tanh(gate / c.mlp_gate_softcap) * torch.sigmoid(gate)
                          * c.mlp_up_softcap * torch.tanh(up / c.mlp_up_softcap))
                out[t] += weight * (w2[expert] @ hidden)
            chosen.append(ranked)
        return out, torch.tensor(chosen)

    def controller(self, layer, stream, dest, message):
        def w(name):
            return self.of(layer, f"controllers.{stream}.{name}_weight")

        hidden = torch.cat((self.unit(dest), self.unit(message)), -1) @ w("down").T
        return (1 - torch.tanh(hidden @ w("forget").T)) * dest + torch.tanh(hidden @ w("gain").T) * message

    def layer(self, layer, blocks, partial, bank):
        c = self.c
        names = ("attn_norm",) * 3 if c.is_local(layer) else ("q_input_norm", "k_input_norm", "v_input_norm")
        mixes = [partial] * 3
        if layer:
            mixes = self.mix(blocks, partial, self.w[f"attn_res.attn_queries.{layer - 1}"])
            mixes = [mixes[0]] * 3 if c.is_local(layer) else list(mixes)
        messages = None
        if bank is not None:
            row = 2 * (layer - c.core_start)
            queries, kv = self.w["middle_cycle_cross_pass_queries"], self.w["middle_cycle_cross_pass_kv_queries"]
            order = torch.stack((queries[row], kv[row], kv[row + 1], queries[row + 1]))  # attn, k, v, mlp
            messages = self.mix(bank[:-1], bank[-1], order)
            mixes = [self.controller(layer, stream, mix, message)
                     for stream, mix, message in zip(("attn", "k", "v"), mixes, messages)]
        inputs = [self.norm(mix, self.of(layer, name + ".weight")) for mix, name in zip(mixes, names)]
        out = self.conv(self.attention(layer, *inputs), self.of(layer, "attn_sconv.weight"))
        if layer % c.n_layers_per_attn_res_block == 0:
            blocks = [*blocks, partial]
        else:
            out = partial + out
        x = self.mix(blocks, out, self.w[f"attn_res.mlp_queries.{layer}"])[0]
        if messages is not None:
            x = self.controller(layer, "mlp", x, messages[3])
        mlp, chosen = self.moe(layer, self.norm(x, self.of(layer, "mlp_norm.weight")))
        return blocks, out + self.conv(mlp, self.of(layer, "mlp_sconv.weight")), chosen

    def run(self, tokens, passes):
        """Logits, the input block of every visit followed by the final-norm output, the blocks every pass
        ends in, and the routed experts of every visit."""
        c = self.c
        core = c.core_layers
        taps, routed, ends = [], [], {}

        def visit(layer, blocks, partial, bank=None):
            taps.append(partial)
            blocks, partial, chosen = self.layer(layer, blocks, partial, bank)
            routed.append(chosen)
            return blocks, partial

        blocks, partial = [], self.norm(self.w["embed.weight"][tokens], self.w["embed_norm.weight"])
        for layer in range(core.start):
            blocks, partial = visit(layer, blocks, partial)
        prelude, bank = (blocks, partial), None
        for depth in range(1, passes + 1):
            blocks, partial = prelude
            for layer in core:
                blocks, partial = visit(layer, blocks, partial, bank)
            bank = [*blocks[len(prelude[0]) + 1:], partial]
            ends[depth] = torch.stack(bank)
        for layer in range(core.stop, c.num_hidden_layers):
            blocks, partial = visit(layer, blocks, partial)
        hidden = self.norm(self.mix(blocks, partial, self.w["attn_res.output_query"])[0], self.w["norm.weight"])
        taps.append(hidden)
        cap = c.final_logit_softcapping
        logits = cap * torch.tanh(hidden @ self.w["unembed.weight"][:c.served_vocab_size].T / cap)
        return logits, taps, ends, torch.stack(routed)


# ---------------------------------------------------------------------------
# The whole model against the independent reference


@pytest.mark.parametrize("passes", DEPTHS)
def test_exact_forward_matches_the_independent_reference(checkpoint, exact, tokens, passes):
    """Guards every formula and index of the forward at once: logits, the block at the input of every visit,
    the final-norm output, the end-of-pass blocks and the routed experts equal an FP64 model written in the
    other formulation, for the model configured with 1, 2 and 3 passes."""
    logits, taps, ends, routed = Reference(checkpoint).run(tokens, passes)
    visits = native.num_layer_visits(checkpoint.config, passes)
    assert visits == 4 + 4 * passes and len(taps) == visits + 1
    out = exact.forward(tokens, num_passes=passes, taps=range(visits + 1), return_logits=True)
    readout = out.readouts[passes]
    torch.testing.assert_close(readout.logits, logits, atol=1e-9, rtol=0)
    assert torch.equal(readout.argmax, logits.argmax(-1))
    for visit, expected in enumerate(taps):
        torch.testing.assert_close(out.features[visit], expected, atol=1e-9, rtol=0, msg=f"visit {visit}")
    for depth, expected in ends.items():
        torch.testing.assert_close(out.pass_end[depth], expected, atol=1e-9, rtol=0)
    experts, layers = out.routed_experts(passes)
    assert torch.equal(experts, routed)
    assert layers == [layer for layer, _ in native.visit_layers(checkpoint.config, passes)]


def test_a_one_block_core_reads_a_one_block_bank(tmp_path, tokens):
    """Guards the bank rule where the sources differ: the transformers port reads every core block of the
    previous pass, vLLM exactly two. On the default tiny checkpoint (core = one block of two layers) the forward
    follows the port, a message being that block normalized."""
    checkpoint = weights.load_inkling2_checkpoint(tiny.write_tiny_checkpoint(tmp_path))
    logits, taps, ends, routed = Reference(checkpoint).run(tokens, 3)
    out = NativeInkling2(checkpoint, "exact").forward(tokens, taps=range(11), return_logits=True)
    assert ends[3].shape[0] == out.pass_end[3].shape[0] == 1 and len(taps) == 11
    torch.testing.assert_close(out.readouts[3].logits, logits, atol=1e-9, rtol=0)
    for visit, expected in enumerate(taps):
        torch.testing.assert_close(out.features[visit], expected, atol=1e-9, rtol=0)
    assert torch.equal(out.routed_experts(3)[0], routed)


EDGE_LAYOUTS = {
    # Sliding layers with other head counts and widths than full layers, and one key head per query head.
    "sliding-heads-differ": (dict(num_hidden_layers=8, local_layer_ids=[0, 2, 4, 6], swa_num_attention_heads=2,
                                  swa_num_key_value_heads=2, swa_head_dim=16), (2, 6)),
    # Layer 0 with full attention (q / k / v norms of the embedding, no mix) and a model that ends in its core.
    "full-first-layer-no-coda": (dict(num_hidden_layers=6, local_layer_ids=[1, 3, 5]), (2, 6)),
    # Blocks of three layers, a window and a relative extent of two keys, a conv over two positions.
    "blocks-of-three": (dict(num_hidden_layers=12, n_layers_per_attn_res_block=3, sliding_window_size=2,
                             rel_extent=2, sconv_kernel_size=2, local_layer_ids=[0, 1, 3, 4, 6, 7, 9, 10]), (3, 9)),
}


@pytest.mark.parametrize("layout", EDGE_LAYOUTS)
def test_other_layouts_match_the_reference_in_both_modes(tmp_path, tokens, layout):
    """Guards the places where Mini's own shape hides an index slip (it has equal sliding and full head
    counts, a sliding first layer, a coda and blocks that end in a full layer): on three other layouts the
    two-pass forward equals the reference, and cached decoding in uneven chunks with attention limited to one
    query per chunk gives the same logits."""
    text, (start, end) = EDGE_LAYOUTS[layout]
    loop = dict(core_start=start, core_end=end, num_passes=3, gate_rank=4, is_varloop=False)
    checkpoint = weights.load_inkling2_checkpoint(
        tiny.write_tiny_checkpoint(tmp_path, tiny.tiny_config(loop_config=loop, **text), mxfp8_layers=()))
    tokens = tokens[:12]
    logits, taps, _, routed = Reference(checkpoint).run(tokens, 2)
    model = NativeInkling2(checkpoint, "exact", attention_elements=16)
    out = model.forward(tokens, num_passes=2, taps=range(len(taps)), return_logits=True)
    torch.testing.assert_close(out.readouts[2].logits, logits, atol=1e-9, rtol=0)
    for visit, expected in enumerate(taps):
        torch.testing.assert_close(out.features[visit], expected, atol=1e-9, rtol=0, msg=f"visit {visit}")
    assert torch.equal(out.routed_experts(2)[0], routed)
    cache = Inkling2Cache()
    steps = [model.forward(part, num_passes=2, cache=cache, return_logits=True) for part in tokens.split((1, 4, 2, 5))]
    torch.testing.assert_close(torch.cat([step.readouts[2].logits for step in steps]), logits, atol=1e-9, rtol=0)


def test_visit_ids_follow_the_shared_contract(checkpoint):
    """Guards the visit numbering other packages rely on: with Mini's layout visits 0-5 are the prelude, 6-17 /
    18-29 / 30-41 the three core passes, 42-47 the coda and 48 the final-norm output."""
    config = tiny.tiny_config(num_hidden_layers=24, n_layers_per_attn_res_block=6,
                              local_layer_ids=[i for i in range(24) if i % 6 != 5],
                              loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=4, is_varloop=False))
    mini = weights.Inkling2Config.from_dict(config)
    visits = native.visit_layers(mini)
    assert native.num_layer_visits(mini) == mini.num_layer_visits == 48
    assert visits[:6] == [(i, 0) for i in range(6)] and visits[42:] == [(i, 0) for i in range(18, 24)]
    assert [visits[6 + 12 * p: 18 + 12 * p] for p in range(3)] == [[(i, p + 1) for i in range(6, 18)] for p in range(3)]
    assert (visits[23], visits[41]) == ((11, 2), (17, 3))  # two of the bundled drafter's taps
    assert native.num_layer_visits(mini, 1) == 24 and native.num_layer_visits(checkpoint.config) == 16


# ---------------------------------------------------------------------------
# (a) The two execution modes


@pytest.mark.parametrize("chunks", [None, (5, 1, 9, 13)], ids=["token-by-token", "uneven-chunks"])
def test_incremental_decoding_equals_the_full_sequence_forward(exact, tokens, full, chunks):
    """Guards the explicit caches: decoding from per-(layer, pass) keys, values and conv states gives the
    logits of teacher forcing with masks at every depth, to FP64 rounding, and the same argmax."""
    _, logits = stepwise(exact, tokens, chunks)
    for depth in DEPTHS:
        torch.testing.assert_close(logits[depth], full.readouts[depth].logits, atol=1e-10, rtol=0)
        assert torch.equal(logits[depth].argmax(-1), full.readouts[depth].argmax)


def test_chunked_attention_and_chunked_logits_change_nothing(checkpoint, tokens, full):
    """Guards the memory bounds: with attention limited to 64 scores per chunk (one query at a time) and the
    head to 256 logits (two rows), every readout is the one of the unchunked forward."""
    model = NativeInkling2(checkpoint, "exact", attention_elements=64, logit_elements=256)
    out = model.forward(tokens, depths=DEPTHS)
    for depth in DEPTHS:
        readout = out.readouts[depth]
        torch.testing.assert_close(readout.hidden, full.readouts[depth].hidden, atol=1e-10, rtol=0)
        assert torch.equal(readout.argmax, full.readouts[depth].argmax) and readout.logits is None


def test_prefill_and_stepwise_caches_hold_the_same_state(exact, tokens):
    """(i) Guards the carried state itself: after 28 tokens the conv inputs, keys and values that token-by-token
    decoding left in every plane are those one prefill leaves; a sliding layer keeps window - 1 = 7 keys and a
    full layer all 28; there are 2 prelude, 12 core and 6 coda planes for 3 passes read out at 3 depths."""
    prefill = Inkling2Cache()
    exact.forward(tokens, depths=DEPTHS, cache=prefill)
    steps, _ = stepwise(exact, tokens)
    assert prefill.length == steps.length == TOKENS
    expected = {(i, 0) for i in (0, 1)} | {(i, p) for i in range(2, 6) for p in DEPTHS} | {
        (i, d) for i in (6, 7) for d in DEPTHS}
    assert set(prefill.planes) == set(steps.planes) == expected
    for key, plane in prefill.planes.items():
        other = steps.planes[key]
        local = exact.config.is_local(key[0])
        held = (21, 7) if local else (0, 28)  # first cached position and number of cached keys
        assert (plane.first, plane.keys.shape[0]) == (other.first, other.keys.shape[0]) == held
        assert set(plane.conv) == set(other.conv) == {"k", "v", "attn", "mlp"}
        for stream, history in plane.conv.items():
            assert history.shape[0] == exact.config.sconv_kernel_size - 1
            torch.testing.assert_close(other.conv[stream], history, atol=1e-10, rtol=0)
        torch.testing.assert_close(other.keys, plane.keys, atol=1e-10, rtol=0)
        torch.testing.assert_close(other.values, plane.values, atol=1e-10, rtol=0)


def test_cache_planes_own_only_the_state_they_keep(exact, tokens):
    """Guards the memory of cached decoding: after a prefill, a plane's conv histories and a sliding layer's
    window of keys and values are copies, not views that keep every activation of the prefill alive (on Mini
    at 16K tokens such views hold about 26 GB)."""
    cache = Inkling2Cache()
    exact.forward(tokens, depths=DEPTHS, cache=cache)
    exact.forward(tokens[:1], depths=DEPTHS, cache=cache)
    for key, plane in cache.planes.items():
        for tensor in (plane.keys, plane.values, *plane.conv.values()):
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(), key


def test_cache_is_bound_to_its_passes_and_depths(exact, tokens, full):
    """Guards against reading planes written by another configuration or for other positions: a cache refuses
    a different (num_passes, depths), and a plane that does not end where the new tokens start. A refused call
    leaves the cache as it was, so decoding continues from the right position with the full-sequence logits."""
    cache = Inkling2Cache()
    exact.forward(tokens[:3], depths=DEPTHS, cache=cache)
    with pytest.raises(ValueError, match="num_passes, depths"):
        exact.forward(tokens[3:4], cache=cache)
    assert cache.length == 3
    cache.length = 5
    with pytest.raises(ValueError, match="ends before position 3, not 5"):
        exact.forward(tokens[5:6], depths=DEPTHS, cache=cache)
    cache.length = 3
    rest = exact.forward(tokens[3:], depths=DEPTHS, cache=cache, return_logits=True)
    for depth in DEPTHS:
        torch.testing.assert_close(rest.readouts[depth].logits, full.readouts[depth].logits[3:], atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# (b) Causality


def test_a_later_token_does_not_change_earlier_outputs(exact, tokens, full):
    """Guards the causal masks and the causal convs: replacing token 17 leaves every output of positions 0-16
    as it was, at every depth, and changes position 17."""
    changed = tokens.clone()
    changed[17] = (changed[17] + 1) % 100
    out = exact.forward(changed, depths=DEPTHS, taps=range(17), return_logits=True)
    for depth in DEPTHS:
        a, b = out.readouts[depth].logits, full.readouts[depth].logits
        torch.testing.assert_close(a[:17], b[:17], atol=1e-12, rtol=0)
        assert (a[17] - b[17]).abs().max() > 1e-3
        torch.testing.assert_close(out.pass_end[depth][:, :17], full.pass_end[depth][:, :17], atol=1e-12, rtol=0)
    for visit, feature in full.features.items():
        torch.testing.assert_close(out.features[visit][:17], feature[:17], atol=1e-12, rtol=0)
    assert torch.equal(out.experts[:, :17], full.experts[:, :17])


# ---------------------------------------------------------------------------
# (c), (d) Window and relative bias, on the attention function with readable inputs


def attention_weights(keys, window, rel_proj=None, query=None):
    """Attention probabilities [T, N] of one head over keys at positions 0..N-1: one-hot values, an open gate
    and an infinitely low sink turn the output into the softmax itself."""
    count, dim = keys.shape[0], keys.shape[-1]
    values = torch.zeros(count, 1, dim, dtype=torch.float64)
    values[torch.arange(count), 0, torch.arange(count)] = 1.0
    rel_proj = torch.zeros(1, 4, dtype=torch.float64) if rel_proj is None else rel_proj
    query = torch.randn(count, 1, dim, dtype=torch.float64) if query is None else query
    out = native.relative_attention(
        query, keys, values, torch.ones(count, 1, 1, dtype=torch.float64), rel_proj,
        torch.full((count, 1, dim), float("inf"), dtype=torch.float64),
        torch.full((count, 1), float("-inf"), dtype=torch.float64), window=window, max_elements=40)
    return out[:, 0, :count]


def test_a_sliding_layer_sees_itself_and_window_minus_one_keys():
    """(c) Guards the window: with window 4 a query attends distances 0..3 only, so a key 4 or more positions
    back cannot change it, while full attention sees every earlier key. max_elements 40 also splits the 12
    queries into several chunks."""
    torch.manual_seed(1)
    keys = torch.randn(12, 1, 16, dtype=torch.float64)
    query = torch.randn(12, 1, 16, dtype=torch.float64)
    distance = torch.arange(12)[:, None] - torch.arange(12)[None, :]
    sliding, everything = attention_weights(keys, 4, query=query), attention_weights(keys, None, query=query)
    assert torch.equal(sliding > 0, (distance >= 0) & (distance < 4))
    assert torch.equal(everything > 0, distance >= 0)
    for weights_ in (sliding, everything):
        torch.testing.assert_close(weights_.sum(-1), torch.ones(12, dtype=torch.float64))
    moved = keys.clone()
    moved[2] += 1.0
    assert torch.equal(attention_weights(moved, 4, query=query)[6:], sliding[6:])  # 6 - 2 = 4: out of the window
    assert not torch.equal(attention_weights(moved, 4, query=query)[5], sliding[5])
    assert (attention_weights(moved, None, query=query)[2:] != everything[2:]).any(-1).all()


def test_relative_bias_ends_at_the_extent():
    """(d) Guards the bias rule: with zero queries the weights are the softmax of the bias alone. Full
    attention gives distances 0..4 their bias and keeps distance 5 and beyond with zero bias; a sliding layer
    (extent = window = 5) gives the same biases and masks distance 5 and beyond."""
    profile = torch.tensor([[0.5, -1.0, 2.0, 0.25, 1.5]], dtype=torch.float64)  # bias by distance, extent 5
    keys = torch.randn(12, 1, 16, dtype=torch.float64)
    query = torch.zeros(12, 1, 16, dtype=torch.float64)
    full_bias = torch.cat((profile[0], torch.zeros(7, dtype=torch.float64)))  # distance 0..11 of the query at 11
    expected = full_bias.softmax(0).flip(0)  # key j is at distance 11 - j
    torch.testing.assert_close(attention_weights(keys, None, profile, query)[11], expected)
    expected = torch.cat((torch.zeros(7, dtype=torch.float64), profile[0].softmax(0).flip(0)))
    torch.testing.assert_close(attention_weights(keys, 5, profile, query)[11], expected)
    # At the boundary itself: the query at position 5 sees key 0 at distance 5, the first one past the extent.
    row = attention_weights(keys, None, profile, query)[5, :6]
    torch.testing.assert_close(row, torch.tensor([0.0, 1.5, 0.25, 2.0, -1.0, 0.5], dtype=torch.float64).softmax(0))
    assert attention_weights(keys, 5, profile, query)[5, 0] == 0


def test_key_heads_are_shared_by_consecutive_query_heads():
    """Guards grouped-query attention: query head h reads key/value head h // group, here heads 0-1 -> 0 and
    2-3 -> 1."""
    torch.manual_seed(2)
    keys, values = torch.randn(3, 2, 4, dtype=torch.float64), torch.randn(3, 2, 4, dtype=torch.float64)
    query = torch.randn(1, 1, 4, dtype=torch.float64).expand(1, 4, 4)
    zeros = torch.zeros(1, 4, 1, dtype=torch.float64)
    out = native.relative_attention(query, keys, values, zeros, torch.zeros(1, 4, dtype=torch.float64),
                                    torch.zeros(1, 4, 4, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64),
                                    first_query=2)
    assert torch.equal(out[0, 0], out[0, 1]) and torch.equal(out[0, 2], out[0, 3])
    assert not torch.allclose(out[0, 0], out[0, 2])


# ---------------------------------------------------------------------------
# (e) Per-depth readouts


@pytest.mark.parametrize("precision", ["exact", "served"])
def test_a_depth_readout_is_the_model_with_that_many_passes(checkpoint, tokens, precision):
    """Guards the per-depth coda planes: every readout of one 3-pass forward equals, bit for bit, the output
    of a forward configured with that many passes, together with its features and routed experts, so an early
    readout neither reads nor leaves state in the plane of another depth."""
    model = NativeInkling2(checkpoint, precision)
    out = model.forward(tokens, depths=DEPTHS, taps=range(17), return_logits=True)
    for passes in DEPTHS:
        visits = native.num_layer_visits(checkpoint.config, passes)
        alone = model.forward(tokens, num_passes=passes, taps=range(visits + 1), return_logits=True)
        mine, readout = alone.readouts[passes], out.readouts[passes]
        assert torch.equal(mine.logits, readout.logits) and torch.equal(mine.hidden, readout.hidden)
        assert torch.equal(mine.argmax, readout.argmax) and torch.equal(mine.argmax, mine.logits.argmax(-1))
        assert torch.equal(alone.features[visits], readout.hidden)
        for visit in range(2 + 4 * passes):  # the prelude and the core visits have the same ids in both
            assert torch.equal(alone.features[visit], out.features[visit])
        assert torch.equal(alone.pass_end[passes], out.pass_end[passes])
        assert torch.equal(alone.routed_experts(passes)[0], out.routed_experts(passes)[0])
    assert not torch.equal(out.readouts[1].logits, out.readouts[3].logits)


def test_a_fourth_pass_leaves_the_first_three_depths_unchanged(exact, tokens, full):
    """Guards the pass count as an argument beyond the checkpoint's own (R = 4 on a 3-pass checkpoint): a pass
    does not depend on how many passes follow, so depths 1-3 of a 4-pass forward are bit for bit those of the
    3-pass forward, depth 4 is the forward configured with 4 passes, and the fourth pass still changes the
    output."""
    out = exact.forward(tokens, num_passes=4, depths=(1, 2, 3, 4), return_logits=True)
    for depth in DEPTHS:
        assert torch.equal(out.readouts[depth].logits, full.readouts[depth].logits)
        assert torch.equal(out.pass_end[depth], full.pass_end[depth])
        assert torch.equal(out.routed_experts(depth)[0], full.routed_experts(depth)[0])
    alone = exact.forward(tokens, num_passes=4, return_logits=True)
    assert torch.equal(alone.readouts[4].logits, out.readouts[4].logits)
    assert (out.readouts[4].logits - out.readouts[3].logits).abs().max() > 1e-3
    assert out.experts.shape[0] == 2 + 4 * 4 and len(out.routed_experts(4)[1]) == native.num_layer_visits(
        exact.config, 4) == 20


def test_readout_requests_are_validated(exact, tokens):
    """Guards the request contract: depths lie in 1..num_passes, a coda tap needs the final depth, ids come
    from the served vocabulary, and any positive pass count runs (here 4 on a 3-pass checkpoint)."""
    with pytest.raises(ValueError, match="depths"):
        exact.forward(tokens, depths=(4,))
    with pytest.raises(ValueError, match="depths"):
        exact.forward(tokens, num_passes=0)
    with pytest.raises(ValueError, match="need a readout at depth 3"):
        exact.forward(tokens, depths=(1,), taps=(14,))
    with pytest.raises(ValueError, match="taps"):
        exact.forward(tokens, taps=(17,))
    with pytest.raises(ValueError, match="token ids"):
        exact.forward(torch.tensor([100]))
    with pytest.raises(ValueError, match="nonempty"):
        exact.forward(torch.tensor([], dtype=torch.long))
    out = exact.forward(tokens[:4], num_passes=4, depths=(2, 4), taps=(20,))
    assert set(out.readouts) == {2, 4} and set(out.pass_end) == {1, 2, 3, 4} and out.experts.shape[0] == 18
    assert torch.equal(out.features[20], out.readouts[4].hidden)


# ---------------------------------------------------------------------------
# (f) Controllers


def test_the_first_pass_has_no_controller(snapshot, exact, tokens, full):
    """Guards the pass-1 bypass and the restart: zeroing every controller weight leaves the depth-1 readout
    bit-identical and changes depths 2 and 3; and a controller that returns its destination unchanged (zero
    gain and forget) makes every later pass repeat the first one exactly, because a pass restarts from the
    prelude's state rather than from the previous pass's output."""
    def zero(name, tensor):
        return torch.zeros_like(tensor) if ".controllers." in name else None

    silent = NativeInkling2(edited(snapshot, zero), "exact").forward(tokens, depths=DEPTHS, return_logits=True)
    assert torch.equal(silent.readouts[1].logits, full.readouts[1].logits)
    assert torch.equal(silent.pass_end[1], full.pass_end[1])
    for depth in (2, 3):
        assert (silent.readouts[depth].logits - full.readouts[depth].logits).abs().max() > 1e-2
        assert torch.equal(silent.readouts[depth].logits, silent.readouts[1].logits)
        assert torch.equal(silent.pass_end[depth], silent.pass_end[1])
    experts = silent.experts[2:].view(3, 4, TOKENS, -1)
    assert torch.equal(experts[1], experts[0]) and torch.equal(experts[2], experts[0])


# ---------------------------------------------------------------------------
# (g) Taps


def test_taps_are_the_running_block_at_a_visit_input(checkpoint, tokens, full):
    """Guards the tap semantics upstream defines. Visit 0 reads the normed embedding; the first layer of a
    block reads the block that just completed (visit 2: the prelude block, visit 4p: B2 of pass p, visit 14: B3
    of pass 3); every pass restart (visits 6 and 10) reads the prelude block again, so the state a pass ends in
    is only in pass_end; the last id is the final-norm output."""
    features, ends = full.features, full.pass_end
    rows = checkpoint.tensors[weights.EMBED_NAME].double()[tokens]
    embedding = rows / torch.sqrt(rows.square().mean(-1, keepdim=True) + 1e-6)
    embedding = embedding * checkpoint.tensors[weights.LLM_PREFIX + "embed_norm.weight"].double()
    assert set(features) == set(range(17)) and ends[1].shape == (2, TOKENS, checkpoint.config.hidden_size)
    torch.testing.assert_close(features[0], embedding, atol=1e-12, rtol=0)
    for restart in (6, 10):
        assert torch.equal(features[restart], features[2])
    for depth in DEPTHS:
        assert torch.equal(features[4 * depth], ends[depth][0])  # layer 4 opens the second core block
        assert not torch.equal(ends[depth][1], features[2])
    assert torch.equal(features[14], ends[3][1]) and torch.equal(features[16], full.readouts[3].hidden)
    assert not torch.equal(ends[1][1], ends[2][1]) and not torch.equal(ends[2][1], ends[3][1])
    # Inside a block the tap grows by one layer's output.
    assert not torch.equal(features[3], features[2]) and not torch.equal(features[5], features[4])


# ---------------------------------------------------------------------------
# (h) MoE


def test_router_breaks_ties_low_and_weights_exclude_the_bias():
    """Guards the routing rule: experts are ranked by sigmoid(logit) + bias with ties to the lower index, best
    first; the weights are the sigmoids alone, renormalized over the selection and summing to route_scale."""
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 0.0]], dtype=torch.float64)
    bias = torch.tensor([0.0, 0.25, 0.25, 0.25, -5.0, 0.3, 0.0, 0.0], dtype=torch.float64)
    ids, share = native.route_experts(logits, bias, 3, 8.0)
    assert ids.tolist() == [[1, 2, 3]]  # 0.75 three times, ahead of expert 5 (0.569) and of expert 4 (bias -5)
    torch.testing.assert_close(share, torch.full((1, 3), 8 / 3, dtype=torch.float64))
    ids, share = native.route_experts(logits, torch.zeros(8, dtype=torch.float64), 2, 2.0)
    assert ids.tolist() == [[4, 0]]  # the largest sigmoid, then the lowest index of the tied rest
    sigmoid = torch.sigmoid(torch.tensor([3.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(share, (sigmoid / sigmoid.sum() * 2.0)[None])
    torch.manual_seed(3)
    logits, bias = torch.randn(50, 16, dtype=torch.float64), 0.1 * torch.randn(16, dtype=torch.float64)
    ids, share = native.route_experts(logits, bias, 8, 8.0)  # the pairwise summation of eight experts
    assert torch.equal(ids, (torch.sigmoid(logits) + bias).topk(8, dim=-1).indices)
    selected = torch.sigmoid(logits.gather(1, ids))
    torch.testing.assert_close(share, selected / selected.sum(-1, keepdim=True) * 8.0)
    torch.testing.assert_close(share.sum(-1), torch.full((50,), 8.0, dtype=torch.float64))


def test_tied_experts_are_routed_in_index_order_inside_the_model(snapshot, tokens):
    """Guards the tie rule end to end: layer 3's experts 4-6 get the same router row and the same bias, large
    enough to win, so every token of each of its three passes routes to experts 4 and 5 and never to 6."""
    def tie(name, tensor):
        if name == "layers.3.mlp.gate.weight":
            return torch.cat((tensor[:4], tensor[4:5].expand(3, -1), tensor[7:]))
        if name == "layers.3.mlp.gate.bias":
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0])

    out = NativeInkling2(edited(snapshot, tie), "served").forward(tokens)
    experts, layers = out.routed_experts(3)
    rows = [visit for visit, layer in enumerate(layers) if layer == 3]
    assert rows == [3, 7, 11] and (experts[rows] == torch.tensor([4, 5])).all()


def test_padded_vocabulary_rows_never_win(snapshot, exact, tokens, full):
    """Guards the served vocabulary: logits cover the 100 served ids of the 112-row table, so padding rows
    made to dominate every position cannot be predicted (over the whole table they would be)."""
    hidden = full.readouts[3].hidden

    def loud(name, tensor):
        if name == "unembed.weight":
            table = tensor.clone()
            table[100:] = (40.0 * hidden[0] / hidden[0].norm()).to(tensor.dtype)
            return table

    patched = edited(snapshot, loud)
    model = NativeInkling2(patched, "exact")
    out = model.forward(tokens, return_logits=True)
    table = patched.tensors[weights.UNEMBED_NAME].double()
    assert (out.readouts[3].hidden[0] @ table.T).argmax() >= 100
    assert out.readouts[3].logits.shape == (TOKENS, 100) and int(out.readouts[3].argmax.max()) < 100
    assert torch.equal(out.readouts[3].logits, full.readouts[3].logits)  # the padding rows are not read at all
    values, ids = model.readout_top(out.readouts[3].hidden, 2)
    assert torch.equal(ids[:, 0], out.readouts[3].argmax) and torch.equal(values, out.readouts[3].logits.topk(2).values)
    loud_states = 50.0 * out.readouts[3].hidden  # raw logits far beyond the cap
    capped = model.readout_logits(loud_states)
    torch.testing.assert_close(capped, 30.0 * torch.tanh(loud_states @ table[:100].T / 30.0))
    assert 29.0 < float(capped.abs().max()) <= 30.0


@pytest.mark.parametrize("precision", ["exact", "served"])
def test_packed_experts_and_released_layers_give_the_same_forward(snapshot, checkpoint, tokens, precision):
    """Guards the two expert containers, the checkpoint dtype and the memory option: a checkpoint loaded in
    FP32 with packed experts (only the routed ones are dequantized) gives the result of the BF16 one with dense
    experts bit for bit, and release_experts leaves no expert layer loaded."""
    dense = NativeInkling2(checkpoint, precision).forward(tokens, depths=DEPTHS, return_logits=True)
    model = NativeInkling2.from_pretrained(snapshot, precision=precision, dtype=torch.float32, experts="packed")
    packed = model.checkpoint
    out = model.forward(tokens, depths=DEPTHS, return_logits=True, release_experts=True)
    assert (packed.expert_mode, packed.dtype, packed.loaded_expert_layers) == ("packed", torch.float32, ())
    for depth in DEPTHS:
        assert torch.equal(out.readouts[depth].logits, dense.readouts[depth].logits)
    with pytest.raises(ValueError, match="experts"):
        NativeInkling2(weights.load_inkling2_checkpoint(snapshot, experts="none"))


# ---------------------------------------------------------------------------
# (i) Short conv


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_conv_state_carried_across_steps_equals_the_full_sequence_conv(dynamic):
    """Guards the conv state and the tap order: the conv over 11 positions equals the definition evaluated tap
    by tap (static: w[c, k] on x[t - 3 + k]; dynamic: (w[l, c] + dyn[t, head, l]) on x[t - l]; zero before
    position 0; skip and scale), and feeding one position or uneven chunks with the returned history gives the
    same values bit for bit."""
    torch.manual_seed(5)
    tokens, channels, heads, kernel = 11, 6, 2, 4
    x = torch.randn(tokens, channels, dtype=torch.float64)
    weight = torch.randn((kernel, channels) if dynamic else (channels, kernel), dtype=torch.float64)
    deltas = torch.randn(tokens, heads, kernel, dtype=torch.float64) if dynamic else None
    expected = torch.zeros_like(x)
    for t in range(tokens):
        for c in range(channels):
            total = 0.0
            for lag in range(kernel):
                if t - lag < 0:
                    continue
                tap = weight[lag, c] + deltas[t, c // 3, lag] if dynamic else weight[c, kernel - 1 - lag]
                total += float(tap) * float(x[t - lag, c])
            expected[t, c] = (total + float(x[t, c])) * (2 * kernel + 1 if dynamic else kernel + 1) ** -0.5
    y, history = native.short_conv(x, weight, None, deltas)
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=0)
    assert torch.equal(history, x[-3:])
    for sizes in ((1,) * tokens, (2, 1, 5, 3)):
        parts, carried, start = [], None, 0
        for size in sizes:
            part, carried = native.short_conv(x[start:start + size], weight, carried,
                                              None if deltas is None else deltas[start:start + size])
            parts.append(part)
            start += size
        assert torch.equal(torch.cat(parts), y) and torch.equal(carried, history)


# ---------------------------------------------------------------------------
# Precision policies


def test_precision_policies_and_the_blackwell_placeholder(checkpoint):
    """Guards the policy table: served rounds to BF16 in FP32, exact is FP64 without rounding, and the
    Blackwell activation quantization is an explicit placeholder that refuses to run."""
    served, exact_ = native.PRECISIONS["served"], native.PRECISIONS["exact"]
    assert (served.dtype, served.storage, served.activation_quantization) == (torch.float32, torch.bfloat16, False)
    assert (exact_.dtype, exact_.storage) == (torch.float64, None)
    assert {p.name: (p.dtype, p.storage) for p in native.PRECISIONS.values()} == {
        "served": (torch.float32, torch.bfloat16), "served-fp64": (torch.float64, torch.bfloat16),
        "exact": (torch.float64, None), "exact-fp32": (torch.float32, None)}
    x = torch.tensor([1.0 + 2.0 ** -10, 3.0])
    assert served.store(x).dtype == torch.float32 and served.store(x).tolist() == [1.0, 3.0]
    assert native.PRECISIONS["exact-fp32"].store(x) is x
    with pytest.raises(NotImplementedError, match="Blackwell"):
        NativeInkling2(checkpoint, Precision("blackwell", torch.float32, torch.bfloat16, activation_quantization=True))


def test_served_states_are_bf16_values_and_logits_stay_fp32(checkpoint, tokens):
    """Guards the rounding points that leave the forward: every state the served policy returns is exactly a
    BF16 value (held in FP32), while the logits come from an FP32 accumulation that is not rounded."""
    out = NativeInkling2(checkpoint, "served").forward(tokens, depths=DEPTHS, taps=range(17), return_logits=True)
    states = [*out.features.values(), *out.pass_end.values(), *(r.hidden for r in out.readouts.values())]
    for state in states:
        assert state.dtype == torch.float32 and torch.equal(state.bfloat16().float(), state)
    logits = out.readouts[3].logits
    assert logits.dtype == torch.float32 and not torch.equal(logits.bfloat16().float(), logits)


@pytest.mark.parametrize("precision", ["served", "served-fp64", "exact-fp32"])
def test_rounded_policies_stay_near_the_exact_forward(snapshot, tokens, precision):
    """Guards against an extra or misplaced rounding: with routing fixed (no expert can flip), the logits of a
    rounded policy stay within BF16 noise of the exact ones at every depth, and FP32 without rounding within
    FP32 noise."""
    checkpoint = fixed_routing(snapshot)
    reference = NativeInkling2(checkpoint, "exact").forward(tokens, depths=DEPTHS, return_logits=True)
    out = NativeInkling2(checkpoint, precision).forward(tokens, depths=DEPTHS, return_logits=True)
    bound = 1e-4 if precision == "exact-fp32" else BF16_NOISE
    assert torch.equal(out.experts, reference.experts)
    for depth in DEPTHS:
        a, b = out.readouts[depth].logits.double(), reference.readouts[depth].logits
        assert (a - b).abs().max() < bound
        rows = confident(b, 2 * bound)
        assert rows.sum() >= TOKENS // 4 and torch.equal(a.argmax(-1)[rows], b.argmax(-1)[rows])


def test_fp64_between_the_rounding_points_makes_the_states_chunk_invariant(checkpoint, tokens):
    """Guards the rounding points as a policy of their own: rounded from FP64, every BF16 state of token-by-token
    decoding equals the one of the full-sequence forward bit for bit (free routing, all depths), which FP32
    arithmetic gives only up to BF16 noise."""
    model = NativeInkling2(checkpoint, "served-fp64")
    out = model.forward(tokens, depths=DEPTHS, taps=range(17))
    cache, steps = Inkling2Cache(), []
    for token in tokens.split(1):
        steps.append(model.forward(token, depths=DEPTHS, taps=range(17), cache=cache))
    for depth in DEPTHS:
        assert torch.equal(torch.cat([step.readouts[depth].hidden for step in steps]), out.readouts[depth].hidden)
        assert torch.equal(torch.cat([step.pass_end[depth] for step in steps], 1), out.pass_end[depth])
        assert torch.equal(torch.cat([step.readouts[depth].argmax for step in steps]), out.readouts[depth].argmax)
    for visit, feature in out.features.items():
        assert torch.equal(torch.cat([step.features[visit] for step in steps]), feature)
    assert torch.equal(torch.cat([step.experts for step in steps], 1), out.experts)


def test_served_incremental_decoding_matches_the_full_sequence(snapshot, tokens):
    """(a) in the served policy, with routing fixed: decoding step by step and teacher forcing differ only by
    BF16 values that an FP32 last bit moved, and name the same token wherever the top two logits are apart."""
    model = NativeInkling2(fixed_routing(snapshot), "served")
    out = model.forward(tokens, depths=DEPTHS, return_logits=True)
    _, logits = stepwise(model, tokens)
    for depth in DEPTHS:
        full_logits = out.readouts[depth].logits
        assert (logits[depth] - full_logits).abs().max() < BF16_NOISE
        rows = confident(full_logits, 2 * BF16_NOISE)
        assert rows.sum() >= TOKENS // 4
        assert torch.equal(logits[depth].argmax(-1)[rows], full_logits.argmax(-1)[rows])


# ---------------------------------------------------------------------------
# Opt-in: the released weights, and the transformers port as an oracle

# "What is the capital of France?" answered with the open assistant text "The capital of France is", rendered
# by training.common.chat.render_inkling2 with effort "none".
CAPITAL_PROMPT = [200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 1775, 149884,
                  149876, 151451, 149879, 151194, 30, 200010, 200001, 200004, 371, 151451, 149879, 151194, 149884]
PARIS_ANSWER = [152717, 13, 200010, 200006]


def mini_snapshot():
    explicit = os.environ.get("INKLING2_SNAPSHOT")
    if explicit:
        path = Path(explicit)
    else:
        from huggingface_hub.constants import HF_HUB_CACHE

        path = Path(HF_HUB_CACHE) / ("models--" + MINI[0].replace("/", "--")) / "snapshots" / MINI[1]
    if not ((path / "config.json").is_file() and (path / tiny.SHARD_NAME).is_file()):
        pytest.skip(f"no local snapshot of {MINI[0]} (set INKLING2_SNAPSHOT)")
    return path


HF_ORACLE = """
import json, sys
import torch
from transformers import Inkling2LoopForCausalLM

directory, tokens, passes, out = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4]
model = Inkling2LoopForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
logits = {}
for count in passes:
    model.config.loop_num_passes = count
    with torch.no_grad():
        logits[count] = model(input_ids=torch.tensor([tokens]), use_cache=False).logits[0].double().tolist()
with open(out, "w") as handle:
    json.dump(logits, handle)
"""


def test_transformers_port_agrees_on_the_tiny_checkpoint(tmp_path, tokens):
    """(j) Oracle from the other primary source, skipped unless INKLING2_HF_ORACLE_PYTHON names an interpreter
    with transformers PR #51050 (head 6173a2f) installed. That model runs in a subprocess in FP32, once per
    pass count (it reads loop_num_passes on every call); the exact forward must agree within 5e-3 on logits of
    order 4: the port's FP32 arithmetic and its folded FP32 expert weights are the only differences left, far
    below any structural one. Not run where this file was written."""
    python = os.environ.get("INKLING2_HF_ORACLE_PYTHON")
    if not python:
        pytest.skip("set INKLING2_HF_ORACLE_PYTHON to an interpreter with transformers PR #51050")
    # The port takes the attention type of a layer from layer_types (default: full attention every sixth layer).
    kinds = ["hybrid" if layer % 2 else "hybrid_sliding" for layer in range(8)]
    directory = tiny.write_tiny_checkpoint(tmp_path / "model", mini_like_config(layer_types=kinds), mxfp8_layers=(6, 7))
    out = tmp_path / "logits.json"
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    work = tmp_path / "empty"
    work.mkdir()
    subprocess.run([python, "-c", HF_ORACLE, str(directory), json.dumps(tokens.tolist()), json.dumps(list(DEPTHS)),
                    str(out)], check=True, cwd=work, env=env, timeout=900)
    theirs = json.loads(out.read_text())
    model = NativeInkling2(weights.load_inkling2_checkpoint(directory), "exact")
    for passes in DEPTHS:
        mine = model.forward(tokens, num_passes=passes, return_logits=True).readouts[passes].logits
        expected = torch.tensor(theirs[str(passes)], dtype=torch.float64)
        torch.testing.assert_close(mine, expected, atol=5e-3, rtol=0)
        rows = confident(mine, 1e-2)
        assert torch.equal(mine.argmax(-1)[rows], expected.argmax(-1)[rows])


def test_released_mini_weights_answer_a_question():
    """Guards the forward on the real function, skipped without a local snapshot of Inkling-2-Mini. On the
    released weights (packed experts, CPU, about 30 s on 48 cores) the three-pass model continues "The capital
    of France is" with " Paris", ".", <|end_message|> and <|content_model_end_sampling|>, the EOS of the shared
    contract; the first token comes from the prefill and the rest from cached decoding, and the tap at Mini's
    visit 48 is its final-norm output."""
    checkpoint = weights.load_inkling2_checkpoint(mini_snapshot(), experts="packed")
    model = NativeInkling2(checkpoint, "served")
    assert native.num_layer_visits(checkpoint.config) == 48 and checkpoint.config.eos_token_id == PARIS_ANSWER[-1]
    cache = Inkling2Cache()
    out = model.forward(CAPITAL_PROMPT, taps=(5, 23, 41, 48), cache=cache)
    assert set(out.features) == {5, 23, 41, 48} and torch.equal(out.features[48], out.readouts[3].hidden)
    assert out.experts.shape == (42, len(CAPITAL_PROMPT), 8) and out.readouts[3].experts.shape[0] == 6
    generated = [int(out.readouts[3].argmax[-1])]
    for _ in range(3):
        generated.append(int(model.forward(generated[-1:], cache=cache).readouts[3].argmax))
    assert generated == PARIS_ANSWER
    assert cache.length == len(CAPITAL_PROMPT) + 3 and len(cache.planes) == 48
