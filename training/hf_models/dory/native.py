"""The original (per-loop KV) Dory1.0-4B for teacher-forced drafter features and labels.

Dory is block-wise looped: an input block of 16 sublayers runs once, a recurrent block of 40 runs
R times on h = norm(s + x) with the input-block output x and the loop state s (each loop reads its
own KV), and an output block of 16 reads out a depth d from the loop-d state s_d. features() runs
the recurrent block once through depths 1..R and then the output block on s_1..s_R as R
independent sequences in one batch, so every depth-d readout is that of the d-loop model.

The math and its BF16 rounding points (projections, RoPE, nGPT norms, attention output, the head's
FP32 scale after the BF16 GEMM) follow the vLLM fork's vllm/model_executor/models/dory.py.
Parameter names are those of nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16. attention="flash" uses
the fork's FlashAttention (version 3 on Hopper, 2 elsewhere), which keeps 16K-token features
linear in memory; "sdpa" is the masked reference used on CPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ...sharekv.flash import flash_version

EPS = 1e-6
LOOPS = 4  # the paper's R; the checkpoint's own n_recurrent_loops is 2
LAYOUT = (16, 40, 16)  # input, recurrent and output sublayers of Dory1.0-4B
# Drafter feature taps, physical sublayer indices: 3 in the input block (the same in every loop),
# 19, 35 and 51 in the recurrent block of loop d, and 67 in the output block on s_d.
FEATURE_LAYERS = (3, 19, 35, 51, 67)


def norm(x):
    """nGPT unit normalization: FP32 accumulation, result in the input dtype."""
    y = x.float()
    return (y * torch.rsqrt(y.square().sum(-1, keepdim=True) + EPS)).to(x.dtype)


def unrolled_layers(config, loops=LOOPS):
    """Sublayers on the unrolled input + loops x recurrent + output path (a config.json dict)."""
    return config["n_input_layers"] + loops * config["n_recurrent_layers"] + config["n_output_layers"]


def feature_layer_ids(config, depth=LOOPS, layers=FEATURE_LAYERS):
    """Physical taps -> indices on the unrolled path of `depth` loops (vLLM's DFlash target_layer_ids).

    config is a config.json dict. For depth R = 4 the taps are [3, 139, 155, 171, 187] of 192.
    """
    if depth < 1:
        raise ValueError("Dory depth must be positive")
    if tuple(config[k] for k in ("n_input_layers", "n_recurrent_layers", "n_output_layers")) != LAYOUT:
        raise ValueError(f"The Dory feature taps require the {'/'.join(map(str, LAYOUT))} block layout")
    n_in, n_rec = config["n_input_layers"], config["n_recurrent_layers"]
    return [i if i < n_in else i + (depth - 1) * n_rec for i in layers]


def validate_config(c):
    if (c.num_hidden_layers != len(c.layer_types)
            or c.n_input_layers + c.n_recurrent_layers + c.n_output_layers != c.num_hidden_layers
            or min(c.n_input_layers, c.n_recurrent_layers, c.n_output_layers) <= 0):
        raise ValueError("Invalid Dory input/recurrent/output partition")
    if (c.num_attention_heads % c.num_key_value_heads or c.head_dim % 2
            or c.initializer_range <= 0 or c.sliding_window <= 0):
        raise ValueError("Unsupported Dory attention dimensions or scales")
    if c.hidden_act != "silu" or c.attention_bias or c.mlp_bias or c.tie_word_embeddings:
        raise ValueError("Dory requires bias-free SiLU and untied embeddings")
    for kind in set(c.layer_types) - {"mlp"}:
        if kind not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported Dory sublayer: {kind}")
        rope = c.rope_parameters[kind]
        if rope["rope_type"] != "default" or rope.get("partial_rotary_factor", 1) != 1 or rope["rope_theta"] <= 0:
            raise ValueError("Dory features require full default RoPE")


class ScaledHead(nn.Linear):
    """Full-vocabulary head: BF16 GEMM, then the per-token FP32 scale logit_scale / initializer_range."""

    def __init__(self, c):
        super().__init__(c.hidden_size, c.vocab_size, bias=False)
        self.logit_scale = nn.Parameter(torch.empty(c.vocab_size))
        self.initializer_range = c.initializer_range

    def forward(self, h):
        # Kept apart from the GEMM, as vLLM rounds it.
        return super().forward(h.to(self.weight.dtype)).float() * (self.logit_scale.float() / self.initializer_range)


class Attention(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.c, self.kind = c, kind
        q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        for name, width in (("q", q), ("k", kv), ("v", kv), ("g", q)):
            setattr(self, f"{name}_proj", nn.Linear(c.hidden_size, width, bias=False))
        self.o_proj = nn.Linear(q, c.hidden_size, bias=False)
        self.sqk = nn.Parameter(torch.empty(q))
        self.s_gate = nn.Parameter(torch.empty(q))
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self, device=None):
        c = self.c
        theta = c.rope_parameters[self.kind]["rope_theta"]
        return 1 / theta ** (torch.arange(0, c.head_dim, 2, device=device).float() / c.head_dim)

    def rotary(self, h):
        """NeoX rotation of h [sequences, tokens, heads, dim]; positions restart in every sequence."""
        t, d = h.shape[1], h.shape[-1] // 2
        freq = torch.arange(t, device=h.device).float()[:, None] * self.inv_freq.float()[None]
        cos = freq.cos().to(h.dtype).float()[None, :, None]
        sin = freq.sin().to(h.dtype).float()[None, :, None]
        a, b = h[..., :d].float(), h[..., d:].float()
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(h.dtype)

    def forward(self, h, attention):
        c = self.c
        batch, tokens = h.shape[:2]
        q = self.rotary(self.q_proj(h).view(batch, tokens, c.num_attention_heads, c.head_dim))
        k = self.rotary(self.k_proj(h).view(batch, tokens, c.num_key_value_heads, c.head_dim))
        v = self.v_proj(h).view(batch, tokens, c.num_key_value_heads, c.head_dim)
        scale = (self.sqk.float() * (math.sqrt(c.head_dim) / c.initializer_range)).view(
            1, 1, c.num_attention_heads, c.head_dim)
        q = (norm(q).float() * scale.square()).to(h.dtype)
        k = norm(k)
        window = c.sliding_window if self.kind == "sliding_attention" else None
        if attention == "flash":
            if h.device.type != "cuda":
                raise ValueError("FlashAttention Dory features require CUDA; use sdpa on CPU")
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            cu = torch.arange(batch + 1, dtype=torch.int32, device=h.device) * tokens
            out = flash_attn_varlen_func(
                q=q.flatten(0, 1), k=k.flatten(0, 1), v=v.flatten(0, 1),
                cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=tokens, max_seqlen_k=tokens,
                causal=True, softmax_scale=1 / math.sqrt(c.head_dim),
                window_size=(window - 1, 0) if window else (-1, -1), fa_version=flash_version(h.device))
            out = out.view(batch, tokens, -1)
        elif attention == "sdpa":
            mask = None
            if window:
                pos = torch.arange(tokens, device=h.device)
                distance = pos[:, None] - pos[None]
                mask = (distance >= 0) & (distance < window)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, is_causal=mask is None,
                enable_gqa=c.num_attention_heads != c.num_key_value_heads)
            out = out.transpose(1, 2).reshape(batch, tokens, -1)
        else:
            raise ValueError(f"Unknown attention backend: {attention}")
        gate_base = c.attention_output_gate_logit_scale_init or math.sqrt(c.hidden_size)
        gate_scale = (self.s_gate.float() * (math.sqrt(gate_base) / c.initializer_range)).square()
        out = (out.float() * torch.sigmoid(self.g_proj(h).float() * gate_scale)).to(h.dtype)
        return self.o_proj(norm(out))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)
        self.suv_gate = nn.Parameter(torch.empty(c.intermediate_size))
        self.suv_up = nn.Parameter(torch.empty(c.intermediate_size))
        self.factor = math.sqrt(c.hidden_size)

    def forward(self, h):
        g = (self.gate_proj(h).float() * (self.suv_gate.float() * self.factor)).to(h.dtype)
        u = (self.up_proj(h).float() * (self.suv_up.float() * self.factor)).to(h.dtype)
        return self.down_proj(norm((F.silu(g.float()) * u.float()).to(h.dtype)))


class Layer(nn.Module):
    """One sublayer (attention or MLP) with the nGPT residual: slerp from h toward its output."""

    def __init__(self, c, kind):
        super().__init__()
        self.is_mlp = kind == "mlp"
        self.mixer = MLP(c) if self.is_mlp else Attention(c, kind)
        self.register_parameter("mlp_alpha" if self.is_mlp else "attn_alpha", nn.Parameter(torch.empty(c.hidden_size)))
        self.alpha_scale = c.ngpt_alpha_init / c.initializer_range

    def forward(self, h, attention):
        out = self.mixer(h) if self.is_mlp else self.mixer(h, attention)
        alpha = self.mlp_alpha if self.is_mlp else self.attn_alpha
        a, b = norm(h).float(), norm(out).float()
        # The slerp's final normalization runs in FP32 before the cast.
        return norm(a + (alpha.float() * self.alpha_scale).abs() * (b - a)).to(h.dtype)


class NativeDory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(**config) if isinstance(config, dict) else config
        c = self.config
        validate_config(c)
        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Embedding(c.vocab_size, c.hidden_size)
        self.backbone.layers = nn.ModuleList(Layer(c, kind) for kind in c.layer_types)
        self.lm_head = ScaledHead(c)

    @classmethod
    def from_pretrained(cls, directory, device, dtype=torch.bfloat16):
        """Frozen model of a checkpoint directory, streamed one tensor at a time (no second weight copy)."""
        from safetensors import safe_open

        directory = Path(directory)
        with torch.device("meta"):
            model = cls(json.loads((directory / "config.json").read_text()))
        model.to(dtype=dtype).to_empty(device=device)
        expected = dict(model.named_parameters())
        index = directory / "model.safetensors.index.json"
        shards = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists()
                  else ["model.safetensors"])
        loaded = set()
        for shard in shards:
            with safe_open(directory / shard, framework="pt") as f:
                for name in f.keys():
                    if name not in expected:
                        raise ValueError(f"Unexpected Dory weight: {name}")
                    weight = f.get_tensor(name)
                    if weight.shape != expected[name].shape:
                        raise ValueError(f"Dory weight shape mismatch: {name}")
                    expected[name].data = weight.to(device=device, dtype=dtype)
                    loaded.add(name)
        if loaded != set(expected):
            raise ValueError(f"Missing Dory weights: {sorted(set(expected) - loaded)}")
        # to_empty also materialized the non-persistent RoPE buffers from meta; rebuild them in FP32.
        for module in model.modules():
            if isinstance(module, Attention):
                module.inv_freq = module.frequencies(device)
        return model.eval().requires_grad_(False)

    @torch.no_grad()
    def features(self, tokens, layers=FEATURE_LAYERS, loops=LOOPS, attention="flash"):
        """tokens [T] -> taps [R, T, len(layers) * H] and output-block readout states [R, T, H].

        Row d - 1 holds depth d: the taps of the input block, of loop d's recurrent block and of the
        output block on s_d, and the output block's result on s_d (the scaled head's input).
        """
        c = self.config
        if tokens.ndim != 1 or not tokens.numel() or loops < 1:
            raise ValueError("Dory features require a nonempty token vector and positive loops")
        if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= c.num_hidden_layers for i in layers):
            raise ValueError("Dory feature layers must be distinct physical sublayer indices")
        n_in, end_rec = c.n_input_layers, c.n_input_layers + c.n_recurrent_layers
        h = self.backbone.embeddings(tokens)[None]
        features = h.new_empty(loops, tokens.numel(), len(layers) * c.hidden_size)
        slots = {layer: slice(j * c.hidden_size, (j + 1) * c.hidden_size) for j, layer in enumerate(layers)}
        for i in range(n_in):
            h = self.backbone.layers[i](h, attention)
            if i in slots:
                features[:, :, slots[i]] = h.expand(loops, -1, -1)
        x = state = h
        states = []
        for r in range(loops):
            h = norm(state + x)
            for i in range(n_in, end_rec):
                h = self.backbone.layers[i](h, attention)
                if i in slots:
                    features[r, :, slots[i]] = h[0]
            state = norm(h)
            states.append(state[0])
        h = torch.stack(states)
        for i in range(end_rec, c.num_hidden_layers):
            h = self.backbone.layers[i](h, attention)
            if i in slots:
                features[:, :, slots[i]] = h
        return features, h
