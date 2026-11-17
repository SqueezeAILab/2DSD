"""Tiny Dory checkpoint for the CPU tests of the Dory tree runner.

The model keeps the 72-sublayer layout of Dory1.0-4B (16 input, 40 recurrent
and 16 output sublayers; MLP sublayers alternating with sliding- and
full-attention ones; grouped-query attention; nGPT scales) at hidden size 16
and a 96-token vocabulary. Parameter names are those of the checkpoint, so
``DoryWeights.from_checkpoint`` and the FP32 reference load the file as they
would the real model. The weights are seeded random values: the tests compare
implementations, not checkpoint quality.
"""

import json

import torch
from safetensors.torch import save_file
from torch import nn


def tiny_config():
    return dict(
        model_type="dory",
        vocab_size=96,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=72,
        n_input_layers=16,
        n_recurrent_layers=40,
        n_output_layers=16,
        n_recurrent_loops=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        initializer_range=0.0177,
        ngpt_alpha_init=0.05,
        attention_output_gate_logit_scale_init=None,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        max_position_embeddings=128,
        layer_types=[
            "mlp" if i % 2 else "full_attention" if i % 8 == 6 else "sliding_attention"
            for i in range(72)
        ],
        rope_parameters={
            kind: dict(rope_type="default", rope_theta=theta, partial_rotary_factor=1.0)
            for kind, theta in (("sliding_attention", 1e4), ("full_attention", 1e7))
        },
    )


# Parameter containers in the checkpoint's module order. The construction order
# fixes the random stream, so a seed gives the same weights on every machine.
class _Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        q = c["num_attention_heads"] * c["head_dim"]
        kv = c["num_key_value_heads"] * c["head_dim"]
        for name, width in (("q", q), ("k", kv), ("v", kv), ("g", q)):
            setattr(self, f"{name}_proj", nn.Linear(c["hidden_size"], width, bias=False))
        self.o_proj = nn.Linear(q, c["hidden_size"], bias=False)
        self.sqk = nn.Parameter(torch.empty(q))
        self.s_gate = nn.Parameter(torch.empty(q))


class _MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        hidden, inner = c["hidden_size"], c["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)
        self.suv_gate = nn.Parameter(torch.empty(inner))
        self.suv_up = nn.Parameter(torch.empty(inner))


class _Layer(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.mixer = _MLP(c) if kind == "mlp" else _Attention(c)
        alpha = "mlp_alpha" if kind == "mlp" else "attn_alpha"
        self.register_parameter(alpha, nn.Parameter(torch.empty(c["hidden_size"])))


class _Head(nn.Linear):
    def __init__(self, c):
        super().__init__(c["hidden_size"], c["vocab_size"], bias=False)
        self.logit_scale = nn.Parameter(torch.empty(c["vocab_size"]))


class TinyDory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.backbone.layers = nn.ModuleList(_Layer(config, kind) for kind in config["layer_types"])
        self.lm_head = _Head(config)


def tiny_model():
    torch.manual_seed(1729)
    model = TinyDory(tiny_config())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("weight"):
                param.normal_(std=0.12)
            elif name.endswith(("attn_alpha", "mlp_alpha", "sqk", "s_gate")):
                param.fill_(0.0177)
            elif name.endswith(("suv_gate", "suv_up")):
                param.fill_(1 / 4)
            elif name.endswith("logit_scale"):
                param.copy_(torch.linspace(0.01, 0.04, param.numel()))
    return model.eval()


def write_model(directory, model):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(model.config))
    save_file(model.state_dict(), directory / "model.safetensors")
