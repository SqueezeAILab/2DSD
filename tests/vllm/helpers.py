"""Shared fixtures of the 2DSD executor tests: exact oracles and tiny targets.

Nothing here loads a checkpoint. The tiny models have the module layout of the
native Ouro runner (two layers, sandwich norms, packed QKV), so the runners
execute their real code paths on them; results are numerical contracts, not
checkpoint quality or speed measurements.
"""

import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

cuda_test = unittest.skipUnless(
    os.environ.get("TWODSD_CUDA_TESTS") == "1" and torch.cuda.is_available(),
    "CUDA tests are opt-in: set TWODSD_CUDA_TESTS=1 on a machine with a free GPU",
)


def oracle_next(tokens, causal=False, loops=4):
    """Greedy next token of the exact recurrent target that FakeTreeRunner computes."""
    values = list(tokens)
    if causal:
        for loop in range(loops):
            prefix_sum, outputs = 0, []
            for value in values:
                outputs.append((3 * value + 2 * prefix_sum + loop + 1) % 97)
                prefix_sum += value
            values = outputs
    return (values[-1] + 1) % 97


class FakeTreeRunner:
    """Exact recurrent target with isolated node KV and strict pool operations."""

    R, H, nf, max_len = 4, 1, 1, 512
    device, dtype = torch.device("cpu"), torch.float64

    def __init__(self, max_nodes=64, causal=True):
        self.max_nodes, self.causal = max_nodes, causal
        self.stats = {
            "calls": 0,
            "cells": 0,
            "registered_nodes": 0,
            "committed_nodes": 0,
            "released_nodes": 0,
        }
        self.registrations = []
        self.release_roots = []
        self.reset()

    @property
    def prefix_length(self):
        return len(self.prefix_tokens)

    def reset(self):
        self.prefix_tokens = []
        self.prefix_inputs = [[] for _ in range(self.R)]
        self.nodes = {}
        self.keys = {}
        self.trace = []

    def embed(self, tokens):
        return tokens.to(self.dtype).reshape(-1, 1)

    def logits(self, hidden):
        prediction = ((hidden[:, 0] + 1) % 97).long()
        out = torch.full((len(hidden), 97), -8.0, dtype=self.dtype)
        out.scatter_(1, prediction[:, None], 3.0)
        return out

    def readout(self, hidden):
        return self.logits(hidden).argmax(-1)

    def prefill(self, tokens):
        self.reset()
        self.prefix_tokens = tokens.tolist()
        values = tokens.tolist()
        for loop in range(self.R):
            self.prefix_inputs[loop] = list(values)
            prefix_sum, outputs = 0, []
            for value in values:
                y = value
                if self.causal:
                    y = (3 * value + 2 * prefix_sum + loop + 1) % 97
                outputs.append(y)
                prefix_sum += value
            values = outputs
        hidden = torch.tensor(values, dtype=self.dtype).reshape(-1, 1)
        return SimpleNamespace(hidden=hidden, features=hidden.clone())

    def path(self, node_id):
        path = []
        while node_id is not None:
            path.append(node_id)
            node_id = self.nodes[node_id]["parent"]
        return list(reversed(path))

    def register(self, node_id, parent_id, token_id):
        key = (parent_id, token_id)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if (node["parent"], node["token"]) != key:
                raise AssertionError("Intermediate review replaced an existing node")
            return node_id
        if key in self.keys:
            raise AssertionError("Same-parent same-token candidate was not merged")
        if len(self.nodes) >= self.max_nodes:
            raise AssertionError("The scheduler allocated beyond its node budget")
        if parent_id is not None and parent_id not in self.nodes:
            raise AssertionError("Candidate lost an unverified ancestor")
        self.nodes[node_id] = {
            "parent": parent_id,
            "token": token_id,
            "depth": 0,
            "inputs": {},
        }
        self.keys[key] = node_id
        self.registrations.append((node_id, parent_id, token_id))
        self.stats["registered_nodes"] += 1
        return node_id

    def advance(self, spans):
        if self.stats["calls"] > 2000:
            raise AssertionError("The node budget starved final verification")
        if len({span.node_id for span in spans}) != len(spans):
            raise AssertionError("A node was advanced twice in one physical call")
        unfinished = {
            node_id for node_id, node in self.nodes.items() if node["depth"] < self.R
        }
        if unfinished != {span.node_id for span in spans}:
            raise AssertionError(
                "A physical call did not advance every live unfinished node"
            )
        self.trace.append(
            sorted(
                (
                    tuple(self.nodes[i]["token"] for i in self.path(span.node_id)),
                    span.loop,
                    float(span.hidden[0]),
                )
                for span in spans
            )
        )
        for span in spans:
            if self.nodes[span.node_id]["depth"] != span.loop:
                raise AssertionError("Fork and bonus nodes must start at loop one")
            self.nodes[span.node_id]["inputs"][span.loop] = float(span.hidden[0])
        results = []
        for span in spans:
            node = self.nodes[span.node_id]
            ancestors = self.path(span.node_id)[:-1]
            if any(
                span.loop not in self.nodes[parent]["inputs"] for parent in ancestors
            ):
                raise AssertionError("A child cannot observe unavailable ancestor KV")
            prefix_sum = sum(self.prefix_inputs[span.loop]) + sum(
                self.nodes[parent]["inputs"][span.loop] for parent in ancestors
            )
            value = node["inputs"][span.loop]
            y = value
            if self.causal:
                y = (3 * value + 2 * prefix_sum + span.loop + 1) % 97
            hidden = torch.tensor([[y]], dtype=self.dtype)
            results.append(SimpleNamespace(hidden=hidden, features=hidden.clone()))
            node["depth"] += 1
        self.stats["calls"] += 1
        self.stats["cells"] += len(spans)
        return results

    def release(self, ids):
        removed = set(ids)
        for node_id in self.nodes:
            if node_id not in removed and removed.intersection(self.path(node_id)[:-1]):
                raise AssertionError("Pruning left a descendant with stale KV ancestry")
        for node_id in ids:
            parent = self.nodes[node_id]["parent"]
            if parent is not None and parent not in removed:
                self.release_roots.append(
                    (node_id, parent, self.nodes[parent]["depth"])
                )
        for node_id in ids:
            node = self.nodes.pop(node_id)
            self.keys.pop((node["parent"], node["token"]))
        self.stats["released_nodes"] += len(ids)

    def commit(self, ids):
        if not ids:
            return
        for i, node_id in enumerate(ids):
            node = self.nodes[node_id]
            expected = None if i == 0 else ids[i - 1]
            if node["parent"] != expected or node["depth"] != self.R:
                raise AssertionError(
                    "Only the final verified contiguous path may commit"
                )
        kept = set(ids)
        for node_id in self.nodes:
            if node_id not in kept and ids[-1] not in self.path(node_id):
                raise AssertionError(
                    "A rejected sibling survived accepted-prefix promotion"
                )
        for node_id in ids:
            node = self.nodes[node_id]
            self.prefix_tokens.append(node["token"])
            for loop in range(self.R):
                self.prefix_inputs[loop].append(node["inputs"][loop])
        for node_id in ids:
            del self.nodes[node_id]
        self.keys = {}
        for node_id, node in self.nodes.items():
            if node["parent"] in kept:
                node["parent"] = None
            self.keys[node["parent"], node["token"]] = node_id
        self.stats["committed_nodes"] += len(ids)


class _Norm(nn.Module):
    def forward(self, hidden, residual=None):
        combined = hidden if residual is None else hidden + residual
        normalized = combined * torch.rsqrt(combined.square().mean(-1, True) + 1e-6)
        return normalized if residual is None else (normalized, combined)


class _Linear(nn.Linear):
    """vLLM-style linear layer (returns an output and a bias) that counts calls."""

    calls = 0

    def forward(self, hidden):
        self.calls += 1
        return super().forward(hidden), None


class _MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, width * 2, bias=False)
        self.down = nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        gate, up = self.gate(hidden).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        attn = self.self_attn
        attn.num_heads, attn.num_kv_heads, attn.head_dim = 2, 1, 4
        attn.q_size, attn.kv_size, attn.scaling = 8, 4, 0.5
        attn.qkv_proj, attn.o_proj = (
            _Linear(8, 16, bias=False),
            _Linear(8, 8, bias=False),
        )
        attn.rotary_emb = lambda positions, query, key: (query, key)
        attn.loop_kv = None
        attn.attn = [SimpleNamespace(impl=SimpleNamespace(sliding_window=None))] * 4
        self.mlp = _MLP(8)
        self.input_layernorm = _Norm()
        self.input_layernorm_2 = _Norm()
        self.post_attention_layernorm = _Norm()
        self.post_attention_layernorm_2 = _Norm()


class _Model(nn.Module):
    """Two Ouro-shaped layers (R=4) without embeddings or a head."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8, num_hidden_layers=2, num_attention_heads=2, total_ut_steps=4
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])
        self.model.norm = _Norm()


class _CausalLM(_Model):
    def __init__(self):
        super().__init__()
        self.model.embed_tokens = nn.Embedding(97, 8)
        self.head = nn.Linear(8, 97, bias=False)

    def embed_input_ids(self, tokens):
        return self.model.embed_tokens(tokens)

    def compute_logits(self, hidden):
        return self.head(hidden)


class _TinyLM(nn.Module):
    """CUDA fixture with vLLM RMSNorm/RoPE and a high-margin greedy readout.

    Embeddings are one-hot and the head maps token t to t + 1 (mod 32); small
    attention/MLP weights keep every argmax well separated from drift.
    """

    def __init__(self, heads=4, kv_heads=2, head_dim=16):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        width = heads * head_dim
        kv_width = kv_heads * head_dim
        self.config = SimpleNamespace(
            hidden_size=width,
            num_hidden_layers=2,
            num_attention_heads=heads,
            total_ut_steps=4,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, width)
        layers = []
        for _ in range(2):
            layer = nn.Module()
            attn = layer.self_attn = nn.Module()
            attn.num_heads, attn.num_kv_heads, attn.head_dim = heads, kv_heads, head_dim
            attn.q_size, attn.kv_size, attn.scaling = width, kv_width, head_dim**-0.5
            attn.qkv_proj, attn.o_proj = (
                _Linear(width, width + 2 * kv_width, bias=False),
                _Linear(width, width, bias=False),
            )
            attn.rotary_emb = RotaryEmbedding(
                head_dim, head_dim, 128, 1e6, True, torch.bfloat16
            )
            attn.loop_kv = None
            attn.attn = [SimpleNamespace(impl=SimpleNamespace(sliding_window=None))] * 4
            layer.mlp = _MLP(width)
            for name in (
                "input_layernorm",
                "input_layernorm_2",
                "post_attention_layernorm",
                "post_attention_layernorm_2",
            ):
                setattr(layer, name, RMSNorm(width, eps=1e-6))
            layers.append(layer)
        self.model.layers = nn.ModuleList(layers)
        self.model.norm = RMSNorm(width, eps=1e-6)
        self.head = nn.Linear(width, 32, bias=False)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.eye(32, width))
            self.head.weight.zero_()
            for token in range(32):
                self.head.weight[(token + 1) % 32, token] = 1
            for layer in layers:
                for parameter in layer.self_attn.qkv_proj.parameters():
                    parameter.mul_(0.002)
                for parameter in layer.self_attn.o_proj.parameters():
                    parameter.mul_(0.002)
                for parameter in layer.mlp.parameters():
                    parameter.mul_(0.002)

    def embed_input_ids(self, tokens):
        return self.model.embed_tokens(tokens)

    def compute_logits(self, hidden):
        return self.head(hidden)
