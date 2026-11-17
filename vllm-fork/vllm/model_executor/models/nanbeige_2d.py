# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Packed R2 Nanbeige target with native or last-loop shared prefix attention.

The tree reuses Ouro's paged prefix and ancestor kernels, graph capture, and
growing node storage. Nanbeige's physical layers have two residual norms and
optional per-head QK normalization. No sandwich norms are introduced.

Shared-last prefill uses native loop KV. Decode selects final-loop KV at
``j <= t - W`` and same-loop KV above that boundary, except the first layer's
first loop, which retains native KV. Shared tree storage retains these two full
histories and fixed recent rings; native targets keep their full loop planes.
"""

import torch

from .ouro_2d import LoopResult, NativeOuro2DRunner
from .ouro_finalized import NativeOuroFinalizedRunner
from .ouro_tree import TreeSpan

__all__ = ["NativeNanbeige2DRunner", "NativeNanbeigeTreeRunner", "TreeSpan"]


def _nanbeige_layers(owner, prefix, inputs) -> LoopResult:
    hidden = inputs.hidden.clone()
    residual = None
    features = torch.empty(
        hidden.shape[0], prefix.nf * prefix.H,
        device=prefix.device, dtype=prefix.dtype,
    )
    for idx, layer in enumerate(prefix.layers):
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        attn = layer.self_attn
        qkv, _ = attn.qkv_proj(hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
        if getattr(attn, "qk_norm", False):
            q = attn.q_layernorm(q.reshape(-1, attn.num_heads, attn.head_dim))
            k = attn.k_layernorm(k.reshape(-1, attn.num_kv_heads, attn.head_dim))
            q, k = q.reshape(-1, attn.q_size), k.reshape(-1, attn.kv_size)
        q, k = prefix._apply_rope(attn.rotary_emb, inputs.positions, q, k)
        hidden, _ = attn.o_proj(owner._attention(idx, q, k, v, inputs))
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        hidden = layer.mlp(hidden)
        slot = prefix._feature_slots.get(idx)
        if slot is not None:
            features[:, slot * prefix.H : (slot + 1) * prefix.H] = hidden + residual
    hidden, _ = prefix.backbone.norm(hidden, residual)
    return LoopResult(hidden, features)


class NativeNanbeige2DRunner(NativeOuro2DRunner):
    """Nanbeige native prefix computation and independent full-KV loop spans.

    For a shared-last target this class is a prefill/cache owner only. Shared
    decode must use ``NativeNanbeigeTreeRunner`` to select KV by query position.
    Input and output lifetime follow ``NativeOuro2DRunner``.
    """

    @staticmethod
    def _loop_count(config) -> int:
        return getattr(config, "num_loops", 2)

    def _validate_model(self, config) -> None:
        super()._validate_model(config)
        if self.R != 2:
            raise ValueError("The Nanbeige 2D runner requires num_loops=2")
        if getattr(config, "skip_loop_final_norm", False):
            raise ValueError("Nanbeige 2D requires norm at each loop boundary")
        unsupported = (
            "enable_double_loop_split", "enable_depth_attention", "enable_mhc",
            "enable_hyper_connection", "loop_share_kv",
        )
        if any(getattr(config, name, False) for name in unsupported) or getattr(
            config, "emb_neighbor_num", None
        ) is not None:
            raise ValueError("Only standard recurrent Nanbeige layers are supported")
        for layer in self.layers:
            attn = layer.self_attn
            if attn.num_kv_heads != config.num_key_value_heads:
                raise ValueError("Tensor parallel Nanbeige execution is unsupported")
            if len(attn.attn) != self.R:
                raise ValueError("Nanbeige attention loop count does not match config")
            if not callable(layer.post_attention_layernorm):
                raise ValueError("Nanbeige requires post-attention residual norm")
        if getattr(config, "loopedtf_kv_policy", "full") == "shared_last" and (
            type(config.loopedtf_kv_window) is not int or config.loopedtf_kv_window < 1
        ):
            raise ValueError("Shared-last requires a positive integer window")

    def _forward_layers(self, inputs) -> LoopResult:
        return _nanbeige_layers(self, self, inputs)

    def advance(self, spans):
        if self.shared_prefill and any(span.start != 0 for span in spans):
            raise ValueError("Shared-last decode requires the Nanbeige tree runner")
        return super().advance(spans)


class NativeNanbeigeTreeRunner(NativeOuroFinalizedRunner):
    """Native Nanbeige node KV with shared-last attention and geometric growth.

    ``max_nodes`` is an initial physical allocation, never a pruning budget.
    All active node/loop cells share one QKV and MLP invocation per physical
    layer. The inherited shared-last path guard rejects a speculative path
    longer than W rather than reading uncomputed final-loop ancestor KV.
    Stage 1 R2 with block sizes up to 16 stays within W64.
    """

    prefix_runner_type = NativeNanbeige2DRunner

    def _forward_layers(self, inputs) -> LoopResult:
        return _nanbeige_layers(self, self.prefix, inputs)
