# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Inkling 2 with its matrix products on tensor cores.

``NativeInkling2`` in the served precision holds BF16-rounded values in FP32
and multiplies them with FP32 GEMMs, which do not run on tensor cores.
``FastInkling2`` is the same forward with:

  - TF32 products wherever both operands hold BF16 values: every linear
    projection, the attention scores and mixing, the router and the
    unembedding. TF32 keeps 10 mantissa bits of each operand and BF16 has 7,
    so the operands are not rounded. Against FP64 on an H100 the TF32 product
    is off by 5e-6 of the largest result and the FP32 GEMM by 1.4e-6, where
    rounding to BF16 is 2e-3;
  - vLLM's grouped expert GEMM for the routed experts (dense BF16 experts on
    a GPU): BF16 inputs and weights, an FP32 accumulator and the expert's
    scale applied before the result is rounded. Packed experts on a GPU run
    on Humming's packed expert GEMM instead, when it is installed
    (inkling2_humming), as the served Hopper path does;
  - the layer weights converted to the arithmetic dtype once;
  - on a GPU, the attention as one Triton kernel (inkling2_attention): scores,
    softmax, mixing and gates of a layer in one launch with no score tensor.
    Like the served kernel it rounds a probability relative to the running
    maximum of the keys read so far, where the reference uses the row maximum.

The rounding points are the reference's. AttnRes keeps FP32 products, because
its sources and scaled queries are not BF16 values. A product that rounds to
another BF16 value can change a routed expert, so states differ from the
reference's by more than rounding at some tokens, as the reference's differ
from upstream's (whose accumulation order it does not reproduce either).
Agreement with the served rollouts is therefore measured, not derived
(scripts/inkling2/runner_check.py --tensor-cores).

A tree runner given a ``FastInkling2`` runs its own products the same way.
On a CPU the model is the reference bit for bit.
"""

import contextlib

import torch

from vllm.model_executor.models.inkling2_reference import (
    ROUTER_GROUP,
    NativeInkling2,
    route_experts,
    situ,
)
from vllm.model_executor.models.inkling2_humming import supported as humming_supported
from vllm.model_executor.models.inkling2_weights import GATE_ROWS, UP_ROWS

EXPERT_ROWS = 4096  # tokens of a sequence forward per grouped expert GEMM
ATTENTION_ROWS = 4096  # queries per attention launch: their relative logits take 0.2 GiB


@contextlib.contextmanager
def tensor_cores(enabled=True):
    """FP32 matrix products inside the block run on TF32 tensor cores, or with
    ``enabled`` false as FP32 GEMMs."""
    matmul = torch.backends.cuda.matmul
    before = matmul.allow_tf32
    matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        matmul.allow_tf32 = before


def fused_experts_supported(model):
    """Whether the routed experts of ``model`` can run as vLLM's grouped expert
    GEMM: dense BF16 experts on a GPU and the served precision."""
    checkpoint, p = model.checkpoint, model.precision
    return (model.device.type == "cuda" and checkpoint.expert_mode == "dense"
            and checkpoint.dtype == torch.bfloat16
            and p.storage == torch.bfloat16 and p.dtype == torch.float32)


def fused_experts(config, precision, experts, x, ids, weights, kernel_configs):
    """Output [n, H] of the routed experts ``ids`` [n, top_k] (contiguous) with
    routing ``weights`` on the normed inputs x [n, H], by vLLM's grouped expert
    GEMM. The expert's scale is the per-row weight that the kernel applies to
    its FP32 accumulator before rounding. ``kernel_configs`` caches the kernel
    configuration by row count."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
        invoke_fused_moe_triton_kernel,
        try_get_optimal_moe_config,
    )
    from vllm.triton_utils import tl

    c, p = config, precision
    rows, top = ids.shape
    kernel = kernel_configs.get(rows)
    if kernel is None:
        kernel = kernel_configs[rows] = try_get_optimal_moe_config(
            experts.w13.shape, experts.w2.shape, top, None, rows)
    assignment = _prepare_expert_assignment(ids, kernel, rows, top, c.n_routed_experts, None,
                                            ignore_invalid_experts=True)
    flags = dict(compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
                 use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    half = torch.bfloat16
    gate_up = torch.empty((rows, top, experts.w13.shape[1]), dtype=half, device=x.device)
    invoke_fused_moe_triton_kernel(x.to(half), experts.w13, gate_up, None, None,
                                   experts.w13_scale[ids], *assignment, True, top, kernel, **flags)
    gate_up = gate_up.to(p.dtype)
    hidden = p.store(situ(gate_up[..., GATE_ROWS], gate_up[..., UP_ROWS], weights[..., None],
                          c.mlp_gate_softcap, c.mlp_up_softcap))
    down = torch.empty((rows, top, c.hidden_size), dtype=half, device=x.device)
    invoke_fused_moe_triton_kernel(hidden.to(half).view(rows * top, -1), experts.w2, down, None, None,
                                   experts.w2_scale[ids].view(-1, 1), *assignment, True, 1, kernel,
                                   **flags)
    down = down.to(p.dtype)
    out = down[:, 0]
    for slot in range(1, top):
        out = out + down[:, slot]
    return p.store(out)


class FastInkling2(NativeInkling2):
    """``NativeInkling2`` in the served precision with tensor-core products."""

    tensor_cores = True

    def __init__(self, checkpoint, precision="served", *, fused_attention=None, **options):
        super().__init__(checkpoint, precision, **options)
        p = self.precision
        if p.storage != torch.bfloat16 or p.dtype != torch.float32:
            raise ValueError("FastInkling2 runs the served precision (BF16 values held in FP32)")
        self.fused_experts = fused_experts_supported(self)
        # Packed experts run on Humming's expert GEMM when it is installed (inkling2_humming).
        self.humming = humming_supported(self)
        self.fused_attention = self.device.type == "cuda" if fused_attention is None else bool(fused_attention)
        self._weights, self._moe_configs, self._humming = {}, {}, {}

    def humming_experts(self, layer):
        """The packed experts of ``layer`` as Humming tensors, built on first use."""
        found = self._humming.get(layer)
        if found is None:
            from vllm.model_executor.models.inkling2_humming import HummingExperts

            c = self.config
            shapes = ((2 * c.intermediate_size, c.hidden_size), (c.hidden_size, c.intermediate_size))
            found = self._humming[layer] = HummingExperts(self.checkpoint.experts(layer), shapes)
        return found

    def _lw(self, layer, name):
        key = (layer, name)
        found = self._weights.get(key)
        if found is None:
            found = self._weights[key] = super()._lw(layer, name)
        return found

    def _linear(self, x, layer, name):
        with tensor_cores():
            return x @ self._lw(layer, name).T

    def _attention(self, run, layer, plane, inputs):
        with tensor_cores():
            return super()._attention(run, layer, plane, inputs)

    def _attend(self, query, keys, values, rel_states, rel_proj, gate, sink, *, first_query, first_key, window):
        if not self.fused_attention:
            return super()._attend(query, keys, values, rel_states, rel_proj, gate, sink,
                                   first_query=first_query, first_key=first_key, window=window)
        from vllm.model_executor.models.inkling2_attention import sequence_attention

        half, tokens = torch.bfloat16, query.shape[0]
        keys, values = keys.to(half), values.to(half)
        out = torch.empty(query.shape, dtype=half, device=query.device)
        for a in range(0, tokens, ATTENTION_ROWS):
            b = min(tokens, a + ATTENTION_ROWS)
            low = first_key if window is None else max(first_key, first_query + a - (window - 1))
            span = slice(low - first_key, first_query + b - first_key)
            rel = (rel_states[a:b] @ rel_proj).to(half)  # the relative logits, at their rounding point
            sequence_attention(query[a:b].to(half), rel, keys[span], values[span], gate[a:b].to(half),
                               sink[a:b].to(half), first_query=first_query + a, first_key=low, window=window,
                               out=out[a:b])
        return out.to(self.precision.dtype)

    def _moe(self, layer, x):
        if not (self.fused_experts or self.humming):
            with tensor_cores():
                return super()._moe(layer, x)
        c, p = self.config, self.precision
        gate = self._lw(layer, "mlp.gate.weight")
        with tensor_cores():
            logits = sum(x[:, k:k + ROUTER_GROUP] @ gate[:, k:k + ROUTER_GROUP].T
                         for k in range(0, c.hidden_size, ROUTER_GROUP))
        ids, weights = route_experts(logits, self._lw(layer, "mlp.gate.bias"),
                                     c.num_experts_per_tok, c.route_scale)
        ids = ids.contiguous()  # a slice of the sorted experts; the kernel reads it flat
        if self.humming:
            experts, half = self.humming_experts(layer), torch.bfloat16
            parts = [experts.products(x[a:a + EXPERT_ROWS].to(half), ids[a:a + EXPERT_ROWS],
                                      weights[a:a + EXPERT_ROWS], c).to(p.dtype)
                     for a in range(0, x.shape[0], EXPERT_ROWS)]
            return (parts[0] if len(parts) == 1 else torch.cat(parts)), ids
        experts = self.checkpoint.experts(layer)
        parts = [fused_experts(c, p, experts, x[a:a + EXPERT_ROWS], ids[a:a + EXPERT_ROWS],
                               weights[a:a + EXPERT_ROWS], self._moe_configs)
                 for a in range(0, x.shape[0], EXPERT_ROWS)]
        return (parts[0] if len(parts) == 1 else torch.cat(parts)), ids

    def readout_hidden(self, blocks, partial):
        with tensor_cores(False):  # AttnRes, also when a runner's call enabled TF32
            return super().readout_hidden(blocks, partial)

    def readout_logits(self, hidden):
        with tensor_cores():
            return super().readout_logits(hidden)

    def forward(self, tokens, **options):
        with tensor_cores(False):  # AttnRes keeps FP32 products; the methods above enable TF32
            return super().forward(tokens, **options)
