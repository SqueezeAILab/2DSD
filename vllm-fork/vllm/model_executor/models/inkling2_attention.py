# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Triton kernels of Inkling 2's gated relative-bias attention.

``sequence_attention(query, rel, keys, values, gate, sink, ...) -> out`` is the
attention of consecutive queries over consecutive cached keys, the function of
``inkling2_reference.relative_attention`` followed by the rounding of its gated
output, in one launch: no score tensor is materialised.

  query, gate [T, heads, D]     the clipped (and tau-scaled) queries and the raw gates
  rel         [T, heads, extent] relative logits by backward distance, as stored
  sink        [T, heads]
  keys, values [N, kv_heads, D]  as cached; head h reads KV head h // (heads / kv_heads)

All of them hold BF16 values in BF16. For backward distance d = pos_q - pos_k the
score is q . k D ** -0.5 + rel[d] (no bias from ``extent`` on); keys with d < 0,
and with ``window`` those with d >= window, are masked. The output [T, heads, D]
(BF16) is softmax-weighted v times sigmoid(gate) times sigmoid(logsumexp - sink).

Arithmetic. Scores, the softmax and the accumulators are FP32. A probability is
rounded to BF16 before its product with v, relative to the running maximum of
the keys read so far, as the served attention kernel does (the reference rounds
relative to the row maximum); the denominator sums the unrounded probabilities.

Order. Keys are read in tiles of ``BLOCK_K`` consecutive positions aligned to
absolute position 0, in ascending order, and a row's result depends on nothing
but its own visible keys: a tile without a visible key leaves the row's state
unchanged bit for bit. A query therefore gets the same bits however the queries
are grouped into launches and blocks, and whichever key range a launch covers.

Launch grids depend on shapes only, so a call can be captured in a CUDA graph.
This module imports no vLLM code.
"""

import torch
import triton
import triton.language as tl

BLOCK_K = 64  # keys per tile; part of the arithmetic (the order of the FP32 sums)


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _pair_sum(x):
    """Sum over the last axis of [ROWS, n] (n a power of two) by a fixed pairwise tree: the order of an FP32
    reduction otherwise follows the thread layout, which changes with the block shape and the warp count."""
    ROWS: tl.constexpr = x.shape[0]
    LEVELS: tl.constexpr = x.shape[1].bit_length() - 1
    for level in tl.static_range(LEVELS):
        x = tl.sum(tl.reshape(x, (ROWS, x.shape[1] // 2, 2)), 2)
    return tl.reshape(x, (ROWS,))


@triton.jit
def _merge_tile(q, k, v, scores_bias, visible, maximum, denominator, numerator, SCALE: tl.constexpr):
    """Online-softmax update of the rows ``q`` [ROWS, D] with one tile of keys [BLOCK_K, D]."""
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE + scores_bias
    scores = tl.where(visible, scores, float("-inf"))
    peak = tl.maximum(maximum, tl.max(scores, 1))
    weights = tl.where(visible, tl.exp(scores - peak[:, None]), 0.0)
    decay = tl.where(maximum == float("-inf"), 0.0, tl.exp(maximum - peak))
    numerator = tl.dot(weights.to(tl.bfloat16), v, acc=numerator * decay[:, None], out_dtype=tl.float32)
    denominator = denominator * decay + _pair_sum(weights)
    return peak, denominator, numerator


@triton.jit(do_not_specialize=["queries", "first_query", "first_key", "num_keys"])
def _sequence_attention(
    query,
    rel,
    keys,
    values,
    gate,
    sink,
    output,
    queries,
    first_query,
    first_key,
    num_keys,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    SLOTS: tl.constexpr,
    TILE: tl.constexpr,
):
    block = tl.program_id(0)
    kv_head = tl.program_id(1)
    GROUP: tl.constexpr = HEADS // KV_HEADS
    ROWS: tl.constexpr = BLOCK_Q * SLOTS
    rows = tl.arange(0, ROWS)
    token = block * BLOCK_Q + rows // SLOTS  # index among this call's queries
    slot = rows % SLOTS
    live = (token < queries) & (slot < GROUP)
    head = kv_head * GROUP + tl.minimum(slot, GROUP - 1)
    dims = tl.arange(0, HEAD_DIM)
    cells = (token * HEADS + head)[:, None].to(tl.int64) * HEAD_DIM + dims[None, :]
    q = tl.load(query + cells, mask=live[:, None], other=0)
    position = first_query + token
    rel_row = (token * HEADS + head).to(tl.int64) * EXTENT

    maximum = tl.full((ROWS,), float("-inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    numerator = tl.zeros((ROWS, HEAD_DIM), tl.float32)

    last = first_query + tl.minimum(block * BLOCK_Q + BLOCK_Q, queries) - 1
    last = tl.minimum(last, first_key + num_keys - 1)
    lowest = first_key
    if WINDOW > 0:
        lowest = tl.maximum(lowest, first_query + block * BLOCK_Q - (WINDOW - 1))
    lowest = tl.maximum(lowest, 0)
    column = tl.arange(0, TILE)
    for tile in range(lowest // TILE, last // TILE + 1):
        at = tile * TILE + column
        inside = (at >= first_key) & (at < first_key + num_keys)
        offsets = ((at - first_key) * KV_HEADS + kv_head)[:, None].to(tl.int64) * HEAD_DIM + dims[None, :]
        k = tl.load(keys + offsets, mask=inside[:, None], other=0)
        v = tl.load(values + offsets, mask=inside[:, None], other=0)
        distance = position[:, None] - at[None, :]
        visible = live[:, None] & inside[None, :] & (distance >= 0)
        if WINDOW > 0:
            visible = visible & (distance < WINDOW)
        near = visible & (distance < EXTENT)
        bias = tl.load(rel + rel_row[:, None] + distance, mask=near, other=0).to(tl.float32)
        maximum, denominator, numerator = _merge_tile(
            q, k, v, bias, visible, maximum, denominator, numerator, SCALE
        )

    mixed = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    lse = maximum + tl.log(denominator)
    gates = tl.load(gate + cells, mask=live[:, None], other=0).to(tl.float32)
    sinks = tl.load(sink + token * HEADS + head, mask=live, other=0).to(tl.float32)
    result = mixed * _sigmoid(gates) * _sigmoid(lse - sinks)[:, None]
    tl.store(output + cells, result.to(output.dtype.element_ty), mask=live[:, None])


def sequence_attention(
    query: torch.Tensor,
    rel: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    *,
    first_query: int = 0,
    first_key: int = 0,
    window: int | None = None,
    block_q: int = 16,
    out: torch.Tensor | None = None,
    tuning: tuple[int, int] = (3, 8),
) -> torch.Tensor:
    """Gated relative-bias attention of consecutive queries (module docstring).

    Args:
        first_query, first_key: absolute positions of ``query[0]`` and ``keys[0]``.
        window: sliding-window size, or None for full attention.
        block_q: queries per program; it does not change the result.
        out: BF16 or FP32 tensor [T, heads, D] to write (FP32 skips the last rounding).
        tuning: (pipeline stages, warps).
    """
    tokens, heads, dim = query.shape
    kv_heads, extent = keys.shape[1], rel.shape[2]
    group = heads // kv_heads
    if out is None:
        out = torch.empty_like(query)
    tensors = (query, rel, keys, values, gate, sink, out)
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert all(tensor.dtype == torch.bfloat16 for tensor in tensors[:-1])
    assert heads == group * kv_heads and dim >= 16 and dim & (dim - 1) == 0
    assert gate.shape == query.shape and sink.shape == (tokens, heads)
    assert rel.shape == (tokens, heads, extent) and values.shape == keys.shape
    assert tokens * heads * max(extent, dim) < 2**62
    stages, warps = tuning
    _sequence_attention[(triton.cdiv(tokens, block_q), kv_heads)](
        query, rel, keys, values, gate, sink, out,
        tokens, first_query, first_key, keys.shape[0],
        HEADS=heads, KV_HEADS=kv_heads, HEAD_DIM=dim, EXTENT=extent,
        WINDOW=0 if window is None else window, SCALE=dim**-0.5,
        BLOCK_Q=block_q, SLOTS=triton.next_power_of_2(group), TILE=BLOCK_K,
        num_stages=stages, num_warps=warps,
    )
    return out
