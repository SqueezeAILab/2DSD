# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Tensor movement for padded loop-context drafting."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_context_kv(
    span_key,
    span_value,
    block_key,
    block_value,
    indices,
    out_key,
    out_value,
    sk0,
    sk1,
    sk2,
    sv0,
    sv1,
    sv2,
    bk0,
    bk1,
    bk2,
    bv0,
    bv1,
    bv2,
    SPAN_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head, dim = columns // HEAD_DIM, columns % HEAD_DIM
    index = tl.load(indices + row)
    in_span = index < SPAN_ROWS
    mask = columns < WIDTH
    sk = tl.load(
        span_key + index * sk0 + head * sk1 + dim * sk2, mask=mask & in_span, other=0
    )
    sv = tl.load(
        span_value + index * sv0 + head * sv1 + dim * sv2, mask=mask & in_span, other=0
    )
    block_row = index - SPAN_ROWS
    bk = tl.load(
        block_key + block_row * bk0 + head * bk1 + dim * bk2,
        mask=mask & ~in_span,
        other=0,
    )
    bv = tl.load(
        block_value + block_row * bv0 + head * bv1 + dim * bv2,
        mask=mask & ~in_span,
        other=0,
    )
    tl.store(out_key + row * WIDTH + columns, tl.where(in_span, sk, bk), mask=mask)
    tl.store(out_value + row * WIDTH + columns, tl.where(in_span, sv, bv), mask=mask)


def gather_context_kv(span_key, span_value, key, value, indices):
    """Gather rows of cat(span, block) without materializing either concatenation."""
    span_rows = 0 if span_key is None else span_key.shape[0]
    if span_key is None:
        span_key, span_value = key, value
    shape = (indices.numel(), *key.shape[1:])
    keys = torch.empty(shape, device=key.device, dtype=key.dtype)
    values = torch.empty_like(keys)
    width = key.shape[1] * key.shape[2]
    _gather_context_kv[(indices.numel(), triton.cdiv(width, 256))](
        span_key,
        span_value,
        key,
        value,
        indices,
        keys,
        values,
        *span_key.stride(),
        *span_value.stride(),
        *key.stride(),
        *value.stride(),
        SPAN_ROWS=span_rows,
        WIDTH=width,
        HEAD_DIM=key.shape[2],
        BLOCK=256,
    )
    return keys, values


@triton.jit
def _scatter_context_kv(
    key,
    value,
    rows,
    out_key,
    out_value,
    kg,
    kr,
    kh,
    kd,
    vg,
    vr,
    vh,
    vd,
    out_group,
    count,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    group, row = program // count, program % count
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head, dim = columns // HEAD_DIM, columns % HEAD_DIM
    mask = columns < WIDTH
    k = tl.load(key + group * kg + row * kr + head * kh + dim * kd, mask=mask, other=0)
    v = tl.load(value + group * vg + row * vr + head * vh + dim * vd, mask=mask, other=0)
    target = group * out_group + tl.load(rows + row) * WIDTH + columns
    tl.store(out_key + target, k, mask=mask)
    tl.store(out_value + target, v, mask=mask)


def scatter_context_kv(key, value, out_key, out_value, rows):
    """out[g, rows[r]] = source[g, r] for keys and values of every group g in one launch.

    Sources are [G, R, heads, head_dim] with any strides; outputs are contiguous
    [G, capacity, heads, head_dim]. Rows that repeat a target leave one of their values there.
    """
    groups, count, heads, head_dim = key.shape
    if not count:
        return
    width = heads * head_dim
    _scatter_context_kv[(groups * count, triton.cdiv(width, 256))](
        key,
        value,
        rows,
        out_key,
        out_value,
        *key.stride(),
        *value.stride(),
        out_key.stride(0),
        count,
        WIDTH=width,
        HEAD_DIM=head_dim,
        BLOCK=256,
    )


@triton.jit
def _norm_rope_place(
    source,
    q_weight,
    k_weight,
    positions,
    rows,
    rope,
    q_out,
    k_out,
    v_out,
    source_group,
    source_row,
    k_weight_group,
    out_group,
    count,
    eps,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    group, row = program // count, program % count
    slot = tl.program_id(1)
    is_q = slot < Q_HEADS
    is_k = (slot >= Q_HEADS) & (slot < Q_HEADS + KV_HEADS)
    is_v = slot >= Q_HEADS + KV_HEADS
    columns = tl.arange(0, HALF)
    base = source + group * source_group + row * source_row + slot * HEAD_DIM
    a = tl.load(base + columns)
    b = tl.load(base + HALF + columns)
    # RMS norm as vllm's rms_norm: FP32 statistics, one rounding of x * scale * weight.
    af, bf = a.to(tl.float32), b.to(tl.float32)
    scale = tl.rsqrt((tl.sum(af * af, axis=0) + tl.sum(bf * bf, axis=0)) / HEAD_DIM + eps)
    k_norm = k_weight + group * k_weight_group
    wa = tl.where(is_q, tl.load(q_weight + columns), tl.load(k_norm + columns)).to(tl.float32)
    wb = tl.where(is_q, tl.load(q_weight + HALF + columns), tl.load(k_norm + HALF + columns)).to(tl.float32)
    na = (af * scale * wa).to(a.dtype).to(tl.float32)
    nb = (bf * scale * wb).to(a.dtype).to(tl.float32)
    # Neox rotation of the rounded norm output, as vllm's rotary_embedding.
    angle = rope + tl.load(positions + row) * HEAD_DIM
    cos = tl.load(angle + columns).to(tl.float32)
    sin = tl.load(angle + HALF + columns).to(tl.float32)
    oa = tl.where(is_v, a, (na * cos - nb * sin).to(a.dtype))
    ob = tl.where(is_v, b, (nb * cos + na * sin).to(a.dtype))
    every = columns >= 0
    q_at = q_out + row * (Q_HEADS * HEAD_DIM) + slot * HEAD_DIM
    tl.store(q_at + columns, oa, mask=every & is_q)
    tl.store(q_at + HALF + columns, ob, mask=every & is_q)
    target = group * out_group + tl.load(rows + row) * (KV_HEADS * HEAD_DIM)
    k_at = k_out + target + (slot - Q_HEADS) * HEAD_DIM
    tl.store(k_at + columns, oa, mask=every & is_k)
    tl.store(k_at + HALF + columns, ob, mask=every & is_k)
    v_at = v_out + target + (slot - Q_HEADS - KV_HEADS) * HEAD_DIM
    tl.store(v_at + columns, oa, mask=every & is_v)
    tl.store(v_at + HALF + columns, ob, mask=every & is_v)


def norm_rope_place(source, q_weight, k_weight, positions, rows, rope, q_out, k_out, v_out, q_heads, kv_heads,
                    head_dim, eps):
    """Query/key RMS norm, RoPE and placement of one projection output in a single launch.

    ``source`` is [R, G, (q_heads + 2 * kv_heads) * head_dim]: per row and group the query heads,
    the key heads, then the value heads. Queries are normed with ``q_weight`` [head_dim], rotated by
    ``positions[r]`` and stored at q_out[r]; keys are normed with ``k_weight`` [G, head_dim], rotated
    and stored at k_out[g, rows[r]]; values are copied to v_out[g, rows[r]]. Outputs are contiguous;
    ``q_weight`` and ``q_out`` are unused when q_heads is 0.
    """
    count, groups = source.shape[0], source.shape[1]
    if not count:
        return
    if q_weight is None:
        q_weight, q_out = k_weight, k_out
    _norm_rope_place[(groups * count, q_heads + 2 * kv_heads)](
        source,
        q_weight,
        k_weight,
        positions,
        rows,
        rope,
        q_out,
        k_out,
        v_out,
        source.stride(1),
        source.stride(0),
        k_weight.stride(0) if k_weight.ndim == 2 else 0,
        k_out.stride(0) if k_out.ndim == 4 else 0,
        count,
        eps,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        HALF=head_dim // 2,
    )


@triton.jit
def _add_rms_norm(a, b, weight, out_sum, out_norm, eps, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * WIDTH
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    x = (tl.load(a + base + columns, mask=mask, other=0).to(tl.float32)
         + tl.load(b + base + columns, mask=mask, other=0).to(tl.float32)).to(a.dtype.element_ty)
    tl.store(out_sum + base + columns, x, mask=mask)
    x = x.to(tl.float32)  # the norm reads the rounded sum, as separate add and rms_norm kernels do
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / WIDTH + eps)
    w = tl.load(weight + columns, mask=mask, other=0).to(tl.float32)
    tl.store(out_norm + base + columns, (x * scale * w).to(a.dtype.element_ty), mask=mask)


def add_rms_norm(a, b, weight, eps):
    """(a + b, rms_norm(a + b) * weight) for contiguous [R, width] inputs in one launch."""
    width = a.shape[1]
    out_sum, out_norm = torch.empty_like(a), torch.empty_like(a)
    _add_rms_norm[(a.shape[0],)](a, b, weight, out_sum, out_norm, eps, WIDTH=width,
                                 BLOCK=triton.next_power_of_2(width))
    return out_sum, out_norm


@triton.jit
def _scaled_max(logits, scale, out_value, out_index, row_stride, chunks, VOCAB: tl.constexpr,
                CHUNK: tl.constexpr, SCALED: tl.constexpr):
    row, chunk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    columns = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = columns < VOCAB
    x = tl.load(logits + row * row_stride + columns, mask=mask, other=float("-inf")).to(tl.float32)
    if SCALED:
        x = x * tl.load(scale + columns, mask=mask, other=1.0)
    tl.store(out_value + row * chunks + chunk, tl.max(x, axis=0))
    tl.store(out_index + row * chunks + chunk, (tl.argmax(x, axis=0) + chunk * CHUNK).to(tl.int64))


@triton.jit
def _first_max(value, index, out, chunks, CHUNKS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, CHUNKS)
    x = tl.load(value + row * chunks + columns, mask=columns < chunks, other=float("-inf"))
    tl.store(out + row, tl.load(index + row * chunks + tl.argmax(x, axis=0)))


def scaled_argmax(logits, scale=None, chunk=4096):
    """argmax over the last dimension of logits * scale (FP32 product), first index on ties.

    The same IDs as ``(logits.float() * scale).argmax(-1)`` without materialising the product.
    """
    rows, vocab = logits.shape
    chunks = triton.cdiv(vocab, chunk)
    value = torch.empty(rows, chunks, device=logits.device, dtype=torch.float32)
    index = torch.empty(rows, chunks, device=logits.device, dtype=torch.long)
    out = torch.empty(rows, device=logits.device, dtype=torch.long)
    _scaled_max[(rows, chunks)](logits, logits if scale is None else scale, value, index, logits.stride(0), chunks,
                                VOCAB=vocab, CHUNK=chunk, SCALED=scale is not None)
    _first_max[(rows,)](value, index, out, chunks, CHUNKS=triton.next_power_of_2(chunks))
    return out


@triton.jit
def _local_merge_attention(
    q,
    keys,
    values,
    gather,
    bounds,
    prefix_out,
    prefix_lse,
    out,
    key_stride,
    lse_stride,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    KEYS: tl.constexpr,
):
    leaf = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    rows = tl.arange(0, ROWS)
    row_live = rows < BLOCK
    dims = tl.arange(0, HEAD_DIM)
    cell = (leaf * BLOCK + rows)[:, None] * (HEADS * HEAD_DIM) + head * HEAD_DIM + dims[None, :]
    query = tl.load(q + cell, mask=row_live[:, None], other=0)
    start = tl.load(bounds + leaf)
    slots = tl.arange(0, KEYS)
    live = slots < tl.load(bounds + leaf + 1) - start
    source = tl.load(gather + start + slots, mask=live, other=0)[:, None] * key_stride \
        + (head // GROUP) * HEAD_DIM + dims[None, :]
    key = tl.load(keys + source, mask=live[:, None], other=0)
    value = tl.load(values + source, mask=live[:, None], other=0)
    # FlashAttention's forward for one key block: FP32 scores, exp2 against the row maximum, the
    # weights rounded to the input type for the value product, FP32 accumulation, one division.
    scores = tl.where(live[None, :], tl.dot(query, tl.trans(key)), float("-inf"))
    peak = tl.max(scores, axis=1)
    weights = tl.exp2(scores * SCALE_LOG2 - peak[:, None] * SCALE_LOG2)
    total = tl.sum(weights, axis=1)
    local = tl.dot(weights.to(query.dtype), value) * (1.0 / total)[:, None]
    local = local.to(query.dtype).to(tl.float32)
    local_lse = peak * SCALE + tl.log(total)
    # merge_attn_states with the prefix attention of the same rows.
    lse = tl.load(prefix_lse + head * lse_stride + leaf * BLOCK + rows, mask=row_live, other=0.0)
    top = tl.maximum(lse, local_lse)
    prefix_weight, local_weight = tl.exp(lse - top), tl.exp(local_lse - top)
    norm = prefix_weight + local_weight
    prefix = tl.load(prefix_out + cell, mask=row_live[:, None], other=0).to(tl.float32)
    merged = prefix * (prefix_weight / norm)[:, None] + local * (local_weight / norm)[:, None]
    tl.store(out + cell, merged.to(query.dtype), mask=row_live[:, None])


def local_merge_attention(q, keys, values, gather, bounds, prefix_out, prefix_lse, block, limit):
    """Each leaf's block attends to its own keys, merged with its prefix attention, in one launch.

    ``q`` and ``prefix_out`` are [L * block, heads, head_dim], ``prefix_lse`` [heads, L * block];
    leaf i reads rows gather[bounds[i]:bounds[i + 1]] (between one and ``limit``) of
    ``keys``/``values`` [N, kv_heads, head_dim]. The arithmetic follows FlashAttention's one-block
    forward and merge_attn_states, so the result differs from those two kernels by FP32 rounding only.
    """
    rows, heads, head_dim = q.shape
    out = torch.empty_like(q)
    _local_merge_attention[(rows // block, heads)](
        q,
        keys,
        values,
        gather,
        bounds,
        prefix_out,
        prefix_lse,
        out,
        keys.stride(0),
        prefix_lse.stride(0),
        SCALE=head_dim**-0.5,
        SCALE_LOG2=head_dim**-0.5 * 1.4426950408889634,
        HEADS=heads,
        GROUP=heads // keys.shape[1],
        HEAD_DIM=head_dim,
        BLOCK=block,
        ROWS=max(16, triton.next_power_of_2(block)),
        KEYS=max(16, triton.next_power_of_2(limit)),
    )
    return out
