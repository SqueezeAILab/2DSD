# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Triton kernels of the Dory tree runner's paged attention.

``range_path_attention(q, prefix_k, prefix_v, tree_k, tree_v, plane, lo, hi, path,
path_mask) -> (out, lse)`` with q ``[n, heads, d]``, prefix K/V ``[planes, C, kv, d]``,
tree K/V ``[planes, N, kv, d]`` (contiguous views of one layer), plane/lo/hi ``[n]``
and path/path_mask ``[n, w]``.

A compact prefix (``ring`` R > 0) is one layer's rows ``[rows, kv, d]``: plane p
keeps position j at row ``p * R + j % R``, except the plane ``bank``, which keeps
every position at ``bank * R + j`` (``capacity`` positions after the rings).

Row i attends to ``prefix_k[plane[i], j]`` for ``lo[i] <= j < hi[i]`` (clamped to
``[0, C)``; none when ``lo >= hi``) and to ``tree_k[plane[i], path[i, e]]`` wherever
``path_mask[i, e]``. Query head h reads KV head ``h // (heads // kv)``. Scores are
``q . k / sqrt(d)``; K/V/q are loaded in their own dtype (BF16) and the softmax runs
in FP32. ``out`` ``[n, heads, d]`` is normalised over these keys and ``lse``
``[n, heads]`` (FP32) is their log-sum-exp, for merging with another partial result.

plane/lo/hi may also be ``[r, n]``: row i then reads the r ranges in order, each from
its own plane, and the path from the last range's plane (``r = 0``: no prefix keys).

``partials=(out, lse, base, live, stride)`` continues the softmax of results already
normalised over disjoint keys (FlashAttention, or this kernel with ``segments``):
``out`` ``[T, heads, d]`` and ``lse`` ``[heads, T]``. Row i owns the entries
``base[i] + p * stride`` for ``p < live[i]`` (``p`` below the static bound
``parts``); they enter the running maximum, denominator and numerator exactly as a
chunk of keys does, so the result is their LSE merge with the keys read here.

``segments=(count, keys)`` reads in parallel over keys: program ``(i, s)`` takes the
keys ``lo[i] + s * keys <= j < lo[i] + (s + 1) * keys`` of every range (the last one
also the path) and the call returns the ``n * count`` partial results in the layout
of ``partials`` (FP32 ``out`` ``[n * count, heads, d]``, ``lse``
``[heads, n * count]``; row i owns ``i * count + s``).

``store_rows`` writes the rows' new K/V into the tree, ``commit_rows`` copies
committed tokens into the prefix (rings included) and ``plan_reads`` derives the
per-row ranges and the FlashAttention sequences of a call (see the functions).

Launch grids depend on shapes only and all row quantities are data, so every call
can be captured in a CUDA graph. This module imports no vLLM code.
"""

import math

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

# (keys per chunk, pipeline stages, warps). The first needs ~137 KB of shared
# memory, i.e. one resident program per SM; the second packs several per SM.
_ONE_PER_SM = (64, 3, 4)
_PACKED = (32, 2, 4)
_packed_only: set[torch.device] = set()

SLIDING, SHARED, NATIVE = 0, 1, 2  # read rules of ``plan_reads``


@triton.jit
def _merge_chunk(
    q, k, v, visible, maximum, denominator, numerator, SCALE: tl.constexpr
):
    """Online-softmax update with one chunk of keys ``[BLOCK, d]``.

    An FP32 weight is the exact sum of three BF16 pieces. Every (piece, query head)
    takes one row of the tensor-core tile, so the BF16 values are accumulated in
    FP32 by a single product without an FP32 copy.
    """
    GROUP: tl.constexpr = q.shape[0]
    BLOCK: tl.constexpr = k.shape[0]
    PIECES: tl.constexpr = numerator.shape[0] // GROUP
    scores = tl.dot(q, tl.trans(k)) * SCALE
    scores = tl.where(visible[None, :], scores, float("-inf"))
    peak = tl.maximum(maximum, tl.max(scores, 1))
    weights = tl.where(visible[None, :], tl.exp(scores - peak[:, None]), 0.0)
    decay = tl.where(maximum == float("-inf"), 0.0, tl.exp(maximum - peak))
    piece_of = tl.arange(0, PIECES)[:, None, None]
    residual = weights
    pieces = tl.zeros((PIECES, GROUP, BLOCK), tl.bfloat16)
    for piece in tl.static_range(3):
        part = residual.to(tl.bfloat16)
        pieces = tl.where(piece_of == piece, part[None, :, :], pieces)
        residual -= part.to(tl.float32)
    carried = tl.reshape(
        tl.broadcast_to(decay[None, :], (PIECES, GROUP)), (PIECES * GROUP,)
    )
    numerator = tl.dot(
        tl.reshape(pieces, (PIECES * GROUP, BLOCK)),
        v,
        acc=numerator * carried[:, None],
    )
    denominator = denominator * decay + tl.sum(weights, 1)
    return peak, denominator, numerator


@triton.jit(
    do_not_specialize=["rows", "prefix_len", "tree_len", "part_stride", "part_total"]
)  # prefix_len: positions of a full plane (the capacity)
def _range_path_attention(
    query,
    prefix_k,
    prefix_v,
    tree_k,
    tree_v,
    planes,
    lows,
    highs,
    paths,
    path_mask,
    part_out,
    part_lse,
    part_base,
    part_live,
    output,
    logsumexp,
    rows,
    prefix_len,
    tree_len,
    part_stride,
    part_total,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RANGES: tl.constexpr,
    PARTS: tl.constexpr,
    SEGMENTS: tl.constexpr,
    SEGMENT_KEYS: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PATH_BLOCK: tl.constexpr,
    RING: tl.constexpr,
    BANK: tl.constexpr,
):
    cell = tl.program_id(0)
    kv_head = tl.program_id(1)
    row = cell // SEGMENTS
    segment = cell % SEGMENTS
    GROUP: tl.constexpr = HEADS // KV_HEADS
    PIECES: tl.constexpr = 4
    STRIDE: tl.constexpr = KV_HEADS * HEAD_DIM
    heads = kv_head * GROUP + tl.arange(0, GROUP)
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(query + (row * HEADS + heads[:, None]) * HEAD_DIM + dims[None, :])
    maximum = tl.full((GROUP,), float("-inf"), tl.float32)
    denominator = tl.zeros((GROUP,), tl.float32)
    numerator = tl.zeros((PIECES * GROUP, HEAD_DIM), tl.float32)

    if PARTS > 0:
        # Results normalised elsewhere: each enters as one key of weight exp(lse).
        first = tl.load(part_base + row)
        live = tl.load(part_live + row)
        merged = tl.zeros((GROUP, HEAD_DIM), tl.float32)
        for part in tl.static_range(PARTS):
            item = first + part * part_stride
            alive = (heads >= 0) & (part < live)
            lse = tl.load(
                part_lse + heads * part_total + item, mask=alive, other=float("-inf")
            )
            cells = (item * HEADS + heads[:, None]) * HEAD_DIM + dims[None, :]
            value = tl.load(part_out + cells, mask=alive[:, None], other=0)
            peak = tl.maximum(maximum, lse)
            decay = tl.where(maximum == float("-inf"), 0.0, tl.exp(maximum - peak))
            gain = tl.where(lse == float("-inf"), 0.0, tl.exp(lse - peak))
            merged = merged * decay[:, None] + value.to(tl.float32) * gain[:, None]
            denominator = denominator * decay + gain
            maximum = peak
        piece_of = tl.arange(0, PIECES)[:, None, None]
        numerator = tl.reshape(
            tl.where(
                piece_of == 0,
                tl.broadcast_to(merged[None, :, :], (PIECES, GROUP, HEAD_DIM)),
                0.0,
            ),
            (PIECES * GROUP, HEAD_DIM),
        )

    keys = tl.arange(0, BLOCK_K)
    plane = 0
    for span in tl.static_range(RANGES):
        plane = tl.load(planes + span * rows + row)
        low = tl.maximum(tl.load(lows + span * rows + row), 0)
        high = tl.minimum(tl.load(highs + span * rows + row), prefix_len)
        if SEGMENTS > 1:
            low += segment * SEGMENT_KEYS
            high = tl.minimum(high, low + SEGMENT_KEYS)
        if RING > 0:
            first_row = plane * RING
        else:
            first_row = plane * prefix_len
        base = (first_row * KV_HEADS + kv_head) * HEAD_DIM
        for start in range(low, high, BLOCK_K):
            position = start + keys
            visible = position < high
            row_of = position
            if RING > 0:
                if BANK >= 0:
                    row_of = tl.where(plane == BANK, position, position % RING)
                else:
                    row_of = position % RING
            offsets = base + row_of[:, None] * STRIDE + dims[None, :]
            k = tl.load(prefix_k + offsets, mask=visible[:, None], other=0)
            v = tl.load(prefix_v + offsets, mask=visible[:, None], other=0)
            maximum, denominator, numerator = _merge_chunk(
                q, k, v, visible, maximum, denominator, numerator, SCALE
            )

    if PATH_WIDTH > 0:
        entries = tl.arange(0, PATH_BLOCK)
        base = (plane * tree_len * KV_HEADS + kv_head) * HEAD_DIM
        for start in tl.static_range(0, PATH_WIDTH, PATH_BLOCK):
            entry = start + entries
            inside = (entry < PATH_WIDTH) & (segment == SEGMENTS - 1)
            slots = tl.load(paths + row * PATH_WIDTH + entry, mask=inside, other=0)
            visible = tl.load(
                path_mask + row * PATH_WIDTH + entry, mask=inside, other=0
            )
            if tl.sum(visible.to(tl.int32)) > 0:
                offsets = base + slots[:, None] * STRIDE + dims[None, :]
                k = tl.load(tree_k + offsets, mask=visible[:, None], other=0)
                v = tl.load(tree_v + offsets, mask=visible[:, None], other=0)
                maximum, denominator, numerator = _merge_chunk(
                    q, k, v, visible, maximum, denominator, numerator, SCALE
                )

    total = tl.sum(tl.reshape(numerator, (PIECES, GROUP, HEAD_DIM)), 0)
    result = total / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        output + (cell * HEADS + heads[:, None]) * HEAD_DIM + dims[None, :], result
    )
    total_lse = maximum + tl.log(denominator)
    if SEGMENTS > 1:
        tl.store(logsumexp + heads * rows * SEGMENTS + cell, total_lse)
    else:
        tl.store(logsumexp + row * HEADS + heads, total_lse)


def range_path_attention(
    q: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    tree_k: torch.Tensor,
    tree_v: torch.Tensor,
    plane: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    path: torch.Tensor,
    path_mask: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    tuning: tuple[int, int, int] | None = None,
    partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    | None = None,
    parts: int | None = None,
    segments: tuple[int, int] | None = None,
    ring: int = 0,
    bank: int = -1,
    capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend each row to prefix ranges and its in-flight path (module docstring).

    Args:
        out_dtype: dtype of ``out``; defaults to q's (FP32 with ``segments``). FP32
            skips the final rounding.
        tuning: ``(keys per chunk, pipeline stages, warps)``; chosen from the
            program count and the device when omitted.
        partials: ``(out, lse, base, live, stride)`` of results to continue from.
        parts: static bound on ``live``; defaults to ``T // stride``.
        segments: ``(count, keys)``: return ``count`` partial results per row, each
            over ``keys`` positions of every range (``count * keys`` covers them).
        ring, bank: compact prefix rows ``[rows, kv, d]`` (module docstring).
        capacity: positions of a full plane; defaults to ``prefix_k.shape[1]`` of a
            ``[planes, C, kv, d]`` prefix. Ranges are clamped to it.

    Returns:
        ``out`` ``[n, heads, d]`` and ``lse`` ``[n, heads]`` (FP32). A row without
        any key gets zeros and ``-inf``. With ``segments``: the module docstring.
    """
    n, heads, dim = q.shape
    kv_heads, width = prefix_k.shape[-2], path.shape[1]
    if capacity is None:
        assert prefix_k.dim() == 4 and ring == 0
        capacity = prefix_k.shape[1]
    group = heads // kv_heads
    tensors = (q, prefix_k, prefix_v, tree_k, tree_v, plane, lo, hi, path, path_mask)
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert dim >= 16 and dim & (dim - 1) == 0 and group & (group - 1) == 0
    assert plane.shape == lo.shape == hi.shape and plane.shape[-1] == n
    ranges = 1 if plane.dim() == 1 else plane.shape[0]
    count, span = segments or (1, 0)
    cells = n * count
    if segments is not None:
        assert count > 1 and partials is None
        out_dtype = out_dtype or torch.float32
    out = torch.empty((cells, heads, dim), device=q.device, dtype=out_dtype or q.dtype)
    shape = (cells, heads) if segments is None else (heads, cells)
    lse = torch.empty(shape, device=q.device, dtype=torch.float32)
    if partials is None:
        part_out, part_lse, base, live, stride, total, parts = q, lse, q, q, 0, 0, 0
    else:
        part_out, part_lse, base, live, stride = partials
        total = part_out.shape[0]
        parts = max(total // stride, 1) if parts is None else parts
        assert part_lse.shape == (heads, total) and part_out.shape[1:] == q.shape[1:]
        assert base.shape == live.shape == (n,)
        continued = (part_out, part_lse, base, live)
        assert all(tensor.is_contiguous() for tensor in continued)

    def launch(block: int, stages: int, warps: int) -> None:
        _range_path_attention[(cells, kv_heads)](
            q, prefix_k, prefix_v, tree_k, tree_v, plane, lo, hi, path, path_mask,
            part_out, part_lse, base, live, out, lse,
            n, capacity, tree_k.shape[1], stride, total,
            HEADS=heads, KV_HEADS=kv_heads, HEAD_DIM=dim, RANGES=ranges, PARTS=parts,
            SEGMENTS=count, SEGMENT_KEYS=span, PATH_WIDTH=width,
            SCALE=1.0 / math.sqrt(dim), BLOCK_K=block,
            PATH_BLOCK=max(16, min(block, triton.next_power_of_2(width))),
            RING=ring, BANK=bank, num_stages=stages, num_warps=warps,
        )

    if tuning is not None:
        launch(*tuning)
        return out, lse
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    if cells * kv_heads <= sms and q.device not in _packed_only:
        try:
            launch(*_ONE_PER_SM)
            return out, lse
        except OutOfResources:
            _packed_only.add(q.device)
    launch(*_PACKED)
    return out, lse


@triton.jit(do_not_specialize=["tree_len", "key_stride", "value_stride"])
def _store_rows(
    keys,
    values,
    tree_k,
    tree_v,
    planes,
    slots,
    tree_len,
    key_stride,
    value_stride,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cells = tl.arange(0, BLOCK)
    inside = cells < WIDTH
    target = (tl.load(planes + row) * tree_len + tl.load(slots + row)) * WIDTH + cells
    k = tl.load(keys + row * key_stride + cells, mask=inside)
    v = tl.load(values + row * value_stride + cells, mask=inside)
    tl.store(tree_k + target, k, mask=inside)
    tl.store(tree_v + target, v, mask=inside)


def store_rows(
    k: torch.Tensor,
    v: torch.Tensor,
    tree_k: torch.Tensor,
    tree_v: torch.Tensor,
    plane: torch.Tensor,
    slot: torch.Tensor,
) -> None:
    """``tree_k[plane[i], slot[i]] = k[i]`` and the same for V, in one launch.

    k/v are ``[n, kv, d]`` (rows may be strided), tree K/V ``[planes, N, kv, d]``.
    Rows that share a target (padding rows on the scratch slot) race benignly.
    """
    n, kv_heads, dim = k.shape
    width = kv_heads * dim
    inner = (dim, 1)
    assert k.shape == v.shape and k.stride()[1:] == inner and v.stride()[1:] == inner
    assert tree_k.is_contiguous() and tree_v.is_contiguous()
    assert plane.is_contiguous() and slot.is_contiguous()
    _store_rows[(n,)](
        k, v, tree_k, tree_v, plane, slot, tree_k.shape[1], k.stride(0), v.stride(0),
        WIDTH=width, BLOCK=triton.next_power_of_2(width),
    )


@triton.jit(do_not_specialize=["slot", "length", "nodes", "rows"])
def _commit_rows(
    tree,
    prefix,
    slots,
    slot,
    bases,
    rings,
    length,
    nodes,
    rows,
    PLANES: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE: tl.constexpr,
):
    plane = tl.program_id(0)
    index = tl.program_id(1)
    kv = tl.program_id(2)
    source_slot = slot if SINGLE else tl.load(slots + index)
    position = length + index
    ring = tl.load(rings + plane)
    row = tl.load(bases + plane) + tl.where(
        ring > 0, position % tl.maximum(ring, 1), position
    )
    cells = tl.arange(0, BLOCK)
    inside = cells < WIDTH
    source = ((kv * PLANES + plane) * nodes + source_slot).to(tl.int64) * WIDTH
    target = (kv * rows + row).to(tl.int64) * WIDTH
    values = tl.load(tree + source + cells, mask=inside)
    tl.store(prefix + target + cells, values, mask=inside)


def commit_rows(
    tree: torch.Tensor,
    prefix: torch.Tensor,
    slots: torch.Tensor | int,
    count: int,
    bases: torch.Tensor,
    rings: torch.Tensor,
    length: int,
) -> None:
    """Copy committed tokens from the tree into the prefix in one launch.

    ``tree`` is ``[2, P, N, kv, d]`` (every layer and plane, by slot), ``prefix``
    ``[2, rows, kv, d]``. Token i (slot ``slots[i]``, or the int ``slots`` when
    ``count`` is 1) goes to position ``length + i`` of each (layer, plane) p:
    row ``bases[p] + position``, or ``bases[p] + position % rings[p]`` on a ring.
    """
    planes, nodes = tree.shape[1], tree.shape[2]
    width = tree.shape[3] * tree.shape[4]
    assert tree.is_contiguous() and prefix.is_contiguous()
    assert bases.shape == rings.shape == (planes,)
    single = isinstance(slots, int)
    _commit_rows[(planes, count, 2)](
        tree, prefix, bases if single else slots, slots if single else 0, bases,
        rings, length, nodes, prefix.shape[1],
        PLANES=planes, WIDTH=width, BLOCK=triton.next_power_of_2(width),
        SINGLE=single,
    )


@triton.jit(do_not_specialize=["rows", "per_plane"])
def _plan_reads(
    positions,
    planes,
    padding,
    length,
    gather,
    base,
    bounds,
    used,
    table,
    plane_out,
    lows,
    highs,
    live,
    rows,
    per_plane,
    ROWS: tl.constexpr,
    PLANES: tl.constexpr,
    RULE: tl.constexpr,
    REACH: tl.constexpr,
    COPIES: tl.constexpr,
    KEYS: tl.constexpr,
    BLOCKS: tl.constexpr,
    SPAN: tl.constexpr,
    TABLE: tl.constexpr,
    RING_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    others = tl.arange(0, ROWS)
    inside = others < rows
    every_plane = tl.load(planes + others, mask=inside, other=PLANES)
    position = tl.load(positions + row)
    plane = tl.load(planes + row)
    end = tl.load(length)
    GROUPS: tl.constexpr = 1 if RULE == 1 else PLANES

    # Rows grouped by plane: a row's place among them (its own index when every
    # row reads one plane).
    place = row
    if GROUPS > 1:
        ahead = (every_plane < plane) | ((every_plane == plane) & (others < row))
        place = tl.sum(ahead.to(tl.int32))
        for copy in tl.static_range(COPIES):
            tl.store(gather + copy * rows + place, row)
    tl.store(base + row, place)

    origin = end * 0
    if RULE == 0:
        # Sliding: the window start of the latest row, up to a block boundary.
        latest = tl.max(tl.load(positions + others, mask=inside, other=0))
        origin = ((tl.maximum(latest - REACH, 0) + 15) // 16) * 16
        total = end - origin
        tl.store(plane_out + row, plane)
        tl.store(lows + row, tl.maximum(position - REACH, 0))
        tl.store(highs + row, origin)
    elif RULE == 1:
        # Shared: final plane up to t - W, the row's plane above (W = REACH + 1).
        total = end - REACH
        skip = tl.load(padding + row).to(tl.int32)
        start = tl.maximum(total, 0)
        split = tl.minimum(tl.maximum(position - REACH, start), end)
        split = tl.where((plane != PLANES - 1) & (skip == 0), split, end)
        start = tl.where(skip != 0, end, start)
        tl.store(plane_out + row, PLANES - 1)
        tl.store(lows + row, start)
        tl.store(highs + row, split)
        tl.store(plane_out + rows + row, plane)
        tl.store(lows + rows + row, split)
        tl.store(highs + rows + row, end)
    else:
        total = end
        tl.store(plane_out + row, plane)
        tl.store(lows + row, end)
        tl.store(highs + row, end)
    count = tl.minimum(tl.maximum((total + KEYS - 1) // KEYS, 0), COPIES)
    tl.store(live + row, count)

    if row == 0:
        cells = tl.arange(0, SPAN)
        for group in tl.static_range(GROUPS):
            before = 0
            if GROUPS > 1:
                before = tl.sum((every_plane < group).to(tl.int32))
            source = PLANES - 1 if RULE == 1 else group
            for copy in tl.static_range(COPIES):
                sequence = copy * GROUPS + group
                tl.store(bounds + sequence, copy * rows + before)
                keys = tl.minimum(tl.maximum(total - copy * KEYS, 1), KEYS)
                tl.store(used + sequence, keys)
                if TABLE:
                    block = (origin + copy * KEYS) // 16 + cells
                    if RING_BLOCKS > 0:  # a compact plane: a ring of blocks
                        block = block % RING_BLOCKS
                    block = tl.minimum(source * per_plane + block, PLANES * per_plane - 1)
                    entry = table + sequence * BLOCKS + cells
                    tl.store(entry, block, mask=cells < BLOCKS)
        tl.store(bounds + COPIES * GROUPS, COPIES * rows)


def plan_reads(
    positions: torch.Tensor,
    planes: torch.Tensor,
    padding: torch.Tensor,
    length: torch.Tensor,
    *,
    rule: int,
    plane_count: int,
    reach: int,
    copies: int,
    keys: int,
    per_plane: int,
    table: torch.Tensor | None = None,
    ring_blocks: int = 0,
) -> dict[str, torch.Tensor | None]:
    """Reads of one call for one attention kind, from the rows' metadata.

    FlashAttention reads ``copies`` sequences of ``keys`` prefix positions per
    group of rows, the kernel continues with ``plane``/``lo``/``hi`` per row:

    - ``SLIDING`` (``reach`` = window - 1): a group is a plane's rows. The sequences
      start at the window start of the latest row, rounded up to a block of 16;
      a row's range is the leading edge of its own window before that.
    - ``SHARED`` (``reach`` = W - 1): one group of all rows reading the final plane
      below ``length - reach``; a row's ranges are the final plane up to its
      ``t - W`` and its own plane above (final-plane and padding rows: none).
    - ``NATIVE``: a group is a plane's rows, the sequences cover the prefix and
      the ranges are empty.

    Returns ``gather`` (rows by group, tiled per copy; ``None`` for one group),
    ``cu`` (query bounds of the ``copies * groups`` sequences, copy-major),
    ``used`` (keys per sequence, at least one), ``table`` (blocks of 16 positions
    of a ``[planes, C]`` prefix; computed only without a caller-supplied one, whose
    sequences must start at ``copy * keys``; ``ring_blocks`` > 0: each plane of
    ``per_plane`` = ``ring_blocks`` blocks is a ring), ``base``/``live`` (a row's first
    partial result and their count, stride ``n``) and ``plane``/``lo``/``hi``.
    """
    n, device = positions.shape[0], positions.device
    groups = 1 if rule == SHARED else plane_count
    ranges = 2 if rule == SHARED else 1
    sequences = copies * groups
    blocks = -(-keys // 16)

    def empty(*shape, dtype=torch.int64):
        return torch.empty(shape, device=device, dtype=dtype)

    plan = {
        "gather": empty(copies * n) if groups > 1 else None,
        "cu": empty(sequences + 1, dtype=torch.int32),
        "used": empty(sequences, dtype=torch.int32),
        "table": (
            empty(sequences, blocks, dtype=torch.int32) if table is None else table
        ),
        "base": empty(n),
        "live": empty(n),
        "plane": empty(ranges, n) if ranges > 1 else empty(n),
        "lo": empty(ranges, n) if ranges > 1 else empty(n),
        "hi": empty(ranges, n) if ranges > 1 else empty(n),
    }
    inputs = (positions, planes, padding, length)
    assert all(tensor.is_contiguous() for tensor in inputs)
    assert padding.dtype == torch.bool
    _plan_reads[(n,)](
        positions, planes, padding, length,
        plan["gather"] if groups > 1 else plan["base"], plan["base"], plan["cu"],
        plan["used"], plan["table"], plan["plane"], plan["lo"], plan["hi"],
        plan["live"], n, per_plane,
        ROWS=triton.next_power_of_2(n), PLANES=plane_count, RULE=rule, REACH=reach,
        COPIES=copies, KEYS=keys, BLOCKS=blocks, SPAN=triton.next_power_of_2(blocks),
        TABLE=table is None, RING_BLOCKS=ring_blocks,
    )
    return plan
