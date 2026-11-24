# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Triton kernel of the Inkling 2 static runner's attention.

``tree_attention(query, rel, gate, sink, prefix_k, prefix_v, tree_k, tree_v,
position, lowest, plane, low_plane, cut, path, path_mask, length, origin) -> out``
is the gated relative-bias attention of every row of a call over its context,
in one launch and with the keys and values read in place:

  query, gate [n, heads, D]      the clipped (and tau-scaled) queries, the raw gates
  rel         [n, heads, extent]  relative logits by backward distance, as stored
  sink        [n, heads]          (gate and sink may be row-strided)
  prefix K/V  [planes, C, kv, D]  one layer's committed keys and values; row j of a
                                  plane holds absolute position ``origin + j``
  tree K/V    [planes, N, kv, D]  the same layer's in-flight keys and values by slot
  position, lowest, plane, low_plane, cut [n]; path, path_mask [n, w]
  length, origin [1]              positions committed, position of prefix row 0

All floating tensors hold BF16 values in BF16. Row i at absolute position
``position[i]`` reads the positions ``lowest[i] <= p <= position[i]``: a position
below ``length`` from the prefix, in ``low_plane[i]`` when ``p < cut[i]`` (last-loop
KV sharing) and in ``plane[i]`` otherwise, and position ``length + e`` from the
tree slot ``path[i, e]`` of ``plane[i]`` where ``path_mask[i, e]`` (a row's path
holds consecutive positions from ``length`` on, its own slot last). The output
[n, heads, D] (BF16) is softmax-weighted v times sigmoid(gate) times
sigmoid(logsumexp - sink).

The arithmetic and the order are those of ``inkling2_attention``: tiles of
``BLOCK_K`` positions aligned to absolute position 0, in ascending order, merged
by its ``_merge_tile``. A row therefore gets the bits that ``sequence_attention``
gives the token at its position over the same keys, whichever positions are in
the prefix and whichever in the tree, and whatever the other rows of the call
are.

``pack=True`` runs the kernel that holds the query heads of one KV head for
several rows per program (``LANES`` over the heads of a group). A committed tile
that rows of a program read from one plane is loaded once and merged for them
together; a tile at a sharing cut or in the tree is read row by row. Every pass
goes through one ``_merge_tile`` site and a lane's sums do not involve the other
lanes, so the rows that share a program do not change a row's bits. Reading a
long prefix once per program instead of once per row is what keeps a call of
many rows cheap on a long context; on a short window the per-row kernel is
faster. Use one of the two for a given layer in every call: they are compiled
apart, and only the same kernel is certain to round a row the same way.

``segments=(count, tiles)`` reads a row's tiles in ``count`` parallel programs
of ``tiles`` tiles each (the last one takes what is left) and a second launch
merges their results in order, as the served kernel splits its keys. The
softmax of each segment is then normalised on its own, so the bits are no
longer those of ``sequence_attention``; the arithmetic is the same.

Launch grids depend on shapes only and all row quantities are data, so a call
can be captured in a CUDA graph. This module imports no vLLM code besides that
kernel.
"""

import torch
import triton
import triton.language as tl

from vllm.model_executor.models.inkling2_attention import BLOCK_K, _merge_tile, _sigmoid

LANES = 16  # rows of a program's tensor-core tile: the query heads of one KV head, padded


@triton.jit(do_not_specialize=["prefix_len", "tree_len"])
def _tree_attention(
    query,
    rel,
    gate,
    sink,
    prefix_k,
    prefix_v,
    tree_k,
    tree_v,
    positions,
    lowest,
    planes,
    low_planes,
    cuts,
    paths,
    path_mask,
    length_at,
    origin_at,
    output,
    prefix_len,
    tree_len,
    stride_gate,
    stride_sink,
    part_lse,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EXTENT: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    ROWS: tl.constexpr,
    TILE: tl.constexpr,
    SEGMENTS: tl.constexpr,
    SEGMENT_TILES: tl.constexpr,
):
    cell = tl.program_id(0)
    row = cell // SEGMENTS
    segment = cell % SEGMENTS
    kv_head = tl.program_id(1)
    GROUP: tl.constexpr = HEADS // KV_HEADS
    lanes = tl.arange(0, ROWS)
    live = lanes < GROUP
    head = kv_head * GROUP + tl.minimum(lanes, GROUP - 1)
    dims = tl.arange(0, HEAD_DIM)
    cells = (row * HEADS + head)[:, None].to(tl.int64) * HEAD_DIM + dims[None, :]
    q = tl.load(query + cells, mask=live[:, None], other=0)
    rel_row = (row * HEADS + head).to(tl.int64) * EXTENT
    position = tl.load(positions + row)
    low = tl.maximum(tl.load(lowest + row), 0)
    plane = tl.load(planes + row)
    low_plane = tl.load(low_planes + row)
    cut = tl.load(cuts + row)
    length = tl.load(length_at)
    origin = tl.load(origin_at)

    maximum = tl.full((ROWS,), float("-inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    numerator = tl.zeros((ROWS, HEAD_DIM), tl.float32)

    column = tl.arange(0, TILE)
    first_tile = low // TILE
    end_tile = position // TILE + 1
    if SEGMENTS > 1:
        first_tile += segment * SEGMENT_TILES
        if segment < SEGMENTS - 1:
            end_tile = tl.minimum(end_tile, first_tile + SEGMENT_TILES)
    for tile in range(first_tile, end_tile):
        at = tile * TILE + column
        seen = (at >= low) & (at <= position)
        visible = seen & (at < length)
        source = tl.where(at < cut, low_plane, plane)
        offsets = ((source * prefix_len + (at - origin)) * KV_HEADS + kv_head)[:, None] * HEAD_DIM + dims[None, :]
        k = tl.load(prefix_k + offsets, mask=visible[:, None], other=0)
        v = tl.load(prefix_v + offsets, mask=visible[:, None], other=0)
        if tile * TILE + TILE > length:  # the tile reaches the tree
            entry = at - length
            inside = seen & (entry >= 0) & (entry < PATH_WIDTH)
            shown = tl.load(path_mask + row * PATH_WIDTH + entry, mask=inside, other=0)
            in_tree = inside & (shown != 0)
            slots = tl.load(paths + row * PATH_WIDTH + entry, mask=in_tree, other=0)
            offsets = ((plane * tree_len + slots) * KV_HEADS + kv_head)[:, None] * HEAD_DIM + dims[None, :]
            k = tl.where(in_tree[:, None], tl.load(tree_k + offsets, mask=in_tree[:, None], other=0), k)
            v = tl.where(in_tree[:, None], tl.load(tree_v + offsets, mask=in_tree[:, None], other=0), v)
            visible = visible | in_tree
        distance = position - at
        reads = live[:, None] & visible[None, :]
        near = reads & (distance[None, :] < EXTENT)
        bias = tl.load(rel + rel_row[:, None] + distance[None, :], mask=near, other=0).to(tl.float32)
        maximum, denominator, numerator = _merge_tile(
            q, k, v, bias, reads, maximum, denominator, numerator, SCALE
        )

    mixed = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    lse = maximum + tl.log(denominator)
    if SEGMENTS > 1:  # a partial result: the merge launch applies the gates
        tl.store(output + (cell * HEADS + head[:, None]).to(tl.int64) * HEAD_DIM + dims[None, :], mixed, mask=live[:, None])
        tl.store(part_lse + cell * HEADS + head, lse, mask=live)
        return
    gates = tl.load(
        gate + row * stride_gate + head[:, None].to(tl.int64) * HEAD_DIM + dims[None, :], mask=live[:, None], other=0
    ).to(tl.float32)
    sinks = tl.load(sink + row * stride_sink + head, mask=live, other=0).to(tl.float32)
    result = mixed * _sigmoid(gates) * _sigmoid(lse - sinks)[:, None]
    tl.store(output + cells, result.to(output.dtype.element_ty), mask=live[:, None])


@triton.jit(do_not_specialize=["prefix_len", "tree_len", "count"])
def _tree_attention_packed(
    query, rel, gate, sink, prefix_k, prefix_v, tree_k, tree_v, positions, lowest, planes, low_planes, cuts,
    paths, path_mask, length_at, origin_at, output, prefix_len, tree_len, count, stride_gate, stride_sink, part_lse,
    HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, EXTENT: tl.constexpr,
    PATH_WIDTH: tl.constexpr, SCALE: tl.constexpr, ROWS: tl.constexpr, PACK: tl.constexpr, TILE: tl.constexpr,
    SEGMENTS: tl.constexpr, SEGMENT_TILES: tl.constexpr,
):
    cell = tl.program_id(0)
    block = cell // SEGMENTS
    segment = cell % SEGMENTS
    kv_head = tl.program_id(1)
    GROUP: tl.constexpr = HEADS // KV_HEADS
    lanes = tl.arange(0, ROWS)
    slot = lanes // GROUP
    row = block * PACK + slot
    live = (slot < PACK) & (row < count)
    row = tl.minimum(row, count - 1)
    head = kv_head * GROUP + lanes % GROUP
    dims = tl.arange(0, HEAD_DIM)
    cells = (row * HEADS + head)[:, None].to(tl.int64) * HEAD_DIM + dims[None, :]
    q = tl.load(query + cells, mask=live[:, None], other=0)
    rel_row = ((row * HEADS + head) * EXTENT).to(tl.int32)
    # 32-bit positions: the per-key masks below are [lanes, tile] integer comparisons.
    position = tl.load(positions + row, mask=live, other=0).to(tl.int32)
    low = tl.maximum(tl.load(lowest + row, mask=live, other=0), 0).to(tl.int32)
    plane = tl.load(planes + row, mask=live, other=0).to(tl.int32)
    low_plane = tl.load(low_planes + row, mask=live, other=0).to(tl.int32)
    cut = tl.load(cuts + row, mask=live, other=0).to(tl.int32)
    length = tl.load(length_at).to(tl.int32)
    origin = tl.load(origin_at).to(tl.int32)

    maximum = tl.full((ROWS,), float("-inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    numerator = tl.zeros((ROWS, HEAD_DIM), tl.float32)

    column = tl.arange(0, TILE)
    first = low // TILE
    end = position // TILE + 1
    if SEGMENTS > 1:
        first += segment * SEGMENT_TILES
        if segment < SEGMENTS - 1:
            end = tl.minimum(end, first + SEGMENT_TILES)
    first = tl.where(live, first, 1073741824)
    end = tl.where(live, end, 0)
    for tile in range(tl.min(first), tl.max(end)):
        base = tile * TILE
        at = base + column
        inside = live & (first <= tile) & (tile < end)
        distance = position[:, None] - at[None, :]
        above = inside[:, None] & (at[None, :] >= low[:, None])
        # The plane a row reads this tile from, or a code of its own (-2 - its place in the program) when
        # it must read alone: its sharing cut is inside the tile, or the tile reaches the tree.
        private = (base + TILE > length) | ((cut > base) & (cut < base + TILE))
        lane_source = tl.where(private, -2 - slot, tl.where(cut <= base, plane, low_plane))
        lane_source = tl.where(inside, lane_source, -1)
        # One pass per distinct source among the rows here: the rows of one plane share the load of a
        # committed tile. Every pass goes through the same merge, so a row's sums are compiled once.
        lowest_source = tl.min(tl.where(inside, lane_source, 1073741824))
        highest_source = tl.max(tl.where(inside, lane_source, -1073741824))
        uniform = lowest_source == highest_source  # the usual case: one pass
        for s in range(tl.where(uniform, 1, PACK)):
            if uniform:
                mine_source = highest_source
                earlier = 0
            else:
                mine_source = tl.max(tl.where(slot == s, lane_source, -1073741824))
                earlier = tl.max(tl.where((slot < s) & (lane_source == mine_source), 1, 0))
            if (mine_source != -1) & (earlier == 0):
                group = lane_source == mine_source
                if mine_source >= 0:  # every key of the tile is at or before every row's position
                    reads = above & group[:, None]
                    wanted = at >= tl.min(tl.where(group, low, 1073741824))
                    offsets = ((mine_source.to(tl.int64) * prefix_len + (at - origin)) * KV_HEADS + kv_head)[:, None] \
                        * HEAD_DIM + dims[None, :]
                    k = tl.load(prefix_k + offsets, mask=wanted[:, None], other=0)
                    v = tl.load(prefix_v + offsets, mask=wanted[:, None], other=0)
                else:
                    r = block * PACK - 2 - mine_source  # the row whose code this is
                    r_plane = tl.load(planes + r)  # 64-bit: offsets into the stores
                    mine = above & (distance >= 0) & group[:, None]
                    theirs = tl.max(mine.to(tl.int32), 0) > 0
                    visible = theirs & (at < length)
                    r_source = tl.where(at < tl.load(cuts + r), tl.load(low_planes + r), r_plane)
                    offsets = ((r_source * prefix_len + (at - origin).to(tl.int64)) * KV_HEADS + kv_head)[:, None] \
                        * HEAD_DIM + dims[None, :]
                    k = tl.load(prefix_k + offsets, mask=visible[:, None], other=0)
                    v = tl.load(prefix_v + offsets, mask=visible[:, None], other=0)
                    entry = at - length
                    reach = theirs & (entry >= 0) & (entry < PATH_WIDTH)
                    shown = tl.load(path_mask + r * PATH_WIDTH + entry, mask=reach, other=0)
                    in_tree = reach & (shown != 0)
                    slots = tl.load(paths + r * PATH_WIDTH + entry, mask=in_tree, other=0)
                    offsets = ((r_plane * tree_len + slots) * KV_HEADS + kv_head)[:, None] * HEAD_DIM + dims[None, :]
                    k = tl.where(in_tree[:, None], tl.load(tree_k + offsets, mask=in_tree[:, None], other=0), k)
                    v = tl.where(in_tree[:, None], tl.load(tree_v + offsets, mask=in_tree[:, None], other=0), v)
                    reads = mine & (visible | in_tree)[None, :]
                near = reads & (distance < EXTENT)
                bias = tl.load(rel + rel_row[:, None] + distance, mask=near, other=0).to(tl.float32)
                maximum, denominator, numerator = _merge_tile(
                    q, k, v, bias, reads, maximum, denominator, numerator, SCALE)

    mixed = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    lse = maximum + tl.log(denominator)
    if SEGMENTS > 1:
        part = (row * SEGMENTS + segment) * HEADS + head
        tl.store(output + part[:, None].to(tl.int64) * HEAD_DIM + dims[None, :], mixed, mask=live[:, None])
        tl.store(part_lse + part, lse, mask=live)
        return
    gates = tl.load(
        gate + (row * stride_gate + head.to(tl.int64) * HEAD_DIM)[:, None] + dims[None, :], mask=live[:, None], other=0
    ).to(tl.float32)
    sinks = tl.load(sink + row * stride_sink + head, mask=live, other=0).to(tl.float32)
    result = mixed * _sigmoid(gates) * _sigmoid(lse - sinks)[:, None]
    tl.store(output + cells, result.to(output.dtype.element_ty), mask=live[:, None])


@triton.jit
def _merge_segments(
    part_out,
    part_lse,
    gate,
    sink,
    output,
    stride_gate,
    stride_sink,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROWS: tl.constexpr,
    SEGMENTS: tl.constexpr,
):
    """One (row, KV head): the segments' results, each normalised over its own
    keys, merged in order by their log-sum-exp, then the gates."""
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    GROUP: tl.constexpr = HEADS // KV_HEADS
    lanes = tl.arange(0, ROWS)
    live = lanes < GROUP
    head = kv_head * GROUP + tl.minimum(lanes, GROUP - 1)
    dims = tl.arange(0, HEAD_DIM)
    maximum = tl.full((ROWS,), float("-inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    numerator = tl.zeros((ROWS, HEAD_DIM), tl.float32)
    for segment in tl.static_range(SEGMENTS):
        cell = row * SEGMENTS + segment
        lse = tl.load(part_lse + cell * HEADS + head, mask=live, other=float("-inf"))
        value = tl.load(part_out + (cell * HEADS + head[:, None]).to(tl.int64) * HEAD_DIM + dims[None, :],
                        mask=live[:, None], other=0.0)
        peak = tl.maximum(maximum, lse)
        decay = tl.where(maximum == float("-inf"), 0.0, tl.exp(maximum - peak))
        gain = tl.where(lse == float("-inf"), 0.0, tl.exp(lse - peak))
        numerator = numerator * decay[:, None] + value * gain[:, None]
        denominator = denominator * decay + gain
        maximum = peak
    mixed = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    total = maximum + tl.log(denominator)
    cells = (row * HEADS + head)[:, None].to(tl.int64) * HEAD_DIM + dims[None, :]
    gates = tl.load(
        gate + row * stride_gate + head[:, None].to(tl.int64) * HEAD_DIM + dims[None, :], mask=live[:, None], other=0
    ).to(tl.float32)
    sinks = tl.load(sink + row * stride_sink + head, mask=live, other=0).to(tl.float32)
    result = mixed * _sigmoid(gates) * _sigmoid(total - sinks)[:, None]
    tl.store(output + cells, result.to(output.dtype.element_ty), mask=live[:, None])


def tree_attention(
    query: torch.Tensor,
    rel: torch.Tensor,
    gate: torch.Tensor,
    sink: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    tree_k: torch.Tensor,
    tree_v: torch.Tensor,
    position: torch.Tensor,
    lowest: torch.Tensor,
    plane: torch.Tensor,
    low_plane: torch.Tensor,
    cut: torch.Tensor,
    path: torch.Tensor,
    path_mask: torch.Tensor,
    length: torch.Tensor,
    origin: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    tuning: tuple[int, int] = (3, 4),
    segments: tuple[int, int] | None = None,
    pack: bool = False,
) -> torch.Tensor:
    """Gated relative-bias attention of the rows of a call (module docstring).

    Args:
        out: BF16 tensor [n, heads, D] to write.
        tuning: (pipeline stages, warps); it does not change the result.
        segments: (count, tiles) to read a row's tiles in parallel programs.
        pack: several rows per program, which share the loads of a committed
            tile they read from one plane (module docstring). A caller keeps
            one value per layer: the two kernels are compiled apart.
    """
    n, heads, dim = query.shape
    kv_heads, extent, width = prefix_k.shape[2], rel.shape[2], path.shape[1]
    group = heads // kv_heads
    if out is None:
        out = torch.empty_like(query)
    floats = (query, rel, gate, sink, prefix_k, prefix_v, tree_k, tree_v, out)
    rows = (position, lowest, plane, low_plane, cut)
    whole = (query, rel, prefix_k, prefix_v, tree_k, tree_v, out)
    assert all(tensor.is_contiguous() for tensor in (*whole, *rows, path, path_mask, length, origin))
    # The gates and the sinks may be columns of a wider tensor: contiguous within a row.
    assert gate.stride(1) == dim and gate.stride(2) == 1 and sink.stride(1) == 1
    assert all(tensor.dtype == torch.bfloat16 for tensor in floats)
    assert all(tensor.dtype == torch.int64 and tensor.shape == (n,) for tensor in rows)
    assert path.dtype == torch.int64 and path_mask.dtype == torch.bool and path_mask.shape == path.shape
    assert length.dtype == origin.dtype == torch.int64 and length.numel() == origin.numel() == 1
    assert heads == group * kv_heads and group <= LANES and dim >= 16 and dim & (dim - 1) == 0
    assert gate.shape == query.shape and sink.shape == (n, heads) and rel.shape == (n, heads, extent)
    assert prefix_k.dim() == tree_k.dim() == 4 and prefix_k.shape == prefix_v.shape and tree_k.shape == tree_v.shape
    stages, warps = tuning
    count, tiles = segments or (1, 0)
    partial = out
    lse = out
    if count > 1:
        partial = torch.empty((n * count, heads, dim), dtype=torch.float32, device=query.device)
        lse = torch.empty((n * count, heads), dtype=torch.float32, device=query.device)
    shapes = dict(HEADS=heads, KV_HEADS=kv_heads, HEAD_DIM=dim, EXTENT=extent, PATH_WIDTH=width, SCALE=dim**-0.5,
                  ROWS=LANES, TILE=BLOCK_K, SEGMENTS=count, SEGMENT_TILES=tiles, num_stages=stages, num_warps=warps)
    if pack:
        rows_per_program = LANES // group
        _tree_attention_packed[(triton.cdiv(n, rows_per_program) * count, kv_heads)](
            query, rel, gate, sink, prefix_k, prefix_v, tree_k, tree_v,
            position, lowest, plane, low_plane, cut, path, path_mask, length, origin, partial,
            prefix_k.shape[1], tree_k.shape[1], n, gate.stride(0), sink.stride(0), lse,
            PACK=rows_per_program, **shapes,
        )
    else:
        _tree_attention[(n * count, kv_heads)](
            query, rel, gate, sink, prefix_k, prefix_v, tree_k, tree_v,
            position, lowest, plane, low_plane, cut, path, path_mask, length, origin, partial,
            prefix_k.shape[1], tree_k.shape[1], gate.stride(0), sink.stride(0), lse, **shapes,
        )
    if count > 1:
        _merge_segments[(n, kv_heads)](
            partial, lse, gate, sink, out, gate.stride(0), sink.stride(0),
            HEADS=heads, KV_HEADS=kv_heads, HEAD_DIM=dim, ROWS=LANES, SEGMENTS=count, num_warps=1,
        )
    return out
