# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Packed Ouro computation with shared prefix KV and isolated token trees.

Every distinct (parent, token) node owns one KV vector per loop/layer. All active
nodes share physical projection calls. CUDA attention combines paged prefix
FlashAttention with a small ancestor gather and softmax merge; sibling branches
never read each other's KV.
"""

import heapq
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

from .ouro_2d import LoopResult, NativeOuro2DRunner


@triton.jit(do_not_specialize=["prefix_begin"])
def _commit_native_prefix(
    tree_k,
    tree_v,
    prefix_k,
    prefix_v,
    slots,
    prefix_begin,
    COUNT: tl.constexpr,
    WIDTH: tl.constexpr,
    TREE_STRIDE: tl.constexpr,
    PREFIX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy committed K/V directly from each layer/loop's tree bank."""
    row = tl.program_id(0)
    node = row % COUNT
    bank = (row // COUNT).to(tl.int64)
    slot = tl.load(slots + node)
    dims = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = (bank * TREE_STRIDE + slot) * WIDTH + dims
    target = (bank * PREFIX_STRIDE + prefix_begin + node) * WIDTH + dims
    key = tl.load(tree_k + source, dims < WIDTH, other=0)
    value = tl.load(tree_v + source, dims < WIDTH, other=0)
    tl.store(prefix_k + target, key, dims < WIDTH)
    tl.store(prefix_v + target, value, dims < WIDTH)


@triton.jit(do_not_specialize=["LSE_HEAD_STRIDE"])
def _ancestor_attention_merge(
    query,
    keys,
    values,
    paths,
    loops,
    prefix_output,
    prefix_lse,
    output,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    LSE_HEAD_STRIDE,
    LSE_QUERY_STRIDE: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
    QUERY_ROW_STRIDE: tl.constexpr = 0,
    QUERY_HEAD_STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, DIM_BLOCK)
    ancestors = tl.arange(0, PATH_WIDTH)
    slots = tl.load(paths + row * PATH_WIDTH + ancestors)
    loop = tl.load(loops + row)
    kv_head = head // (HEADS // KV_HEADS)
    # Native RoPE returns the Q view of packed QKV, whose row stride includes
    # the K/V columns. Synthetic contiguous queries use the default strides.
    row_stride = QUERY_ROW_STRIDE if QUERY_ROW_STRIDE else HEADS * HEAD_DIM
    head_stride = QUERY_HEAD_STRIDE if QUERY_HEAD_STRIDE else HEAD_DIM
    q = tl.load(
        query + row * row_stride + head * head_stride + dims,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    offsets = ((loop * POOL_SIZE + slots[:, None]) * KV_HEADS + kv_head) * HEAD_DIM
    valid = (slots[:, None] >= 0) & (dims[None, :] < HEAD_DIM)
    k = tl.load(keys + offsets + dims[None, :], mask=valid, other=0).to(tl.float32)
    v = tl.load(values + offsets + dims[None, :], mask=valid, other=0).to(tl.float32)
    scores = tl.sum(k * q[None, :], 1) * SCALE
    scores = tl.where(slots >= 0, scores, float("-inf"))
    prefix_logsum = tl.load(
        prefix_lse + head * LSE_HEAD_STRIDE + row * LSE_QUERY_STRIDE
    )
    maximum = tl.maximum(prefix_logsum, tl.max(scores, 0))
    weights = tl.exp(scores - maximum)
    prefix_weight = tl.exp(prefix_logsum - maximum)
    prefix = tl.load(
        prefix_output + (row * HEADS + head) * HEAD_DIM + dims,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    numerator = tl.sum(weights[:, None] * v, 0) + prefix_weight * prefix
    denominator = tl.sum(weights, 0) + prefix_weight
    tl.store(
        output + (row * HEADS + head) * HEAD_DIM + dims,
        numerator / denominator,
        mask=dims < HEAD_DIM,
    )


@triton.jit(do_not_specialize=["LSE_HEAD_STRIDE", "PREFIX_LAYER_BASE"])
def _shared_band_ancestor_merge(
    query,
    keys,
    values,
    paths,
    loops,
    positions,
    prefix_keys,
    prefix_values,
    share,
    prefix_output,
    prefix_lse,
    output,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    LSE_HEAD_STRIDE,
    LSE_QUERY_STRIDE: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
    PREFIX_LOOP_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    BAND: tl.constexpr,
    FINAL_LOOP: tl.constexpr,
    LEXICAL: tl.constexpr,
    QUERY_ROW_STRIDE: tl.constexpr = 0,
    QUERY_HEAD_STRIDE: tl.constexpr = 0,
    COMPACT_RING: tl.constexpr = 0,
    PREFIX_LAYER_BASE=0,
):
    """Merge the shared prefix partial with the recent prefix band and ancestors.

    The FlashAttention partial covers final-loop KV at j <= P - W, which every
    query may read. Band positions P - W < j < P read final-loop KV once
    j <= t - W and the query's own loop otherwise; the final loop always reads
    itself. Tree ancestors lie within W of every query and read its own loop.
    Layer-1/loop-1 queries (LEXICAL) took their whole native prefix already.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, DIM_BLOCK)
    ancestors = tl.arange(0, PATH_WIDTH)
    slots = tl.load(paths + row * PATH_WIDTH + ancestors)
    loop = tl.load(loops + row)
    position = tl.load(positions + row)
    prefix_length = tl.load(share)
    kv_head = head // (HEADS // KV_HEADS)
    row_stride = QUERY_ROW_STRIDE if QUERY_ROW_STRIDE else HEADS * HEAD_DIM
    head_stride = QUERY_HEAD_STRIDE if QUERY_HEAD_STRIDE else HEAD_DIM
    q = tl.load(
        query + row * row_stride + head * head_stride + dims,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    offsets = ((loop * POOL_SIZE + slots[:, None]) * KV_HEADS + kv_head) * HEAD_DIM
    valid = (slots[:, None] >= 0) & (dims[None, :] < HEAD_DIM)
    k = tl.load(keys + offsets + dims[None, :], mask=valid, other=0).to(tl.float32)
    v = tl.load(values + offsets + dims[None, :], mask=valid, other=0).to(tl.float32)
    scores = tl.sum(k * q[None, :], 1) * SCALE
    scores = tl.where(slots >= 0, scores, float("-inf"))
    offset = tl.arange(0, BAND)
    band = prefix_length - (WINDOW - 1) + offset
    lexical = (loop == 0) & (LEXICAL == 1)
    in_band = (band >= 0) & (band < prefix_length) & (offset < WINDOW - 1) & ~lexical
    final = (loop == FINAL_LOOP) | (band <= position - WINDOW)
    if COMPACT_RING:
        band_position = tl.maximum(band, 0)
        band_rows = (PREFIX_LAYER_BASE + tl.where(
            final, band_position,
            PREFIX_LOOP_STRIDE + loop * COMPACT_RING + band_position % COMPACT_RING,
        )).to(tl.int64)
    else:
        band_rows = tl.where(final, FINAL_LOOP, loop) * PREFIX_LOOP_STRIDE + tl.maximum(
            band, 0
        )
    band_offsets = (band_rows[:, None] * KV_HEADS + kv_head) * HEAD_DIM
    band_valid = in_band[:, None] & (dims[None, :] < HEAD_DIM)
    band_k = tl.load(
        prefix_keys + band_offsets + dims[None, :], mask=band_valid, other=0
    ).to(tl.float32)
    band_v = tl.load(
        prefix_values + band_offsets + dims[None, :], mask=band_valid, other=0
    ).to(tl.float32)
    band_scores = tl.where(in_band, tl.sum(band_k * q[None, :], 1) * SCALE, float("-inf"))
    # With P < W the common range is empty. FlashAttention's output and LSE for
    # an empty key range are not relied on: that partial is dropped explicitly.
    empty = (tl.load(share + 1) == 0) & ~lexical
    prefix_logsum = tl.load(
        prefix_lse + head * LSE_HEAD_STRIDE + row * LSE_QUERY_STRIDE
    )
    prefix_logsum = tl.where(empty, float("-inf"), prefix_logsum)
    maximum = tl.maximum(
        prefix_logsum, tl.maximum(tl.max(scores, 0), tl.max(band_scores, 0))
    )
    weights = tl.exp(scores - maximum)
    band_weights = tl.exp(band_scores - maximum)
    prefix_weight = tl.where(empty, 0.0, tl.exp(prefix_logsum - maximum))
    prefix = tl.load(
        prefix_output + (row * HEADS + head) * HEAD_DIM + dims,
        mask=(dims < HEAD_DIM) & ~empty,
        other=0,
    ).to(tl.float32)
    numerator = (
        tl.sum(weights[:, None] * v, 0)
        + tl.sum(band_weights[:, None] * band_v, 0)
        + prefix_weight * prefix
    )
    denominator = tl.sum(weights, 0) + tl.sum(band_weights, 0) + prefix_weight
    tl.store(
        output + (row * HEADS + head) * HEAD_DIM + dims,
        numerator / denominator,
        mask=dims < HEAD_DIM,
    )


SHARE_INTS = 12  # P, P - W + 1 clamped, layer-1 cu_seqlens[3], block rows[2], lengths[2], pad


@dataclass(frozen=True)
class TreeSpan:
    node_id: int
    loop: int
    hidden: torch.Tensor


@dataclass(eq=False)
class _Node:
    node_id: int
    slot: int
    token_id: int
    position: int
    parent: Any = None
    path: tuple = ()
    depth: int = 0


@dataclass
class _TreeInputs:
    hidden: torch.Tensor
    metadata: torch.Tensor
    cpu_metadata: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    loops: torch.Tensor
    paths: torch.Tensor
    sequence_lengths: torch.Tensor
    cu_query: torch.Tensor
    block_table: torch.Tensor
    transfer_event: Any = None
    share: torch.Tensor | None = None
    shared_cu: torch.Tensor | None = None
    max_query_len: int = 0


@dataclass
class _TreeGraph:
    inputs: _TreeInputs
    graph: Any
    output: LoopResult


class NativeOuroTreeRunner:
    """Run token-tree cells with one physical projection pass per layer.

    Args:
        model: Loaded native Ouro model, with tensor/pipeline parallel size one.
        max_len: Maximum committed prefix plus node-path token position.
        max_nodes: Number of simultaneously allocated speculative nodes.
        feat_layers: Zero-based physical target layers in drafter feature order.
        use_cuda_graph: Capture decode graphs indexed by active cell count.
        attention_backend: ``flash_attn`` for CUDA, ``sdpa`` for CPU reference.
        max_graph_shapes: Bound graph memory; additional cell counts run eagerly.
        use_fused_norm: Use scoped CUDA normalization kernels for prefix and tree.
        use_fused_rope: Call the CUDA rotary kernel explicitly for prefix and tree.
        compact_shared_prefix: Store shared-last histories and recent rings compactly.

    Register unique nodes before advancing them. A wave may include ancestors
    and descendants at the same loop: their KVs are all written before attention.
    Graph outputs are views valid until that graph's next replay. All input
    states are copied before replay, so previous outputs can be direct inputs.
    """

    prefix_runner_type = NativeOuro2DRunner

    def __init__(
        self,
        model,
        max_len: int,
        max_nodes: int = 64,
        feat_layers: list[int] | None = None,
        use_cuda_graph: bool = True,
        attention_backend: str = "flash_attn",
        max_graph_shapes: int = 64,
        use_fused_norm: bool = True,
        use_fused_rope: bool = True,
        compact_shared_prefix: bool = True,
    ) -> None:
        if max_nodes < 1 or max_graph_shapes < 0:
            raise ValueError(
                "Positive node capacity and nonnegative graph cap required"
            )
        self.prefix = self.prefix_runner_type(
            model,
            max_len,
            feat_layers=feat_layers,
            use_cuda_graph=False,
            attention_backend=attention_backend,
            use_fused_norm=use_fused_norm,
            use_fused_rope=use_fused_rope,
            shared_prefill=getattr(model.config, "loopedtf_kv_policy", "full")
            == "shared_last",
            compact_shared_prefix=compact_shared_prefix
            and getattr(model.config, "loopedtf_kv_policy", "full") == "shared_last",
        )
        self.model = model
        self.use_fused_norm = use_fused_norm
        self.use_fused_rope = use_fused_rope
        # Shared-last targets read final-loop prefix KV outside each query's W band.
        policy = getattr(model.config, "loopedtf_kv_policy", "full")
        if policy not in ("full", "shared_last"):
            raise ValueError("The tree runner supports full or shared_last loop KV")
        self.window = (
            int(model.config.loopedtf_kv_window) if policy == "shared_last" else None
        )
        self.R, self.H, self.L, self.nf = (
            self.prefix.R,
            self.prefix.H,
            self.prefix.L,
            self.prefix.nf,
        )
        self.device, self.dtype = self.prefix.device, self.prefix.dtype
        self.max_len, self.max_nodes = max_len, max_nodes
        self.feat_layers = self.prefix.feat_layers
        self.heads, self.kv_heads, self.head_dim = (
            self.prefix.heads,
            self.prefix.kv_heads,
            self.prefix.head_dim,
        )
        self.use_cuda_graph = (
            use_cuda_graph
            and self.device.type == "cuda"
            and attention_backend == "flash_attn"
        )
        self.max_graph_shapes = max_graph_shapes
        self.path_width = triton.next_power_of_2(max_nodes)
        self.padded_nodes = ((max_nodes + 15) // 16) * 16
        shape = (self.L, self.R * self.padded_nodes, self.kv_heads, self.head_dim)
        # FA3 reads V rows past a sequence's length within a tile; keep unwritten rows finite.
        self.k_tree = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v_tree = torch.zeros_like(self.k_tree)
        self._final_block_table = self.prefix._block_table[self.R - 1 :].contiguous()
        self._graphs: dict[int, _TreeGraph] = {}
        self._capture_stream = None
        self.stats = {
            "calls": 0,
            "cells": 0,
            "graph_replays": 0,
            "captures": 0,
            "capture_seconds": 0.0,
            "query_bound_growths": 0,
            "graph_cache_fallbacks": 0,
            "registered_nodes": 0,
            "released_nodes": 0,
            "committed_nodes": 0,
        }
        # Calls per cell count, for choosing graph shapes; not reset with the request state.
        self.cell_counts: dict[int, int] = {}
        self.reset()

    @property
    def prefix_length(self) -> int:
        return self.prefix.length[0]

    @staticmethod
    def _retire_nodes(nodes) -> None:
        # Each path includes its own node. Retired nodes must not depend on
        # cyclic GC after all surviving paths and parents have been rebased.
        for node in nodes:
            node.path = ()
            node.parent = None

    def reset(self) -> None:
        self.prefix.reset()
        self._retire_nodes(getattr(self, "_nodes", {}).values())
        self._nodes: dict[int, _Node] = {}
        self._keys: dict[tuple, int] = {}
        self._free = list(range(self.max_nodes))
        heapq.heapify(self._free)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.prefix.embed(tokens)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.prefix.logits(hidden)

    def readout(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.prefix.readout(hidden)

    @torch.inference_mode()
    def prefill(self, tokens: torch.Tensor) -> LoopResult:
        self.reset()
        return self.prefix.prefill(tokens)

    def register(self, node_id: int, parent_id: int | None, token_id: int) -> int:
        """Allocate a unique (parent, token) node and derive its absolute position."""
        if self.prefix_length < 1:
            raise ValueError("A nonempty verified prefix must be prefetched first")
        key = (parent_id, token_id)
        if node_id in self._nodes:
            node = self._nodes[node_id]
            expected_parent = None if node.parent is None else node.parent.node_id
            if (expected_parent, node.token_id) != key:
                raise ValueError("An existing node ID cannot change parent or token")
            return node_id
        if key in self._keys:
            raise ValueError("Duplicate (parent, token); reuse the existing node ID")
        if not self._free:
            raise ValueError("Speculative node budget exceeded")
        if parent_id is not None and parent_id not in self._nodes:
            raise ValueError("Parent node is not allocated")
        parent = None if parent_id is None else self._nodes[parent_id]
        position = self.prefix_length if parent is None else parent.position + 1
        if position >= self.max_len:
            raise ValueError("Node path exceeds target token capacity")
        if self.window is not None and position - self.prefix_length >= self.window:
            # Positions up to t - W must already hold final-loop KV in the prefix.
            raise ValueError("Shared-last node path exceeds the read window")
        node = _Node(node_id, heapq.heappop(self._free), token_id, position, parent)
        node.path = (*(() if parent is None else parent.path), node)
        self._nodes[node_id] = node
        self._keys[key] = node_id
        self.stats["registered_nodes"] += 1
        return node_id

    def release(self, node_ids: list[int]) -> None:
        """Free pruned nodes; the caller must include all surviving descendants."""
        removed = set(node_ids)
        if any(node_id not in self._nodes for node_id in removed):
            raise ValueError("Cannot release an unallocated node")
        for node in self._nodes.values():
            if node.node_id not in removed and any(
                ancestor.node_id in removed for ancestor in node.path[:-1]
            ):
                raise ValueError(
                    "Release must include every descendant of pruned nodes"
                )
        retired = []
        for node_id in removed:
            node = self._nodes.pop(node_id)
            parent_id = None if node.parent is None else node.parent.node_id
            self._keys.pop((parent_id, node.token_id))
            heapq.heappush(self._free, node.slot)
            retired.append(node)
        self._retire_nodes(retired)
        self.stats["released_nodes"] += len(removed)

    @torch.inference_mode()
    def commit(self, node_ids: list[int]) -> None:
        """Append fully computed accepted nodes to the verified shared prefix.

        Prune incompatible branches first. Remaining nodes must descend from the
        last accepted node; their paths and immediate parent links are rebased.
        """
        if not node_ids:
            return
        nodes = [self._nodes[node_id] for node_id in node_ids]
        begin = self.prefix_length
        for i, node in enumerate(nodes):
            if node.position != begin + i or node.depth != self.R:
                raise ValueError("Commit requires contiguous fully computed nodes")
            expected_parent = None if i == 0 else nodes[i - 1]
            if node.parent is not expected_parent:
                raise ValueError("Committed nodes must form the root's accepted path")
        selected = set(node_ids)
        last = nodes[-1]
        for node in self._nodes.values():
            if node.node_id not in selected and last not in node.path:
                raise ValueError("Prune branches incompatible with the accepted prefix")
        compact = self.prefix.compact_prefix
        if compact is not None:
            compact.commit(
                self.k_tree, self.v_tree, [node.slot for node in nodes],
                begin, self.padded_nodes,
            )
        elif self.device.type == "cuda":
            slots = torch.tensor(
                [node.slot for node in nodes], device=self.device, dtype=torch.long
            )
            width = self.kv_heads * self.head_dim
            _commit_native_prefix[
                (self.L * self.R * len(nodes), triton.cdiv(width, 256))
            ](
                self.k_tree, self.v_tree, self.prefix.k, self.prefix.v, slots,
                begin, COUNT=len(nodes), WIDTH=width,
                TREE_STRIDE=self.padded_nodes, PREFIX_STRIDE=self.prefix.padded_len,
                BLOCK=256,
            )
        else:
            sources, destinations = [], []
            for loop in range(self.R):
                sources.extend(loop * self.padded_nodes + node.slot for node in nodes)
                destinations.extend(
                    loop * self.prefix.padded_len + node.position for node in nodes
                )
            source = torch.tensor(sources, device=self.device, dtype=torch.long)
            destination = torch.tensor(
                destinations, device=self.device, dtype=torch.long
            )
            for target, bank in (
                (self.prefix.k, self.k_tree),
                (self.prefix.v, self.v_tree),
            ):
                target.view(self.L, -1, self.kv_heads, self.head_dim).index_copy_(
                    1, destination, bank.index_select(1, source)
                )
        self.prefix.length = [begin + len(nodes)] * self.R
        for node_id in node_ids:
            node = self._nodes.pop(node_id)
            heapq.heappush(self._free, node.slot)
        self._keys.clear()
        for node in self._nodes.values():
            node.path = tuple(
                ancestor
                for ancestor in node.path
                if ancestor.position >= self.prefix_length
            )
            if node.parent is not None and node.parent.position < self.prefix_length:
                node.parent = None
            parent_id = None if node.parent is None else node.parent.node_id
            self._keys[parent_id, node.token_id] = node.node_id
        self._retire_nodes(nodes)
        self.stats["committed_nodes"] += len(nodes)

    def _validate_spans(self, spans: list[TreeSpan]) -> None:
        if not spans or len({span.node_id for span in spans}) != len(spans):
            raise ValueError("Advance requires distinct nonempty node cells")
        current = {(span.node_id, span.loop) for span in spans}
        for span in spans:
            node = self._nodes.get(span.node_id)
            if node is None or not 0 <= span.loop < self.R or node.depth != span.loop:
                raise ValueError("Node cell must advance its next unfinished loop")
            if span.hidden.shape != (self.H,):
                raise ValueError(f"Tree hidden must have shape [{self.H}]")
            if span.hidden.device != self.device or span.hidden.dtype != self.dtype:
                raise ValueError("Tree hidden must match target device and dtype")
            for ancestor in node.path[:-1]:
                if (
                    ancestor.depth <= span.loop
                    and (ancestor.node_id, span.loop) not in current
                ):
                    raise ValueError(
                        "Ancestor KV must exist or advance in the same wave"
                    )

    def _metadata_views(self, metadata: torch.Tensor, count: int):
        positions = metadata[: count * 8].view(torch.int64)
        slots = metadata[count * 8 : count * 16].view(torch.int64)
        loops = metadata[count * 16 : count * 24].view(torch.int64)
        path_end = count * 24 + count * self.path_width * 4
        paths = (
            metadata[count * 24 : path_end]
            .view(torch.int32)
            .view(count, self.path_width)
        )
        lengths = metadata[path_end : path_end + self.R * 4].view(torch.int32)
        cu_end = path_end + self.R * 4 + (self.R + 1) * 4
        cu_query = metadata[path_end + self.R * 4 : cu_end].view(torch.int32)
        share = metadata[cu_end:].view(torch.int32) if self.window is not None else None
        return positions, slots, loops, paths, lengths, cu_query, share

    def _make_inputs(self, spans: list[TreeSpan]) -> _TreeInputs:
        count = len(spans)
        share_bytes = SHARE_INTS * 4 if self.window is not None else 0
        cpu = torch.empty(
            count * (24 + self.path_width * 4) + (2 * self.R + 1) * 4 + share_bytes,
            dtype=torch.uint8,
            pin_memory=self.use_cuda_graph,
        )
        metadata = torch.empty_like(cpu, device=self.device)
        positions, slots, loops, paths, lengths, cu_query, share = (
            self._metadata_views(metadata, count)
        )
        inputs = _TreeInputs(
            torch.empty(count, self.H, device=self.device, dtype=self.dtype),
            metadata,
            cpu,
            positions,
            slots,
            loops,
            paths,
            lengths,
            cu_query,
            self.prefix._block_table,
            torch.cuda.Event() if self.use_cuda_graph else None,
            share,
            torch.tensor([0, count], device=self.device, dtype=torch.int32)
            if self.window is not None
            else None,
            self._query_bound(spans),
        )
        self._update_inputs(inputs, spans)
        return inputs

    def _query_bound(self, spans: list[TreeSpan]) -> int:
        """A bucketed upper bound for queries sharing one native prefix bank."""
        if self.window is not None:
            return len(spans)
        counts = [0] * self.R
        for span in spans:
            counts[span.loop] += 1
        bound = triton.next_power_of_2(max(counts))
        # FA2's single-query GQA fastpath requires a query in every sequence.
        if len(spans) > 1 and bound == 1 and 0 in counts:
            bound = 2
        return min(len(spans), bound)

    def _update_inputs(self, inputs: _TreeInputs, spans: list[TreeSpan]) -> None:
        if inputs.transfer_event is not None:
            inputs.transfer_event.synchronize()
        views = self._metadata_views(inputs.cpu_metadata, len(spans))
        positions, slots, loops, paths, lengths, cu_query = (
            tensor.numpy() for tensor in views[:6]
        )
        paths.fill(-1)
        lengths.fill(self.prefix_length)
        counts = [0] * self.R
        for i, span in enumerate(spans):
            node = self._nodes[span.node_id]
            positions[i], loops[i] = node.position, span.loop
            slots[i] = span.loop * self.padded_nodes + node.slot
            path = [ancestor.slot for ancestor in node.path]
            paths[i, : len(path)] = path
            counts[span.loop] += 1
        cumulative = [0]
        for count in counts:
            cumulative.append(cumulative[-1] + count)
        cu_query[:] = cumulative
        if max(counts) > inputs.max_query_len:
            raise ValueError("Native prefix queries exceed the captured graph bound")
        if self.window is not None:
            share = views[6].numpy()
            prefix, common = self.prefix_length, max(self.prefix_length - self.window + 1, 0)
            share[:] = 0
            share[0], share[1] = prefix, common
            if len(spans) == 1:
                # One sequence: FA2's single-query fastpath rejects empty groups.
                lexical = spans[0].loop == 0
                share[2:4] = (0, 1)
                share[5] = 0 if lexical else self.R - 1
                share[7] = prefix if lexical else common
            else:
                # Layer 1 reads its native loop-1 prefix; other loops share.
                share[2:5] = (0, counts[0], len(spans))
                share[5:7] = (0, self.R - 1)
                share[7:9] = (prefix, common)
        inputs.metadata.copy_(inputs.cpu_metadata, non_blocking=self.use_cuda_graph)
        if inputs.transfer_event is not None:
            inputs.transfer_event.record(torch.cuda.current_stream(self.device))
        torch.stack([span.hidden for span in spans], out=inputs.hidden)

    def _attention(self, layer: int, q, k, v, inputs: _TreeInputs) -> torch.Tensor:
        count = q.shape[0]
        q = q.view(count, self.heads, self.head_dim)
        k = k.view(count, self.kv_heads, self.head_dim)
        v = v.view_as(k)
        if self.prefix._flash is not None:
            cache_shape = (-1, 16, self.kv_heads, self.head_dim)
            self.prefix._write_cache(
                k,
                v,
                self.k_tree[layer].view(cache_shape),
                self.v_tree[layer].view(cache_shape),
                inputs.slots,
                "auto",
                self.prefix._scale,
                self.prefix._scale,
            )
            if self.window is not None:
                return self._shared_attention(layer, q, inputs)
            prefix_output, lse = self.prefix._flash(
                q=q,
                k=self.prefix.k[layer],
                v=self.prefix.v[layer],
                cu_seqlens_q=inputs.cu_query,
                max_seqlen_q=inputs.max_query_len,
                seqused_k=inputs.sequence_lengths,
                max_seqlen_k=self.max_len,
                block_table=inputs.block_table,
                causal=False,
                softmax_scale=self.head_dim**-0.5,
                fa_version=self.prefix.fa_version,
                return_softmax_lse=True,
            )
            output = torch.empty_like(q)
            _ancestor_attention_merge[(count, self.heads)](
                q,
                self.k_tree[layer],
                self.v_tree[layer],
                inputs.paths,
                inputs.loops,
                prefix_output,
                lse,
                output,
                self.heads,
                self.kv_heads,
                self.head_dim,
                self.padded_nodes,
                self.path_width,
                self.head_dim**-0.5,
                lse.stride(0),
                lse.stride(1),
                triton.next_power_of_2(self.head_dim),
                QUERY_ROW_STRIDE=q.stride(0),
                QUERY_HEAD_STRIDE=q.stride(1),
                num_warps=4,
            )
            return output.reshape(count, -1)
        self.k_tree[layer].index_copy_(0, inputs.slots, k)
        self.v_tree[layer].index_copy_(0, inputs.slots, v)
        compact = self.prefix.compact_prefix
        prefix_k = self.prefix.k if compact is not None else self.prefix.k[layer]
        prefix_v = self.prefix.v if compact is not None else self.prefix.v[layer]
        prefix_keys = prefix_k.view(-1, self.kv_heads, self.head_dim)
        prefix_values = prefix_v.view_as(prefix_keys)
        output = []
        for row in range(count):
            loop = int(inputs.loops[row])
            path = inputs.paths[row]
            path = path[path >= 0].long() + loop * self.padded_nodes
            begin = loop * self.prefix.padded_len
            positions = torch.arange(self.prefix_length, device=self.device)
            rows = (
                compact.rows(layer, loop, positions)
                if compact is not None else positions + begin
            )
            native = loop == self.R - 1 or (layer == 0 and loop == 0)
            if self.window is not None and not native:
                # Final-loop KV at j <= t - W, the query's own loop above.
                split = min(
                    max(int(inputs.positions[row]) - self.window + 1, 0),
                    self.prefix_length,
                )
                final = (self.R - 1) * self.prefix.padded_len
                rows = torch.cat(
                    (
                        compact.rows(layer, self.R - 1, positions[:split])
                        if compact is not None else positions[:split] + final,
                        rows[split:],
                    )
                )
            keys = torch.cat((prefix_keys[rows], self.k_tree[layer][path]))
            values = torch.cat((prefix_values[rows], self.v_tree[layer][path]))
            result = F.scaled_dot_product_attention(
                q[row : row + 1].transpose(0, 1)[None],
                keys.transpose(0, 1)[None],
                values.transpose(0, 1)[None],
                enable_gqa=self.heads != self.kv_heads,
            )
            output.append(result[0].transpose(0, 1))
        return torch.cat(output).reshape(count, -1)

    def _shared_attention(self, layer: int, q, inputs: _TreeInputs) -> torch.Tensor:
        """FlashAttention over the common final-loop prefix, then band and ancestors."""
        count = q.shape[0]
        share = inputs.share
        compact = self.prefix.compact_prefix
        if layer == 0:
            sequences = 1 if count == 1 else 2
            cu = share[2 : 3 + sequences]
            tables = (
                compact.first_layer_tables
                if compact is not None else self.prefix._block_table
            )
            table = tables.index_select(
                0, share[5 : 5 + sequences].long()
            )
            lengths = share[7 : 7 + sequences]
        else:
            cu, lengths = inputs.shared_cu, share[1:2]
            table = (
                compact.final_tables[layer]
                if compact is not None else self._final_block_table
            )
        prefix_k = self.prefix.k if compact is not None else self.prefix.k[layer]
        prefix_v = self.prefix.v if compact is not None else self.prefix.v[layer]
        prefix_output, lse = self.prefix._flash(
            q=q,
            k=prefix_k,
            v=prefix_v,
            cu_seqlens_q=cu,
            max_seqlen_q=count,
            seqused_k=lengths,
            max_seqlen_k=self.max_len,
            block_table=table,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            fa_version=self.prefix.fa_version,
            return_softmax_lse=True,
        )
        output = torch.empty_like(q)
        _shared_band_ancestor_merge[(count, self.heads)](
            q,
            self.k_tree[layer],
            self.v_tree[layer],
            inputs.paths,
            inputs.loops,
            inputs.positions,
            prefix_k,
            prefix_v,
            share,
            prefix_output,
            lse,
            output,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.padded_nodes,
            self.path_width,
            self.head_dim**-0.5,
            lse.stride(0),
            lse.stride(1),
            triton.next_power_of_2(self.head_dim),
            PREFIX_LOOP_STRIDE=self.prefix.padded_len,
            WINDOW=self.window,
            BAND=triton.next_power_of_2(max(self.window - 1, 1)),
            FINAL_LOOP=self.R - 1,
            LEXICAL=int(layer == 0),
            QUERY_ROW_STRIDE=q.stride(0),
            QUERY_HEAD_STRIDE=q.stride(1),
            COMPACT_RING=compact.ring_size if compact is not None else 0,
            PREFIX_LAYER_BASE=(
                layer * compact.layer_stride if compact is not None else 0
            ),
            num_warps=4,
        )
        return output.reshape(count, -1)

    def _forward(self, inputs: _TreeInputs) -> LoopResult:
        with self.prefix._norm_context():
            return self._forward_layers(inputs)

    def _forward_layers(self, inputs: _TreeInputs) -> LoopResult:
        hidden = inputs.hidden.clone()
        residual = None
        features = torch.empty(
            hidden.shape[0], self.nf * self.H, device=self.device, dtype=self.dtype
        )
        for idx, layer in enumerate(self.prefix.layers):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(hidden)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
            q, k = self.prefix._apply_rope(attn.rotary_emb, inputs.positions, q, k)
            hidden, _ = attn.o_proj(self._attention(idx, q, k, v, inputs))
            hidden = layer.input_layernorm_2(hidden)
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.post_attention_layernorm_2(layer.mlp(hidden))
            slot = self.prefix._feature_slots.get(idx)
            if slot is not None:
                features[:, slot * self.H : (slot + 1) * self.H] = hidden + residual
        hidden, _ = self.prefix.backbone.norm(hidden, residual)
        return LoopResult(hidden, features)

    def _capture(self, inputs: _TreeInputs) -> _TreeGraph:
        started = time.perf_counter()
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=self.device)
        stream = self._capture_stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward(inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = self._forward(inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.stats["captures"] += 1
        self.stats["capture_seconds"] += time.perf_counter() - started
        return _TreeGraph(inputs, graph, result)

    @torch.inference_mode()
    def prewarm_graphs(self, max_cells: int) -> dict[str, Any]:
        """Capture empty-request decode shapes; retain graphs, clear dummy KV state.

        This is a startup-only operation. Graph pools remain independent;
        capture streams are shared, and all graph replays must be serial.
        """
        if type(max_cells) is not int or max_cells < 0:
            raise ValueError("Prewarm max_cells must be a nonnegative integer")
        report = {
            "enabled": bool(self.use_cuda_graph),
            "requested_max_cells": max_cells,
            "prepared_max_cells": 0,
            "graphs_before": len(self._graphs),
            "graphs_after": len(self._graphs),
            "elapsed_seconds": 0.0,
            "tree_stats_delta": {key: 0 for key in self.stats},
            "prefix_stats_delta": {key: 0 for key in self.prefix.stats},
            "warmup_forward_evaluations": 0,
            "capture_forward_recordings": 0,
        }
        if not self.use_cuda_graph or max_cells == 0:
            return report
        if any(self.prefix.length) or self._nodes:
            raise ValueError("Graph prewarm requires empty startup sequence state")
        limit = max(
            0, min(max_cells, self.max_nodes, self.max_graph_shapes, self.max_len - 1)
        )
        report["prepared_max_cells"] = limit
        missing = [count for count in range(1, limit + 1) if count not in self._graphs]
        if len(missing) > self.max_graph_shapes - len(self._graphs):
            raise ValueError(
                "Prewarm requires available graph slots for the requested range"
            )
        if not missing:
            return report
        before_tree, before_prefix = dict(self.stats), dict(self.prefix.stats)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        try:
            token = torch.tensor([0], device=self.device, dtype=torch.long)
            self.prefill(token)
            hidden = self.embed(token)[0]
            for count in missing:
                for node_id in range(count):
                    self.register(node_id, None if node_id == 0 else node_id - 1, 0)
                spans = [TreeSpan(node_id, 0, hidden) for node_id in range(count)]
                self._validate_spans(spans)
                self._graphs[count] = self._capture(self._make_inputs(spans))
                self.release(list(range(count)))
        finally:
            self.reset()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            report["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                self.device
            )
            report["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                self.device
            )
            report["retained_memory_allocated_bytes"] = torch.cuda.memory_allocated(
                self.device
            )
            report["retained_memory_reserved_bytes"] = torch.cuda.memory_reserved(
                self.device
            )
        report["elapsed_seconds"] = time.perf_counter() - started
        report["graphs_after"] = len(self._graphs)
        report["tree_stats_delta"] = {
            key: value - before_tree.get(key, 0) for key, value in self.stats.items()
        }
        report["prefix_stats_delta"] = {
            key: value - before_prefix.get(key, 0)
            for key, value in self.prefix.stats.items()
        }
        # _capture executes two warmup forwards and records one graph forward;
        # graph recording does not count as an executed decode graph replay.
        report["warmup_forward_evaluations"] = 2 * len(missing)
        report["capture_forward_recordings"] = len(missing)
        return report

    @torch.inference_mode()
    def advance(self, spans: list[TreeSpan]) -> list[LoopResult]:
        """Advance distinct node cells together, preserving branch visibility."""
        self._validate_spans(spans)
        count = len(spans)
        # Prefix attention sees one query sequence per loop. Sorting shares
        # prefix KV tiles across that loop's nodes; ancestor attention remains
        # independent for every query. Restore the caller's result order below.
        order = sorted(range(count), key=lambda i: spans[i].loop)
        ordered_spans = [spans[i] for i in order]
        bound = self._query_bound(ordered_spans)
        captured = self._graphs.get(count) if self.use_cuda_graph else None
        if self.use_cuda_graph and (
            captured is not None or len(self._graphs) < self.max_graph_shapes
        ):
            if captured is None or bound > captured.inputs.max_query_len:
                # Grow the bound if this count acquires a less balanced loop
                # partition. Retain one graph per count, preserving the cache
                # budget and allowing every later narrower partition to reuse it.
                self.stats["query_bound_growths"] += int(captured is not None)
                inputs = self._make_inputs(ordered_spans)
                captured = self._capture(inputs)
                self._graphs[count] = captured
            else:
                self._update_inputs(captured.inputs, ordered_spans)
            captured.graph.replay()
            result = captured.output
            self.stats["graph_replays"] += 1
        else:
            self.stats["graph_cache_fallbacks"] += int(self.use_cuda_graph)
            result = self._forward(self._make_inputs(ordered_spans))
        for span in spans:
            self._nodes[span.node_id].depth += 1
        self.stats["calls"] += 1
        self.stats["cells"] += count
        self.cell_counts[count] = self.cell_counts.get(count, 0) + 1
        output = [None] * count
        for packed, original in enumerate(order):
            output[original] = LoopResult(
                result.hidden[packed : packed + 1], result.features[packed : packed + 1]
            )
        return output
