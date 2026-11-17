# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Ouro tree storage that grows without changing scheduling decisions."""

import heapq
import time
from dataclasses import replace

import torch

from vllm.triton_utils import triton

from .ouro_tree import NativeOuroTreeRunner, TreeSpan

__all__ = ["NativeOuroFinalizedRunner", "TreeSpan"]


class NativeOuroFinalizedRunner(NativeOuroTreeRunner):
    """Keep the packed tree computation while growing its physical KV capacity.

    ``max_nodes`` is the initial allocation, not a candidate pruning budget.
    ``ensure_capacity`` reserves an absolute live-node count; registration also
    grows automatically. Allocation failures propagate without dropping nodes.
    Prefix storage, node identity, slots and computed depths survive growth.

    Ancestor path width grows independently of the node pool. Changed KV
    addresses or path layout invalidate decode graphs. The inherited graph
    cache limit bounds retained captures; further shapes execute eagerly.
    Calls and graph replays must remain serial, as in the base runner. Retained
    graph outputs still need caller-owned copies before another replay.
    """

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
        super().__init__(
            model,
            max_len,
            max_nodes=max_nodes,
            feat_layers=feat_layers,
            use_cuda_graph=use_cuda_graph,
            attention_backend=attention_backend,
            max_graph_shapes=max_graph_shapes,
            use_fused_norm=use_fused_norm,
            use_fused_rope=use_fused_rope,
            compact_shared_prefix=compact_shared_prefix,
        )
        self.path_width = min(self.path_width, 64)
        self._single_prefix_cu = (
            torch.tensor([0, 1], device=self.device, dtype=torch.int32)
            if self.prefix._flash is not None
            else None
        )
        self.stats.update(
            capacity=self.max_nodes,
            capacity_growths=0,
            capacity_growth_seconds=0.0,
            kv_reallocations=0,
            tree_kv_bytes=self._tree_kv_bytes(),
            path_width_growths=0,
            graph_layout_generation=0,
            invalidated_graphs=0,
            peak_live_nodes=0,
        )

    def _tree_kv_bytes(self) -> int:
        return (self.k_tree.numel() + self.v_tree.numel()) * self.k_tree.element_size()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _invalidate_graph_layout(self) -> None:
        self.stats["invalidated_graphs"] += len(self._graphs)
        self._graphs.clear()
        self._capture_stream = None
        self.stats["graph_layout_generation"] += 1

    @torch.inference_mode()
    def ensure_capacity(self, required: int) -> int:
        """Reserve slots without pruning or changing existing node state.

        Args:
            required: Absolute number of live speculative nodes to accommodate.

        Returns:
            Current slot capacity, grown geometrically if necessary.

        Raises:
            ValueError: The requested capacity is not a nonnegative integer.
            torch.OutOfMemoryError: New KV storage could not be allocated. The
                previous banks, graphs, slots and nodes remain usable.
        """
        if type(required) is not int or required < 0:
            raise ValueError("Required node capacity must be a nonnegative integer")
        if required <= self.max_nodes:
            return self.max_nodes

        started = time.perf_counter()
        capacity = self.max_nodes
        while capacity < required:
            capacity *= 2
        padded = ((capacity + 15) // 16) * 16
        free = [*self._free, *range(self.max_nodes, capacity)]
        heapq.heapify(free)
        changed_bank = padded != self.padded_nodes
        if changed_bank:
            self._synchronize()
            shape = (self.L, self.R * padded, self.kv_heads, self.head_dim)
            # FA3 reads V rows past a sequence's length within a tile; keep unwritten rows finite.
            keys = torch.zeros(shape, device=self.device, dtype=self.dtype)
            values = torch.zeros_like(keys)
            old_shape = (
                self.L,
                self.R,
                self.padded_nodes,
                self.kv_heads,
                self.head_dim,
            )
            new_shape = (self.L, self.R, padded, self.kv_heads, self.head_dim)
            # Preserve slots within each loop plane, rather than copying the
            # flattened bank whose loop stride changes. Reserved slots are also
            # copied so growth needs no GPU gather metadata or node remapping.
            keys.view(new_shape)[:, :, : self.padded_nodes].copy_(
                self.k_tree.view(old_shape)
            )
            values.view(new_shape)[:, :, : self.padded_nodes].copy_(
                self.v_tree.view(old_shape)
            )
            self._synchronize()
            self.k_tree, self.v_tree = keys, values
            self.padded_nodes = padded
            self._invalidate_graph_layout()
            self.stats["kv_reallocations"] += 1

        self.max_nodes, self._free = capacity, free
        self.stats["capacity"] = capacity
        self.stats["tree_kv_bytes"] = self._tree_kv_bytes()
        self.stats["capacity_growths"] += 1
        self.stats["capacity_growth_seconds"] += time.perf_counter() - started
        return capacity

    def _ensure_path_width(self, required: int) -> None:
        if required <= self.path_width:
            return
        self._synchronize()
        self.path_width = triton.next_power_of_2(required)
        self._invalidate_graph_layout()
        self.stats["path_width_growths"] += 1

    def _forward(self, inputs):
        if self._single_prefix_cu is not None and inputs.hidden.shape[0] == 1:
            # FA2's single-query GQA fastpath requires one query per sequence.
            # Empty loop groups violate that shape. Select the actual loop's
            # prefix once per forward; GPU indices remain dynamic on replay.
            inputs = replace(
                inputs,
                cu_query=self._single_prefix_cu,
                sequence_lengths=inputs.sequence_lengths[:1],
                block_table=inputs.block_table.index_select(0, inputs.loops[:1]),
            )
        return super()._forward(inputs)

    def register(self, node_id: int, parent_id: int | None, token_id: int) -> int:
        """Register without a scheduling budget; invalid nodes do not grow storage."""
        if node_id in self._nodes:
            return super().register(node_id, parent_id, token_id)
        if self.prefix_length < 1:
            raise ValueError("A nonempty verified prefix must be prefetched first")
        if (parent_id, token_id) in self._keys:
            raise ValueError("Duplicate (parent, token); reuse the existing node ID")
        if parent_id is not None and parent_id not in self._nodes:
            raise ValueError("Parent node is not allocated")
        parent = None if parent_id is None else self._nodes[parent_id]
        position = self.prefix_length if parent is None else parent.position + 1
        if position >= self.max_len:
            raise ValueError("Node path exceeds target token capacity")
        self.ensure_capacity(len(self._nodes) + 1)
        self._ensure_path_width(1 if parent is None else len(parent.path) + 1)
        result = super().register(node_id, parent_id, token_id)
        self.stats["peak_live_nodes"] = max(
            self.stats["peak_live_nodes"], len(self._nodes)
        )
        return result
