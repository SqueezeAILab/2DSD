# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Exact shared-last prefix storage: final history, lexical history, recent rings."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["count", "begin", "pool_size"])
def _commit_shared_prefix(
    tree_k,
    tree_v,
    prefix_k,
    prefix_v,
    slots,
    count,
    begin,
    pool_size,
    LAYERS: tl.constexpr,
    LOOPS: tl.constexpr,
    HISTORY: tl.constexpr,
    RING: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    item = tl.program_id(0)
    index = item % count
    loop = item // count % LOOPS
    layer = item // (count * LOOPS)
    position = begin + index
    lexical = (layer == 0) & (loop == 0)
    full = lexical | (loop == LOOPS - 1)
    stride = HISTORY + (LOOPS - 1) * RING
    base = tl.where(
        lexical,
        LAYERS * stride,
        layer * stride + tl.where(loop == LOOPS - 1, 0, HISTORY + loop * RING),
    )
    destination = (base + tl.where(full, position, position % RING)).to(tl.int64)
    source = ((layer * LOOPS + loop) * pool_size + tl.load(slots + index)).to(tl.int64)
    dims = tl.arange(0, BLOCK)
    keep = (dims < WIDTH) & (full | (position >= begin + count - RING))
    key = tl.load(tree_k + source * WIDTH + dims, mask=keep, other=0)
    value = tl.load(tree_v + source * WIDTH + dims, mask=keep, other=0)
    tl.store(prefix_k + destination * WIDTH + dims, key, mask=keep)
    tl.store(prefix_v + destination * WIDTH + dims, value, mask=keep)


class SharedPrefixKV:
    """Paged full histories and rings in one allocation with stable addresses.

    Each layer has a final-loop history followed by one ring per earlier loop.
    The first layer's lexical history follows all layers. Its otherwise unused
    ring keeps the per-layer stride uniform. Uncommitted nodes live separately.
    """

    def __init__(
        self,
        layers,
        loops,
        padded_len,
        window,
        block_size,
        kv_heads,
        head_dim,
        *,
        device,
        dtype,
    ):
        if type(window) is not int or window < 1:
            raise ValueError("Shared prefix storage requires a positive window")
        self.layers, self.loops = layers, loops
        self.padded_len, self.block_size = padded_len, block_size
        self.ring_size = ((window + block_size - 1) // block_size) * block_size
        self.layer_stride = padded_len + (loops - 1) * self.ring_size
        self.lexical_base = layers * self.layer_stride
        self.kv_heads, self.head_dim = kv_heads, head_dim
        shape = (
            (self.lexical_base + padded_len) // block_size,
            block_size,
            kv_heads,
            head_dim,
        )
        # Keep unwritten/padded rows finite for FlashAttention's vectorized loads.
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        blocks = torch.arange(
            padded_len // block_size, device=device, dtype=torch.int32
        )
        self.final_tables = (
            torch.arange(layers, device=device, dtype=torch.int32)[:, None]
            * (self.layer_stride // block_size)
            + blocks[None, :]
        )[:, None, :]
        self.first_layer_tables = blocks[None, :].repeat(loops, 1)
        self.first_layer_tables[0].add_(self.lexical_base // block_size)

    @property
    def nbytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    def rows(self, layer, loop, positions):
        """Map absolute positions already known to be live to physical rows."""
        if layer == 0 and loop == 0:
            return self.lexical_base + positions
        base = layer * self.layer_stride
        if loop == self.loops - 1:
            return base + positions
        return (
            base + self.padded_len + loop * self.ring_size + positions % self.ring_size
        )

    def store_prefill(self, layer, loop, keys, values):
        """Persist only the histories needed after native full-prompt attention."""
        end = keys.shape[0]
        if loop == self.loops - 1 or (layer == 0 and loop == 0):
            base = (
                self.lexical_base
                if layer == 0 and loop == 0
                else layer * self.layer_stride
            )
            segments = [(base, 0, end)]
        else:
            begin = max(0, end - self.ring_size)
            slot = begin % self.ring_size
            size = min(end - begin, self.ring_size - slot)
            base = layer * self.layer_stride + self.padded_len + loop * self.ring_size
            segments = [(base + slot, begin, size)]
            if begin + size < end:
                segments.append((base, begin + size, end - begin - size))
        for target, source in ((self.k, keys), (self.v, values)):
            flat = target.view(-1, self.kv_heads, self.head_dim)
            for destination, start, count in segments:
                flat[destination : destination + count].copy_(
                    source[start : start + count]
                )

    def commit(self, keys, values, node_slots, begin, pool_size):
        """Copy accepted KV, overwriting only prefix rows outside the live ring."""
        count = len(node_slots)
        if not count:
            return
        if begin < 0 or begin + count > self.padded_len:
            raise ValueError("Shared prefix commit exceeds history capacity")
        if keys.device.type == "cuda":
            slots = torch.tensor(node_slots, dtype=torch.long, pin_memory=True).to(
                keys.device, non_blocking=True
            )
            width = self.kv_heads * self.head_dim
            _commit_shared_prefix[(self.layers * self.loops * count,)](
                keys,
                values,
                self.k,
                self.v,
                slots,
                count,
                begin,
                pool_size,
                self.layers,
                self.loops,
                self.padded_len,
                self.ring_size,
                width,
                triton.next_power_of_2(width),
            )
            return
        positions = torch.arange(begin, begin + count, device=keys.device)
        slots = torch.tensor(node_slots, device=keys.device, dtype=torch.long)
        for layer in range(self.layers):
            for loop in range(self.loops):
                full = loop == self.loops - 1 or (layer == 0 and loop == 0)
                start = 0 if full else max(0, count - self.ring_size)
                destination = self.rows(layer, loop, positions[start:])
                source = loop * pool_size + slots[start:]
                for target, bank in ((self.k, keys), (self.v, values)):
                    target.view(-1, self.kv_heads, self.head_dim).index_copy_(
                        0, destination, bank[layer].index_select(0, source)
                    )
