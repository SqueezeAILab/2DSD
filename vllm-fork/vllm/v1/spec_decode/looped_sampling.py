# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Small probability readbacks used by the finalized 2D scheduler."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_probabilities(addresses, output, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pointers = tl.load(addresses + indices, mask=indices < COUNT, other=0)
    values = tl.load(
        pointers.to(tl.pointer_type(tl.float32)), mask=indices < COUNT, other=0
    )
    tl.store(output + indices, values, mask=indices < COUNT)


def gather_probabilities(items, buffers=None):
    """Read one element per owned distribution view with one GPU round trip.

    The item list keeps every source tensor alive until the synchronous readback
    completes. Addresses include the view's offset and stride; distributions
    need not share a base matrix or be contiguous.
    """
    if not items:
        return []
    if len(items) == 1:
        dist, token = items[0]
        return [float(dist[token])]
    device = items[0][0].device
    if device.type != "cuda" or any(
        dist.dtype != torch.float32 or dist.device != device for dist, _ in items
    ):
        return [float(dist[token]) for dist, token in items]
    addresses = []
    for dist, token in items:
        if dist.ndim != 1 or not 0 <= token < dist.numel():
            raise IndexError("A probability read requires a valid distribution index")
        addresses.append(dist.data_ptr() + token * dist.stride(0) * dist.element_size())
    if buffers is None:
        buffers = {}
    count = len(items)
    if buffers.get("capacity", 0) < count or buffers.get("device") != device:
        capacity = triton.next_power_of_2(count)
        cpu = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        buffers.update(
            capacity=capacity,
            device=device,
            cpu=cpu,
            addresses=cpu.numpy(),
            metadata=torch.empty(capacity, dtype=torch.int64, device=device),
            output=torch.empty(capacity, dtype=torch.float32, device=device),
        )
    buffers["addresses"][:count] = addresses
    metadata = buffers["metadata"][:count]
    metadata.copy_(buffers["cpu"][:count], non_blocking=True)
    output = buffers["output"][:count]
    _gather_probabilities[(triton.cdiv(len(items), 128),)](
        metadata, output, COUNT=len(items), BLOCK=128
    )
    return output.tolist()
