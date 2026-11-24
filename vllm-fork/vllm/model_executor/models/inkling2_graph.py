# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Static-shape tree runner of Inkling 2 for the 2DSD executor.

The computation of ``Inkling2BatchRunner`` with tensor shapes and control flow
that do not depend on the data, so that a call can replay from CUDA graphs. A
call has up to three phases, each over its own rows padded to a size bucket:

  prelude  new nodes: the layers before the core, in the layers' one plane
  core     every node: one pass of the core, in the plane of its pass
  coda     nodes that reach a readout depth: the layers after the core in the
           plane of that depth, the readout and the node's features

Rows carry only indices: their slot, position and plane, the slots of their
path in the tree (padded to a width bucket), where the last ``kernel - 1``
positions of the path are, and for the core whether a previous pass exists.
Padding rows use a scratch slot. Everything else is state on the device:

  - prefix keys and values of every plane, full-attention layers over the
    whole context and sliding layers over a moving range, and the prefix
    short-conv histories;
  - tree keys, values and conv inputs by plane and slot;
  - node states by slot: after the prelude, after every pass (pass p reads
    pass p - 1 and writes p, so a phase can run twice with the same result),
    the feature taps, and the features of every readout depth.

Attention reads a full-attention layer's prefix over a capacity bucket in
every plane of the layer and takes each row's plane afterwards (and, under
last-loop KV sharing, the final plane below the row's cut); a sliding layer
gathers each row's window. The arithmetic is the batched runner's.

Routed experts. With dense BF16 experts on a GPU (they need about 46 GiB, so
an H100 or larger) the two expert products are vLLM's grouped expert GEMM
(``fused_moe``): BF16 inputs and weights, an FP32 accumulator, the expert's
FP32 scale applied to it before the result is rounded, which is the served
arithmetic. Its shapes are static, so a phase is then captured as a CUDA graph
per (phase, row bucket, path bucket, prefix bucket) on first use and replayed
afterwards; a phase never reads a row it writes, so the warm-up runs before a
capture leave the state unchanged. Otherwise (packed experts, the CPU, another
precision) the experts are the batched runner's and every phase runs eagerly.

All layers must have the same attention geometry (true of the released
checkpoints).
"""

import time
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.models.inkling2_batch import Inkling2BatchRunner
from vllm.model_executor.models import inkling2_ops as ops
from vllm.model_executor.models.inkling2_fast import fused_experts, fused_experts_supported
from vllm.model_executor.models.inkling2_reference import (
    ROUTER_GROUP,
    Inkling2Cache,
    Inkling2Plane,
    l2_clip,
    rms,
    route_experts,
    sigmoid,
    softcap,
)
from vllm.model_executor.models.inkling2_attention import BLOCK_K as TILE_KEYS
from vllm.model_executor.models.inkling2_tree_attention import tree_attention
from vllm.model_executor.models.inkling2_weights import LOOP_STREAMS, layer_prefix

ROW_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
PATH_BUCKETS = (2, 4, 8, 16, 32, 64)
PREFIX_BUCKETS = (256, 1024, 4096, 8192, 16384, 32768, 65536)
SLIDE = 1024  # positions a sliding layer keeps beyond its window before its range moves
# Tiles (of TILE_KEYS keys) of a full layer's context that one program of the attention kernel reads. One
# value for every prefix length: a query's sums are then grouped the same way whether the positions before
# it are committed or still in the tree (a width chosen from the committed length differed between plain
# decoding and a block across a bucket boundary).
SEGMENT_TILES = 8
COMMIT_GRAPH_ROWS = 16  # a commit of at most this many nodes replays a graph
FLAG_LATER, FLAG_PADDING = 1, 2


def _bucket(value, buckets, limit=None):
    """The smallest bucket holding ``value``. Beyond the largest one a call runs at its own size, rounded
    up to 64 (a wide Stage 2 fork can exceed any fixed list; its graph is then captured for that size)."""
    for size in buckets:
        if value <= size:
            return size if limit is None else min(size, limit)
    if limit is not None:
        if value <= limit:
            return limit
        raise ValueError(f"{value} exceeds the limit {limit}")
    return -(-value // 64) * 64


class Inkling2GraphRunner(Inkling2BatchRunner):
    """Node-state runner with fused readouts and static shapes.

    Takes the arguments of ``Inkling2BatchRunner``. ``fused_experts`` selects
    vLLM's grouped expert GEMM (default: whenever the model allows it) and
    ``use_cuda_graph`` graph replay of the phases, which needs it.
    ``kernel_attention`` reads a row's keys and values in place with the Triton
    kernel of ``inkling2_tree_attention`` and keeps them in BF16 (default: with
    a tensor-core model that allows fused experts). A row then gets the bits of
    the sequence forward's attention kernel at its position. ``row_kernels``
    computes the small steps of a layer (AttnRes mixes with their norm, the
    message controllers, the clip of the query and relative states) with the
    row-wise kernels of ``inkling2_ops`` and the stored projections as BF16
    products (default: with kernel attention).
    """

    def __init__(self, model, readout_depths, *, use_cuda_graph=True, fused_experts=None,
                 kernel_attention=None, row_kernels=None, **options):
        dims = {model.config.attention_dims(layer)
                for layer in range(model.config.num_hidden_layers)}
        if len(dims) != 1:
            raise ValueError("The static runner needs one attention geometry in every layer")
        self.heads, self.kv_heads, self.head_dim = next(iter(dims))
        humming = bool(getattr(model, "humming", False))
        capable = fused_experts_supported(model) or humming
        if fused_experts and not capable:
            raise ValueError("Fused experts need dense BF16 experts on a GPU and the served precision")
        self.fused_experts = capable if fused_experts is None else bool(fused_experts)
        # The kernel's tensor-core tiles need a power-of-two head dimension of at least 16.
        readable = capable and self.head_dim >= 16 and self.head_dim & (self.head_dim - 1) == 0
        if kernel_attention and not readable:
            raise ValueError("Kernel attention needs a GPU, the served precision and a power-of-two "
                             "head dimension of at least 16")
        self.kernel_attention = (readable and bool(getattr(model, "tensor_cores", False))
                                 if kernel_attention is None else bool(kernel_attention))
        if row_kernels and not self.kernel_attention:
            raise ValueError("Row kernels run with kernel attention")
        self.row_kernels = self.kernel_attention if row_kernels is None else bool(row_kernels)
        if humming and self.fused_experts and not self.row_kernels:
            raise ValueError("Humming experts run with row kernels")
        self._rows_cache = {}
        self.use_cuda_graph = bool(use_cuda_graph) and self.fused_experts
        self._graphs, self._pool, self._moe_configs = {}, None, {}
        self._commit_graphs = {}
        # Passes run ahead of their calls for a schedule that reads out at depth R only (advance_nodes).
        self.fuse_passes, self._ahead = bool(options.pop("fuse_passes", True)), None
        super().__init__(model, readout_depths, **options)
        self.stats.update(graph_captures=0, graph_replays=0, capture_seconds=0.0)

    # State.

    def _allocate(self):
        c, core = self.config, self.core
        device, dtype = self.device, self.dtype
        self.depth_list = sorted(self.readout_depths)
        self.slots = self.max_nodes
        scratch = self.slots + 1  # the last slot belongs to padding rows
        kernel, block = c.sconv_kernel_size, c.n_layers_per_attn_res_block
        self.counts, self.base, self.fa_index, self.sw_index = [], [], {}, []
        planes = full = sliding = 0
        for layer in range(self.layers):
            count = (1 if layer < core.start else self.R if layer < core.stop
                     else len(self.depth_list))
            self.counts.append(count)
            self.base.append(planes)
            if c.is_local(layer):
                self.sw_index += range(sliding, sliding + count)
                sliding += count
            else:
                self.fa_index[layer] = full
                self.sw_index += [0] * count
                full += count
            planes += count
        self.planes = planes
        self._base = torch.tensor(self.base, dtype=torch.long, device=device)
        self.window_size = c.sliding_window_size
        self.sw_capacity = self.window_size - 1 + SLIDE
        kv, dim, hidden = self.kv_heads, self.head_dim, self.H
        zeros = lambda *shape: torch.zeros(shape, dtype=dtype, device=device)  # noqa: E731
        # Keys and values are BF16 values: the kernel reads them as stored.
        half = torch.bfloat16 if self.kernel_attention else dtype
        store = lambda *shape: torch.zeros(shape, dtype=half, device=device)  # noqa: E731
        self.PKf, self.PVf = store(full, self.max_len, kv, dim), store(full, self.max_len, kv, dim)
        self.PKs = store(sliding, self.sw_capacity, kv, dim)
        self.PVs = store(sliding, self.sw_capacity, kv, dim)
        self.sw_first = {layer: self.sw_index[self.base[layer]] for layer in range(self.layers)
                         if c.is_local(layer)}
        self.channels = {"k": kv * dim, "v": kv * dim, "attn": hidden, "mlp": hidden}
        # The conv inputs are BF16 values too; the row kernels read them as stored.
        inputs = store if self.row_kernels else zeros
        self.PC = {s: inputs(planes, kernel - 1, n) for s, n in self.channels.items()}
        self.TK, self.TV = store(planes, scratch, kv, dim), store(planes, scratch, kv, dim)
        self.TC = {s: inputs(planes, scratch, n) for s, n in self.channels.items()}
        index = lambda values: torch.tensor(values, dtype=torch.long, device=device)  # noqa: E731
        self.sw_index = index(self.sw_index)
        is_full = [not c.is_local(layer) for layer in range(self.layers) for _ in range(self.counts[layer])]
        self.full_planes = index([g for g, flag in enumerate(is_full) if flag])
        self.sliding_planes = index([g for g, flag in enumerate(is_full) if not flag])
        self.pre_count = sum(layer % block == 0 for layer in range(core.start))
        self.core_count = sum(layer % block == 0 for layer in core)
        self.pre_blocks = zeros(scratch, self.pre_count, hidden)
        self.pre_partial = zeros(scratch, hidden)
        self.core_blocks = zeros(self.R, scratch, self.core_count, hidden)
        self.core_partial = zeros(self.R, scratch, hidden)
        self.pre_taps = [layer for layer in self.feature_layers if layer < core.start]
        self.core_taps = [layer for layer in self.feature_layers if core.start <= layer < core.stop]
        self.pre_tap = zeros(scratch, len(self.pre_taps), hidden)
        self.core_tap = zeros(self.R, scratch, len(self.core_taps), hidden)
        self.feat = zeros(len(self.depth_list), scratch, self.nf * hidden)
        self.slot_pos = torch.zeros(scratch, dtype=torch.long, device=device)
        self._arange = torch.arange(max(self.max_len, self.window_size, c.rel_extent, 256), device=device)
        self._length = torch.zeros((), dtype=torch.long, device=device)
        self._origin = torch.zeros((), dtype=torch.long, device=device)
        # The same two numbers as the kernel takes them, and the origin of a full-attention prefix.
        self._length1, self._origin1 = self._length.view(1), self._origin.view(1)
        self._zero1 = torch.zeros(1, dtype=torch.long, device=device)
        self._allocated = True

    def _grow(self, slots):
        def wider(x, dim):
            shape = list(x.shape)
            shape[dim] = slots - self.slots
            return torch.cat((x, x.new_zeros(shape)), dim)

        self.TK, self.TV = wider(self.TK, 1), wider(self.TV, 1)
        self.TC = {s: wider(x, 1) for s, x in self.TC.items()}
        for name, dim in (("pre_blocks", 0), ("pre_partial", 0), ("core_blocks", 1), ("core_partial", 1),
                          ("pre_tap", 0), ("core_tap", 1), ("feat", 1), ("slot_pos", 0)):
            setattr(self, name, wider(getattr(self, name), dim))
        # The scratch slot stays last: nodes take the slots before it.
        self.free.extend(range(slots - 1, self.slots - 1, -1))
        self.slots = slots
        # Captured graphs hold the old tensors, and the outputs of the last replay, which the executor may still
        # hold, live in the old pool: the next capture starts a new pool, as DoryGraphRunner does on growth.
        self._graphs.clear()
        self._commit_graphs.clear()
        self._pool = None

    def reset(self):
        if not self._allocated:
            self._allocate()
        self.length = self.origin = 0
        self._ahead = None
        self._length.zero_()
        self._origin.zero_()
        self.nodes = {}
        self.free = list(range(self.slots - 1, -1, -1))

    @property
    def prefix(self):
        """The prefix in the reference cache's form (copies; for inspection)."""
        planes, length, keep = {}, self.length, self.window_size - 1
        for layer in range(self.layers):
            for i in range(self.counts[layer]):
                g = self.base[layer] + i
                index = (0 if layer < self.core.start else i + 1 if layer < self.core.stop
                         else self.depth_list[i])
                if self.config.is_local(layer):
                    first = max(0, length - keep)
                    span = slice(first - self.origin, length - self.origin)
                    at = int(self.sw_index[g])
                    keys, values = self.PKs[at, span], self.PVs[at, span]
                else:
                    first, at = 0, self.fa_index[layer] + i
                    keys, values = self.PKf[at, :length], self.PVf[at, :length]
                planes[(layer, index)] = Inkling2Plane(keys.clone(), values.clone(), first,
                                                       {s: x[g].clone() for s, x in self.PC.items()})
        return SimpleNamespace(length=length, planes=planes)

    def _set_length(self, length, origin):
        self.length, self.origin = length, origin
        self._length.fill_(length)
        self._origin.fill_(origin)

    # Prompt.

    @torch.no_grad()
    def prefill(self, token_ids):
        self.reset()
        core, cache = self.core, Inkling2Cache()
        visits = [layer if layer < core.start else layer + (self.R - 1) * len(core)
                  for layer in self.feature_layers]
        out = self.model.forward(token_ids, num_passes=self.R, depths=tuple(self.depth_list), taps=visits,
                                 cache=cache, pass_end=False)
        origin = 0
        for (layer, index), plane in cache.planes.items():
            i = (0 if layer < core.start else index - 1 if layer < core.stop
                 else self.depth_list.index(index))
            g, count = self.base[layer] + i, plane.keys.shape[0]
            if self.config.is_local(layer):
                at, origin = int(self.sw_index[g]), plane.first
                self.PKs[at, :count], self.PVs[at, :count] = plane.keys, plane.values
            else:
                at = self.fa_index[layer] + i
                self.PKf[at, :count], self.PVf[at, :count] = plane.keys, plane.values
            for stream, history in plane.conv.items():
                self.PC[stream][g] = history
        self._set_length(cache.length, origin)
        hidden = out.readouts[self.R].hidden
        features = torch.cat([*(out.features[visit] for visit in visits),
                              *([hidden] if self.final_feature else [])], -1)
        return SimpleNamespace(hidden=hidden, features=features)

    # Tree.

    @torch.no_grad()
    def commit(self, node_ids, features=None):
        self._check_commit(node_ids)
        if not node_ids:
            return
        slots = torch.tensor([self.nodes[node_id].slot for node_id in node_ids], device=self.device)
        keep = self.window_size - 1
        for begin in range(0, len(node_ids), SLIDE):
            part = slots[begin:begin + SLIDE]
            count, length, origin = part.numel(), self.length, self.origin
            start = length - origin
            if start + count > self.sw_capacity:
                self.PKs[:, :keep] = self.PKs[:, start - keep:start].clone()
                self.PVs[:, :keep] = self.PVs[:, start - keep:start].clone()
                origin = length - keep
                self._origin.fill_(origin)  # the move below reads it
            if self.use_cuda_graph and count <= COMMIT_GRAPH_ROWS:
                entry = self._commit_graphs.get(count)
                if entry is None:
                    entry = self._commit_graphs[count] = {"slots": part.clone(), "graph": torch.cuda.CUDAGraph()}
                    torch.cuda.synchronize(self.device)
                    if self._pool is None:
                        self._pool = torch.cuda.graph_pool_handle()
                    with torch.cuda.graph(entry["graph"], pool=self._pool):
                        self._move(entry["slots"])  # recorded, not run: the replay below moves the rows
                else:
                    entry["slots"].copy_(part)
                entry["graph"].replay()
            else:
                self._move(part)
            self._set_length(length + count, origin)
        if features is not None:
            features.copy_(self.feat[-1][slots])  # depth R is the last readout depth
        for node_id in node_ids:
            node = self.nodes.pop(node_id)
            self.free.append(node.slot)
            for child in node.children:
                self.nodes[child].parent = None
        self.stats["committed_nodes"] += len(node_ids)

    def _move(self, part):
        """The keys, values and conv inputs of the tree slots ``part`` become the
        prefix rows from the committed length on. The length and the sliding
        origin are read on the device, so the body can be a graph."""
        rows = self._length1 + torch.arange(part.numel(), device=self.device)
        full, sliding = self.full_planes[:, None], self.sliding_planes[:, None]
        self.PKf.index_copy_(1, rows, self.TK[full, part[None, :]])
        self.PVf.index_copy_(1, rows, self.TV[full, part[None, :]])
        at = rows - self._origin1
        self.PKs.index_copy_(1, at, self.TK[sliding, part[None, :]])
        self.PVs.index_copy_(1, at, self.TV[sliding, part[None, :]])
        for stream, history in self.PC.items():  # in place: graphs hold these tensors
            joined = torch.cat((history, self.TC[stream][:, part]), 1)
            history.copy_(joined[:, joined.shape[1] - history.shape[1]:])

    # Execution.

    def _stage(self, nodes, chains, planes, flags=None, extra=None):
        """Rows of one phase on the device: the nodes padded to a row bucket."""
        history = self.config.sconv_kernel_size - 1
        count = _bucket(len(nodes), ROW_BUCKETS)
        width = _bucket(max(len(chains[id(node)]) for node in nodes) + 1, PATH_BUCKETS)
        scratch = self.slots
        meta = np.zeros((count, 5 + 4 * history + 2 * width), np.int64)
        meta[:, 0], meta[:, 1], meta[:, 3] = scratch, self.length, FLAG_PADDING
        meta[:, 5 + 3 * history:5 + 3 * history + width] = scratch
        meta[:, 5 + 3 * history + width] = 1  # a padding row reads its own scratch key
        meta[:, 5 + 3 * history + 2 * width:] = -1
        row_of = {node.slot: row for row, node in enumerate(nodes)}
        for row, node in enumerate(nodes):
            ancestors = [member.slot for member in chains[id(node)]]
            line = meta[row]
            line[0], line[1], line[2] = node.slot, node.position, planes[row]
            line[3] = flags[row] if flags is not None else 0
            line[4] = extra[row] if extra is not None else 0
            for j in range(history):
                behind = len(ancestors) - history + j
                if behind >= 0:
                    line[5 + j], line[5 + history + j] = 1, ancestors[behind]
                    # An earlier input computed by a row of this call in the same plane: its row.
                    source = row_of.get(ancestors[behind])
                    if source is not None and planes[source] == planes[row]:
                        line[5 + 3 * history + 2 * width + j] = source
                else:
                    line[5 + 2 * history + j] = history + behind
            path = ancestors + [node.slot]
            start = 5 + 3 * history
            line[start:start + width] = scratch
            line[start:start + len(path)] = path
            line[start + width:start + 2 * width] = 0
            line[start + width:start + width + len(path)] = 1
        return meta, width

    def _unpack(self, meta, width):
        history = self.config.sconv_kernel_size - 1
        start = 5 + 3 * history
        r = SimpleNamespace(
            slot=meta[:, 0], pos=meta[:, 1], plane=meta[:, 2], later=(meta[:, 3] & FLAG_LATER) != 0,
            extra=meta[:, 4], hist_tree=meta[:, 5:5 + history] != 0,
            hist_slot=meta[:, 5 + history:5 + 2 * history],
            hist_prefix=meta[:, 5 + 2 * history:start],
            path=meta[:, start:start + width], path_mask=meta[:, start + width:start + 2 * width] != 0,
            hist_row=meta[:, start + 2 * width:start + 2 * width + history])
        if self.kernel_attention:  # the kernel reads contiguous vectors; one set serves every layer
            r.pos, r.plane, r.path = r.pos.contiguous(), r.plane.contiguous(), r.path.contiguous()
            r.reads = {}
        if self.row_kernels:
            r.slot = r.slot.contiguous()
            r.hist_slot, r.hist_prefix = r.hist_slot.contiguous(), r.hist_prefix.contiguous()
            r.hist_row = r.hist_row.contiguous()
            r.planes = self._base[:, None] + r.plane[None, :]  # [layers, n]: a layer's row of plane indices
            r.stacks, r.tau = {}, None
            r.padding = (meta[:, 3] & FLAG_PADDING) != 0  # these rows take row 0's experts
            c = self.config
            if c.log_scaling_n_floor is not None:  # the full layers' query scale, the same for all of them
                ratio = (r.pos + 1).to(self.model.precision.dtype) * (1.0 / c.log_scaling_n_floor)
                r.tau = 1.0 + c.log_scaling_alpha * torch.log(ratio.clamp(min=1.0))
        return r

    @torch.no_grad()
    def advance_nodes(self, node_ids, loops):
        ahead, self._ahead = self._ahead, None
        chains = self._check_advance(node_ids, loops)
        self._count(loops)
        nodes = [self.nodes[node_id] for node_id in node_ids]
        if ahead is not None and ahead["ids"] == tuple(node_ids) and all(loop == ahead["loop"] for loop in loops):
            # The pass of this call ran with the rows' first call (below): only the bookkeeping is left.
            for node in nodes:
                node.depth += 1
            ahead["loop"] += 1
            if ahead["loop"] < self.R:
                self._ahead = ahead
                return [], None, None
            return ahead["out"]
        chains = {id(node): chains[node_id] for node, node_id in zip(nodes, node_ids)}
        capacity = _bucket(self.length, PREFIX_BUCKETS, self.max_len)
        fresh = [node for node, loop in zip(nodes, loops) if loop == 0]
        if fresh:
            self._run("prelude", capacity, False,
                      *self._stage(fresh, chains, [0] * len(fresh), extra=[n.token for n in fresh]))
        host, width = self._stage(nodes, chains, loops, flags=[FLAG_LATER if loop else 0 for loop in loops])
        self._run("core", capacity, any(loops), host, width)
        for node in nodes:
            node.depth += 1
        if self.fuse_passes and self.R > 1 and len(fresh) == len(nodes) and self.readout_depths == {self.R}:
            # A schedule that reads out after the last pass only (plain decoding, block drafting) calls
            # again with the same rows until then: the later passes and the readout run now, in this
            # host round, and those calls find their results here. Rows that change in between (none
            # do in such a schedule) are computed again by the ordinary path, which gives the same state.
            # The rows of a later pass and of the readout differ from the first pass's in three columns.
            count = len(nodes)
            for loop in range(1, self.R):
                host[:count, 2], host[:count, 3] = loop, FLAG_LATER
                self._run("core", capacity, True, host, width)
            host[:count, 2], host[:count, 3], host[:count, 4] = self.depth_list.index(self.R), 0, self.R - 1
            logits, argmax = self._run("coda", capacity, False, host, width)
            self._ahead = {"ids": tuple(node_ids), "loop": 1,
                           "out": (list(range(count)), argmax[:count], logits[:count])}
            return [], None, None
        read = [row for row, loop in enumerate(loops) if loop + 1 in self.readout_depths]
        if not read:
            return read, None, None
        logits, argmax = self._run(
            "coda", capacity, False,
            *self._stage([nodes[row] for row in read], chains,
                         [self.depth_list.index(loops[row] + 1) for row in read],
                         extra=[loops[row] for row in read]))
        return read, argmax[:len(read)], logits[:len(read)]

    def _run(self, phase, capacity, later, host, width):
        """One phase over staged rows: eagerly, or by replaying its graph."""
        if not self.use_cuda_graph:
            meta = torch.from_numpy(host).to(self.device)
            return self._body(phase, meta, width, capacity, later)
        key = (phase, host.shape[0], width, capacity, later)
        entry = self._graphs.get(key)
        if entry is None:
            meta = torch.zeros(host.shape, dtype=torch.long, device=self.device)
            entry = self._graphs[key] = {"meta": meta, "graph": None, "out": None}
        entry["meta"].copy_(torch.from_numpy(host))
        if entry["graph"] is None:
            begun = time.perf_counter()
            for _ in range(2):  # also compiles the kernels of this shape
                self._body(phase, entry["meta"], width, capacity, later)
            torch.cuda.synchronize(self.device)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                entry["out"] = self._body(phase, entry["meta"], width, capacity, later)
            entry["graph"] = graph
            self.stats["graph_captures"] += 1
            self.stats["capture_seconds"] += time.perf_counter() - begun
        entry["graph"].replay()
        self.stats["graph_replays"] += 1
        return entry["out"]

    def _body(self, phase, meta, width, capacity, later):
        r = self._unpack(meta, width)
        with self._products():
            if phase == "prelude":
                return self._prelude(r, capacity)
            if phase == "core":
                return self._core(r, capacity, later)
            return self._coda(r, capacity)

    def _prelude(self, r, capacity):
        blocks, partial = [], self._embed(r.extra)
        for layer in range(self.core.start):
            if layer in self.pre_taps:
                self.pre_tap[r.slot, self.pre_taps.index(layer)] = partial
            blocks, partial = self._layer_step(layer, r, capacity, blocks, partial)
        if blocks:
            self.pre_blocks[r.slot] = torch.stack(blocks, 1)
        self.pre_partial[r.slot] = partial

    def _embed(self, tokens):
        """The normed embeddings of the rows. With row kernels the norm is a program per row: a torch
        reduction sums in an order that depends on the number of rows."""
        if not self.row_kernels:
            return self.model.embed_tokens(tokens)
        model = self.model
        weight = self._cached("embed_norm", lambda: model._w("embed_norm.weight").contiguous())
        return ops.rms_norm(model.checkpoint.embedding[tokens], weight, self.config.rms_norm_eps,
                            out_dtype=torch.bfloat16).to(self.dtype)

    def _core(self, r, capacity, later):
        blocks = list(self.pre_blocks[r.slot].unbind(1))
        partial = self.pre_partial[r.slot]
        bank = None
        if later:  # rows of the first pass read a stale bank and discard the blend
            before = (r.plane - 1).clamp(min=0)
            bank = [*self.core_blocks[before, r.slot][:, 1:].unbind(1), self.core_partial[before, r.slot]]
        for layer in self.core:
            if layer in self.core_taps:
                self.core_tap[r.plane, r.slot, self.core_taps.index(layer)] = partial
            blocks, partial = self._layer_step(layer, r, capacity, blocks, partial, bank)
        if self.core_count:
            self.core_blocks[r.plane, r.slot] = torch.stack(blocks[self.pre_count:], 1)
        self.core_partial[r.plane, r.slot] = partial

    def _coda(self, r, capacity):
        model, depth = self.model, r.extra  # the pass the readout follows, counted from 0
        blocks = [*self.pre_blocks[r.slot].unbind(1), *self.core_blocks[depth, r.slot].unbind(1)]
        partial = self.core_partial[depth, r.slot]
        coda = {}
        for layer in range(self.core.stop, self.layers):
            if layer in self.feature_layers:
                coda[layer] = partial
            blocks, partial = self._layer_step(layer, r, capacity, blocks, partial)
        hidden = self._readout_hidden(blocks, partial)
        taps = [self.pre_tap[r.slot, self.pre_taps.index(layer)] if layer in self.pre_taps
                else self.core_tap[depth, r.slot, self.core_taps.index(layer)] if layer in self.core_taps
                else coda[layer]
                for layer in self.feature_layers]
        self.feat[r.plane, r.slot] = torch.cat([*taps, *([hidden] if self.final_feature else [])], -1)
        logits = self.logits(hidden)
        return logits, logits.argmax(-1)

    def _readout_hidden(self, blocks, partial):
        """The final-norm state of the rows, the unembedding's input (held in the arithmetic dtype)."""
        if not self.row_kernels:
            return self.model.readout_hidden(blocks, partial)
        c = self.config
        norm = self._cached("final_norm", lambda: self.model._w("norm.weight")[None].contiguous())
        return ops.attn_res(torch.stack(blocks), partial, None, self._query("attn_res.output_query"), norm,
                            c.rms_norm_eps, True, c.attn_res_eps, out_dtype=torch.bfloat16)[0].to(self.dtype)

    def raw_logits(self, hidden):
        """Logits before the softcap. With row kernels: the BF16 unembedding read as stored and
        accumulated in FP32, a row's logits the same in any call."""
        if not self.row_kernels:
            return super().raw_logits(hidden)
        return ops.gemv_any(hidden.to(torch.bfloat16), self.model.checkpoint.unembedding)

    def vocabulary_logits(self, ids):
        """A callable from hidden states [n, H] to the logits before the softcap of the tokens ``ids``
        alone [n, len(ids)]: their unembedding rows, gathered once (a drafter's restricted readout, in BF16
        with row kernels: the dtype a DSpark head sums in)."""
        if not self.row_kernels:
            return lambda hidden: self.raw_logits(hidden)[:, ids]
        table = self.model.checkpoint.unembedding[ids].contiguous()
        return lambda hidden: ops.gemv_any(hidden.to(torch.bfloat16), table, torch.bfloat16)

    def logits(self, hidden):
        if not self.row_kernels:
            return super().logits(hidden)
        return softcap(self.raw_logits(hidden), self.config.final_logit_softcapping)

    def _moe(self, layer, x):
        """Routed experts: [n, H]. With fused experts, vLLM's grouped expert GEMM
        with the expert's scale as the per-row weight it applies to the FP32
        accumulator before rounding."""
        if not self.fused_experts:
            return super()._moe(layer, x)
        c, weight = self.config, self.model._lw
        gate = weight(layer, "mlp.gate.weight")
        logits = sum(x[:, k:k + ROUTER_GROUP] @ gate[:, k:k + ROUTER_GROUP].T
                     for k in range(0, c.hidden_size, ROUTER_GROUP))
        ids, weights = route_experts(logits, weight(layer, "mlp.gate.bias"),
                                     c.num_experts_per_tok, c.route_scale)
        # ids is a slice of the sorted experts; the kernel's row grouping reads it flat.
        return fused_experts(c, self.model.precision, self.model.checkpoint.experts(layer), x,
                             ids.contiguous(), weights, self._moe_configs)

    def _layer_step(self, layer, r, capacity, blocks, partial, bank=None):
        step = self._kernel_layer if self.row_kernels else self._static_layer
        return step(layer, r, capacity, blocks, partial, bank)

    # The layer with row kernels.

    def _half(self, layer, name):
        """A layer weight as the checkpoint stores it (BF16)."""
        return self.model.checkpoint.tensors[layer_prefix(layer) + name]

    def _cached(self, key, build):
        found = self._rows_cache.get(key)
        if found is None:
            found = self._rows_cache[key] = build()
        return found

    def _norm_rows(self, layer):
        """The norm weights of the attention inputs, one row per input: [1 or 3, H]."""
        names = (("attn_norm",) if self.config.is_local(layer)
                 else ("q_input_norm", "k_input_norm", "v_input_norm"))
        return self._cached((layer, "norms"), lambda: torch.stack(
            [self.model._lw(layer, name + ".weight") for name in names]).contiguous())

    def _controllers(self, layer):
        """The four controllers of a core layer in LOOP_STREAMS order, as stored
        (BF16): down [4, rank, 2H] and [gain; forget] [4, 2H, rank]; then the
        same for the three attention streams alone, contiguous."""
        def build():
            down = torch.stack([self._half(layer, f"controllers.{s}.down_weight") for s in LOOP_STREAMS])
            gain_forget = torch.stack([torch.cat((self._half(layer, f"controllers.{s}.gain_weight"),
                                                  self._half(layer, f"controllers.{s}.forget_weight")))
                                       for s in LOOP_STREAMS])
            down, gain_forget = down.contiguous(), gain_forget.contiguous()
            return {(0, 3): (down[:3].contiguous(), gain_forget[:3].contiguous()),
                    (3, 4): (down[3:].contiguous(), gain_forget[3:].contiguous())}
        return self._cached((layer, "controllers"), build)

    def _projections(self, layer, shared):
        """The stored projections of a layer grouped by their input, each group
        one BF16 weight [sum of widths, H] with the widths. With one ``shared``
        input (a sliding layer without messages) all six are one group; else q,
        r, gate and sink read the query input, the keys theirs and the values
        theirs (a full layer's with the taps of the dynamic conv)."""
        def build():
            local = self.config.is_local(layer)
            query = ["attn.wq_du.weight", "attn.wr_du.weight", "attn.gate_attn.weight", "attn.attn_sink_proj.weight"]
            keys = ["attn.wk_dv.weight"] + ([] if local else ["attn.k_sconv.weight_proj"])
            values = ["attn.wv_dv.weight"] + ([] if local else ["attn.v_sconv.weight_proj"])
            built = []
            for names in ([query + keys + values] if shared else [query, keys, values]):
                weights = [self._half(layer, name) for name in names]
                built.append((torch.cat(weights).contiguous() if len(weights) > 1 else weights[0],
                              [w.shape[0] for w in weights]))
            return built
        return self._cached((layer, "projections", shared), build)

    def _clip_scale(self, layer, name, heads, width):
        return self._cached((layer, name), lambda: self.model._lw(layer, f"attn.{name}.scale")
                            .expand(heads, width).contiguous())

    def _blend_rows(self, layer, streams, dest, message, norm_weight, later):
        """The controllers ``streams`` on the FP32 mixes ``dest`` and messages
        [S, n, H], then the norm: BF16 [S, n, H]. A row of the first pass has no
        message: its gates are zero, which leaves its own mix."""
        down, gain_forget = self._controllers(layer)[(streams.start, streams.stop)]
        pair = ops.pair_rms_norm(dest, message)
        # One launch per product, the first in parallel chunks of its inputs. A row's products do not
        # depend on the other rows of the call.
        gates = ops.gemv_streams(ops.gemv_streams(pair, down, chunked=True), gain_forget)
        return ops.blend_rms_norm(dest, message, gates, norm_weight, self.config.rms_norm_eps, later)

    @staticmethod
    def _stacked(r, sources, name="blocks"):
        """The sources of an AttnRes mix as one tensor [S, n, H]. Within a phase a
        list only grows, so its length names its content: one copy per length."""
        key = (name, len(sources))
        found = r.stacks.get(key)
        if found is None:
            found = r.stacks[key] = torch.stack(sources)
        return found

    def _kernel_layer(self, layer, r, capacity, blocks, partial, bank=None):
        """``_static_layer`` with the small steps as row kernels. Mixes, norms and
        controllers are one launch each, the clip of q and r one each, and a
        projection whose result is stored is a BF16 product on the checkpoint's
        own weight."""
        model, c = self.model, self.config
        p, half = model.precision, torch.bfloat16
        weight, local, core = model._lw, c.is_local(layer), layer in self.core
        norms, count = self._norm_rows(layer), partial.shape[0]
        eps, norm_eps = c.attn_res_eps, c.rms_norm_eps
        messages = None
        if layer == 0:
            x = torch.stack([ops.rms_norm(partial, norm, norm_eps, out_dtype=half) for norm in norms])
        else:
            stack = self._stacked(r, blocks)
            queries = self._query(f"attn_res.attn_queries.{layer - 1}")
            if not core:
                x = ops.attn_res(stack, partial, None, queries, norms, norm_eps, True, eps, out_dtype=half)
            else:
                # A core layer norms its mixes in the blend kernel whether the call carries messages or
                # not: a first-pass row then gets the same bits alone and among later-pass rows.
                mixes = ops.attn_res(stack, partial, None, queries, None, 0.0, True, eps)
                if bank is None:
                    x = ops.blend_rms_norm(mixes, None, None, norms, norm_eps)
                else:
                    messages = ops.attn_res(self._stacked(r, bank[:-1], "bank"), bank[-1], None,
                                            self._query(layer), None, 0.0, True, eps)
                    x = self._blend_rows(layer, slice(0, 3), mixes.expand(3, -1, -1) if local else mixes,
                                         messages[:3], norms.expand(3, -1) if local else norms, r.later)
        x_q, x_k, x_v = x.unbind(0) if x.shape[0] == 3 else (x[0],) * 3
        heads, dim = self.heads, self.head_dim
        tau = None if local else r.tau
        # One product per input: the columns of each are the projections that read it. A sliding
        # layer has one input for all of them unless messages were blended into q, k and v.
        shared = x.shape[0] == 1
        # GEMV launches: an output is its row's ordered sum with its weight row, so it is the same in a
        # call of one row or many, and in a merged product or its own (cuBLAS picks kernels by shape).
        products = [ops.gemv_any(x_in, weights, half).split(widths, dim=1)
                    for x_in, (weights, widths) in zip((x_q, x_k, x_v), self._projections(layer, shared))]
        wq, wr, gate, sink = products[0][:4]
        query = ops.clip_heads(wq, self._clip_scale(layer, "q_l2_clip", heads, dim), heads,
                               c.qk_l2_clip_threshold, c.l2_clip_eps, tau)
        rel_states = ops.clip_heads(wr, self._clip_scale(layer, "rel_l2_clip", heads, c.d_rel), heads,
                                    c.rel_l2_clip_threshold, c.l2_clip_eps, tau)
        gate = gate.view(count, heads, dim)
        g = r.planes[layer]
        if shared:
            (keys, values), k_taps, v_taps = products[0][4:], None, None
        elif local:
            (keys,), (values,), k_taps, v_taps = products[1], products[2], None, None
        else:
            (keys, k_taps), (values, v_taps) = products[1], products[2]
        keys = self._kernel_key_value(r, g, layer, "k", keys, k_taps)
        values = self._kernel_key_value(r, g, layer, "v", values, v_taps)
        first_plane, planes = self.base[layer], self.counts[layer]
        ops.store_key_value(keys, values, self.TK[first_plane:first_plane + planes],
                            self.TV[first_plane:first_plane + planes], r.plane, r.slot,
                            c.qk_l2_clip_threshold, c.l2_clip_eps, c.rms_norm_eps)
        out = self._kernel_attention(r, layer, query, rel_states, gate, sink, rounded=True, capacity=capacity)
        attn = ops.gemv_any(out.view(count, -1), self._half(layer, "attn.wo_ud.weight"), half)
        attn = self._kernel_conv(r, g, "attn", attn, weight(layer, "attn_sconv.weight").squeeze(1))
        first = layer % c.n_layers_per_attn_res_block == 0
        if first:  # the partial closes a block: the sources are the blocks and the attention output
            blocks, last, addend = [*blocks, partial], attn, None
        else:  # the last source is partial + attn, summed in the kernel
            last, addend = partial, attn
        mlp_query = self._query(f"attn_res.mlp_queries.{layer}")
        mlp_norm = self._cached((layer, "mlp_norm"), lambda: weight(layer, "mlp_norm.weight")[None].contiguous())
        stack = self._stacked(r, blocks)
        if not core:
            y = ops.attn_res(stack, last, addend, mlp_query, mlp_norm, norm_eps, True, eps, out_dtype=half)
        else:
            mix = ops.attn_res(stack, last, addend, mlp_query, None, 0.0, True, eps)
            y = (ops.blend_rms_norm(mix, None, None, mlp_norm, norm_eps) if messages is None
                 else self._blend_rows(layer, slice(3, 4), mix, messages[3:], mlp_norm, r.later))
        mlp = self._kernel_moe(layer, y[0], r.padding)
        mlp = self._kernel_conv(r, g, "mlp", mlp, weight(layer, "mlp_sconv.weight").squeeze(1))
        if first:
            return blocks, ops.sum_rows(attn, mlp, out_dtype=p.dtype)
        return blocks, ops.sum_rows(partial, attn, mlp, out_dtype=p.dtype)

    def _kernel_moe(self, layer, y, padding=None):
        """Routed experts of the BF16 normed MLP input y [n, H]: BF16 [n, H]. The
        router's choice, the activation and the sum of a row's experts are row
        kernels around vLLM's grouped expert GEMM."""
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            _prepare_expert_assignment,
            invoke_fused_moe_triton_kernel,
            try_get_optimal_moe_config,
        )
        from vllm.triton_utils import tl

        c, p, weight = self.config, self.model.precision, self.model._lw
        rows, top = y.shape[0], c.num_experts_per_tok
        ids, weights = ops.router_rows(y, self._half(layer, "mlp.gate.weight"), weight(layer, "mlp.gate.bias"),
                                       top, c.route_scale, padding)
        if self.model.humming:  # the same sums as the sequence forward's, scheduled for a few rows
            return self.model.humming_experts(layer).products(y, ids, weights, c, few=True)
        experts = self.model.checkpoint.experts(layer)
        kernel = self._moe_configs.get(rows)
        if kernel is None:
            kernel = self._moe_configs[rows] = try_get_optimal_moe_config(
                experts.w13.shape, experts.w2.shape, top, None, rows)
        assignment = _prepare_expert_assignment(ids, kernel, rows, top, c.n_routed_experts, None,
                                                ignore_invalid_experts=True)
        flags = dict(compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
                     use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
        half = torch.bfloat16
        gate_up = torch.empty((rows, top, experts.w13.shape[1]), dtype=half, device=y.device)
        invoke_fused_moe_triton_kernel(y, experts.w13, gate_up, None, None, experts.w13_scale[ids],
                                       *assignment, True, top, kernel, **flags)
        hidden = ops.situ_rows(gate_up.view(rows * top, -1), weights.view(-1), c.mlp_gate_softcap,
                               c.mlp_up_softcap)
        down = torch.empty((rows, top, c.hidden_size), dtype=half, device=y.device)
        invoke_fused_moe_triton_kernel(hidden, experts.w2, down, None, None, experts.w2_scale[ids].view(-1, 1),
                                       *assignment, True, 1, kernel, **flags)
        return ops.sum_slots(down)

    def _kernel_conv(self, r, g, stream, x, weight, dynamic=None):
        """One stream's short conv of the rows' BF16 inputs x [n, C]: BF16. The
        launch also stores the inputs; rows of this visit read each other's from x."""
        return ops.conv_rows(x, self.TC[stream], self.PC[stream], g, r.hist_tree, r.hist_slot, r.hist_prefix,
                             weight, dynamic, store=(r.slot, r.hist_row))

    def _kernel_key_value(self, r, g, layer, stream, projected, taps):
        """The short conv of the k or v projection [n, kv_heads * D] (BF16), with
        the projected taps of a full layer's dynamic conv."""
        name = f"attn.{stream}_sconv."
        if taps is None:
            return self._kernel_conv(r, g, stream, projected, self.model._lw(layer, name + "weight").squeeze(1))
        taps = taps.contiguous().view(-1, self.kv_heads, self.config.sconv_kernel_size)
        return self._kernel_conv(r, g, stream, projected, self.model._lw(layer, name + "static_w"), taps)

    def _static_layer(self, layer, r, capacity, blocks, partial, bank=None):
        """``Inkling2BatchRunner._layer`` without data-dependent shapes: a row of
        the first pass computes the blend with the previous pass and keeps its
        own mix."""
        model, c = self.model, self.config
        p, weight = model.precision, model._lw
        local = c.is_local(layer)
        names = ("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm")
        norms = [weight(layer, name + ".weight") for name in names]
        messages = None
        if layer == 0:
            inputs = [model._norm(partial, norm) for norm in norms]
        else:
            mixes = self._res(blocks, partial, self._query(f"attn_res.attn_queries.{layer - 1}"))
            if bank is None:
                inputs = [model._norm(mix, norm) for mix, norm in zip(mixes, norms)]
            else:
                messages = self._res(bank[:-1], bank[-1], self._query(layer))
                inputs = []
                for s, stream in enumerate(LOOP_STREAMS[:3]):
                    mix = mixes[0 if local else s]
                    blend = model._blend(layer, stream, mix, messages[s])
                    inputs.append(model._norm(torch.where(r.later[:, None], blend, mix),
                                              norms[0 if local else s]))
        x_q, x_k, x_v = inputs if len(inputs) == 3 else inputs * 3
        heads, kv_heads, dim = self.heads, self.kv_heads, self.head_dim
        count = x_q.shape[0]
        tau = None
        if not local and c.log_scaling_n_floor is not None:
            ratio = (r.pos + 1).to(p.dtype) * (1.0 / c.log_scaling_n_floor)
            tau = (1.0 + c.log_scaling_alpha * torch.log(ratio.clamp(min=1.0)))[:, None, None]

        def clipped(name, width, scale, threshold):
            x = p.store(x_q @ weight(layer, f"attn.{name}.weight").T).view(count, heads, width)
            x = p.store(l2_clip(x * weight(layer, f"attn.{scale}.scale"), threshold, c.l2_clip_eps))
            return x if tau is None else p.store(x * tau)

        query = clipped("wq_du", dim, "q_l2_clip", c.qk_l2_clip_threshold)
        rel_states = clipped("wr_du", c.d_rel, "rel_l2_clip", c.rel_l2_clip_threshold)
        gate = p.store(x_q @ weight(layer, "attn.gate_attn.weight").T).view(count, heads, dim)
        sink = p.store(x_q @ weight(layer, "attn.attn_sink_proj.weight").T)
        g = self.base[layer] + r.plane
        keys = self._static_key_value(r, g, layer, "k", x_k)
        self.TK[g, r.slot] = self._kept(l2_clip(keys, c.qk_l2_clip_threshold, c.l2_clip_eps))
        values = self._static_key_value(r, g, layer, "v", x_v)
        self.TV[g, r.slot] = self._kept(rms(values, c.rms_norm_eps))
        out = self._static_attention(r, g, layer, capacity, query, rel_states, gate, sink)
        attn = p.store(p.store(out).view(count, -1) @ weight(layer, "attn.wo_ud.weight").T)
        attn = self._static_conv(r, g, "attn", attn, weight(layer, "attn_sconv.weight").squeeze(1))
        if layer % c.n_layers_per_attn_res_block == 0:
            blocks, last = [*blocks, partial], attn
        else:
            last = partial + attn
        mix = self._res(blocks, last, self._query(f"attn_res.mlp_queries.{layer}"))[0]
        if messages is not None:
            blend = model._blend(layer, LOOP_STREAMS[3], mix, messages[3])
            mix = torch.where(r.later[:, None], blend, mix)
        mlp = self._moe(layer, model._norm(mix, weight(layer, "mlp_norm.weight")))
        mlp = self._static_conv(r, g, "mlp", mlp, weight(layer, "mlp_sconv.weight").squeeze(1))
        return blocks, p.store(last + mlp)

    def _kept(self, x):
        """A row's key or value as the tree keeps it: rounded to BF16."""
        return x.to(torch.bfloat16) if self.kernel_attention else self.model.precision.store(x)

    def _static_key_value(self, r, g, layer, stream, x):
        c, p, weight = self.config, self.model.precision, self.model._lw
        projected = p.store(x @ weight(layer, f"attn.w{stream}_dv.weight").T)
        name = f"attn.{stream}_sconv."
        if c.is_local(layer):
            y = self._static_conv(r, g, stream, projected, weight(layer, name + "weight").squeeze(1))
        else:
            taps = p.store(x @ weight(layer, name + "weight_proj").T)
            y = self._static_conv(r, g, stream, projected, weight(layer, name + "static_w"),
                                  taps.view(-1, self.kv_heads, c.sconv_kernel_size))
        return y.view(-1, self.kv_heads, self.head_dim)

    def _static_conv(self, r, g, stream, x, weight, dynamic=None):
        self.TC[stream][g, r.slot] = x
        column = g[:, None]
        history = torch.where(r.hist_tree[..., None], self.TC[stream][column, r.hist_slot],
                              self.PC[stream][column, r.hist_prefix])
        context = torch.cat((history, x[:, None]), 1)
        kernel = context.shape[1]
        acc = torch.zeros_like(x)
        if dynamic is None:
            for k in range(kernel):
                acc = acc + context[:, k] * weight[:, k]
            scale = (kernel + 1) ** -0.5
        else:
            head_dim = x.shape[1] // dynamic.shape[1]
            for lag in range(kernel):
                taps = weight[lag] + dynamic[:, :, lag].repeat_interleave(head_dim, dim=1)
                acc = acc + context[:, kernel - 1 - lag] * taps
            scale = (2 * kernel + 1) ** -0.5
        return self.model.precision.store((acc + x) * scale)

    def _reads(self, r, layer):
        """What the kernel reads for the rows in ``layer``: the layer's prefix and
        tree planes, and per row the lowest position it sees, the plane of the
        prefix positions below its sharing cut and that cut, with the position of
        prefix row 0. The row vectors are the same for all layers of a kind, so
        they are computed once per phase."""
        local, count = self.config.is_local(layer), self.counts[layer]
        shares = not local and self.window is not None and layer in self.share_layers and count > 1
        kind = "local" if local else ("shared", count) if shares else "full"
        found = r.reads.get(kind)
        if found is None:
            zero = torch.zeros_like(r.pos)
            if local:
                found = (r.pos - (self.window_size - 1), r.plane, zero, self._origin1)
            elif shares:  # the final plane below the row's cut, the row's own plane from it on
                final = count - 1
                cut = torch.where(r.plane != final, (r.pos - self.window + 1).clamp(min=0), 0)
                found = (zero, torch.full_like(r.plane, final), cut, self._zero1)
            else:
                found = (zero, r.plane, zero, self._zero1)
            r.reads[kind] = found
        first = self.base[layer]
        if local:
            at = self.sw_first[layer]
            prefix = self.PKs[at:at + count], self.PVs[at:at + count]
        else:
            at = self.fa_index[layer]
            prefix = self.PKf[at:at + count], self.PVf[at:at + count]
        return prefix, (self.TK[first:first + count], self.TV[first:first + count]), found

    def _segments(self, layer, capacity):
        """How the row kernels split a row's keys into parallel programs: the
        window of a sliding layer in four, a full layer's context in runs of
        SEGMENT_TILES tiles from position 0. None keeps one program per row,
        whose bits are the sequence kernel's.

        The split of a full layer depends on the capacity only in the number
        of programs: tile t is always read by program t // SEGMENT_TILES, and
        programs past a row's last tile are empty, which the merge skips. A
        row may be up to two tiles past the capacity (its tree positions)."""
        if not self.row_kernels or capacity is None:
            return None
        if self.config.is_local(layer):
            tiles = (self.window_size - 1) // TILE_KEYS + 2
            return (4, -(-tiles // 4)) if tiles > 4 else None
        tiles = capacity // TILE_KEYS + 2
        return (-(-tiles // SEGMENT_TILES), SEGMENT_TILES) if tiles > SEGMENT_TILES else None

    def _kernel_attention(self, r, layer, query, rel_states, gate, sink, rounded=False, capacity=None):
        """The rows' attention at its rounding point (BF16 with ``rounded``). A
        row's path holds the positions from the prefix length on, which is how
        the kernel finds a position's slot."""
        half = torch.bfloat16
        if rel_states.dtype == half:  # [n, heads, d_rel] with the stored projection [d_rel, extent]
            proj = self._cached((layer, "rel_proj"), lambda: self._half(layer, "attn.rel_logits_proj.proj")
                                .T.contiguous())
            rel = ops.gemv_any(rel_states.reshape(-1, rel_states.shape[-1]), proj, half).view(
                *rel_states.shape[:-1], -1)
        else:
            rel = (rel_states @ self.model._lw(layer, "attn.rel_logits_proj.proj")).to(half)
        (prefix_k, prefix_v), (tree_k, tree_v), (lowest, low_plane, cut, origin) = self._reads(r, layer)
        out = tree_attention(query.to(half), rel, gate.to(half), sink.to(half), prefix_k, prefix_v,
                             tree_k, tree_v, r.pos, lowest, r.plane, low_plane, cut, r.path, r.path_mask,
                             self._length1, origin, segments=self._segments(layer, capacity),
                             pack=not self.config.is_local(layer))  # a full layer's long prefix: once per program
        return out if rounded else out.to(self.model.precision.dtype)

    def _static_attention(self, r, g, layer, capacity, query, rel_states, gate, sink):
        if self.kernel_attention:
            return self._kernel_attention(r, layer, query, rel_states, gate, sink)
        c, p = self.config, self.model.precision
        rows, heads, dim = query.shape
        kv_heads = self.kv_heads
        group = heads // kv_heads
        rel_proj = self.model._lw(layer, "attn.rel_logits_proj.proj")
        extent = rel_proj.shape[1]
        q = query.view(rows, kv_heads, group, dim)
        rel = p.store(rel_states @ rel_proj)  # [rows, heads, extent]

        def bias_of(dist, dead=None):
            bias = rel.gather(2, dist.clamp(0, extent - 1)[:, None, :].expand(-1, heads, -1))
            off = (dist < 0) | (dist >= extent)
            if dead is not None:
                off = off | dead
            return bias.masked_fill(off[:, None, :], 0.0).view(rows, kv_heads, group, -1)

        length = self._length
        tree_keys, tree_values = self.TK[g[:, None], r.path], self.TV[g[:, None], r.path]
        tree_dist = r.pos[:, None] - self.slot_pos[r.path]
        tree_visible = r.path_mask & (tree_dist >= 0)
        tree_scores = torch.einsum("nhgd,nkhd->nhgk", q, tree_keys) * dim ** -0.5 + bias_of(tree_dist)
        if c.is_local(layer):
            window = c.sliding_window_size
            at = r.pos[:, None] - (window - 1) + self._arange[:window - 1][None, :]
            visible = (at >= self._origin) & (at < length)
            offset = (at - self._origin).clamp(0, self.sw_capacity - 1)
            plane = self.sw_index[g][:, None]
            keys, values = self.PKs[plane, offset], self.PVs[plane, offset]
            dist = r.pos[:, None] - at
            scores = torch.einsum("nhgd,nkhd->nhgk", q, keys) * dim ** -0.5 + bias_of(dist)
            tree_visible = tree_visible & (tree_dist < window)
            scores = torch.cat((scores, tree_scores), -1)
            visible = torch.cat((visible & (dist < window), tree_visible), -1)
            scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
            top = scores.amax(-1, keepdim=True)
            probs = torch.exp(scores - top)
            total = probs.sum(-1, keepdim=True)
            stored = p.store(probs)
            split = keys.shape[1]
            mixed = (torch.einsum("nhgk,nkhd->nhgd", stored[..., :split], values)
                     + torch.einsum("nhgk,nkhd->nhgd", stored[..., split:], tree_values))
        else:
            first, count = self.fa_index[layer], self.counts[layer]
            keys, values = self.PKf[first:first + count, :capacity], self.PVf[first:first + count, :capacity]
            every = torch.einsum("nhgd,pchd->pnhgc", q, keys) * dim ** -0.5
            at = self._arange[:capacity]
            scores = every[r.plane, torch.arange(rows, device=every.device)]  # each row in its own plane
            key_plane = r.plane[:, None].expand(rows, capacity)
            if self.window is not None and layer in self.share_layers:
                final = count - 1
                cut = (r.pos - self.window + 1).clamp(min=0)
                shared = (r.plane != final)[:, None] & (at[None, :] < cut[:, None])
                scores = torch.where(shared[:, None, None, :], every[final], scores)
                key_plane = torch.where(shared, final, key_plane)
            # Only the last `extent` prefix keys are near enough for a relative bias.
            band = length - extent + self._arange[:extent]
            bias = bias_of(r.pos[:, None] - band[None, :], (band < 0)[None, :])
            scores = scores.index_add(-1, band.clamp(min=0), bias)
            scores = torch.cat((scores, tree_scores), -1)
            visible = torch.cat(((at < length)[None, :].expand(rows, capacity), tree_visible), -1)
            scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
            top = scores.amax(-1, keepdim=True)
            probs = torch.exp(scores - top)
            total = probs.sum(-1, keepdim=True)
            stored = p.store(probs)
            mixed = torch.einsum("nhgk,nkhd->nhgd", stored[..., capacity:], tree_values)
            for plane in range(count):
                part = stored[..., :capacity] * (key_plane == plane)[:, None, None, :]
                mixed = mixed + torch.einsum("nhgc,chd->nhgd", part, values[plane])
        lse = (top + torch.log(total)).reshape(rows, heads)
        return ((mixed / total).reshape(rows, heads, dim) * sigmoid(gate)
                * sigmoid(lse - sink)[..., None])
