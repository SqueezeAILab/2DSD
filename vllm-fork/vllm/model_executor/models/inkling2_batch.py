# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Batched tree runner of Inkling 2 for the 2DSD executor.

Computes what ``Inkling2TreeRunner`` (the reference path) computes, with every
row of a call in one batch and the state in preallocated tensors. A call runs
the prelude for its new nodes, one core pass for every node, and the coda for
the nodes that reach a readout depth. Rows of one layer visit may be in
different planes (passes or readout depths); they share the layer's weights and
differ only in the keys, values and short-conv inputs they read and write.

State.
  - Prefix, per layer: keys and values ``[planes, capacity, kv_heads, D]`` and
    the last ``kernel - 1`` short-conv inputs of each stream. A full-attention
    layer holds every position; a sliding layer holds a moving range that always
    covers the last ``window - 1`` positions.
  - Tree, per layer: the same tensors indexed by node slot. A node's row of a
    layer visit is written before any row of that visit reads its path, so a
    parent and its child may run the same pass in one call.
  - Node: the blocks and the partial block after the prelude and after the
    node's latest pass (which is the bank of its next pass and the state its
    readout continues from), and its features per readout depth.

A row attends over the prefix keys of its plane, the keys of its uncommitted
ancestors and its own key. Under last-loop KV sharing a row of a sharing layer
in a plane other than the final one reads the final plane for prefix positions
``<= t - W``. Short-conv histories follow the row's path across the boundary
between the prefix and the tree.

Arithmetic is the reference forward's (same formulas, rounding points and
order of operations inside a row); batching changes only the shape of the
products, which moves an FP32 result in its last bit. The prompt is prefilled
by the reference forward.

Routed experts. ``ExpertBank`` hands out the dense factors of a set of
experts of one layer in the arithmetic dtype: a gather and a cast from a dense
checkpoint, two table lookups from a packed one (48 GB cards), exact in both
cases. The runner computes all routed experts of a layer visit in two batched
products over them, grouped by expert. The bank also keeps recently used single
experts for the reference forward, which prefills the prompt.
"""

import contextlib
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from vllm.model_executor.models.inkling2_2d import Inkling2TreeRunner
from vllm.model_executor.models.inkling2_fast import tensor_cores
from vllm.model_executor.models.inkling2_reference import (
    ROUTER_GROUP,
    Inkling2Cache,
    Inkling2Plane,
    attn_res,
    l2_clip,
    rms,
    route_experts,
    sigmoid,
    situ,
)
from vllm.model_executor.models.inkling2_weights import (
    _E2M1_CODES,
    _E4M3_CODES,
    _MXFP8_NORMAL_EXPONENTS,
    _POW2,
    GATE_ROWS,
    LOOP_STREAMS,
    MXFP8_BLOCK,
    NVFP4_BLOCK,
    UP_ROWS,
    Inkling2PackedExperts,
    _raw_bytes,
    expert_prefix,
    nvfp4_outer_scale,
)

CONV_STREAMS = ("k", "v", "attn", "mlp")
# Positions a sliding layer keeps beyond its window before it moves the range.
SLIDE = 1024
# Attention scores computed at once: rows x heads x keys.
SCORE_ELEMENTS = 1 << 27


# Dense expert factors dequantized at once: experts x rows x columns.
FACTOR_ELEMENTS = 1 << 28


class ExpertBank:
    """Dense factors of routed experts in the model's arithmetic dtype.

    ``factors`` returns the factors ``[n, out, in]`` and scales ``[n]`` of a set
    of experts of one layer and projection, as ``NativeInkling2._expert`` returns
    them for one. ``single`` is that method's replacement: it keeps the factors
    of recently used experts up to ``budget`` bytes, least recently used out
    first. One bank per model.
    """

    def __init__(self, model, budget=None):
        self.budget = budget
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0
        self.dtype, self.device = model.precision.dtype, model.device
        self._experts = model.checkpoint.experts
        self._scales, self._tables, self._direct = {}, None, {}

    @classmethod
    def of(cls, model, budget=None):
        bank = getattr(model, "_expert_bank", None)
        if bank is None:
            bank = model._expert_bank = cls(model, budget)
            model._expert = bank.single
        elif budget is not None:
            bank.budget = budget
        return bank

    def _lookup(self):
        if self._tables is None:
            dtype, device = self.dtype, self.device
            nibble = torch.tensor(_E2M1_CODES, dtype=dtype, device=device)
            byte = torch.arange(256, device=device)
            self._tables = SimpleNamespace(
                pairs=torch.stack((nibble[byte & 0xF], nibble[byte >> 4]), -1),
                e4m3=torch.tensor(_E4M3_CODES, dtype=dtype, device=device),
                pow2=torch.tensor(_POW2, dtype=dtype, device=device))
        return self._tables

    def _lookup_is_exact(self, experts, name):
        """Whether the table lookup gives this tensor exactly: no NaN code, and
        for MXFP8 only exponents whose products are normal numbers."""
        found = self._direct.get(name)
        if found is None:
            codes = _raw_bytes(experts.tensors[name])
            block = _raw_bytes(experts.tensors[name + "_scale"])
            if experts.quant_format == "nvfp4":
                found = not bool(((block & 0x7F) == 0x7F).any())
            else:
                low, high = _MXFP8_NORMAL_EXPONENTS
                found = (self.dtype == torch.float64 or
                         (int(block.min()) >= low and int(block.max()) <= high))
                found = found and not bool(((codes & 0x7F) == 0x7F).any())
                found = found and not bool((block == 0xFF).any())
            self._direct[name] = found
        return found

    def factors(self, layer, projection, index):
        experts = self._experts(layer)
        if not isinstance(experts, Inkling2PackedExperts):
            scale = getattr(experts, projection + "_scale")[index]
            return getattr(experts, projection)[index].to(self.dtype), scale.to(self.dtype)
        name = expert_prefix(layer, projection)
        if not self._lookup_is_exact(experts, name):
            dense, scale = experts.dequantize(projection, index, self.dtype)
            return dense, scale.to(self.dtype)
        tables = self._lookup()
        codes = _raw_bytes(experts.tensors[name])[index].int()
        block = _raw_bytes(experts.tensors[name + "_scale"])[index].int()
        count, rows = codes.shape[:2]
        if experts.quant_format == "nvfp4":
            scale = self._scales.get(name)
            if scale is None:
                scale = self._scales[name] = nvfp4_outer_scale(
                    experts.tensors[name + "_amax"],
                    experts.tensors[name + "_outer_scale_denominator"]).to(self.dtype)
            dense = tables.pairs[codes].view(count, rows, -1, NVFP4_BLOCK)
            dense = dense * tables.e4m3[block].unsqueeze(-1)
            return dense.flatten(2), scale[index]
        dense = tables.e4m3[codes].view(count, rows, -1, MXFP8_BLOCK)
        dense = dense * tables.pow2[block].unsqueeze(-1)
        return dense.flatten(2), torch.ones(count, dtype=self.dtype, device=self.device)

    def single(self, experts, index):
        """Drop-in for ``NativeInkling2._expert``: (w13, scale, w2, scale)."""
        key = (experts.layer_id, int(index))
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return entry
        self.misses += 1
        at = torch.tensor([key[1]], device=self.device)
        w13, s13 = self.factors(key[0], "w13", at)
        w2, s2 = self.factors(key[0], "w2", at)
        entry = (w13[0], s13[0], w2[0], s2[0])
        size = sum(t.numel() * t.element_size() for t in entry)
        if self.budget is not None:
            while self.entries and self.bytes + size > self.budget:
                _, old = self.entries.popitem(last=False)
                self.bytes -= sum(t.numel() * t.element_size() for t in old)
            if size > self.budget:
                return entry
        self.entries[key] = entry
        self.bytes += size
        return entry


def cache_weights(model):
    """Hold the model's layer weights in its arithmetic dtype instead of
    converting them on every use."""
    if getattr(model, "_weight_cache", None) is None:
        cache, load = {}, model._lw

        def weight(layer, name):
            key = (layer, name)
            found = cache.get(key)
            if found is None:
                found = cache[key] = load(layer, name)
            return found

        model._weight_cache, model._lw = cache, weight


def default_expert_budget(model):
    """Bytes of single dense experts to keep on the model's device for the
    reference forward (the runner itself keeps none)."""
    return None if model.device.type != "cuda" else 4 << 30


@dataclass
class _Rows:
    """Rows of one call: slot and position of each node, the slots it attends
    to in the tree (its uncommitted ancestors and itself) and where the last
    ``kernel - 1`` positions of its path are (tree slots or prefix history)."""

    slot: torch.Tensor
    pos: torch.Tensor
    path: torch.Tensor  # [n, slots] bool
    hist_tree: torch.Tensor  # [n, kernel - 1] bool
    hist_slot: torch.Tensor
    hist_prefix: torch.Tensor

    def select(self, index):
        return _Rows(*(getattr(self, name)[index] for name in self.__annotations__))


class _LayerStore:
    """Prefix and tree state of one layer; plane ``i`` is the layer's i-th plane
    (``indices[i]`` in the reference cache's naming)."""

    def __init__(self, indices, capacity, window, kv_heads, dim, hidden, kernel,
                 slots, dtype, device):
        self.indices = list(indices)
        self.index_of = {index: i for i, index in enumerate(self.indices)}
        self.window, self.capacity, self.base = window, capacity, 0
        planes = len(self.indices)
        self.channels = {"k": kv_heads * dim, "v": kv_heads * dim,
                         "attn": hidden, "mlp": hidden}
        zeros = lambda *shape: torch.zeros(shape, dtype=dtype, device=device)  # noqa: E731
        self.pk = zeros(planes, capacity, kv_heads, dim)
        self.pv = zeros(planes, capacity, kv_heads, dim)
        self.pconv = {s: zeros(planes, kernel - 1, c) for s, c in self.channels.items()}
        self.tk = zeros(planes, slots, kv_heads, dim)
        self.tv = zeros(planes, slots, kv_heads, dim)
        self.tconv = {s: zeros(planes, slots, c) for s, c in self.channels.items()}

    def grow(self, slots):
        def wider(x):
            extra = x.new_zeros((x.shape[0], slots - x.shape[1], *x.shape[2:]))
            return torch.cat((x, extra), 1)

        self.tk, self.tv = wider(self.tk), wider(self.tv)
        self.tconv = {s: wider(x) for s, x in self.tconv.items()}

    def load(self, index, plane):
        """Take a plane of the reference cache (after the prompt)."""
        i, count = self.index_of[index], plane.keys.shape[0]
        self.base = plane.first
        self.pk[i, :count] = plane.keys
        self.pv[i, :count] = plane.values
        for stream, history in plane.conv.items():
            self.pconv[stream][i] = history

    def commit(self, slots, length):
        """Append the rows of tree ``slots`` (path order) at ``length``."""
        count = slots.numel()
        start = length - self.base
        if start + count > self.capacity:  # sliding layers only
            keep = self.window - 1
            self.pk[:, :keep] = self.pk[:, start - keep:start].clone()
            self.pv[:, :keep] = self.pv[:, start - keep:start].clone()
            self.base, start = length - keep, keep
        self.pk[:, start:start + count] = self.tk[:, slots]
        self.pv[:, start:start + count] = self.tv[:, slots]
        for stream, history in self.pconv.items():
            joined = torch.cat((history, self.tconv[stream][:, slots]), 1)
            self.pconv[stream] = joined[:, joined.shape[1] - history.shape[1]:].clone()

    def export(self, index, length):
        """The plane as the reference cache holds it after ``length`` tokens."""
        i = self.index_of[index]
        first = 0 if self.window is None else max(0, length - (self.window - 1))
        span = slice(first - self.base, length - self.base)
        return Inkling2Plane(self.pk[i, span].clone(), self.pv[i, span].clone(), first,
                             {s: x[i].clone() for s, x in self.pconv.items()})


class Inkling2BatchRunner(Inkling2TreeRunner):
    """Node-state runner with fused readouts, batched over the rows of a call.

    Takes the arguments of ``Inkling2TreeRunner`` and ``expert_cache_bytes``,
    the single dense experts kept for the reference forward (default 4 GiB on a
    GPU; None keeps all).
    """

    def __init__(self, model, readout_depths, *, expert_cache_bytes=None, **options):
        cache_weights(model)
        self._fast = bool(getattr(model, "tensor_cores", False))
        if expert_cache_bytes is None:
            expert_cache_bytes = default_expert_budget(model)
        self.experts = ExpertBank.of(model, expert_cache_bytes)
        self._queries = {}
        self._allocated = False
        super().__init__(model, readout_depths, **options)

    # State.

    def _allocate(self):
        c, core = self.config, self.core
        device, dtype = self.device, self.dtype
        self.depth_list = sorted(self.readout_depths)
        self.slots = self.max_nodes
        kernel, block = c.sconv_kernel_size, c.n_layers_per_attn_res_block
        self.store = []
        for layer in range(self.layers):
            if layer < core.start:
                indices = [0]
            elif layer < core.stop:
                indices = range(1, self.R + 1)
            else:
                indices = self.depth_list
            _, kv_heads, dim = c.attention_dims(layer)
            window = c.sliding_window_size if c.is_local(layer) else None
            capacity = self.max_len if window is None else window - 1 + SLIDE
            self.store.append(_LayerStore(indices, capacity, window, kv_heads, dim,
                                          self.H, kernel, self.slots, dtype, device))
        zeros = lambda *shape: torch.zeros(shape, dtype=dtype, device=device)  # noqa: E731
        self.pre_count = sum(layer % block == 0 for layer in range(core.start))
        self.core_count = sum(layer % block == 0 for layer in core)
        self.pre_blocks = zeros(self.slots, self.pre_count, self.H)
        self.pre_partial = zeros(self.slots, self.H)
        self.core_blocks = zeros(self.slots, self.core_count, self.H)
        self.core_partial = zeros(self.slots, self.H)
        self.column = {layer: i * self.H for i, layer in enumerate(self.feature_layers)}
        self.pre_taps = [layer for layer in self.feature_layers if layer < core.start]
        self.pre_tap = zeros(self.slots, len(self.pre_taps), self.H)
        self.feat = zeros(len(self.depth_list), self.slots, self.nf * self.H)
        self.slot_pos = torch.zeros(self.slots, dtype=torch.long, device=device)
        self._allocated = True

    def _grow(self, slots):
        def wider(x):
            return torch.cat((x, x.new_zeros((slots - x.shape[0], *x.shape[1:]))))

        for store in self.store:
            store.grow(slots)
        for name in ("pre_blocks", "pre_partial", "core_blocks", "core_partial",
                     "pre_tap", "slot_pos"):
            setattr(self, name, wider(getattr(self, name)))
        self.feat = torch.cat(
            (self.feat, self.feat.new_zeros((self.feat.shape[0], slots - self.slots,
                                             self.feat.shape[2]))), 1)
        self.free.extend(range(slots - 1, self.slots - 1, -1))
        self.slots = slots

    def reset(self):
        if not self._allocated:
            self._allocate()
        self.length = 0
        self.nodes = {}
        self.free = list(range(self.slots - 1, -1, -1))
        for store in self.store:
            store.base = 0

    @property
    def prefix_length(self):
        return self.length

    @property
    def prefix(self):
        """The prefix in the reference cache's form (copies; for inspection)."""
        planes = {(layer, index): store.export(index, self.length)
                  for layer, store in enumerate(self.store) for index in store.indices}
        return SimpleNamespace(length=self.length, planes=planes)

    def _query(self, name):
        found = self._queries.get(name)
        if found is None:
            if isinstance(name, int):
                found = self.model._message_queries(name)
            else:
                found = self.model._query(name)
            self._queries[name] = found
        return found

    # Prompt.

    @torch.no_grad()
    def prefill(self, token_ids):
        self.reset()
        core, cache = self.core, Inkling2Cache()
        visits = [layer if layer < core.start else layer + (self.R - 1) * len(core)
                  for layer in self.feature_layers]
        out = self.model.forward(token_ids, num_passes=self.R,
                                 depths=tuple(self.depth_list), taps=visits,
                                 cache=cache, pass_end=False)
        for (layer, index), plane in cache.planes.items():
            self.store[layer].load(index, plane)
        self.length = cache.length
        hidden = out.readouts[self.R].hidden
        features = torch.cat([*(out.features[visit] for visit in visits),
                              *([hidden] if self.final_feature else [])], -1)
        return SimpleNamespace(hidden=hidden, features=features)

    # Tree.

    def ensure_capacity(self, nodes):
        super().ensure_capacity(nodes)
        if self.max_nodes > self.slots:
            self._grow(self.max_nodes)

    def register_many(self, nodes):
        super().register_many(nodes)
        if len(self.free) < len(nodes):
            self._grow(self.slots + max(len(nodes), self.slots))
        for node_id, _, _ in nodes:
            node = self.nodes[node_id]
            node.slot = self.free.pop()
            self.slot_pos[node.slot] = node.position

    def release(self, node_ids):
        slots = [self.nodes[node_id].slot for node_id in set(node_ids)]
        super().release(node_ids)
        self.free.extend(slots)

    @torch.no_grad()
    def commit(self, node_ids, features=None):
        self._check_commit(node_ids)
        if not node_ids:
            return
        slots = torch.tensor([self.nodes[node_id].slot for node_id in node_ids],
                             device=self.device)
        for start in range(0, len(node_ids), SLIDE):
            part = slots[start:start + SLIDE]
            for store in self.store:
                store.commit(part, self.length)
            self.length += part.numel()
        if features is not None:
            features.copy_(self.feat[-1][slots])  # depth R is the last readout depth
        for node_id in node_ids:
            node = self.nodes.pop(node_id)
            self.free.append(node.slot)
            for child in node.children:
                self.nodes[child].parent = None
        self.stats["committed_nodes"] += len(node_ids)

    def span_features(self, spans):
        return tuple(
            self.feat[self.depth_list.index(depth)][
                torch.tensor([self.nodes[node_id].slot for node_id in node_ids],
                             device=self.device)]
            for depth, node_ids in spans)

    # Execution.

    def _rows(self, node_ids, chains):
        count, history = len(node_ids), self.config.sconv_kernel_size - 1
        slot, pos = [], []
        path = torch.zeros(count, self.slots, dtype=torch.bool)
        hist_tree = torch.zeros(count, history, dtype=torch.bool)
        hist_slot = torch.zeros(count, history, dtype=torch.long)
        hist_prefix = torch.zeros(count, history, dtype=torch.long)
        for row, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            ancestors = [member.slot for member in chains[node_id]]
            slot.append(node.slot)
            pos.append(node.position)
            path[row, ancestors + [node.slot]] = True
            # The last `history` positions of the path before the node: the tail of the
            # prefix history followed by the node's ancestors.
            for j in range(history):
                behind = len(ancestors) - history + j
                if behind >= 0:
                    hist_tree[row, j], hist_slot[row, j] = True, ancestors[behind]
                else:
                    hist_prefix[row, j] = history + behind
        device = self.device
        return _Rows(torch.tensor(slot, device=device), torch.tensor(pos, device=device),
                     path.to(device), hist_tree.to(device), hist_slot.to(device),
                     hist_prefix.to(device))

    def _products(self):
        """The arithmetic of a call's matrix products: the model's."""
        return tensor_cores() if self._fast else contextlib.nullcontext()

    def _res(self, blocks, last, queries):
        """``attn_res``, with FP32 products under a tensor-core model too: its
        sources and queries are not BF16 values."""
        with tensor_cores(False) if self._fast else contextlib.nullcontext():
            return attn_res(blocks, last, queries, self.config.attn_res_eps)

    @torch.no_grad()
    def advance_nodes(self, node_ids, loops):
        """Advance every node by one loop. Returns the rows (indices into
        ``node_ids``) that reached a readout depth, in increasing order, with
        their argmax [rows] and logits [rows, V]. Nothing runs unless the call
        is consistent with the tree."""
        with self._products():
            return self._advance(node_ids, loops)

    def _advance(self, node_ids, loops):
        chains = self._check_advance(node_ids, loops)
        self._count(loops)
        model, core, device = self.model, self.core, self.device
        rows = self._rows(node_ids, chains)
        nodes = [self.nodes[node_id] for node_id in node_ids]

        fresh = [row for row, loop in enumerate(loops) if loop == 0]
        if fresh:
            index = torch.tensor(fresh, device=device)
            sub = rows.select(index)
            plane = torch.zeros(len(fresh), dtype=torch.long, device=device)
            tokens = torch.tensor([nodes[row].token for row in fresh], device=device)
            blocks, partial = [], model.embed_tokens(tokens)
            for layer in range(core.start):
                if layer in self.pre_taps:
                    self.pre_tap[sub.slot, self.pre_taps.index(layer)] = partial
                blocks, partial = self._layer(sub, plane, layer, blocks, partial)
            if blocks:
                self.pre_blocks[sub.slot] = torch.stack(blocks, 1)
            self.pre_partial[sub.slot] = partial

        passes = [loop + 1 for loop in loops]
        plane = torch.tensor([p - 1 for p in passes], device=device)
        blocks = list(self.pre_blocks[rows.slot].unbind(1))
        partial = self.pre_partial[rows.slot]
        later = [row for row, loop in enumerate(loops) if loop > 0]
        bank = later_index = None
        if later:
            later_index = torch.tensor(later, device=device)
            behind = rows.slot[later_index]
            bank = [*self.core_blocks[behind][:, 1:].unbind(1), self.core_partial[behind]]
        read = [row for row, p in enumerate(passes) if p in self.readout_depths]
        read_index = torch.tensor(read, dtype=torch.long, device=device)
        read_depth = torch.tensor([self.depth_list.index(passes[row]) for row in read],
                                  dtype=torch.long, device=device)
        read_slot = rows.slot[read_index]
        for layer in core:
            if read and layer in self.column:
                column = self.column[layer]
                self.feat[read_depth, read_slot, column:column + self.H] = partial[read_index]
            blocks, partial = self._layer(rows, plane, layer, blocks, partial, bank, later_index)
        if self.core_count:
            self.core_blocks[rows.slot] = torch.stack(blocks[self.pre_count:], 1)
        self.core_partial[rows.slot] = partial
        for node in nodes:
            node.depth += 1
        if not read:
            return read, None, None

        sub = rows.select(read_index)
        blocks, partial = [block[read_index] for block in blocks], partial[read_index]
        for layer in range(core.stop, self.layers):
            if layer in self.column:
                column = self.column[layer]
                self.feat[read_depth, read_slot, column:column + self.H] = partial
            blocks, partial = self._layer(sub, read_depth, layer, blocks, partial)
        hidden = model.readout_hidden(blocks, partial)
        for i, layer in enumerate(self.pre_taps):
            column = self.column[layer]
            self.feat[read_depth, read_slot, column:column + self.H] = self.pre_tap[read_slot, i]
        if self.final_feature:
            self.feat[read_depth, read_slot, -self.H:] = hidden
        logits = model.readout_logits(hidden)
        return read, logits.argmax(-1), logits

    def _layer(self, rows, plane, layer, blocks, partial, bank=None, later=None):
        """One visit of ``layer`` for rows in planes ``plane`` (indices into the
        layer's planes); ``bank`` holds the previous pass of the rows ``later``.
        Mirrors ``NativeInkling2._layer``."""
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
                    blend = model._blend(layer, stream, mix[later], messages[s])
                    inputs.append(model._norm(mix.index_copy(0, later, blend),
                                              norms[0 if local else s]))
        x_q, x_k, x_v = inputs if len(inputs) == 3 else inputs * 3
        heads, kv_heads, dim = c.attention_dims(layer)
        count = x_q.shape[0]
        tau = None
        if not local and c.log_scaling_n_floor is not None:
            ratio = (rows.pos + 1).to(p.dtype) * (1.0 / c.log_scaling_n_floor)
            tau = (1.0 + c.log_scaling_alpha * torch.log(ratio.clamp(min=1.0)))[:, None, None]

        def clipped(name, width, scale, threshold):
            x = p.store(x_q @ weight(layer, f"attn.{name}.weight").T).view(count, heads, width)
            x = p.store(l2_clip(x * weight(layer, f"attn.{scale}.scale"), threshold,
                                c.l2_clip_eps))
            return x if tau is None else p.store(x * tau)

        query = clipped("wq_du", dim, "q_l2_clip", c.qk_l2_clip_threshold)
        rel_states = clipped("wr_du", c.d_rel, "rel_l2_clip", c.rel_l2_clip_threshold)
        gate = p.store(x_q @ weight(layer, "attn.gate_attn.weight").T).view(count, heads, dim)
        sink = p.store(x_q @ weight(layer, "attn.attn_sink_proj.weight").T)
        store = self.store[layer]
        keys = self._key_value(rows, plane, layer, "k", x_k)
        store.tk[plane, rows.slot] = p.store(l2_clip(keys, c.qk_l2_clip_threshold, c.l2_clip_eps))
        values = self._key_value(rows, plane, layer, "v", x_v)
        store.tv[plane, rows.slot] = p.store(rms(values, c.rms_norm_eps))
        out = self._attention(rows, plane, layer, query, rel_states, gate, sink)
        attn = p.store(p.store(out).view(count, -1) @ weight(layer, "attn.wo_ud.weight").T)
        attn = self._conv(rows, plane, layer, "attn", attn,
                          weight(layer, "attn_sconv.weight").squeeze(1))
        if layer % c.n_layers_per_attn_res_block == 0:
            blocks, last = [*blocks, partial], attn
        else:
            last = partial + attn
        mix = self._res(blocks, last, self._query(f"attn_res.mlp_queries.{layer}"))[0]
        if messages is not None:
            blend = model._blend(layer, LOOP_STREAMS[3], mix[later], messages[3])
            mix = mix.index_copy(0, later, blend)
        mlp = self._moe(layer, model._norm(mix, weight(layer, "mlp_norm.weight")))
        mlp = self._conv(rows, plane, layer, "mlp", mlp,
                         weight(layer, "mlp_sconv.weight").squeeze(1))
        return blocks, p.store(last + mlp)

    def _moe(self, layer, x):
        """Routed experts on the normed MLP input x [n, H]: [n, H]. The (row,
        slot) pairs are grouped by expert, so each routed expert is read once
        and all of them run in two batched products."""
        c, p, weight = self.config, self.model.precision, self.model._lw
        gate = weight(layer, "mlp.gate.weight")
        logits = sum(x[:, k:k + ROUTER_GROUP] @ gate[:, k:k + ROUTER_GROUP].T
                     for k in range(0, c.hidden_size, ROUTER_GROUP))
        ids, weights = route_experts(logits, weight(layer, "mlp.gate.bias"),
                                     c.num_experts_per_tok, c.route_scale)
        count, top = ids.shape
        unique, group = torch.unique(ids, return_inverse=True)
        group = group.reshape(-1)  # expert group of every (row, slot) pair
        sizes = torch.bincount(group, minlength=unique.numel())
        order = torch.argsort(group, stable=True)
        place = torch.empty_like(group)  # a pair's place within its group
        place[order] = (torch.arange(group.numel(), device=group.device)
                        - (torch.cumsum(sizes, 0) - sizes)[group[order]])
        row = torch.arange(count, device=group.device).repeat_interleave(top)
        width = int(sizes.max())
        inputs = x.new_zeros(unique.numel(), width, x.shape[1])
        inputs[group, place] = x[row]
        routing = x.new_zeros(unique.numel(), width, 1)
        routing[group, place, 0] = weights.reshape(-1)
        down = torch.empty_like(inputs)
        step = max(1, FACTOR_ELEMENTS // (2 * c.intermediate_size * c.hidden_size))
        for start in range(0, unique.numel(), step):
            part = slice(start, start + step)
            w13, s13 = self.experts.factors(layer, "w13", unique[part])
            gate_up = p.store(torch.bmm(inputs[part], w13.transpose(1, 2)) * s13[:, None, None])
            del w13
            hidden = p.store(situ(gate_up[..., GATE_ROWS], gate_up[..., UP_ROWS], routing[part],
                                  c.mlp_gate_softcap, c.mlp_up_softcap))
            w2, s2 = self.experts.factors(layer, "w2", unique[part])
            down[part] = p.store(torch.bmm(hidden, w2.transpose(1, 2)) * s2[:, None, None])
            del w2
        slots = down[group, place].view(count, top, -1)
        out = slots[:, 0]
        for slot in range(1, top):
            out = out + slots[:, slot]
        return p.store(out)

    def _key_value(self, rows, plane, layer, stream, x):
        """The k or v projection of x after its short conv: [n, kv_heads, D]."""
        c, p, weight = self.config, self.model.precision, self.model._lw
        _, kv_heads, dim = c.attention_dims(layer)
        projected = p.store(x @ weight(layer, f"attn.w{stream}_dv.weight").T)
        name = f"attn.{stream}_sconv."
        if c.is_local(layer):
            y = self._conv(rows, plane, layer, stream, projected,
                           weight(layer, name + "weight").squeeze(1))
        else:
            taps = p.store(x @ weight(layer, name + "weight_proj").T)
            y = self._conv(rows, plane, layer, stream, projected,
                           weight(layer, name + "static_w"),
                           taps.view(-1, kv_heads, c.sconv_kernel_size))
        return y.view(-1, kv_heads, dim)

    def _conv(self, rows, plane, layer, stream, x, weight, dynamic=None):
        """One stream's short conv of a row's input x [n, C] over its path; the
        input is stored first, so rows of this visit see each other."""
        store = self.store[layer]
        store.tconv[stream][plane, rows.slot] = x
        column = plane[:, None]
        history = torch.where(rows.hist_tree[..., None],
                              store.tconv[stream][column, rows.hist_slot],
                              store.pconv[stream][column, rows.hist_prefix])
        context = torch.cat((history, x[:, None]), 1)  # oldest first, x last
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

    def _attention(self, rows, plane, layer, query, rel_states, gate, sink):
        """Gated relative-bias attention of every row over its plane: [n, heads, D]."""
        out = torch.empty_like(query)
        rel_proj = self.model._lw(layer, "attn.rel_logits_proj.proj")
        for index in torch.unique(plane).tolist():
            pick = (plane == index).nonzero().squeeze(1)
            keys, values, key_pos, visible = self._sources(layer, index, rows.pos[pick],
                                                           rows.path[pick])
            limit = max(1, SCORE_ELEMENTS // (query.shape[1] * max(1, keys.shape[0])))
            for start in range(0, pick.numel(), limit):
                part = pick[start:start + limit]
                out[part] = self._attend(layer, query[part], rel_states[part], gate[part],
                                         sink[part], rel_proj, keys, values,
                                         rows.pos[part][:, None] - key_pos[None, :],
                                         visible[start:start + limit])
        return out

    def _sources(self, layer, index, pos, path):
        """Keys and values a group of rows of one plane can read, their positions
        [N] and which of them each row reads [rows, N]: the prefix (the final
        plane below a row's sharing cut), then the tree slots on a row's path."""
        store, length, device = self.store[layer], self.length, self.device
        parts = []
        if self.shares(layer, store.indices[index]):
            final = store.index_of[self.R]
            cut = (pos - self.window + 1).clamp(min=0)
            low, high = int(cut.min()), int(cut.max())
            if high > 0:
                at = torch.arange(high, device=device)
                parts.append((store.pk[final, :high], store.pv[final, :high], at,
                              at[None, :] < cut[:, None]))
            if length > low:
                at = torch.arange(low, length, device=device)
                parts.append((store.pk[index, low:length], store.pv[index, low:length], at,
                              at[None, :] >= cut[:, None]))
        else:
            low = store.base
            if store.window is not None:
                low = max(low, int(pos.min()) - (store.window - 1))
            if length > low:
                at = torch.arange(low, length, device=device)
                span = slice(low - store.base, length - store.base)
                parts.append((store.pk[index, span], store.pv[index, span], at, None))
        used = path.any(0).nonzero().squeeze(1)
        used = used[torch.argsort(self.slot_pos[used], stable=True)]
        parts.append((store.tk[index, used], store.tv[index, used], self.slot_pos[used],
                      path[:, used]))
        rows = pos.shape[0]
        visible = torch.cat([part[3] if part[3] is not None
                             else torch.ones(rows, part[2].numel(), dtype=torch.bool,
                                             device=device) for part in parts], 1)
        return (torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts]),
                torch.cat([part[2] for part in parts]), visible)

    def _attend(self, layer, query, rel_states, gate, sink, rel_proj, keys, values, dist,
                visible):
        """``relative_attention`` of the reference for rows with their own key
        sets: ``dist`` [rows, N] is each row's distance to every key."""
        c, p = self.config, self.model.precision
        rows, heads, dim = query.shape
        kv_heads, extent = keys.shape[1], rel_proj.shape[1]
        group = heads // kv_heads
        visible = visible & (dist >= 0)
        if c.is_local(layer):
            visible = visible & (dist < c.sliding_window_size)
        scores = torch.einsum("qhgd,nhd->qhgn", query.view(rows, kv_heads, group, dim),
                              keys) * dim ** -0.5
        rel = p.store(rel_states @ rel_proj)  # [rows, heads, extent]
        near = ((dist >= 0) & (dist < extent)).any(0).nonzero().squeeze(1)
        if near.numel():
            close = dist[:, near]
            bias = rel.gather(2, close.clamp(0, extent - 1)[:, None, :].expand(-1, heads, -1))
            bias = bias.masked_fill(((close < 0) | (close >= extent))[:, None, :], 0.0)
            scores[..., near] += bias.view(rows, kv_heads, group, -1)
        scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
        top = scores.amax(-1, keepdim=True)
        probs = torch.exp(scores - top)
        total = probs.sum(-1, keepdim=True)
        mixed = torch.einsum("qhgn,nhd->qhgd", p.store(probs), values) / total
        lse = (top + torch.log(total)).reshape(rows, heads)
        return (mixed.reshape(rows, heads, dim) * sigmoid(gate)
                * sigmoid(lse - sink)[..., None])
