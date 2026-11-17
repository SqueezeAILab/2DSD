# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Dory target runners for 2DSD: the exact dense runner (``DoryTreeRunner``) and
the paged, CUDA-graph runner used for generation (``DoryGraphRunner``).

One target call (``advance_readout``) runs three phases over the candidate tree:
the input block for nodes entering the tree, one loop of the recurrent block for
every unfinished node, and the output block plus scaled LM head for nodes that
reach a readout depth. Depth-r logits are those of the model run with r loops,
so the output block keeps its own KV per readout depth.

KV read rule of a query at position t (prefix length P, keys j <= t):

- native: same-plane KV (the query's loop or readout depth) everywhere.
- shared (``window`` W): in recurrent full-attention layers below loop R, and in
  output full-attention layers below depth R, keys j <= t - W come from the
  final plane (loop R or depth R) and keys j > t - W from the query's plane.
  Sliding layers and the input block are native. The prompt is native.

In-flight keys (j >= P) are the node's own path, always read from the query's
plane; the path guard ``t - P < W`` keeps every shared read on committed tokens.

Prefix storage (``compact_prefix``, the default) keeps only the keys the rule can
read: full histories for the planes read over the whole prefix (every plane of a
native full-attention layer; the final plane of a shared one), and block-aligned
rings of the last positions for the planes read only near the end (every plane of
a sliding layer, ring >= the window; the planes below the final one of a shared
layer, ring >= W). The values read are the same as with every plane over the
whole context (``compact_prefix=False``).

``DoryTreeRunner`` materialises masked softmax scores over full per-plane
histories: it defines the semantics that ``DoryGraphRunner`` (paged
FlashAttention prefix reads, Triton tree attention, CUDA graphs) reproduces, and
runs the CPU tests.
"""

import functools
import heapq
import math
import time
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6
FEATURE_SUBLAYERS = (3, 19, 35, 51, 67)
STACKS = ("in", "rec", "out")


@dataclass
class LoopResult:
    hidden: torch.Tensor
    features: torch.Tensor


@dataclass
class TreeSpan:
    node_id: int
    loop: int
    hidden: torch.Tensor


@dataclass
class _Rows:
    """Query rows of one phase; ``path`` holds in-flight ancestor slots, self last."""

    slots: torch.Tensor
    positions: torch.Tensor
    planes: torch.Tensor
    path: torch.Tensor
    path_positions: torch.Tensor
    path_mask: torch.Tensor

    def select(self, index, planes=None):
        return _Rows(
            self.slots[index],
            self.positions[index],
            self.planes[index] if planes is None else planes,
            self.path[index],
            self.path_positions[index],
            self.path_mask[index],
        )


def justnorm(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + EPS)).to(x.dtype)


def slerp(a, b, rate):
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + rate * (yb - ya)).to(a.dtype)


def _get(config, name):
    return config[name] if isinstance(config, dict) else getattr(config, name)


ROW_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
PATH_BUCKETS = (4, 16, 64)
PREFIX_BUCKETS = (1024, 4096, 16384, 65536)
SPLIT = 2048  # prefix keys per FlashAttention sequence on the paged path
SPLIT_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
# Sliding layers of the paged path. For these row buckets the kernel alone reads
# a row's window, in this many parallel segments; for larger ones FlashAttention
# reads the part of the window that the rows of a plane share, as WINDOW_COPIES
# sequences.
SEGMENT_ROWS = {1: 8, 2: 8, 4: 8, 8: 4, 12: 8, 16: 8}
WINDOW_COPIES = 6


def _align(value, block=16):
    return -(-value // block) * block


def _bucket(value, buckets, limit=None):
    index = bisect_left(buckets, value)
    size = buckets[index] if index < len(buckets) else value
    return size if limit is None else min(size, limit)


def _block_pre(hidden, cos, sin, qkv, g, sqk2, heads: int, kv_heads: int, dim: int):
    """Attention sublayer up to the attention op: projections, RoPE, q/k norm."""
    n = hidden.shape[0]
    sizes = [heads * dim, kv_heads * dim, kv_heads * dim]
    q, k, v = F.linear(hidden, qkv).split(sizes, -1)
    gate = F.linear(hidden, g)
    half = dim // 2

    def rotate(x):
        a, b = x[..., :half].float(), x[..., half:].float()
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype)

    q = rotate(q.view(n, heads, dim))
    k = rotate(k.view(n, kv_heads, dim))
    q = (sqk2 * justnorm(q).float()).to(hidden.dtype)
    return q, justnorm(k), v.view(n, kv_heads, dim), gate


def _block_post(
    residual, attended, gate, gate_scale, o, alpha, gate_up, suv_gate, suv_up, down,
    mlp_alpha,
):
    """Attention output gate and residual, then the following MLP sublayer."""
    out = attended.reshape(residual.shape[0], -1).float()
    out = (out * torch.sigmoid(gate.float() * gate_scale)).to(residual.dtype)
    hidden = slerp(residual, F.linear(justnorm(out), o), alpha)
    merged = F.linear(hidden, gate_up)
    g, u = merged.chunk(2, -1)
    g = (g.float() * suv_gate).to(merged.dtype)
    u = (u.float() * suv_up).to(merged.dtype)
    mixed = justnorm((F.silu(g.float()) * u.float()).to(merged.dtype))
    return slerp(hidden, F.linear(mixed, down), mlp_alpha)


def _attend_core(
    q, shared_k, shared_v, shared_mask, row_k, row_v, row_mask, kv_heads: int
):
    """Masked softmax over keys shared by all rows [m, kv, d] and per-row keys
    [n, r, kv, d]; FP32 scores, output in q's dtype [n, heads, d]."""
    n, heads, dim = q.shape
    grouped = q.float().view(n, kv_heads, heads // kv_heads, dim)
    scale = math.sqrt(dim)
    shared = torch.einsum("ngcd,mgd->ngcm", grouped, shared_k.float()) / scale
    own = torch.einsum("ngcd,nmgd->ngcm", grouped, row_k.float()) / scale
    scores = torch.cat(
        (
            shared.masked_fill(~shared_mask[:, None, None, :], float("-inf")),
            own.masked_fill(~row_mask[:, None, None, :], float("-inf")),
        ),
        -1,
    )
    weights = torch.softmax(scores, -1)
    keys = shared_k.shape[0]
    out = torch.einsum("ngcm,mgd->ngcd", weights[..., :keys], shared_v.float())
    out = out + torch.einsum("ngcm,nmgd->ngcd", weights[..., keys:], row_v.float())
    return out.reshape(n, heads, dim).to(q.dtype)


def _load_kernels():
    """The sibling Triton module, also when this file was loaded by path."""
    try:
        from . import dory_tree_attention as module
    except ImportError:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).with_name("dory_tree_attention.py")
        spec = importlib.util.spec_from_file_location("dory_tree_attention", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class DoryWeights:
    """Sublayer tensors in dory.py's merged layout plus FP32 scale constants."""

    def __init__(self, config, embeddings, head, logit_scale, layers):
        self.hidden = _get(config, "hidden_size")
        self.heads = _get(config, "num_attention_heads")
        self.kv_heads = _get(config, "num_key_value_heads")
        self.head_dim = _get(config, "head_dim")
        self.sliding_window = _get(config, "sliding_window")
        self.layer_types = list(_get(config, "layer_types"))
        self.n_in = _get(config, "n_input_layers")
        self.n_rec = _get(config, "n_recurrent_layers")
        self.rope = _get(config, "rope_parameters")
        self.std = _get(config, "initializer_range")
        self.embeddings, self.head = embeddings, head
        self.head_scale = logit_scale.float() / self.std
        self.layers = layers
        self.device, self.dtype = embeddings.device, embeddings.dtype

    @staticmethod
    def _layer(config, kind, tensor):
        std = _get(config, "initializer_range")
        hidden = _get(config, "hidden_size")
        alpha = (
            tensor("mlp_alpha" if kind == "mlp" else "attn_alpha").float()
            * (_get(config, "ngpt_alpha_init") / std)
        ).abs()
        if kind == "mlp":
            return {
                "kind": kind,
                "gate_up": tensor("gate_up"),
                "down": tensor("mixer.down_proj.weight"),
                "suv_gate": tensor("mixer.suv_gate").float() * hidden**0.5,
                "suv_up": tensor("mixer.suv_up").float() * hidden**0.5,
                "alpha": alpha,
            }
        heads, dim = _get(config, "num_attention_heads"), _get(config, "head_dim")
        gate = _get(config, "attention_output_gate_logit_scale_init")
        gate = hidden**0.5 if gate is None else gate
        return {
            "kind": kind,
            "qkv": tensor("qkv"),
            "g": tensor("mixer.g_proj.weight"),
            "o": tensor("mixer.o_proj.weight"),
            "sqk2": (tensor("mixer.sqk").float() * (dim**0.5 / std))
            .view(1, heads, dim)
            .square(),
            "gate_scale": (tensor("mixer.s_gate").float() * (gate**0.5 / std)).square(),
            "alpha": alpha,
        }

    @classmethod
    def from_state(cls, config, tensor, device=None, dtype=None):
        """Build from checkpoint names; ``tensor(name)`` returns one tensor."""

        def load(name):
            return tensor(name).to(device=device, dtype=dtype)

        layers = []
        for i, kind in enumerate(_get(config, "layer_types")):
            prefix = f"backbone.layers.{i}."

            def piece(name, prefix=prefix):
                if name == "qkv":
                    parts = [load(f"{prefix}mixer.{p}_proj.weight") for p in "qkv"]
                    return torch.cat(parts)
                if name == "gate_up":
                    names = ("gate", "up")
                    parts = [load(f"{prefix}mixer.{p}_proj.weight") for p in names]
                    return torch.cat(parts)
                return load(prefix + name)

            layers.append(cls._layer(config, kind, piece))
        return cls(
            config,
            load("backbone.embeddings.weight"),
            load("lm_head.weight"),
            load("lm_head.logit_scale"),
            layers,
        )

    @classmethod
    def from_checkpoint(cls, directory, device, dtype=torch.bfloat16):
        import json
        from pathlib import Path

        from safetensors import safe_open

        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        with safe_open(directory / "model.safetensors", framework="pt") as handle:
            return cls.from_state(config, handle.get_tensor, device, dtype)

    @classmethod
    def from_vllm(cls, model):
        """Share the tensors of a loaded ``DoryForCausalLM`` (no copies)."""
        config, layers = model.config, []
        for kind, layer in zip(config.layer_types, model.backbone.layers):

            def tensor(name, layer=layer):
                if name in ("qkv", "gate_up"):
                    return getattr(layer.mixer, f"{name}_proj").weight
                owner = layer
                for part in name.split("."):
                    owner = getattr(owner, part)
                return owner

            layers.append(cls._layer(config, kind, tensor))
        return cls(
            config,
            model.backbone.embeddings.weight,
            model.lm_head.weight,
            model.lm_head.logit_scale,
            layers,
        )


class DoryTreeRunner:
    """Candidate-tree execution of Dory with in-call readouts (see module doc).

    ``readout_depths`` fixes the output-block planes and the prefill depths; it
    must equal the schedule's readout loops. ``window=None`` is the native
    runtime, an integer W the shared one. ``max_nodes`` is an initial physical
    allocation that grows on demand, never a pruning budget. Returned tensors
    are owned by the caller. ``compact_prefix`` selects the prefix storage of
    the module doc; ``prefix_rings`` = (sliding ring, shared ring) overrides the
    ring sizes (defaults: the window and W, aligned to blocks of 16).
    """

    fused_readout = True

    def __init__(
        self,
        weights,
        loops,
        readout_depths,
        window=None,
        max_len=8192,
        max_nodes=64,
        prefill_chunk=1024,
        compact_prefix=True,
        prefix_rings=None,
    ):
        depths = sorted(set(readout_depths))
        if not depths or depths[0] < 1 or depths[-1] != loops:
            raise ValueError("Readout depths lie in [1, R] and include R")
        if window is not None and window < 1:
            raise ValueError("The shared window must be positive")
        w = self.w = weights
        self.R, self.window = loops, window
        self.readout_depths = frozenset(depths)
        self._depth_plane = {depth: i for i, depth in enumerate(depths)}
        self.max_len, self.max_nodes = max_len, max_nodes
        self.max_path = None  # the static path sets a path width
        self.prefill_chunk = prefill_chunk
        self.device, self.dtype = w.device, w.dtype
        self.H, self.nf = w.hidden, len(FEATURE_SUBLAYERS)
        n_in, end_rec = w.n_in, w.n_in + w.n_rec
        self._spans = {
            "in": range(n_in),
            "rec": range(n_in, end_rec),
            "out": range(end_rec, len(w.layer_types)),
        }
        self._planes = {"in": 1, "rec": loops, "out": len(depths)}
        self._attention = {
            stack: [i for i in span if w.layer_types[i] != "mlp"]
            for stack, span in self._spans.items()
        }
        self._taps = {
            sublayer: slot for slot, sublayer in enumerate(FEATURE_SUBLAYERS)
        }
        self._kinds = {
            stack: [w.layer_types[i] for i in self._attention[stack]]
            for stack in STACKS
        }
        self._positions = torch.arange(max_len, device=self.device)
        positions = torch.arange(max_len, device=self.device, dtype=torch.float32)
        self._rope = {}
        for kind, spec in w.rope.items():
            inverse = 1.0 / spec["rope_theta"] ** (
                torch.arange(0, w.head_dim, 2, device=self.device).float() / w.head_dim
            )
            freqs = positions[:, None] * inverse[None]
            self._rope[kind] = (freqs.cos().to(self.dtype), freqs.sin().to(self.dtype))
        self.compact_prefix = compact_prefix
        self._allocate_prefix(prefix_rings)
        self._allocate_tree(max_nodes)
        self.stats = {
            "calls": 0,
            "cells": 0,
            "input_rows": 0,
            "recurrent_rows": 0,
            "output_rows": 0,
            "registered_nodes": 0,
            "committed_nodes": 0,
            "released_nodes": 0,
            "capacity": max_nodes,
        }
        self.reset()

    # --- storage ---------------------------------------------------------------
    def _allocate_prefix(self, rings):
        """Prefix K/V of every attention layer in one store [K or V, row, head, dim].

        A layer's rows: with ring R, plane p holds position j at ``p * R + j % R``,
        except the final plane of a shared layer (the bank), which holds every
        position at ``(planes - 1) * R + j``; without a ring, plane p holds j at
        ``p * max_len + j``. ``_lp_base``/``_lp_ring`` give the first row and ring
        (0: none) of every (layer, plane) in the tree store's order, so that a
        commit writes all of them at once.
        """
        w, capacity = self.w, self.max_len
        if rings is None:
            rings = (_align(w.sliding_window), _align(self.window or 1))
        sliding_ring, shared_ring = rings
        if sliding_ring < w.sliding_window or (
            self.window is not None and shared_ring < self.window
        ):
            raise ValueError("A prefix ring is shorter than the keys it serves")
        self._layouts, self._layer_rows = {}, {}
        bases, lp_rings, start = [], [], 0
        for stack in STACKS:
            planes = self._planes[stack]
            for kind in ("sliding_attention", "full_attention"):
                ring, bank = 0, -1
                if self.compact_prefix and kind == "sliding_attention":
                    ring = sliding_ring
                elif self.compact_prefix and self._shared(stack, kind, -1):
                    ring, bank = shared_ring, planes - 1
                if ring >= capacity:
                    ring, bank = 0, -1  # no saving
                if not ring:
                    rows = planes * capacity
                elif bank < 0:
                    rows = planes * ring
                else:
                    rows = (planes - 1) * ring + capacity
                self._layouts[stack, kind] = (planes, ring, bank, rows)
            for ordinal, kind in enumerate(self._kinds[stack]):
                planes, ring, bank, rows = self._layouts[stack, kind]
                self._layer_rows[stack, ordinal] = (start, rows)
                for plane in range(planes):
                    bases.append(start + plane * (ring or capacity))
                    lp_rings.append(0 if plane == bank else ring)
                start += rows
        self._prefix_store = torch.zeros(
            (2, start, w.kv_heads, w.head_dim), device=self.device, dtype=self.dtype
        )
        self._prefix_k, self._prefix_v = {}, {}
        for key, (first, rows) in self._layer_rows.items():
            self._prefix_k[key] = self._prefix_store[0, first : first + rows]
            self._prefix_v[key] = self._prefix_store[1, first : first + rows]
        self._lp_base = np.array(bases, np.int64)
        self._lp_ring = np.array(lp_rings, np.int64)
        self._lp_wrap = np.maximum(self._lp_ring, 1)

    @property
    def prefix_kv_bytes(self):
        store = self._prefix_store
        return store.numel() * store.element_size()

    def _rows_of(self, stack, kind, plane, positions):
        """Rows of a layer's prefix (``_prefix_k[stack, ordinal]``) that hold the
        positions of a plane; ``plane`` may be a tensor broadcast against them.
        Positions outside a ring map onto newer ones: only masked keys go there."""
        _, ring, bank, _ = self._layouts[stack, kind]
        if not ring:
            return plane * self.max_len + positions
        wrapped = positions % ring
        if bank < 0:
            pass
        elif isinstance(plane, torch.Tensor):
            wrapped = torch.where(plane == bank, positions, wrapped)
        elif plane == bank:
            wrapped = positions
        return plane * ring + wrapped

    def _store_prompt(self, stack, ordinal, plane, k, v):
        """Write a plane's prompt K/V; a ring keeps the last positions."""
        n, kind = k.shape[0], self._kinds[stack][ordinal]
        _, ring, bank, _ = self._layouts[stack, kind]
        keep = n if not ring or plane == bank else min(n, ring)
        rows = self._rows_of(stack, kind, plane, self._positions[n - keep : n])
        self._prefix_k[stack, ordinal].index_copy_(0, rows, k[n - keep :])
        self._prefix_v[stack, ordinal].index_copy_(0, rows, v[n - keep :])

    def _allocate_kv(self, rows):
        """Tree KV of all stacks over ``rows`` slots: one store [K or V, layer and
        plane, row, head, dim], so that a commit reads every plane at once, and
        per stack its view [K or V, layer, plane, row, head, dim]."""
        w = self.w
        sizes = [len(self._attention[stack]) * self._planes[stack] for stack in STACKS]
        store = torch.zeros(
            (2, sum(sizes), rows, w.kv_heads, w.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        views, start = {}, 0
        for stack, size in zip(STACKS, sizes):
            shape = (len(self._attention[stack]), self._planes[stack])
            views[stack] = store[:, start : start + size].unflatten(1, shape)
            start += size
        return store, views

    def _allocate_tree(self, nodes):
        old = getattr(self, "_tree_store", None)
        # Slot ``nodes`` is scratch: padding rows of the static path write there.
        self._tree_store, self._tree = self._allocate_kv(nodes + 1)
        banks = torch.zeros(
            (2, nodes + 1, self.H), device=self.device, dtype=self.dtype
        )
        if old is not None:
            self._tree_store[:, :, : self.max_nodes] = old[:, :, : self.max_nodes]
            banks[:, : self.max_nodes] = self._banks[:, : self.max_nodes]
        self._banks = banks  # [0] input-block output x, [1] tap of sublayer 3
        self._allocate_host(nodes, None if old is None else self.max_nodes)
        self.max_nodes = nodes

    def _allocate_host(self, nodes, old, width=None):
        """Node state on the host, indexed by slot.

        A table row is the node's row metadata: slot, position, finished loops, a
        flag until its first call, then its path and the mask of the path. The
        path holds the slots of the in-flight ancestors from the first uncommitted
        position, the node itself last, then scratch. The scratch slot ``nodes``
        stands for "no parent" and "no path entry" and counts as a finished node.
        """
        before = None if old is None else self._path.shape[1]
        if width is None:  # a path bucket, so that a call's path bucket fits
            width = _bucket(self.window or 64, PATH_BUCKETS) if old is None else before
        table = np.zeros((nodes + 1, 4 + 2 * width), np.int64)
        table[:, 0] = np.arange(nodes + 1)
        table[:, 2] = self.R
        table[:, 4 : 4 + width] = nodes
        token = np.zeros(nodes + 1, np.int64)
        parent = np.full(nodes + 1, nodes, np.int64)
        live = np.zeros(nodes + 1, bool)
        if old is not None:
            table[:old, 1:4] = self._table[:old, 1:4]
            table[:old, 4 : 4 + before] = np.where(
                self._path[:old] == old, nodes, self._path[:old]
            )
            table[:old, 4 + width : 4 + width + before] = self._mask[:old]
            token[:old], live[:old] = self._token[:old], self._live[:old]
            above = self._parent[:old]
            parent[:old] = np.where(above == old, nodes, above)
        self._table, self._token, self._parent, self._live = table, token, parent, live
        self._depth = table[:, 2]
        self._path, self._mask = table[:, 4 : 4 + width], table[:, 4 + width :]
        self._mark = np.zeros(nodes + 1, bool)

    @property
    def prefix_length(self):
        return self.length

    def reset(self):
        self.length = 0
        self._nodes: dict[int, int] = {}  # node id -> slot
        self._free = list(range(self.max_nodes))
        heapq.heapify(self._free)
        self._unfinished = 0
        self._span = 0  # bound on the in-flight path length of the live nodes
        self._live[:] = False

    def _index(self, values):
        """Host integers as a device index. On CUDA the copy is asynchronous from
        pinned memory, which the allocator reuses only after the copy has run."""
        if self.device.type != "cuda":
            return torch.tensor(values, dtype=torch.long, device=self.device)
        pinned = torch.empty(len(values), dtype=torch.long, pin_memory=True)
        pinned.numpy()[:] = values
        return pinned.to(self.device, non_blocking=True)

    def ensure_capacity(self, required):
        nodes = self.max_nodes
        while nodes < required:
            nodes *= 2
        if nodes != self.max_nodes:
            fresh = range(self.max_nodes, nodes)
            self._allocate_tree(nodes)
            for slot in fresh:
                heapq.heappush(self._free, slot)
            self.stats["capacity"] = nodes
        return self.max_nodes

    def register(self, node_id, parent_id, token_id):
        return self.register_many([(node_id, parent_id, token_id)])[0]

    def register_many(self, nodes):
        """Register (node id, parent id, token) triples in order, a parent before
        its children; returns their slots."""
        ids, free = self._nodes, self._free
        if len(free) < len(nodes):
            self.ensure_capacity(self.max_nodes - len(free) + len(nodes))
        table, length, scratch = self._table, self.length, self.max_nodes
        token, above, live = self._token, self._parent, self._live
        width, slots, span = self._path.shape[1], [], self._span
        for node_id, parent_id, token_id in nodes:
            if node_id in ids:
                raise ValueError("A node is registered once")
            if parent_id is None:
                parent, offset = scratch, 0
            else:
                parent = ids[parent_id]
                offset = int(table[parent, 1]) + 1 - length
            if self.max_path is not None and offset >= self.max_path:
                raise ValueError("The in-flight path exceeds the static path width")
            if length + offset >= self.max_len:
                raise ValueError("The candidate exceeds the context capacity")
            if self.window is not None and offset >= self.window:
                raise ValueError("The in-flight path exceeds the shared window")
            if offset >= width:  # the dense native path has no width bound
                self._allocate_host(scratch, scratch, 2 * width)
                table, width = self._table, 2 * width
                token, above, live = self._token, self._parent, self._live
            slot = heapq.heappop(free)
            ids[node_id] = slot
            table[slot, 1] = length + offset
            table[slot, 2] = 0
            table[slot, 3] = 1
            # The parent's path and mask, then the node itself.
            table[slot, 4:] = table[parent, 4:]
            table[slot, 4 + offset] = slot
            table[slot, 4 + width + offset] = 1
            token[slot] = token_id
            above[slot] = parent
            live[slot] = True
            if offset >= span:
                self._span = span = offset + 1
            self._unfinished += 1
            slots.append(slot)
        self.stats["registered_nodes"] += len(slots)
        return slots

    def release(self, node_ids):
        if not node_ids:
            return
        nodes = self._nodes
        slots = np.array([nodes[node_id] for node_id in node_ids], np.int64)
        removed = self._mark
        removed[slots] = True
        orphans = self._live & removed[self._parent]
        orphans[slots] = False
        removed[slots] = False
        if np.count_nonzero(orphans):
            raise ValueError("A released node has a live descendant")
        self._unfinished -= int(np.count_nonzero(self._depth[slots] < self.R))
        self._live[slots] = False
        free = self._free
        for node_id, slot in zip(node_ids, slots.tolist()):
            del nodes[node_id]
            heapq.heappush(free, slot)
        self.stats["released_nodes"] += len(node_ids)

    def commit(self, node_ids, features=None):
        """Move the verified path into the prefix. A runner that keeps node features
        writes the loop-R features of the committed tokens to ``features``."""
        if not node_ids:
            return
        nodes, table, parent = self._nodes, self._table, self._parent
        count, scratch, length = len(node_ids), self.max_nodes, self.length
        slots = [nodes[node_id] for node_id in node_ids]
        before = scratch
        for i, slot in enumerate(slots):
            if parent[slot] != before or table[slot, 1] != length + i:
                raise ValueError("Only the contiguous verified path may commit")
            # Every node passes each readout depth on its way to loop R, so a node
            # at loop R holds all output planes.
            if table[slot, 2] != self.R:
                raise ValueError("A committed token lacks a loop or readout plane")
            before = slot
        index = before if count == 1 else self._index(slots)
        self._commit_prefix(index, count, length)
        if features is not None:
            self._commit_features(index, features)
        free, live = self._free, self._live
        for node_id, slot in zip(node_ids, slots):
            del nodes[node_id]
            heapq.heappush(free, slot)
            live[slot] = False
        # Every other live node descends from the last committed token: its path
        # holds that token at the last committed position (a shorter path holds
        # scratch there).
        path, mask, span = self._path, self._mask, self._span
        if np.count_nonzero(path[live, count - 1] != before):
            raise ValueError("A rejected sibling survived the commit")
        parent[parent == before] = scratch
        path[:, : span - count] = path[:, count:span]
        path[:, span - count : span] = scratch
        mask[:, : span - count] = mask[:, count:span]
        mask[:, span - count : span] = 0
        self._span = span - count
        self.length += count
        self.stats["committed_nodes"] += count

    def _commit_prefix(self, index, count, length):
        """Copy the committed slots (an int for one) to positions ``length`` on, in
        every (layer, plane); ring rows overwrite positions no rule reads any more."""
        if count == 1:
            source = self._tree_store.narrow(2, index, 1)
        else:
            source = self._tree_store.index_select(2, index)
        position = length + np.arange(count)
        wrapped = np.where(
            self._lp_ring[:, None] > 0,
            position[None] % self._lp_wrap[:, None],
            position[None],
        )
        rows = self._index((self._lp_base[:, None] + wrapped).reshape(-1))
        self._prefix_store.index_copy_(1, rows, source.flatten(1, 2))

    def _commit_features(self, index, features):
        raise ValueError("The dense runner returns node features with each call")

    # --- model math (dory.py order and rounding points) ------------------------
    def embed(self, tokens):
        return F.embedding(tokens, self.w.embeddings)

    def logits(self, hidden):
        return F.linear(hidden, self.w.head).float() * self.w.head_scale

    def _rotate(self, x, positions, kind):
        cos, sin = self._rope[kind]
        cos = cos[positions].float()[:, None]
        sin = sin[positions].float()[:, None]
        half = self.w.head_dim // 2
        a, b = x[..., :half].float(), x[..., half:].float()
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype)

    def _project(self, layer, hidden, positions):
        w, n = self.w, hidden.shape[0]
        q_size, kv_size = w.heads * w.head_dim, w.kv_heads * w.head_dim
        q, k, v = F.linear(hidden, layer["qkv"]).split([q_size, kv_size, kv_size], -1)
        gate = F.linear(hidden, layer["g"])
        q = self._rotate(q.view(n, w.heads, w.head_dim), positions, layer["kind"])
        k = self._rotate(k.view(n, w.kv_heads, w.head_dim), positions, layer["kind"])
        q = (layer["sqk2"] * justnorm(q).float()).to(hidden.dtype)
        return q, justnorm(k), v.view(n, w.kv_heads, w.head_dim), gate

    def _finish(self, layer, residual, attended, gate):
        out = attended.reshape(residual.shape[0], -1).float()
        gated = torch.sigmoid(gate.float() * layer["gate_scale"])
        out = (out * gated).to(residual.dtype)
        return slerp(residual, F.linear(justnorm(out), layer["o"]), layer["alpha"])

    def _mlp(self, layer, hidden):
        merged = F.linear(hidden, layer["gate_up"])
        g, u = merged.chunk(2, -1)
        g = (g.float() * layer["suv_gate"]).to(merged.dtype)
        u = (u.float() * layer["suv_up"]).to(merged.dtype)
        mixed = justnorm((F.silu(g.float()) * u.float()).to(merged.dtype))
        return slerp(hidden, F.linear(mixed, layer["down"]), layer["alpha"])

    def _softmax_attend(self, q, sources):
        """q [n, heads, d]; sources: (K, V, mask [n, m]) with K/V [m, kv, d] shared
        by all rows or [n, m, kv, d] per row. Returns [n, heads, d]."""
        w = self.w
        n = q.shape[0]
        grouped = q.float().view(n, w.kv_heads, w.heads // w.kv_heads, w.head_dim)
        scores = []
        for keys, _, mask in sources:
            equation = "ngcd,mgd->ngcm" if keys.dim() == 3 else "ngcd,nmgd->ngcm"
            score = torch.einsum(equation, grouped, keys.float())
            scores.append(
                (score / math.sqrt(w.head_dim)).masked_fill(
                    ~mask[:, None, None, :], float("-inf")
                )
            )
        weights = torch.softmax(torch.cat(scores, -1), -1)
        out, start = 0, 0
        for (_, values, _), score in zip(sources, scores):
            part = weights[..., start : start + score.shape[-1]]
            equation = "ngcm,mgd->ngcd" if values.dim() == 3 else "ngcm,nmgd->ngcd"
            out = out + torch.einsum(equation, part, values.float())
            start += score.shape[-1]
        return out.reshape(n, w.heads, w.head_dim).to(q.dtype)

    def _shared(self, stack, kind, plane):
        return (
            self.window is not None
            and stack != "in"
            and kind == "full_attention"
            and plane != self._planes[stack] - 1
        )

    def _tree_attention(self, stack, ordinal, kind, q, k, v, rows):
        """Write the rows' KV, then attend over the committed prefix and the path."""
        tree_k, tree_v = self._tree[stack][0, ordinal], self._tree[stack][1, ordinal]
        tree_k[rows.planes, rows.slots] = k
        tree_v[rows.planes, rows.slots] = v
        prefix_k = self._prefix_k[stack, ordinal]
        prefix_v = self._prefix_v[stack, ordinal]
        out = torch.empty_like(q)
        sliding = self.w.sliding_window if kind == "sliding_attention" else None
        for plane in rows.planes.unique().tolist():
            index = (rows.planes == plane).nonzero().flatten()
            t = rows.positions[index][:, None]
            low = 0 if sliding is None else max(0, int(t.min()) - sliding + 1)
            low = min(low, self.length)
            j = torch.arange(low, self.length, device=self.device)[None]
            visible = j <= t if sliding is None else j > t - sliding
            path_visible = rows.path_mask[index]
            if sliding is not None:
                path_visible = path_visible & (rows.path_positions[index] > t - sliding)
            sources = []
            if self._shared(stack, kind, plane):
                final = self._planes[stack] - 1
                old = j <= t - self.window
                history = self._rows_of(stack, kind, final, j[0])
                sources.append(
                    (prefix_k[history], prefix_v[history], visible & old)
                )
                visible = visible & ~old
            history = self._rows_of(stack, kind, plane, j[0])
            sources.append((prefix_k[history], prefix_v[history], visible))
            path = rows.path[index]
            sources.append((tree_k[plane][path], tree_v[plane][path], path_visible))
            out[index] = self._softmax_attend(q[index], sources)
        return out

    def _prefill_attention(self, stack, ordinal, plane, kind, q, k, v):
        """Native causal attention of the prompt; fills one prefix plane."""
        n = q.shape[0]
        self._store_prompt(stack, ordinal, plane, k, v)
        sliding = self.w.sliding_window if kind == "sliding_attention" else None
        out = torch.empty_like(q)
        for start in range(0, n, self.prefill_chunk):
            end = min(start + self.prefill_chunk, n)
            low = 0 if sliding is None else max(0, start - sliding + 1)
            t = torch.arange(start, end, device=self.device)[:, None]
            j = torch.arange(low, end, device=self.device)[None]
            visible = j <= t
            if sliding is not None:
                visible = visible & (j > t - sliding)
            out[start:end] = self._softmax_attend(
                q[start:end], [(k[low:end], v[low:end], visible)]
            )
        return out

    def _run(self, stack, hidden, positions, attend, features=None, rows=None):
        """Apply one block; ``attend(ordinal, kind, q, k, v)`` supplies attention.
        Feature taps of the block go to ``features`` (rows ``rows`` when given)."""
        ordinal = 0
        for sublayer in self._spans[stack]:
            layer = self.w.layers[sublayer]
            if layer["kind"] == "mlp":
                hidden = self._mlp(layer, hidden)
            else:
                q, k, v, gate = self._project(layer, hidden, positions)
                attended = attend(ordinal, layer["kind"], q, k, v)
                hidden = self._finish(layer, hidden, attended, gate)
                ordinal += 1
            if features is not None and sublayer in self._taps:
                start = self._taps[sublayer] * self.H
                columns = slice(start, start + self.H)
                if rows is None:
                    features[:, columns] = hidden
                else:
                    features[rows, columns] = hidden
        return hidden

    # --- execution -------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, tokens):
        """Native pass over the prompt, all loops and readout depths; depth-R
        output-block state and loop-R features of every prompt position."""
        if self.length or self._nodes:
            raise ValueError("Prefill starts from an empty runner")
        n = tokens.numel()
        if not 0 < n <= self.max_len:
            raise ValueError("Invalid prompt length")
        positions = torch.arange(n, device=self.device)
        features = torch.empty(
            (n, self.nf * self.H), device=self.device, dtype=self.dtype
        )

        def native(stack, plane):
            return lambda ordinal, kind, q, k, v: self._prefill_attention(
                stack, ordinal, plane, kind, q, k, v
            )

        x = self._run("in", self.embed(tokens), positions, native("in", 0), features)
        state, states = x, {}
        for loop in range(self.R):
            taps = features if loop == self.R - 1 else None
            hidden = self._run(
                "rec", justnorm(state + x), positions, native("rec", loop), taps
            )
            state = justnorm(hidden)
            states[loop + 1] = state
        for depth, plane in self._depth_plane.items():
            taps = features if depth == self.R else None
            out = self._run("out", states[depth], positions, native("out", plane), taps)
            if depth == self.R:
                final = out
        self.length = n
        return LoopResult(final, features)

    def _rows(self, slots, table):
        position = table[:, 1]
        width = int(position.max()) - self.length + 1  # the longest in-flight path
        mask = table[:, self._path.shape[1] + 4 :][:, :width] != 0
        path = np.where(mask, table[:, 4 : 4 + width], 0)
        path_positions = self.length + torch.arange(width)[None].expand(len(slots), -1)
        tensors = (
            torch.from_numpy(slots),
            torch.from_numpy(position.copy()),
            torch.from_numpy(table[:, 2].copy()),
            torch.from_numpy(path),
            path_positions,
            torch.from_numpy(mask),
        )
        return _Rows(*(tensor.to(self.device) for tensor in tensors))

    def _check(self, node_ids, loops):
        """Slots and table rows of a call that advances every unfinished node once,
        each at the loop the caller expects and behind its parent."""
        nodes = self._nodes
        slots = np.array([nodes[node_id] for node_id in node_ids], np.int64)
        table = self._table[slots]
        depth = table[:, 2]
        # The parent suffices: its own ancestors passed this check before it. It
        # also holds the output KV of every readout depth up to its own loop.
        behind = self._depth[self._parent[slots]] < depth
        if (
            len(node_ids) != self._unfinished
            or len(set(node_ids)) != len(node_ids)
            or depth.tolist() != loops
            or np.count_nonzero(behind | (depth >= self.R))
            or not node_ids
        ):
            if (
                len(node_ids) != self._unfinished
                or len(set(node_ids)) != len(node_ids)
                or np.count_nonzero(depth >= self.R)
                or not node_ids
            ):
                raise ValueError("A call advances every unfinished node once")
            if depth.tolist() != loops:
                raise ValueError("A span loop differs from the node depth")
            raise ValueError("An ancestor lacks KV at the node's loop")
        return slots, table

    def _check_spans(self, spans):
        node_ids = [span.node_id for span in spans]
        return self._check(node_ids, [span.loop for span in spans])

    def _complete(self, slots, depth, fresh, readout):
        """Count the call; ``depth`` holds the loops the nodes had finished before."""
        self._depth[slots] = depth + 1
        if fresh:
            self._table[slots, 3] = 0
        self._unfinished -= int(np.count_nonzero(depth == self.R - 1))
        self.stats["calls"] += 1
        self.stats["cells"] += len(slots)
        self.stats["input_rows"] += fresh
        self.stats["recurrent_rows"] += len(slots)
        self.stats["output_rows"] += readout

    @torch.no_grad()
    def advance_readout(self, spans):
        """Advance every unfinished node by one loop.

        Returns per-span results (recurrent state after the loop and the five
        feature taps; the output tap is NaN off the readout depths), the logits
        of the spans that reached a readout depth in span order, and their ids.
        """
        slots, table = self._check_spans(spans)
        rows = self._rows(slots, table)
        n = len(spans)
        features = torch.full(
            (n, self.nf * self.H), float("nan"), device=self.device, dtype=self.dtype
        )

        def tree(stack, selected):
            return lambda ordinal, kind, q, k, v: self._tree_attention(
                stack, ordinal, kind, q, k, v, selected
            )

        fresh = [i for i, span in enumerate(spans) if span.loop == 0]
        if fresh:
            index = torch.tensor(fresh, device=self.device)
            selected = rows.select(index, torch.zeros_like(index))
            embedded = torch.stack([spans[i].hidden.reshape(-1) for i in fresh])
            taps = torch.empty(
                (len(fresh), self.nf * self.H), device=self.device, dtype=self.dtype
            )
            x = self._run(
                "in", embedded, selected.positions, tree("in", selected), taps
            )
            self._banks[0, selected.slots] = x
            self._banks[1, selected.slots] = taps[:, : self.H]
        x = self._banks[0, rows.slots]
        state = torch.stack(
            [
                x[i] if span.loop == 0 else span.hidden.reshape(-1)
                for i, span in enumerate(spans)
            ]
        )
        features[:, : self.H] = self._banks[1, rows.slots]
        hidden = self._run(
            "rec", justnorm(state + x), rows.positions, tree("rec", rows), features
        )
        state = justnorm(hidden)
        readout = [
            i for i, span in enumerate(spans) if span.loop + 1 in self.readout_depths
        ]
        logits = None
        if readout:
            index = torch.tensor(readout, device=self.device)
            planes = torch.tensor(
                [self._depth_plane[spans[i].loop + 1] for i in readout],
                device=self.device,
            )
            selected = rows.select(index, planes)
            out = self._run(
                "out",
                state[index],
                selected.positions,
                tree("out", selected),
                features,
                index,
            )
            logits = self.logits(out)
        self._complete(slots, table[:, 2], len(fresh), len(readout))
        results = [LoopResult(state[i : i + 1], features[i : i + 1]) for i in range(n)]
        return results, logits, [spans[i].node_id for i in readout]

    def advance(self, spans):
        raise RuntimeError("Dory readouts need the output block; use advance_readout")


class DoryGraphRunner(DoryTreeRunner):
    """The same rule in static shapes: each phase replays from a CUDA graph.

    Rows are padded to a size bucket (padding rows use the scratch slot), the
    in-flight path to a width bucket and dense prefix reads to a capacity
    bucket, so tensor shapes and control flow are fixed per graph; positions,
    planes, slots and the prefix length are data. The nGPT glue around each
    attention op runs as two compiled functions shared by all blocks.

    Prefix attention is still dense (FP32 scores over a capacity bucket):
    suitable for short contexts and as the bridge to paged FlashAttention.
    Graph outputs stay valid until the same phase replays.

    Node state stays on the device, by slot: the recurrent state after each loop
    and the feature taps at each readout depth. The graphs read and write it, so
    a call (``advance_nodes``) sends only row metadata and the tokens of new
    nodes, and returns the readout rows with their argmax tokens and logits.
    ``advance_readout`` is the same call behind the span interface of the dense
    runner.
    """

    keeps_node_state = True

    def __init__(
        self,
        *args,
        use_cuda_graph=True,
        compile_blocks=True,
        attention="auto",
        profile=False,
        **kwargs,
    ):
        self._graphs, self._pools = {}, {}
        self.profile = profile  # synchronises around replays: for attribution only
        # The paged path reads prefix planes in whole splits (blocks of 16 positions).
        kwargs["max_len"] = -(-kwargs.get("max_len", 8192) // SPLIT) * SPLIT
        super().__init__(*args, **kwargs)
        if attention == "auto":
            attention = "paged" if self.device.type == "cuda" else "dense"
        if attention not in ("paged", "dense"):
            raise ValueError("attention is 'auto', 'paged' or 'dense'")
        # "paged": FlashAttention over the prefix keys that rows share and Triton
        # kernels for each row's own keys (``_paged_plan``). "dense": masked softmax
        # (CPU, cross-check).
        self.attention = attention
        if attention == "paged" and any(
            ring % 16 for _, ring, _, _ in self._layouts.values()
        ):
            raise ValueError("Paged reads need prefix rings in blocks of 16")
        if attention == "paged":
            from vllm.vllm_flash_attn import (
                flash_attn_varlen_func,
                is_fa_version_supported,
            )

            self._flash_attention = flash_attn_varlen_func
            self.fa_version = 3 if is_fa_version_supported(3) else 2
            self._kernels = _load_kernels()
            self._constants = {}
            self._lp_base_device = torch.tensor(self._lp_base, device=self.device)
            self._lp_ring_device = torch.tensor(self._lp_ring, device=self.device)
        kinds = self.w.layer_types
        for span in self._spans.values():
            if any((kinds[i] == "mlp") != bool((i - span.start) % 2) for i in span):
                raise ValueError("Blocks must alternate attention and MLP sublayers")
        self.max_path = self.window or 64
        self.use_cuda_graph = use_cuda_graph and self.device.type == "cuda"
        self._pre, self._post, self._attend = _block_pre, _block_post, _attend_core
        if compile_blocks:
            import torch._dynamo as dynamo

            for name in ("recompile_limit", "cache_size_limit"):
                if hasattr(dynamo.config, name):
                    setattr(dynamo.config, name, 64)
            self._pre = torch.compile(_block_pre)
            self._post = torch.compile(_block_post)
            self._attend = torch.compile(_attend_core)
        self._arange = torch.arange(max(self.max_len, 256), device=self.device)
        # Marks the last copies from the pinned host buffers (see _settle).
        self._copied = torch.cuda.Event() if self.device.type == "cuda" else None
        self._copying = False
        # Lookups by loop count after the call: readout depth or not, output plane.
        self._readout = np.zeros(self.R + 1, bool)
        self._plane = np.zeros(self.R + 1, np.int64)
        for depth, plane in self._depth_plane.items():
            self._readout[depth], self._plane[depth] = True, plane
        self.compile_blocks = compile_blocks
        self.stats.update(graph_captures=0, graph_replays=0, capture_seconds=0.0)

    def _allocate_tree(self, nodes):
        old = self.max_nodes if hasattr(self, "_states") else None
        super()._allocate_tree(nodes)
        # Graphs hold the old banks; a pool dies with its last graph.
        self._graphs, self._pools = {}, {}
        self._padding_rows = {}
        # All graphs read the prefix length from one tensor.
        self._length = torch.zeros((), device=self.device, dtype=torch.long)
        self._length_set = 0
        # Node state, rows [plane, slot]. States: the recurrent state after each
        # loop. Features: the five taps at each readout depth, in output-plane
        # order; the last plane takes the taps of the other loops and its output
        # tap stays NaN.
        rows, depths = nodes + 1, sorted(self._depth_plane)
        width, taps = self.nf * self.H, (self.nf - 1) * self.H
        options = dict(device=self.device, dtype=self.dtype)
        states = torch.zeros((self.R, rows, self.H), **options)
        features = torch.zeros((len(depths) + 1, rows, width), **options)
        features[..., taps:] = float("nan")
        if old is not None:
            states[:, :old] = self._states.view(self.R, old + 1, -1)[:, :old]
            features[:, :old] = self._features.view(-1, old + 1, width)[:, :old]
        self._states = states.view(-1, self.H)
        self._features = features.view(-1, width)
        self._features_low = self._features[:, :taps]
        self._features_high = self._features[:, taps:]
        self._features_final = features[self._depth_plane[self.R]]
        # First rows of the planes a phase uses, by the plane column of its rows:
        # the features of the loop a recurrent row runs, and the state an output
        # row reads (the one after the loop of its depth).
        other = len(depths)
        planes = [self._depth_plane.get(loop + 1, other) for loop in range(self.R)]
        self._feature_rows = np.array(planes) * rows
        self._feature_base = torch.tensor(self._feature_rows, device=self.device)
        self._state_base = torch.tensor(
            [(depth - 1) * rows for depth in depths], device=self.device
        )

    def _commit_prefix(self, index, count, length):
        if self.attention != "paged":
            return super()._commit_prefix(index, count, length)
        self._kernels.commit_rows(
            self._tree_store, self._prefix_store, index, count,
            self._lp_base_device, self._lp_ring_device, length,
        )

    def _commit_features(self, index, features):
        if isinstance(index, int):
            features.copy_(self._features_final.narrow(0, index, 1))
        else:
            torch.index_select(self._features_final, 0, index, out=features)

    def node_features(self, node_id, depth):
        """Feature taps of a live node at a readout depth it has reached: a view
        that holds until the node's slot runs that loop again."""
        rows = self.max_nodes + 1
        return self._features[self._depth_plane[depth] * rows + self._nodes[node_id]]

    def span_features(self, spans):
        """Feature taps of several node spans in one gather. ``spans`` holds
        (readout depth, node ids) pairs; each result is an owned
        [len(node ids), F] tensor of the nodes' taps at that depth."""
        nodes, rows, planes = self._nodes, self.max_nodes + 1, self._depth_plane
        index = [
            planes[depth] * rows + nodes[node_id]
            for depth, node_ids in spans
            for node_id in node_ids
        ]
        features = self._features.index_select(0, self._index(index))
        if len(spans) == 1:
            return (features,)
        return features.split([len(node_ids) for _, node_ids in spans])

    def _entry(self, phase, rows, width, capacity):
        key = (phase, rows, width, capacity)
        entry = self._graphs.get(key)
        if entry is None:
            # meta columns: slot, position, plane, flags (1 fresh, 2 padding),
            # path[width], path mask[width]; the input block adds the row's token.
            shape = (rows, 4 + 2 * width + (phase == "in"))
            meta = torch.zeros(shape, device=self.device, dtype=torch.long)
            # The host fills a twin of the metadata: pinned memory copied without
            # waiting on CUDA (see _settle), the input itself on the CPU.
            cuda = self.device.type == "cuda"
            twin = meta
            if cuda:
                twin = torch.zeros(shape, dtype=torch.long, pin_memory=True)
            entry = {
                "io": {"meta": meta, "length": self._length},
                "capacity": capacity,
                "graph": None,
                "out": None,
                "host": twin.numpy(),
                "twin": twin if cuda else None,
                "views": {},
            }
            self._graphs[key] = entry
        return entry

    def _padding(self, width):
        """Metadata of a padding row: the scratch slot, an empty prefix range and
        its own scratch key."""
        row = self._padding_rows.get(width)
        if row is None:
            scratch = self.max_nodes
            row = self._padding_rows[width] = np.zeros(4 + 2 * width, np.int64)
            row[0], row[3], row[4 : 4 + width], row[4 + width] = scratch, 2, scratch, 1
        row[1] = self.length
        return row

    def _set_length(self):
        if self._length_set != self.length:
            self._length.fill_(self.length)
            self._length_set = self.length

    def _settle(self):
        """Wait for the last copies from the pinned host buffers before a call
        rewrites them. A greedy call has read its readouts back by then."""
        if self._copying:
            if not self._copied.query():
                self._copied.synchronize()
            self._copying = False

    def _load(self, entry):
        if entry["twin"] is not None:
            entry["io"]["meta"].copy_(entry["twin"], non_blocking=True)

    def _loaded(self):
        if self.device.type == "cuda":
            # On the stream the copies were queued on, whichever the caller uses.
            self._copied.record(torch.cuda.current_stream(self.device))
            self._copying = True

    def _fill(self, entry, nodes, planes, hidden, fresh=None):
        """Inputs of an entry made of padding rows only (graph capture before
        requests); ``advance_nodes`` stages the rows of a call."""
        if len(nodes):
            raise ValueError("Calls stage their rows in advance_nodes")
        self._settle()
        host = entry["host"]
        width = (host.shape[1] - 4) // 2
        host[:, : 4 + 2 * width] = self._padding(width)
        self._set_length()
        self._load(entry)
        self._loaded()

    @staticmethod
    def _unpack(io):
        meta = io["meta"]
        width = (meta.shape[1] - 4) // 2
        return {
            "length": io["length"],
            "slots": meta[:, 0].contiguous(),
            "positions": meta[:, 1].contiguous(),
            "planes": meta[:, 2].contiguous(),
            "fresh": (meta[:, 3] & 1) != 0,
            "padding": (meta[:, 3] & 2) != 0,
            "path": meta[:, 4 : 4 + width].contiguous(),
            "path_mask": meta[:, 4 + width : 4 + 2 * width] != 0,
        }

    def _plan(self, stack, io, capacity):
        """Key indices and masks of one phase, per attention kind; every layer of
        the stack reuses them."""
        w = self.w
        planes, final = self._planes[stack], self._planes[stack] - 1
        t, plane, length = io["positions"][:, None], io["planes"], io["length"]
        path_position = (length + self._arange[: io["path"].shape[1]])[None]
        plans = {}
        kinds = {w.layer_types[i] for i in self._attention[stack]}
        if self.attention == "paged":
            return self._paged_plan(stack, io, kinds, capacity)
        if "sliding_attention" in kinds:
            span = min(w.sliding_window - 1, self.max_len)
            index = length - span + self._arange[:span]
            valid = index >= 0
            index = index.clamp_min(0)
            key_plane = self._arange[:planes].repeat_interleave(span)
            plans["sliding_attention"] = {
                "mode": "sliding",
                "index": index,
                "shared_mask": valid.repeat(planes)[None]
                & (key_plane[None] == plane[:, None])
                & (index.repeat(planes)[None] > t - w.sliding_window),
                "row_mask": io["path_mask"] & (path_position > t - w.sliding_window),
            }
        if "full_attention" in kinds and self._shared(stack, "full_attention", -1):
            position = self._arange[:capacity][None]
            own = (plane != final)[:, None]
            band = min(self.window, self.max_len)
            index = length - band + self._arange[:band]
            valid = index >= 0
            index = index.clamp_min(0)[None]
            band_mask = valid[None] & (index > t - self.window) & own
            plans["full_attention"] = {
                "mode": "shared",
                "index": index,
                "shared_mask": (position < length)
                & ((position <= t - self.window) | ~own),
                "row_mask": torch.cat((band_mask, io["path_mask"]), 1),
            }
        elif "full_attention" in kinds:
            key_plane = self._arange[:planes].repeat_interleave(capacity)
            position = self._arange[:capacity].repeat(planes)[None]
            plans["full_attention"] = {
                "mode": "native",
                "shared_mask": (position < length)
                & (key_plane[None] == plane[:, None]),
                "row_mask": io["path_mask"],
            }
        return plans

    def _prefill_attention(self, stack, ordinal, plane, kind, q, k, v):
        if self.attention != "paged":
            return super()._prefill_attention(stack, ordinal, plane, kind, q, k, v)
        n = q.shape[0]
        self._store_prompt(stack, ordinal, plane, k, v)
        sliding = kind == "sliding_attention"
        cu = torch.tensor([0, n], device=self.device, dtype=torch.int32)
        return self._flash_attention(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=n,
            cu_seqlens_q=cu,
            max_seqlen_k=n,
            cu_seqlens_k=cu,
            causal=True,
            window_size=(self.w.sliding_window - 1, 0) if sliding else (-1, -1),
            softmax_scale=self.w.head_dim**-0.5,
            fa_version=self.fa_version,
        )

    def _constant(self, key, build):
        """A tensor that depends on shapes only: built once, outside any graph."""
        if key not in self._constants:
            self._constants[key] = build()
        return self._constants[key]

    def _paged_plan(self, stack, io, kinds, capacity):
        """Reads of the paged path, per attention kind (see ``_plan``).

        FlashAttention reads the prefix keys that the rows of a plane share, as
        sequences that each hold those rows: parallel over keys and nearly
        independent of the row count. The kernel continues every row's softmax
        from these partial results over the keys only that row sees and its path
        (``plan_reads`` in ``dory_tree_attention.py`` has the ranges per rule).
        With few rows a sliding window is cheaper in the kernel alone, split
        into segments that run in parallel and merged by a second launch.
        """
        w, kernels = self.w, self._kernels
        planes, final = self._planes[stack], self._planes[stack] - 1
        t, n = io["positions"], io["positions"].shape[0]
        rows = (t, io["planes"], io["padding"], io["length"])
        per_plane = self.max_len // 16  # blocks of 16 positions in a prefix plane
        plans = {}
        if "sliding_attention" in kinds:
            reach = w.sliding_window - 1
            ring = self._layouts[stack, "sliding_attention"][1] // 16
            count = SEGMENT_ROWS.get(n)
            if count is not None:
                plan = {
                    "segments": (count, -(-reach // count)),
                    "plane": io["planes"],
                    "lo": (t - reach).clamp_min(0),
                    "hi": io["length"] + torch.zeros_like(t),
                    "base": self._constant(
                        ("base", n, count), lambda: self._arange[:n] * count
                    ),
                    "live": self._constant(
                        ("live", n, count), lambda: torch.full_like(t, count)
                    ),
                    "none": self._constant(("none", n), lambda: t.new_zeros((0, n))),
                }
            else:
                keys = -(-reach // (16 * WINDOW_COPIES)) * 16
                plan = kernels.plan_reads(
                    *rows, rule=kernels.SLIDING, plane_count=planes, reach=reach,
                    copies=WINDOW_COPIES, keys=keys, per_plane=ring or per_plane,
                    ring_blocks=ring,
                )
                plan.update(copies=WINDOW_COPIES, keys=keys, splits=1)
            plan["mask"] = io["path_mask"]
            if self.max_path > w.sliding_window:
                position = io["length"] + self._arange[: io["path"].shape[1]]
                plan["mask"] = plan["mask"] & (
                    position[None] > t[:, None] - w.sliding_window
                )
            plans["sliding_attention"] = plan
        if "full_attention" in kinds:
            shared = self._shared(stack, "full_attention", -1)
            sources = [final] if shared else list(range(planes))
            splits = capacity // SPLIT

            _, ring, _, _ = self._layouts[stack, "full_attention"]

            def table():
                # A source plane's first block: a native plane, or a shared bank.
                first = [
                    source * (ring or self.max_len) // 16 + split * (SPLIT // 16)
                    for split in range(splits)
                    for source in sources
                ]
                first = torch.tensor(first, device=self.device)[:, None]
                return (first + self._arange[None, : SPLIT // 16]).to(torch.int32)

            plan = kernels.plan_reads(
                *rows, rule=kernels.SHARED if shared else kernels.NATIVE,
                plane_count=planes, reach=self.window - 1 if shared else 0,
                copies=splits, keys=SPLIT, per_plane=per_plane,
                table=self._constant(("table", stack, capacity), table),
            )
            plan.update(copies=splits, keys=SPLIT, splits=0, mask=io["path_mask"])
            plans["full_attention"] = plan
        return plans

    def _paged_layer(self, stack, ordinal, q, io, plan):
        prefix_k = self._prefix_k[stack, ordinal]
        prefix_v = self._prefix_v[stack, ordinal]
        tree_k, tree_v = self._tree[stack][0, ordinal], self._tree[stack][1, ordinal]
        _, ring, bank, _ = self._layouts[stack, self._kinds[stack][ordinal]]
        kernel = functools.partial(
            self._kernels.range_path_attention,
            ring=ring, bank=bank, capacity=self.max_len,
        )
        store = (prefix_k, prefix_v, tree_k, tree_v)
        n, heads, dim = q.shape
        if "segments" in plan:
            count, none = plan["segments"][0], plan["none"]
            partials = kernel(
                q, *store, plan["plane"], plan["lo"], plan["hi"], io["path"],
                plan["mask"], segments=plan["segments"],
            )
            return kernel(
                q, *store, none, none, none, io["path"][:, :0], plan["mask"][:, :0],
                partials=(*partials, plan["base"], plan["live"], 1), parts=count,
            )[0]
        copies = plan["copies"]
        if plan["gather"] is not None:
            queries = q.index_select(0, plan["gather"])
        elif copies > 1:
            queries = q[None].expand(copies, -1, -1, -1).reshape(-1, heads, dim)
        else:
            queries = q
        blocks = (-1, 16, self.w.kv_heads, dim)
        # FlashAttention 2 treats a longest query sequence of one as one query per
        # sequence, which a plane without rows breaks.
        longest = max(n, 2) if self.fa_version == 2 and self._planes[stack] > 1 else n
        old, old_lse = self._flash_attention(
            q=queries,
            k=prefix_k.view(blocks),
            v=prefix_v.view(blocks),
            max_seqlen_q=longest,
            cu_seqlens_q=plan["cu"],
            max_seqlen_k=plan["keys"],
            seqused_k=plan["used"],
            block_table=plan["table"],
            causal=False,
            softmax_scale=dim**-0.5,
            return_softmax_lse=True,
            num_splits=plan["splits"],
            fa_version=self.fa_version,
        )
        return kernel(
            q, *store, plan["plane"], plan["lo"], plan["hi"], io["path"],
            plan["mask"], partials=(old, old_lse, plan["base"], plan["live"], n),
            parts=copies,
        )[0]

    def _attention_layer(self, stack, ordinal, q, io, plan, capacity):
        if self.attention == "paged":
            return self._paged_layer(stack, ordinal, q, io, plan)
        kv = (self.w.kv_heads, self.w.head_dim)
        prefix_k = self._prefix_k[stack, ordinal]
        prefix_v = self._prefix_v[stack, ordinal]
        tree_k, tree_v = self._tree[stack][0, ordinal], self._tree[stack][1, ordinal]
        kind = self._kinds[stack][ordinal]
        planes = self._planes[stack]
        rows = io["planes"][:, None]
        row_k, row_v = tree_k[rows, io["path"]], tree_v[rows, io["path"]]
        if plan["mode"] == "sliding":
            every = self._arange[:planes, None]
            index = self._rows_of(stack, kind, every, plan["index"][None]).reshape(-1)
            shared_k, shared_v = prefix_k[index], prefix_v[index]
        elif plan["mode"] == "shared":
            first = int(self._rows_of(stack, kind, planes - 1, 0))
            shared_k = prefix_k.narrow(0, first, capacity)
            shared_v = prefix_v.narrow(0, first, capacity)
            band = self._rows_of(stack, kind, rows, plan["index"])
            row_k = torch.cat((prefix_k[band], row_k), 1)
            row_v = torch.cat((prefix_v[band], row_v), 1)
        else:
            shared_k = prefix_k.view(planes, self.max_len, *kv)[:, :capacity]
            shared_v = prefix_v.view(planes, self.max_len, *kv)[:, :capacity]
            shared_k, shared_v = shared_k.reshape(-1, *kv), shared_v.reshape(-1, *kv)
        return self._attend(
            q, shared_k, shared_v, plan["shared_mask"], row_k, row_v, plan["row_mask"],
            self.w.kv_heads,
        )

    def _blocks(self, stack, hidden, io, capacity):
        w = self.w
        positions = io["positions"]
        rope = {
            kind: (cos[positions].float()[:, None], sin[positions].float()[:, None])
            for kind, (cos, sin) in self._rope.items()
        }
        taps = []
        tree = self._tree[stack]
        plans = self._plan(stack, io, capacity)
        for ordinal, sublayer in enumerate(self._attention[stack]):
            layer, mlp = w.layers[sublayer], w.layers[sublayer + 1]
            q, k, v, gate = self._pre(
                hidden, *rope[layer["kind"]], layer["qkv"], layer["g"], layer["sqk2"],
                w.heads, w.kv_heads, w.head_dim,
            )
            if self.attention == "paged":
                self._kernels.store_rows(
                    k, v, tree[0, ordinal], tree[1, ordinal], io["planes"], io["slots"]
                )
            else:
                tree[0, ordinal][io["planes"], io["slots"]] = k
                tree[1, ordinal][io["planes"], io["slots"]] = v
            attended = self._attention_layer(
                stack, ordinal, q, io, plans[layer["kind"]], capacity
            )
            hidden = self._post(
                hidden, attended, gate, layer["gate_scale"], layer["o"], layer["alpha"],
                mlp["gate_up"], mlp["suv_gate"], mlp["suv_up"], mlp["down"],
                mlp["alpha"],
            )
            if sublayer + 1 in self._taps:
                taps.append(hidden)
        return hidden, taps

    def _body(self, phase, io, capacity):
        """One phase over the rows of ``io``. Inputs and results other than the
        readouts are node state on the device; a body never reads a row it writes,
        so running it again leaves the same state."""
        meta = io["meta"]
        io = self._unpack(io)
        slots, rows = io["slots"], self.max_nodes + 1
        if phase == "in":  # the last metadata column holds the tokens
            x, taps = self._blocks("in", self.embed(meta[:, -1]), io, capacity)
            self._banks[0, slots] = x
            self._banks[1, slots] = taps[0]
            return {}
        if phase == "rec":
            x, loop = self._banks[0, slots], io["planes"]
            # A fresh row starts from the input block's output.
            before = self._states[(loop - 1).clamp_min(0) * rows + slots]
            state = torch.where(io["fresh"][:, None], x, before)
            hidden, taps = self._blocks("rec", justnorm(state + x), io, capacity)
            self._states[loop * rows + slots] = justnorm(hidden)
            features = torch.cat([self._banks[1, slots], *taps], -1)
            self._features_low[self._feature_base[loop] + slots] = features
            return {}
        state = self._states[self._state_base[io["planes"]] + slots]
        hidden, taps = self._blocks("out", state, io, capacity)
        self._features_high[io["planes"] * rows + slots] = taps[0]
        logits = self.logits(hidden)
        return {"logits": logits, "argmax": logits.argmax(-1)}

    def _execute(self, phase, entry):
        if not self.use_cuda_graph:
            return self._body(phase, entry["io"], entry["capacity"])
        if entry["graph"] is None:
            # Phase bodies are idempotent, so the warm-up runs (which also trigger
            # compilation) leave the same KV and banks as the replay.
            begun = time.perf_counter()
            for _ in range(2):
                self._body(phase, entry["io"], entry["capacity"])
            torch.cuda.synchronize(self.device)
            if phase not in self._pools:
                self._pools[phase] = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pools[phase]):
                entry["out"] = self._body(phase, entry["io"], entry["capacity"])
            entry["graph"] = graph
            self.stats["graph_captures"] += 1
            self.stats["capture_seconds"] += time.perf_counter() - begun
        started = self._clock()
        entry["graph"].replay()
        self._clock(f"seconds_{phase}", started)
        self.stats["graph_replays"] += 1
        return entry["out"]

    def _capacity(self, length):
        """Prefix capacity bucket of a call over ``length`` committed positions."""
        if self.attention == "dense":
            return _bucket(length, PREFIX_BUCKETS, self.max_len)
        # Whole splits covering the prefix, doubled per bucket.
        splits = _bucket(-(-length // SPLIT), SPLIT_BUCKETS)
        return min(splits * SPLIT, self.max_len)

    def prewarm_graphs(self, rows, width=None, length=None):
        """Capture every graph a schedule can reach before requests are timed.

        Covers each phase for the row buckets up to ``rows``, the path buckets up
        to ``width`` and the prefix buckets up to ``length`` (defaults: the widest
        path and the context capacity). Captures use padding rows only, which
        read their own scratch key and write the scratch slot, so runner state is
        unchanged.
        """
        report = {"enabled": False, "graphs": 0, "seconds": 0.0}
        if not self.use_cuda_graph or rows < 1:
            return report
        if self.length or self._nodes:
            raise ValueError("Graphs are prepared on an empty runner")
        width = self.max_path if width is None else min(width, self.max_path)
        length = self.max_len if length is None else min(length, self.max_len)
        row_buckets = [b for b in ROW_BUCKETS if b < rows]
        row_buckets.append(_bucket(rows, ROW_BUCKETS))
        widths = [b for b in PATH_BUCKETS if b < width]
        widths.append(_bucket(width, PATH_BUCKETS))
        capacities = {self._capacity(n) for n in range(1, length + 1, SPLIT)}
        capacities = sorted(capacities | {self._capacity(length)})
        before = self.stats["graph_captures"]
        begun = time.perf_counter()
        empty = torch.zeros((0, self.H), device=self.device, dtype=self.dtype)
        for capacity in capacities:
            for path in widths:
                for count in row_buckets:
                    for phase in ("in", "rec", "out"):
                        entry = self._entry(phase, count, path, capacity)
                        if entry["graph"] is None:
                            self._fill(entry, [], [], empty)
                            self._execute(phase, entry)
        torch.cuda.synchronize(self.device)
        report.update(
            enabled=True,
            graphs=self.stats["graph_captures"] - before,
            seconds=time.perf_counter() - begun,
            rows=row_buckets,
            widths=widths,
            capacities=capacities,
        )
        return report

    def _clock(self, name=None, started=None):
        if not self.profile:
            return None
        torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        if name is not None:
            self.stats[name] = self.stats.get(name, 0.0) + now - started
        return now

    @torch.no_grad()
    def advance_nodes(self, node_ids, loops):
        """Advance every unfinished node by one loop from the state kept here.

        ``node_ids`` fixes the row order of the call and ``loops`` are the loops the
        caller believes the nodes have finished. Returns the rows (indices into
        ``node_ids``) that reached a readout depth, their argmax tokens and their
        logits; both tensors are graph outputs, valid until the next call. Node
        features are read with ``node_features``, ``span_features`` and ``commit``.
        """
        started = self._clock()
        slots, table = self._check(node_ids, loops)
        n, length = len(node_ids), self.length
        depth, position = table[:, 2], table[:, 1]
        width = _bucket(int(position.max()) - length + 1, PATH_BUCKETS)
        capacity = self._capacity(length)
        fresh_rows = table[:, 3].nonzero()[0]
        readout_rows = self._readout[depth + 1].nonzero()[0]
        new, read = fresh_rows.size, readout_rows.size
        self._settle()
        self._set_length()
        # Rows of the recurrent block first; the other phases take subsets of them.
        padding = self._padding(width)
        rec = self._entry("rec", _bucket(n, ROW_BUCKETS), width, capacity)
        host = rec["host"]
        if 2 * width + 4 == table.shape[1]:
            host[:n] = table
        else:  # the first ``width`` entries of the path and of its mask
            mask = self._path.shape[1] + 4
            host[:n, : 4 + width] = table[:, : 4 + width]
            host[:n, 4 + width :] = table[:, mask : mask + width]
        if n < len(host):
            host[n:] = padding
        try:
            if new:
                entry = self._entry("in", _bucket(new, ROW_BUCKETS), width, capacity)
                staged = entry["host"]
                staged[:new, :-1] = host[fresh_rows]
                staged[:new, 3] = 0
                staged[:new, -1] = self._token[slots[fresh_rows]]
                if new < len(staged):
                    staged[new:, :-1] = padding
                self._load(entry)
                self._execute("in", entry)
            self._load(rec)
            self._execute("rec", rec)
            argmax = logits = None
            if read:
                entry = self._entry("out", _bucket(read, ROW_BUCKETS), width, capacity)
                staged = entry["host"]
                staged[:read] = host[readout_rows]
                staged[:read, 2] = self._plane[depth[readout_rows] + 1]
                staged[:read, 3] = 0
                if read < len(staged):
                    staged[read:] = padding
                self._load(entry)
                out = self._execute("out", entry)
                if self.use_cuda_graph:
                    views = entry["views"].get(read)
                    if views is None:
                        views = (out["argmax"][:read], out["logits"][:read])
                        entry["views"][read] = views
                    argmax, logits = views
                else:  # eager outputs are new tensors every call
                    argmax, logits = out["argmax"][:read], out["logits"][:read]
        finally:
            # Also after a failed phase: its queued copies must finish before
            # the next call rewrites the host buffers.
            self._loaded()
        self._complete(slots, depth, new, read)
        self._clock("seconds_call", started)
        return readout_rows.tolist(), argmax, logits

    @torch.no_grad()
    def advance_readout(self, spans):
        """``advance_nodes`` behind the span interface: the caller's hidden states
        replace the kept ones and the results are copies it owns. The input graph
        embeds a fresh node's registered token itself, so a fresh span must carry
        that embedding."""
        node_ids = [span.node_id for span in spans]
        loops = [span.loop for span in spans]
        slots, table = self._check(node_ids, loops)
        depth, rows = table[:, 2], self.max_nodes + 1
        hidden = torch.stack([span.hidden.reshape(-1) for span in spans])
        fresh = table[:, 3].nonzero()[0]
        if fresh.size:
            embedded = self.embed(self._index(self._token[slots[fresh]]))
            given = hidden.index_select(0, self._index(fresh)).to(embedded.dtype)
            if not torch.equal(given, embedded):
                raise ValueError(
                    "A fresh span enters with the embedding of its registered token"
                )
        # The state after loop d is row [d - 1, slot]; a fresh row's is not read.
        self._states.index_copy_(
            0, self._index(np.maximum(depth - 1, 0) * rows + slots), hidden
        )
        readout, _, logits = self.advance_nodes(node_ids, loops)
        state = self._states.index_select(0, self._index(depth * rows + slots))
        features = self._features.index_select(
            0, self._index(self._feature_rows[depth] + slots)
        )
        results = [
            LoopResult(state[i : i + 1], features[i : i + 1]) for i in range(len(spans))
        ]
        return results, logits, [node_ids[row] for row in readout]
