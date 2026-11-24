# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Tree runner of Inkling 2 for the 2DSD executor: the reference path.

A node of the token tree is one token. Each of its layer visits is the reference
forward's (``NativeInkling2._layer``) on a plane assembled for that node: the
keys, values and short-conv inputs of the verified prefix followed by those of
the node's uncommitted ancestors. A node therefore computes exactly what
incremental decoding of its own path computes, and the runner adds only the
bookkeeping of a tree: where a node's context comes from, which rows it leaves
behind, and how a verified node joins the prefix.

Planes are those of the reference cache: ``(layer, 0)`` for a prelude layer,
``(layer, pass)`` for a core layer and ``(layer, depth)`` for a coda layer run
on the state after ``depth`` passes. A depth-d readout is the coda on the node's
pass-d state in the depth-d planes; verification reads depth R.

Each pass restarts from the node's prelude state and receives the previous pass
as a bank of its core blocks, so a node carries its prelude state and one bank
between calls. Rows of one call may be at different passes; a node at pass p
needs its ancestors' pass-p rows, which exist because an ancestor is never
behind its descendants.

Last-loop KV sharing (``window`` W, paper Eq. 6) changes what a plane reads on
the layers that share: a query at position t in a plane other than the final one
reads the final plane's keys and values for positions j <= t - W and its own
plane's for nearer positions and for its in-flight ancestors. Only full-attention
layers of the core and the coda may share; their sliding layers are bounded by
the window and the prelude has one plane. Short-conv state is never shared. The
prompt is prefilled without sharing. Depth-R coda rows read only the depth-R
plane, so sharing on the coda changes early readouts, not the verified output.

This path is written for agreement with the reference, not for speed: one token
at a time, with the context copied for every visit.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from vllm.model_executor.models.inkling2_reference import (
    Inkling2Cache,
    Inkling2Plane,
    NativeInkling2,
    drafter_feature_layers,
)


@dataclass
class _Row:
    """What one token leaves in one plane: its key and value rows [1, kv, D]
    and its input to every short conv [1, C]."""

    keys: torch.Tensor
    values: torch.Tensor
    conv: dict


@dataclass
class _Node:
    token: int
    parent: int | None  # None: the parent is the last prefix token
    position: int
    depth: int = 0  # completed passes
    prelude: tuple | None = None  # (blocks, partial) after the prelude
    bank: list | None = None  # core blocks of the last completed pass
    rows: dict = field(default_factory=dict)  # (layer, plane index) -> _Row
    taps: dict = field(default_factory=dict)  # (layer, plane index) -> [H]
    features: dict = field(default_factory=dict)  # readout depth -> [F]
    children: set = field(default_factory=set)


class _NodeRun:
    """The ``run`` argument of ``NativeInkling2._layer`` for one node: its
    position, and planes that hold the node's context when a visit starts and
    its own row at their end when the visit returns."""

    def __init__(self, runner, node, chain):
        self.runner, self.node, self.chain = runner, node, chain
        self.start = node.position
        self.positions = torch.tensor([node.position], device=runner.device)
        self.views = {}

    def plane(self, layer, index):
        runner = self.runner
        base = runner.prefix.planes[(layer, index)]
        rows = [ancestor.rows[(layer, index)] for ancestor in self.chain]
        keys, values = base.keys, base.values
        if runner.shares(layer, index):
            cut = self.node.position - runner.window + 1
            if cut > runner.prefix.length:
                raise RuntimeError("The in-flight path exceeds the sharing window")
            if cut > 0:
                final = runner.prefix.planes[(layer, runner.R)]
                keys = torch.cat((final.keys[:cut], base.keys[cut:]))
                values = torch.cat((final.values[:cut], base.values[cut:]))
        keys = torch.cat((keys, *[row.keys for row in rows]))
        values = torch.cat((values, *[row.values for row in rows]))
        conv = {}
        for stream, history in base.conv.items():
            joined = torch.cat((history, *[row.conv[stream] for row in rows]))
            conv[stream] = joined[joined.shape[0] - history.shape[0] :]
        view = self.views[(layer, index)] = Inkling2Plane(
            keys, values, base.first, conv
        )
        return view

    def store(self, layer, index):
        view = self.views.pop((layer, index))
        self.node.rows[(layer, index)] = _Row(
            view.keys[-1:].clone(),
            view.values[-1:].clone(),
            {stream: history[-1:].clone() for stream, history in view.conv.items()},
        )


class Inkling2TreeRunner:
    """Node-state runner with fused readouts over a ``NativeInkling2`` model.

    ``readout_depths`` are the loop depths the schedule reads out (they include
    R). ``window`` enables last-loop KV sharing on ``share_layers`` (default:
    the full-attention layers of the core and the coda). A node's features at a
    readout depth are the inputs of ``feature_layers``: prelude layers once,
    core layers in that pass, coda layers in the coda on that pass's state
    (default ``drafter_feature_layers``). ``final_feature`` appends the depth's
    final-norm state.
    """

    fused_readout = True
    keeps_node_state = True

    def __init__(
        self,
        model: NativeInkling2,
        readout_depths,
        *,
        loops=None,
        window=None,
        share_layers=None,
        feature_layers=None,
        max_len=8192,
        max_nodes=64,
        final_feature=False,
    ):
        config = model.config
        self.model, self.config = model, config
        self.R = config.num_passes if loops is None else int(loops)
        self.readout_depths = frozenset(int(depth) for depth in readout_depths)
        if not self.readout_depths or self.R not in self.readout_depths:
            raise ValueError("Readout depths include the final loop")
        if min(self.readout_depths) < 1 or max(self.readout_depths) > self.R:
            raise ValueError("Readout depths lie in [1, R]")
        if config.sliding_window_size < 2 or config.sconv_kernel_size < 2:
            raise ValueError("The tree runner needs a window and a conv kernel of 2+")
        core = config.core_layers
        self.core, self.layers = core, config.num_hidden_layers
        full = [
            layer
            for layer in range(core.start, self.layers)
            if not config.is_local(layer)
        ]
        self.window = None if window is None else int(window)
        self.share_layers = frozenset(full if share_layers is None else share_layers)
        if self.window is not None:
            if self.window < 1:
                raise ValueError("The sharing window is at least one position")
            if not self.share_layers <= set(full):
                raise ValueError(
                    "Only full-attention layers of the core and the coda share"
                )
        if feature_layers is None:
            feature_layers = drafter_feature_layers(config.num_hidden_layers)
        self.feature_layers = tuple(feature_layers)
        if (len(set(self.feature_layers)) != len(self.feature_layers)
                or any(not 0 <= layer < config.num_hidden_layers for layer in self.feature_layers)):
            raise ValueError("Feature layers are distinct layers of the model")
        self.final_feature = bool(final_feature)
        self.H = config.hidden_size
        self.nf = len(self.feature_layers) + self.final_feature
        self.max_len, self.max_nodes = int(max_len), int(max_nodes)
        self.device, self.dtype = model.device, model.precision.dtype
        # A call runs the prelude for its new nodes, one core pass for every node
        # (cells) and the coda for the nodes that reach a readout depth.
        self.stats = {
            "calls": 0,
            "cells": 0,
            "prelude_calls": 0,
            "prelude_rows": 0,
            "readout_calls": 0,
            "readout_rows": 0,
            "registered_nodes": 0,
            "committed_nodes": 0,
            "released_nodes": 0,
            "peak_live_nodes": 0,
        }
        self.reset()

    @property
    def prefix_length(self):
        return self.prefix.length

    def reset(self):
        self.prefix = Inkling2Cache()
        self.nodes = {}

    def shares(self, layer, index):
        """Whether a query in plane ``index`` of ``layer`` reads old positions
        from the final plane."""
        return (
            self.window is not None and index != self.R and layer in self.share_layers
        )

    # Prompt and readout head.

    def prefill(self, token_ids):
        """Run the prompt without sharing; its planes become the prefix."""
        self.reset()
        core = self.core
        visits = [
            layer if layer < core.start else layer + (self.R - 1) * len(core)
            for layer in self.feature_layers
        ]
        out = self.model.forward(
            token_ids,
            num_passes=self.R,
            depths=tuple(sorted(self.readout_depths)),
            taps=visits,
            cache=self.prefix,
            pass_end=False,
        )
        hidden = out.readouts[self.R].hidden
        features = torch.cat([*(out.features[visit] for visit in visits),
                              *([hidden] if self.final_feature else [])], -1)
        return SimpleNamespace(hidden=hidden, features=features)

    def logits(self, hidden):
        return self.model.readout_logits(hidden)

    def raw_logits(self, hidden):
        """Logits before the softcap, for a drafter that adds its own terms."""
        model = self.model
        if model._unembedding is None:
            model._unembedding = model.checkpoint.unembedding.to(model.precision.dtype)
        return hidden.to(model.precision.dtype) @ model._unembedding.T

    def embed(self, token_ids):
        return self.model.embed_tokens(token_ids)

    # Tree.

    def ensure_capacity(self, nodes):
        self.max_nodes = max(self.max_nodes, int(nodes))

    def register_many(self, nodes):
        """Add nodes (node id, parent id or None for the prefix, token); a
        parent may be an earlier entry of the same call. Nothing is added
        unless every node is valid."""
        positions = {}
        for node_id, parent_id, _ in nodes:
            if node_id in self.nodes or node_id in positions:
                raise RuntimeError("A node is registered once")
            if parent_id is None:
                position = self.prefix_length
            elif parent_id in positions:
                position = positions[parent_id] + 1
            else:
                position = self.nodes[parent_id].position + 1
            if position >= self.max_len:
                raise RuntimeError("Node position exceeds the context length")
            if self.window is not None and position - self.window >= self.prefix_length:
                raise RuntimeError("The in-flight path exceeds the sharing window")
            positions[node_id] = position
        for node_id, parent_id, token in nodes:
            self.nodes[node_id] = _Node(int(token), parent_id, positions[node_id])
            if parent_id is not None:
                self.nodes[parent_id].children.add(node_id)
        self.stats["registered_nodes"] += len(nodes)
        self.stats["peak_live_nodes"] = max(
            self.stats["peak_live_nodes"], len(self.nodes)
        )

    def release(self, node_ids):
        removed = set(node_ids)
        for node_id in removed:
            if self.nodes[node_id].children - removed:
                raise RuntimeError("A released node has live children")
        for node_id in removed:
            node = self.nodes.pop(node_id)
            if node.parent in self.nodes:
                self.nodes[node.parent].children.discard(node_id)
        self.stats["released_nodes"] += len(removed)

    def commit(self, node_ids, features=None):
        """Append verified nodes to the prefix, in path order; ``features``
        [n, F] receives their loop-R features. Nothing is appended unless the
        nodes are a finished path that starts after the prefix."""
        self._check_commit(node_ids)
        for row, node_id in enumerate(node_ids):
            node = self.nodes.pop(node_id)
            for (layer, index), own in node.rows.items():
                self._append(layer, index, node.position, own)
            self.prefix.length += 1
            if features is not None:
                features[row].copy_(node.features[self.R])
            for child in node.children:
                self.nodes[child].parent = None
        self.stats["committed_nodes"] += len(node_ids)

    def _check_commit(self, node_ids):
        previous = None
        for offset, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            if node.parent != previous or node.position != self.prefix_length + offset:
                raise RuntimeError("Only the path after the prefix can be committed")
            if node.depth != self.R:
                raise RuntimeError("A committed node has finished every loop")
            previous = node_id

    def _append(self, layer, index, position, own):
        plane = self.prefix.planes[(layer, index)]
        keys = torch.cat((plane.keys, own.keys))
        values = torch.cat((plane.values, own.values))
        if self.config.is_local(layer):
            # The next query, at position + 1, sees window - 1 earlier keys.
            window = self.config.sliding_window_size
            drop = max(0, position + 1 - (window - 1) - plane.first)
            if drop:
                keys, values = keys[drop:].clone(), values[drop:].clone()
                plane.first += drop
        plane.keys, plane.values = keys, values
        for stream, history in plane.conv.items():
            joined = torch.cat((history, own.conv[stream]))
            plane.conv[stream] = joined[joined.shape[0] - history.shape[0] :]

    def _chain(self, node):
        """Uncommitted ancestors of a node, oldest first."""
        chain, parent = [], node.parent
        while parent is not None:
            chain.append(self.nodes[parent])
            parent = chain[-1].parent
        chain.reverse()
        root = chain[0] if chain else node
        if root.position != self.prefix_length:
            raise RuntimeError("The node's branch does not start after the prefix")
        return chain

    # Execution.

    def advance_nodes(self, node_ids, loops):
        """Advance every node by one loop. Returns the rows (indices into
        ``node_ids``) that reached a readout depth, in increasing order, with
        their argmax [rows] and logits [rows, V]. Nothing runs unless the call
        is consistent with the tree."""
        chains = self._check_advance(node_ids, loops)
        self._count(loops)
        order = sorted(
            range(len(node_ids)), key=lambda row: self.nodes[node_ids[row]].position
        )
        hidden = {}
        for row in order:
            node = self.nodes[node_ids[row]]
            run = _NodeRun(self, node, chains[node_ids[row]])
            state = self._pass(run, node)
            if node.depth in self.readout_depths:
                hidden[row] = self._readout(run, node, state)
        rows = sorted(hidden)
        if not rows:
            return rows, None, None
        logits = self.model.readout_logits(torch.cat([hidden[row] for row in rows]))
        return rows, logits.argmax(-1), logits

    def _count(self, loops):
        fresh = sum(loop == 0 for loop in loops)
        read = sum(loop + 1 in self.readout_depths for loop in loops)
        stats = self.stats
        stats["calls"] += 1
        stats["cells"] += len(loops)
        stats["prelude_calls"] += fresh > 0
        stats["prelude_rows"] += fresh
        stats["readout_calls"] += read > 0
        stats["readout_rows"] += read

    def _check_advance(self, node_ids, loops):
        """Uncommitted ancestors (oldest first) of every node of a call that is
        consistent with the tree."""
        called = set(node_ids)
        if len(called) != len(node_ids):
            raise RuntimeError("A node advances once per call")
        chains = {}
        for node_id, loop in zip(node_ids, loops):
            node = self.nodes[node_id]
            if node.depth != loop or node.depth >= self.R:
                raise RuntimeError("Runner loops differ from the schedule")
            chain = chains[node_id] = self._chain(node)
            # An ancestor's rows of this pass exist, or this call writes them first.
            if any(
                ancestor.depth + (ancestor_id in called) <= node.depth
                for ancestor_id, ancestor in self._ids(node, chain)
            ):
                raise RuntimeError("An ancestor is behind its descendant")
        return chains

    @staticmethod
    def _ids(node, chain):
        """(node id, node) of a chain, whose ids are its members' parents'."""
        ids = [member.parent for member in chain[1:]] + [node.parent]
        return zip(ids, chain)

    def _pass(self, run, node):
        model, core = self.model, self.core
        if node.depth == 0:
            token = torch.tensor([node.token], device=self.device)
            blocks, partial = [], model.embed_tokens(token)
            for layer in range(core.start):
                self._tap(node, layer, 0, partial)
                blocks, partial, _ = model._layer(run, layer, 0, blocks, partial, None)
                run.store(layer, 0)
            node.prelude = (blocks, partial)
        depth = node.depth + 1
        blocks, partial = node.prelude
        for layer in core:
            self._tap(node, layer, depth, partial)
            blocks, partial, _ = model._layer(
                run, layer, depth, blocks, partial, node.bank
            )
            run.store(layer, depth)
        node.bank = [*blocks[len(node.prelude[0]) + 1 :], partial]
        node.depth = depth
        return blocks, partial

    def _readout(self, run, node, state):
        """Coda on the state after the node's current pass: final-norm state
        [1, H]; the node's features of this depth are stored."""
        model, depth = self.model, node.depth
        blocks, partial = state
        for layer in range(self.core.stop, self.layers):
            self._tap(node, layer, depth, partial)
            blocks, partial, _ = model._layer(run, layer, depth, blocks, partial, None)
            run.store(layer, depth)
        hidden = model.readout_hidden(blocks, partial)
        taps = [
            node.taps[(layer, 0 if layer < self.core.start else depth)]
            for layer in self.feature_layers
        ]
        node.features[depth] = torch.cat([*taps, *([hidden[0]] if self.final_feature else [])])
        return hidden

    def _tap(self, node, layer, index, partial):
        if layer in self.feature_layers:
            node.taps[(layer, index)] = partial[0]

    def span_features(self, spans):
        """Features [n, F] of each (depth, node ids) span."""
        return tuple(
            torch.stack([self.nodes[node_id].features[depth] for node_id in node_ids])
            for depth, node_ids in spans
        )

    def prewarm_graphs(self, *args, **kwargs):
        return {}
