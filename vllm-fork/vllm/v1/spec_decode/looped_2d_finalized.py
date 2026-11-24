# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Two-dimensional speculative decoding (2DSD) for looped language models.

Block review at an early loop readout with frozen suffixes, deeper-loop
promotion and forking, and immediate drafting of every new anchor. A policy
never deletes candidates before final verification; physical capacity belongs
to the target runner, not to a probability ranking.

Greedy decoding compares readout argmaxes. Temperature sampling is the
probability version of the same rules (recursive rejection sampling, paper
Sec. 4.1 and App. B): every child keeps the proposal it was drawn from and the
last loop verifies main, frozen and fork candidates in that order. Drafter
tokens are argmax drafts, so their proposal is one-hot.
"""

import random
import time
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _token_ids_tensor(token_ids, device):
    target = torch.device(device)
    if target.type != "cuda":
        return torch.tensor(token_ids, device=target, dtype=torch.long)
    # copy_kernel_cuda records the pinned allocation's stream, preventing
    # allocator reuse before the asynchronous H2D copy has completed.
    return torch.tensor(
        token_ids, device="cpu", dtype=torch.long, pin_memory=True
    ).to(target, non_blocking=True)


@dataclass
class _TreeSpan:
    node_id: int
    loop: int
    hidden: torch.Tensor


@dataclass(eq=False)
class FinalizedNode:
    """One token of the candidate trie; ``depth`` counts its completed loops."""

    node_id: int
    parent: Any
    token: int
    position: int
    depth: int = 0
    hidden: torch.Tensor | None = None
    features: torch.Tensor | None = None
    children: dict[int, Any] = field(default_factory=dict)
    registered: bool = False
    role: str = "main"
    origin: str = "initial"
    growing: bool = True
    ever_frozen: bool = False
    promoted: bool = False
    proposal_depth: int = 0
    readouts: dict[int, int] = field(default_factory=dict)
    frozen_suffix: list[int] = field(default_factory=list)
    loop_features: dict[int, torch.Tensor] = field(default_factory=dict)
    # Sampling only: loop-d readout distributions of this node's next token, and
    # an explicit proposal for children not drawn from a parent readout (forks).
    dists: dict[int, torch.Tensor] = field(default_factory=dict)
    proposal_dist: torch.Tensor | None = None
    # Sampling with prefetch: what a readout or a draw already brought to the host. ``picked`` holds
    # values of ``dists`` by (loop, token), ``anchors`` a token drawn from a loop's distribution without
    # the excluded token, with its probability, as (excluded, token, probability) by loop, ``modes`` the
    # most probable token by loop.
    picked: dict[tuple[int, int], float] = field(default_factory=dict)
    anchors: dict[int, tuple] = field(default_factory=dict)
    modes: dict[int, int] = field(default_factory=dict)
    order: tuple[int, int] = (0, 0)  # (position, node id): the row order of a call


ROLE_ORDER = {"main": 0, "frozen": 1, "fork": 2}
ROW_ORDER = attrgetter("order")


def _normalized(weights, fallback):
    total = weights.sum()
    return torch.where(total > 0, weights / total.clamp_min(1e-30), fallback)


@dataclass
class DraftEvent:
    head: FinalizedNode
    parent_position: int
    parent_features: torch.Tensor
    depth: int
    cause: str


class FinalizedTreeExecutor:
    """Batch-one greedy Stage 1, promotion-only Stage 2a, or full Stage 2b.

    ``k`` includes the readout-selected anchor. Stage 1 with k=1 is depth-axis
    speculation (LoopSpec d1) and never invokes the drafter; Stage 2b with k=1 is
    LoopSpec (d1, d2). Stage 1 with d1=R performs direct final verification, the
    ordinary block-drafting (DFlash) boundary.

    A loop-context drafter reads the in-flight span from the cursor to the
    parent as the parent's loop-d features, so nodes keep their features at
    every drafting depth below R until they commit.
    """

    def __init__(
        self,
        runner,
        drafter=None,
        k=4,
        stage="stage2b",
        d1=1,
        d2=2,
        record_trace=False,
        temperature=0.0,
        record_sampling_check=False,
        stage2_gate="argmax",
        top_p=1.0,
        proposal_top_p=None,
        draft_confidence=None,
        frozen_suffix=True,
        nucleus_width=None,
        prefetch=False,
    ):
        if stage not in ("stage1", "stage2a", "stage2b"):
            raise ValueError("Unknown finalized stage")
        if nucleus_width is not None and (type(nucleus_width) is not int or nucleus_width < 1):
            raise ValueError("nucleus_width is a positive number of tokens")
        # None reads the top-p cut of a target row from vLLM's sort of the vocabulary; a number reads
        # it from that many most probable tokens of the row (_nucleus).
        self.nucleus_width = nucleus_width
        self.nucleus_calls = self.nucleus_sorts = 0
        # Sampling: False reads a probability from the GPU when a review asks for it. True reads, with
        # the readout of a call and in one transfer, what its reviews will ask for (_prefetch); the
        # draws then use the generator in another order, so a seed gives other tokens of the same law.
        self.prefetch = bool(prefetch)
        if not frozen_suffix and (stage != "stage1" or temperature > 0):
            raise ValueError("Only greedy Stage 1 can drop the suffix its review rejects")
        # A suffix rejected at the d1 review stays in flight as a frozen candidate for the last loop
        # (and for Stage 2's promotion). False drops it at the review: its rows leave the later calls.
        self.frozen_suffix = bool(frozen_suffix)
        if temperature < 0:
            raise ValueError("Temperature must be nonnegative")
        proposal_top_p = top_p if proposal_top_p is None else proposal_top_p
        if not (0 < top_p <= 1 and 0 < proposal_top_p <= 1):
            raise ValueError("top_p and proposal_top_p must be in (0, 1]")
        if min(top_p, proposal_top_p) < 1 and temperature == 0:
            raise ValueError("top_p applies only when sampling")
        if stage2_gate != "argmax" and not 0 <= float(stage2_gate) < 1:
            raise ValueError("stage2_gate is 'argmax' or a threshold in [0, 1)")
        if draft_confidence is not None and not 0 <= float(draft_confidence) <= 1:
            raise ValueError("draft_confidence is a probability")
        # DSpark's static confidence threshold: a drafted block ends before its first slot whose
        # predicted acceptance is below it. None keeps every drafted slot.
        self.draft_confidence = None if draft_confidence is None else float(draft_confidence)
        self.stage2_gate = stage2_gate
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.proposal_top_p = float(proposal_top_p)
        self.sampling = self.temperature > 0
        self.record_sampling_check = record_sampling_check and self.sampling
        if not 1 <= k <= 16 or not 1 <= d1 <= runner.R:
            raise ValueError("Require 1 <= k <= 16 and 1 <= d1 <= R")
        if stage != "stage1" and not d1 < d2 < runner.R:
            raise ValueError("Stage 2 requires d1 < d2 < R")
        if k > 1 and drafter is None:
            raise ValueError("Blocks larger than one require a drafter")
        self.runner, self.drafter = runner, drafter
        self.k, self.stage, self.d1, self.d2 = k, stage, d1, d2
        self.record_trace = record_trace
        self._probability_buffers = {}
        if drafter is not None and getattr(drafter, "input_format", None) != "loop-context":
            raise ValueError("The drafter must use the loop-context input format")
        self.loop_context = drafter is not None
        depths = {d1} if stage == "stage1" else {d1, d2}
        self.context_depths = depths - {runner.R} if self.loop_context else set()
        self.readout_depths = frozenset(depths | {runner.R})
        # A fused-readout runner (Dory) computes depth-r logits inside the target
        # call, through an output block with its own KV; head(hidden) is not p_r.
        self.fused_readout = bool(getattr(runner, "fused_readout", False))
        if self.fused_readout and (
            frozenset(runner.readout_depths) != self.readout_depths
        ):
            raise ValueError("Runner readout depths differ from the schedule")
        # Such a runner may also keep each node's recurrent state and feature taps
        # itself (DoryGraphRunner): a call then passes node ids only, and features
        # are read from the runner where a draft or a commit needs them.
        self.node_state = self.fused_readout and bool(
            getattr(runner, "keeps_node_state", False)
        )
        self._reset_candidates()

    def _reset_candidates(self):
        for node in getattr(self, "nodes", {}).values():
            self._retire(node)
        self.nodes, self._next_id = {}, 0
        self.cursor = None
        self.blocks = {}
        self.events = []

    @staticmethod
    def _retire(node):
        # Parent/child cycles can otherwise keep entire CUDA readout batches
        # alive until Python's cyclic collector happens to run.
        node.parent = None
        node.children.clear()
        node.hidden = node.features = None
        node.readouts.clear()
        node.frozen_suffix.clear()
        node.loop_features.clear()
        node.dists.clear()
        node.picked.clear()
        node.anchors.clear()
        node.modes.clear()
        node.proposal_dist = None

    def _child(
        self, parent, token, *, role="main", origin="raw_draft", proposal_depth=0
    ):
        if parent is not None and token in parent.children:
            return parent.children[token]
        position = self.runner.prefix_length if parent is None else parent.position + 1
        node = FinalizedNode(
            self._next_id,
            parent,
            int(token),
            position,
            role=role,
            origin=origin,
            proposal_depth=proposal_depth,
            order=(position, self._next_id),
        )
        self._next_id += 1
        self.nodes[node.node_id] = node
        if parent is not None:
            parent.children[node.token] = node
        return node

    def _drop(self, removed: set[int]) -> None:
        registered = [i for i in removed if self.nodes[i].registered]
        if registered:
            self.runner.release(registered)
        for node_id in removed:
            node = self.nodes[node_id]
            if node.parent is not None:
                node.parent.children.pop(node.token, None)
        for node_id in removed:
            self._retire(self.nodes.pop(node_id))
        self.stats["pruned_nodes"] += len(removed)

    @staticmethod
    def _descendants(node) -> set[int]:
        pending, result = [node], set()
        while pending:
            current = pending.pop()
            result.add(current.node_id)
            pending.extend(current.children.values())
        return result

    @staticmethod
    def _path(node) -> list[FinalizedNode]:
        result = []
        while node is not None:
            result.append(node)
            node = node.parent
        return result[::-1]

    def _register_pending(self):
        if hasattr(self.runner, "ensure_capacity"):
            self.runner.ensure_capacity(len(self.nodes))
        pending = [node for node in self.nodes.values() if not node.registered]
        pending.sort(key=ROW_ORDER)
        if self.node_state:
            self.runner.register_many(
                [
                    (n.node_id, n.parent.node_id if n.parent else None, n.token)
                    for n in pending
                ]
            )
            for node in pending:
                node.registered = True
        else:
            for node in pending:
                self.runner.register(
                    node.node_id,
                    node.parent.node_id if node.parent else None,
                    node.token,
                )
                node.registered = True
        self.stats["max_live_nodes"] = max(
            self.stats["max_live_nodes"], len(self.nodes)
        )

    def _trace(self, event, **values):
        if self.record_trace:
            self.trace.append({"call": self.stats["calls"], "event": event, **values})

    def _queue(self, parent, head, depth, cause):
        if head.position >= self.output_limit:
            return
        if parent.features is None and not self.node_state:
            raise RuntimeError("Draft event lost its conditioning parent feature")
        self.events.append(
            DraftEvent(head, parent.position, parent.features, depth, cause)
        )
        self._trace(
            "anchor", token=head.token, position=head.position, depth=depth, cause=cause
        )

    def _new_anchor(self, parent, depth, cause, role="main"):
        token = parent.readouts[depth]
        if token in parent.children:
            raise RuntimeError("A new anchor would duplicate an existing proposal")
        head = self._child(parent, token, role=role, origin=cause, proposal_depth=depth)
        self._queue(parent, head, depth, cause)
        return head

    def _freeze(self, suffix):
        head = suffix[0]
        head.role = "frozen"
        head.frozen_suffix = [n.node_id for n in suffix]
        for node in suffix:
            node.growing = False
            node.ever_frozen = True
        self.stats["frozen_created"] += 1
        self.stats["frozen_tokens"] += len(suffix)
        self._trace("freeze", tokens=[n.token for n in suffix], position=head.position)

    def _scan(self, head, suffix, depth, cause):
        """Review successors of an already chosen anchor, never the anchor itself."""
        parent, passed = head, 0
        head.growing = True
        for index, child in enumerate(suffix):
            if child.parent is not parent or child.depth < depth:
                raise RuntimeError("Block suffix lost ancestry or its review readout")
            if parent.readouts[depth] != child.token:
                if self.frozen_suffix:
                    self._freeze(suffix[index:])
                else:
                    self._drop(self._descendants(child))
                self._new_anchor(parent, depth, f"{cause}_reject")
                return passed
            child.growing = True
            child.proposal_depth = depth
            child.origin = f"{cause}_accepted"
            parent = child
            passed += 1
        if parent.position + 1 < self.output_limit:
            self._new_anchor(parent, depth, f"{cause}_frontier")
        return passed

    def _stage1(self):
        if self.d1 == self.runner.R:
            return  # Final verification already performed the single R review.
        for head_id, ids in list(self.blocks.items()):
            head = self.nodes.get(head_id)
            if head is None:
                self.blocks.pop(head_id)
                continue
            if head.depth < self.d1:
                continue
            self.blocks.pop(head_id)
            if not head.growing:
                continue
            if head.depth != self.d1:
                raise RuntimeError("A growing block missed its d1 review")
            nodes = [self.nodes[i] for i in ids]
            passed = self._scan(head, nodes[1:], self.d1, "d1")
            self.stats["stage1_blocks"] += 1
            self.stats["stage1_progress_tokens"] += 1 + passed
            self.stats["stage1_pass_lengths"][str(1 + passed)] += 1
            self._trace("stage1", position=head.position, progress=1 + passed)

    def _stage2(self, advanced):
        if self.stage == "stage1":
            return
        handled = set()
        for parent in advanced:
            if (
                parent.depth != self.d2
                or not parent.growing
                or parent.node_id not in self.nodes
                or parent.node_id in handled
            ):
                continue
            main = next((n for n in parent.children.values() if n.role == "main"), None)
            frozen = next(
                (n for n in parent.children.values() if n.role == "frozen"), None
            )
            if main is None:
                continue
            self.stats["stage2_reviews"] += 1
            predicted = parent.readouts[self.d2]
            if predicted == main.token:
                continue
            self.stats["stage2_disagreements"] += 1
            if frozen is not None and predicted == frozen.token:
                ids = list(frozen.frozen_suffix)
                if not ids or ids[0] != frozen.node_id:
                    raise RuntimeError("Promotion has no frozen block suffix")
                suffix = [self.nodes[i] for i in ids]
                if any(n.depth != self.d2 for n in suffix):
                    raise RuntimeError("Promotion suffix is not at d2")
                handled.update(ids)
                frozen.promoted = True
                frozen.frozen_suffix.clear()
                self.stats["promotions"] += 1
                passed = self._scan(frozen, suffix[1:], self.d2, "promotion")
                self.stats["promotion_reused_tokens"] += 1 + passed
                self._trace("promotion", position=frozen.position, tokens=1 + passed)
            elif self.stage == "stage2b" and parent.position + 1 < self.output_limit:
                if predicted in parent.children:
                    raise RuntimeError("A d2 event was executed more than once")
                self._new_anchor(parent, self.d2, "d2_fork", role="fork")
                self.stats["forks"] += 1

    def _verify(self):
        accepted, removed = [], set()
        stopped = False
        while self.cursor.depth == self.runner.R:
            current = self.cursor
            token = current.readouts[self.runner.R]
            chosen = current.children.get(token)
            if chosen is None:
                chosen = self._child(
                    current, token, origin="bonus", proposal_depth=self.runner.R
                )
                source = "bonus"
                self.stats["bonus_tokens"] += 1
            else:
                source = chosen.role
                if (
                    source == "main"
                    and chosen.ever_frozen
                    and chosen.proposal_depth == 0
                ):
                    source = "frozen"
                self.stats["accepted_candidates"] += 1
                if chosen.promoted:
                    self.stats["promoted_heads_committed"] += 1
                if chosen.ever_frozen:
                    self.stats["previously_frozen_tokens_committed"] += 1
            for sibling in current.children.values():
                if sibling is not chosen:
                    removed.update(self._descendants(sibling))
            accepted.append(current)
            self.cursor = chosen
            self.output.append(chosen.token)
            self.stats["committed_decode_tokens"] += 1
            self.stats["commit_sources"][source] += 1
            origins = self.stats["commit_origins"]
            origins[chosen.origin] = origins.get(chosen.origin, 0) + 1
            self._trace(
                "commit",
                token=chosen.token,
                position=chosen.position,
                source=source,
                origin=chosen.origin,
                depth=chosen.depth,
            )
            if chosen.token in self.eos or len(self.output) >= self.max_new:
                stopped = True
                break
            if source == "bonus":
                # Store an owned reference before current is retired below.
                self._queue(current, chosen, self.runner.R, "bonus")
        if accepted:
            self._drop(removed)
            self._commit(accepted)
            self.prefix_tokens.extend(n.token for n in accepted)
            self.cursor.parent = None
            for node in accepted:
                self._retire(self.nodes.pop(node.node_id))
            self.blocks = {
                key: ids for key, ids in self.blocks.items() if key in self.nodes
            }
        return stopped

    def _commit(self, accepted):
        """Commit the verified path; its loop-R features extend the drafter prefix."""
        start = accepted[0].position
        features = self.prefix_features[start : start + len(accepted)]
        node_ids = [n.node_id for n in accepted]
        if self.node_state:
            self.runner.commit(node_ids, features=features)
        else:
            torch.stack([n.features for n in accepted], out=features)
            self.runner.commit(node_ids)

    # Sampling. Each rule below is the probability version of the greedy rule above.

    def _gather(self, items):
        from .looped_sampling import gather_probabilities

        return gather_probabilities(items, self._probability_buffers)

    def _probabilities(self, items):
        """p(token) per item: (node, loop, token) for a readout of the node, or (distribution, token).
        Values that came with a readout or a draw are on the host; the rest take one gather."""
        values = [item[0].picked.get(item[1:]) if len(item) == 3 else None for item in items]
        missing = [i for i, value in enumerate(values) if value is None]
        if missing:
            rest = (items[i] for i in missing)
            read = self._gather([(i[0].dists[i[1]], i[2]) if len(i) == 3 else i for i in rest])
            for i, value in zip(missing, read):
                values[i] = value
        return values

    def _readout(self, logits, nodes):
        """The distributions of a call's readout rows, on their nodes; with prefetch also what the
        reviews of the call read from them."""
        depths = [node.depth for node in nodes]
        if not self.prefetch:
            probs = self._readout_distributions(logits, depths)
        elif self._prefetch(probs := self._readout_distributions(logits, depths, True), nodes, self._short):
            # A row holds less than top_p in its most probable tokens: the sort, and the values again.
            self.nucleus_sorts += 1
            probs = self._sorted_distributions(logits, depths)
            self._prefetch(probs, nodes)
        for i, node in enumerate(nodes):
            node.dists[node.depth] = probs[i]

    def _prefetch(self, probs, nodes, short=None):
        """In one transfer, what the reviews of a call read from its readout rows.

        Each row's probability of its children's tokens: a d1 review and the final verification test
        those. For a row that a review below loop R can stop at (a growing node at d1, a frozen one
        at d2) with at most its drafted child, a token drawn from the row without that child's
        token, with its probability: the anchor the review takes there, after it rejects the child
        or past the last token of a block that passes. The draw of every such row is made before
        the review decides which one it uses; the decision does not read the draws, so the one it
        takes is an exact draw of its row. For Stage 2's argmax gate, the most probable token of
        the rows at d2. ``short`` is the pending flag of _nucleus: True is returned, and nothing
        kept, when it is set.
        """
        R, width = self.runner.R, probs.shape[-1]
        picks = [(i, node, token) for i, node in enumerate(nodes) for token in node.children]
        # A d1 review takes its anchor at a growing node, a promotion's review at a frozen one.
        reviewed = {self.d1: True} if self.stage == "stage1" else {self.d1: True, self.d2: False}
        draws = [(i, node, next(iter(node.children), None)) for i, node in enumerate(nodes)
                 if node.depth < R and len(node.children) < 2 and reviewed.get(node.depth) is node.growing]
        excluded = [(j, token) for j, (_, _, token) in enumerate(draws) if token is not None]
        gate = self.stage != "stage1" and self.stage2_gate == "argmax" and any(
            node.depth == self.d2 for node in nodes
        )
        if not (picks or draws or gate or short is not None):
            return False
        index = _token_ids_tensor(
            [i * width + token for i, _, token in picks]
            + [j * width + token for j, token in excluded]
            + [i for i, _, _ in draws],
            probs.device,
        )
        a, b = len(picks), len(picks) + len(excluded)
        parts = [probs.reshape(-1).index_select(0, index[:a])]
        if gate:
            parts.append(probs.argmax(-1).to(probs.dtype))
        if draws:
            weights = probs.index_select(0, index[b:])
            if excluded:
                weights.view(-1).index_fill_(0, index[a:b], 0)
            tokens = self._draw(weights)
            parts += [tokens.to(probs.dtype), weights.gather(1, tokens.unsqueeze(1)).squeeze(1)]
        if short is not None:
            parts.append(short.to(probs.dtype).reshape(1))
        values = torch.cat(parts).tolist()
        if short is not None and values[-1]:
            return True
        for (_, node, token), value in zip(picks, values):
            node.picked[(node.depth, token)] = value
        offset = a
        if gate:
            for i, node in enumerate(nodes):
                if node.depth == self.d2:
                    node.modes[self.d2] = int(values[offset + i])
            offset += len(nodes)
        for j, (_, node, token) in enumerate(draws):
            node.anchors[node.depth] = (token, int(values[offset + j]), values[offset + len(draws) + j])
        return False

    def _distribution(self, logits, top_p):
        """Readout law: temperature, then vLLM's top-p mask, then softmax.

        vLLM AR samples one row per step through the PyTorch sort path of
        apply_top_k_top_p; target rows (the first token and every loop-R readout)
        use that same path here with top_p (not the Triton kernel vLLM picks for
        eight or more rows), so p_R is AR's law. Shallow readouts only propose,
        so any proposal_top_p stays lossless; 1 skips their vocabulary sort.
        """
        scaled = logits.float() / self.temperature
        if top_p < 1:
            from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

            p = torch.full((len(scaled),), top_p, dtype=torch.float32, device=scaled.device)
            scaled = apply_top_k_top_p_pytorch(scaled, None, p)
        return torch.softmax(scaled, dim=-1)

    def _nucleus(self, probs, rows, defer=False):
        """The top-p cut of ``_distribution`` on rows of probabilities (all rows if ``rows`` is None),
        in place and without a sort of the vocabulary.

        vLLM's rule keeps a token while the mass ranked above it is below top_p. The ``nucleus_width``
        most probable tokens of a row, in order, decide that for every token when they hold top_p
        between them; the row is then those of them that stay, renormalized. Returns False, with
        ``probs`` as it was, when a row holds less than top_p in them: the caller sorts every target
        row of the call.

        It is the sort's cut wherever the two were compared, not by construction. Tokens of equal
        probability on the two sides of the cut are kept in topk's order here and in sort's order
        there, and the two add the masses in another order, so a row whose mass meets top_p within
        rounding can be cut one token apart. FP32 logits of a 200,058-token vocabulary showed neither.

        ``defer`` cuts the rows without waiting for the GPU and returns that test as a tensor: when it
        is set the rows are not distributions, and the caller sorts.
        """
        sub = probs if rows is None else probs.index_select(0, rows)
        values, ids = sub.topk(min(self.nucleus_width, sub.shape[-1]), dim=-1)
        mass = values.cumsum(-1)
        short = (mass[:, -1] < self.top_p).any()
        if not defer and bool(short):
            return False
        values[:, 1:].masked_fill_(mass[:, :-1] >= self.top_p, 0)
        values /= values.sum(-1, keepdim=True)
        if rows is None:
            probs.zero_().scatter_(1, ids, values)
        else:
            probs.index_fill_(0, rows, 0)
            flat = rows.unsqueeze(1) * probs.shape[-1] + ids
            probs.view(-1).scatter_(0, flat.view(-1), values.view(-1))
        return short if defer else True

    def _readout_distributions(self, logits, depths, defer=False):
        """Target law for loop-R rows, proposal law for shallower rows. With ``defer`` a cut from the
        most probable tokens leaves its test in ``_short`` (_nucleus) for the caller to read."""
        R = self.runner.R
        target = [i for i, d in enumerate(depths) if d == R]
        # Every row takes top_p: proposals cut like the target, or a call of target rows only.
        every = self.proposal_top_p == self.top_p or len(target) == len(depths)
        self._short = None
        if self.nucleus_width and self.top_p < 1 and (
            every or (target and self.proposal_top_p == 1)
        ):
            probs = torch.softmax(logits.float() / self.temperature, dim=-1)
            rows = None if every else _token_ids_tensor(target, logits.device)
            self.nucleus_calls += 1
            if defer:
                self._short = self._nucleus(probs, rows, defer=True)
                return probs
            if self._nucleus(probs, rows):
                return probs
            self.nucleus_sorts += 1
        return self._sorted_distributions(logits, depths)

    def _sorted_distributions(self, logits, depths):
        """_readout_distributions with vLLM's sort for the top-p cut."""
        R = self.runner.R
        target = [i for i, d in enumerate(depths) if d == R]
        if self.proposal_top_p == self.top_p or len(target) == len(depths):
            return self._distribution(logits, self.top_p)
        probs = self._distribution(logits, self.proposal_top_p)
        if target:
            index = _token_ids_tensor(target, logits.device)
            probs.index_copy_(0, index, self._distribution(logits.index_select(0, index), self.top_p))
        return probs

    def _draw(self, weights):
        """One category per row of nonnegative weights: argmax(w / E), E ~ Exp(1), the exact
        exponential race of vLLM's sampler."""
        noise = torch.empty_like(weights).exponential_(generator=self.generator)
        return torch.div(weights, noise, out=noise).argmax(-1)

    def _proposal(self, child):
        """The distribution child was drawn from; None is a one-hot argmax draft."""
        if child.proposal_dist is not None:
            return child.proposal_dist
        if child.proposal_depth == 0:
            return None
        return child.parent.dists[child.proposal_depth]

    def _proposal_item(self, child):
        """_proposal as an item of _probabilities for child's own token."""
        if child.proposal_dist is not None:
            return (child.proposal_dist, child.token)
        if child.proposal_depth == 0:
            return None
        return (child.parent, child.proposal_depth, child.token)

    def _draw_anchors(self, specs, depth):
        """One token per (parent, cause, excluded token), drawn from the parent's loop-depth
        distribution without the excluded token, in one batched draw."""
        excluded = [(i, t) for i, (_, _, t) in enumerate(specs) if t is not None]
        weights = (
            specs[0][0].dists[depth].unsqueeze(0)
            if len(specs) == 1 and not excluded
            else torch.stack([parent.dists[depth] for parent, _, _ in specs])
        )
        if excluded:
            rows, cols = zip(*excluded)
            weights[
                _token_ids_tensor(list(rows), weights.device),
                _token_ids_tensor(list(cols), weights.device),
            ] = 0
        return self._draw(weights).tolist()

    def _sample_anchors(self, specs, depth, role="main"):
        """New anchors from (parent, cause, excluded token).

        A rejected drafter token x leaves p ⊖ one-hot(x), p without x. The new
        child is still an exact sample of p, so its proposal is p (§7.2.1). A
        parent whose readout came with such a draw (_prefetch) uses it, once.
        """
        if not specs:
            return []
        tokens, rest = [None] * len(specs), []
        for i, (parent, _, excluded) in enumerate(specs):
            ready = parent.anchors.pop(depth, None)
            if ready is not None and ready[0] == excluded and ready[2] > 0:
                tokens[i] = ready[1]
                parent.picked[(depth, ready[1])] = ready[2]
            else:
                rest.append(i)
        if rest:
            for i, token in zip(rest, self._draw_anchors([specs[i] for i in rest], depth)):
                tokens[i] = token
        heads = []
        for (parent, cause, _), token in zip(specs, tokens):
            if token in parent.children:
                raise RuntimeError("A sampled anchor duplicates an existing proposal")
            head = self._child(parent, token, role=role, origin=cause, proposal_depth=depth)
            self._queue(parent, head, depth, cause)
            heads.append(head)
        return heads

    def _review_sampled(self, chains, depth, cause):
        """Rejection-sample drafter suffixes of already chosen anchors (§5.1, §6.2).

        chains are [head, s1, s2, ...]. With a one-hot draft s_j passes with
        probability p_depth(s_j); the first rejection freezes the rest and draws a
        new anchor from p_depth without s_j. A fully passed chain draws its next
        anchor from the last token's readout. Returns the passed count per chain.
        """
        pairs = [(c[j], c[j + 1]) for c in chains for j in range(len(c) - 1)]
        for parent, child in pairs:
            if child.parent is not parent or child.depth < depth:
                raise RuntimeError("Block suffix lost ancestry or its review readout")
        probabilities = self._probabilities([(p, depth, c.token) for p, c in pairs])
        offset, anchors, passed_counts = 0, [], []
        for chain in chains:
            head, suffix = chain[0], chain[1:]
            head.growing = True
            parent, passed = head, 0
            for index, child in enumerate(suffix):
                if self.rng.random() >= probabilities[offset + index]:
                    self._freeze(suffix[index:])
                    anchors.append((parent, f"{cause}_reject", child.token))
                    break
                child.growing = True
                child.proposal_depth = depth
                child.origin = f"{cause}_accepted"
                parent = child
                passed += 1
            else:
                if parent.position + 1 < self.output_limit:
                    anchors.append((parent, f"{cause}_frontier", None))
            offset += len(suffix)
            passed_counts.append(passed)
        self._sample_anchors(anchors, depth)
        return passed_counts

    def _stage1_sampled(self):
        if self.d1 == self.runner.R:
            return  # Final verification performs the single R review.
        heads, chains = [], []
        for head_id, ids in list(self.blocks.items()):
            head = self.nodes.get(head_id)
            if head is None:
                self.blocks.pop(head_id)
                continue
            if head.depth < self.d1:
                continue
            self.blocks.pop(head_id)
            if not head.growing:
                continue
            if head.depth != self.d1:
                raise RuntimeError("A growing block missed its d1 review")
            heads.append(head)
            chains.append([self.nodes[i] for i in ids])
        for head, passed in zip(heads, self._review_sampled(chains, self.d1, "d1")):
            self.stats["stage1_blocks"] += 1
            self.stats["stage1_progress_tokens"] += 1 + passed
            self.stats["stage1_pass_lengths"][str(1 + passed)] += 1
            self._trace("stage1", position=head.position, progress=1 + passed)

    def _stage2_sampled(self, advanced):
        """Promotion of the frozen x*, else a fork from the d2 residual (§6.1).

        stage2_gate only decides whether a candidate exists, so every setting is
        lossless. The default "argmax" acts when greedy would: promote when argmax
        p_d2 = x*, fork when it is neither y1 nor x*. The LoopSpec gate (0) opens on
        any decrease of p_d2(y1) and, with a k-token block per fork, made T=1 Stage
        2b slower than Stage 1 on Ouro 2.6B; a threshold tau acts when the d2
        estimate of y1's rejection, 1 - p_d2(y1) / p_d1(y1), is at least tau. The
        fork token is drawn from p_d2 ⊖ p_d1 without y1 and x*.
        """
        if self.stage == "stage1":
            return
        events = []
        for parent in advanced:
            if (
                parent.depth != self.d2
                or not parent.growing
                or parent.node_id not in self.nodes
            ):
                continue
            main = next((n for n in parent.children.values() if n.role == "main"), None)
            frozen = next(
                (n for n in parent.children.values() if n.role == "frozen"), None
            )
            if main is not None:
                events.append((parent, main, frozen))
        if not events:
            return
        self.stats["stage2_reviews"] += len(events)
        if self.stage2_gate == "argmax":  # decisions read only argmax p_d2
            modes = [p.modes.get(self.d2) for p, _, _ in events]
            if None in modes:
                distributions = (
                    events[0][0].dists[self.d2].unsqueeze(0)
                    if len(events) == 1
                    else torch.stack([p.dists[self.d2] for p, _, _ in events])
                )
                modes = distributions.argmax(-1).tolist()
            values = iter(())
        else:
            items = []
            for parent, main, frozen in events:
                items += [(parent.dists[self.d2], main.token), (parent.dists[self.d1], main.token)]
                if frozen is not None:
                    items.append((parent.dists[self.d2], frozen.token))
            values, modes = iter(self._gather(items)), [None] * len(events)
        promotions, forks = [], []
        for (parent, main, frozen), mode in zip(events, modes):
            if self.stage2_gate == "argmax":
                promote = frozen is not None and mode == frozen.token
                fork = not promote and mode != main.token
            else:
                deep, shallow = next(values), next(values)
                frozen_deep = next(values) if frozen is not None else None
                tau = float(self.stage2_gate)
                gate = deep < shallow if tau == 0 else deep <= (1 - tau) * shallow
                promote = frozen_deep is not None and frozen_deep > deep and (tau == 0 or gate)
                fork = not promote and gate
            if promote:
                promotions.append(frozen)
            elif fork:
                self.stats["stage2_disagreements"] += 1
                if self.stage == "stage2b" and parent.position + 1 < self.output_limit:
                    forks.append((parent, main, frozen))
        self.stats["stage2_disagreements"] += len(promotions)
        chains = []
        for frozen in promotions:
            ids = list(frozen.frozen_suffix)
            if not ids or ids[0] != frozen.node_id:
                raise RuntimeError("Promotion has no frozen block suffix")
            suffix = [self.nodes[i] for i in ids]
            if any(n.depth != self.d2 for n in suffix):
                raise RuntimeError("Promotion suffix is not at d2")
            frozen.promoted = True
            frozen.frozen_suffix.clear()
            self.stats["promotions"] += 1
            chains.append(suffix)
        for frozen, passed in zip(
            promotions, self._review_sampled(chains, self.d2, "promotion")
        ):
            self.stats["promotion_reused_tokens"] += 1 + passed
            self._trace("promotion", position=frozen.position, tokens=1 + passed)
        if not forks:
            return
        # z ~ (p_d2 ⊖ p_d1) ⊖ (q_D ⊖ p_d1): the residual without the frozen token. y1
        # has no residual mass when p_d2(y1) < p_d1(y1); the argmax gate removes it too.
        residual = (
            torch.stack([p.dists[self.d2] for p, _, _ in forks])
            - torch.stack([p.dists[self.d1] for p, _, _ in forks])
        ).clamp_min_(0)
        excluded = [(i, m.token) for i, (_, m, _) in enumerate(forks)]
        excluded += [(i, f.token) for i, (_, _, f) in enumerate(forks) if f is not None]
        if excluded:
            rows, cols = zip(*excluded)
            residual[
                _token_ids_tensor(list(rows), residual.device),
                _token_ids_tensor(list(cols), residual.device),
            ] = 0
        totals = residual.sum(-1)
        # An empty exponential race returns token zero; that row is discarded
        # below. It consumes the same noise without materializing a placeholder.
        tokens = self._draw(residual)
        proposals = residual / totals.clamp_min(1e-30)[:, None]
        drawn = torch.cat([totals, tokens.to(totals.dtype)]).tolist()
        for i, (parent, _, _) in enumerate(forks):
            if drawn[i] <= 0:
                continue  # The only residual mass was the main or frozen token.
            token = int(drawn[len(forks) + i])
            if token in parent.children:
                raise RuntimeError("A d2 fork duplicates an existing proposal")
            head = self._child(
                parent, token, role="fork", origin="d2_fork", proposal_depth=self.d2
            )
            head.proposal_dist = proposals[i]
            self._queue(parent, head, self.d2, "d2_fork")
            self.stats["forks"] += 1

    def _main_ratios(self):
        """t(c) / g(c) of the main child along the chain of loop-R nodes from the cursor, in one
        transfer. Usually the main child is accepted and the chain commits without a readback per
        token; later candidates of a rejected node are read on demand."""
        chain, node = [], self.cursor
        while node.depth == self.runner.R:
            main = next((c for c in node.children.values() if c.role == "main"), None)
            if main is None:
                break
            chain.append((node, main))
            node = main
        if not chain:
            return {}
        items = [(n, self.runner.R, m.token) for n, m in chain]
        proposals = [self._proposal_item(m) for _, m in chain]
        values = self._probabilities(items + [g for g in proposals if g is not None])
        ratios, offset = {}, len(chain)
        for (n, _), g, t in zip(chain, proposals, values):
            if g is None:
                ratios[n.node_id] = t
            else:
                if values[offset] <= 0:
                    raise RuntimeError("A candidate has no mass in its own proposal")
                ratios[n.node_id] = t / values[offset]
                offset += 1
        return ratios

    def _verify_sampled(self):
        """Recursive rejection sampling at loop R in role order main, frozen, fork (§7.2).

        Candidate c with proposal g passes with min(1, t(c) / g(c)); a rejection
        leaves t ⊖ g, or t without c for a one-hot draft. All rejected: the bonus
        token is drawn from the last target.
        """
        accepted, removed = [], set()
        stopped = False
        R = self.runner.R
        main_ratios = self._main_ratios()
        while self.cursor.depth == R:
            current = self.cursor
            target = original = current.dists[R]
            candidates = sorted(
                current.children.values(), key=lambda n: ROLE_ORDER[n.role]
            )
            if len({n.role for n in candidates}) != len(candidates):
                raise RuntimeError("A verified node has two children with one role")
            chosen = None
            for index, child in enumerate(candidates):
                proposal = self._proposal(child)
                if index == 0 and current.node_id in main_ratios:
                    ratio = main_ratios[current.node_id]
                elif proposal is None:
                    ratio = float(target[child.token])
                else:
                    t, g = torch.stack((target[child.token], proposal[child.token])).tolist()
                    if g <= 0:
                        raise RuntimeError("A candidate has no mass in its own proposal")
                    ratio = t / g
                if self.rng.random() < ratio:
                    chosen = child
                    break
                if proposal is None:
                    rest = target.clone()
                    rest[child.token] = 0
                else:
                    rest = (target - proposal).clamp_min_(0)
                target = _normalized(rest, target)
            if chosen is None:
                token = int(self._draw(target))
                chosen = self._child(current, token, origin="bonus", proposal_depth=R)
                source = "bonus"
                self.stats["bonus_tokens"] += 1
            else:
                source = chosen.role
                if (
                    source == "main"
                    and chosen.ever_frozen
                    and chosen.proposal_depth == 0
                ):
                    source = "frozen"
                self.stats["accepted_candidates"] += 1
                if chosen.promoted:
                    self.stats["promoted_heads_committed"] += 1
                if chosen.ever_frozen:
                    self.stats["previously_frozen_tokens_committed"] += 1
            if self.record_sampling_check:
                # Under exact sampling p(x) - sum p^2 has mean zero, and the mass ranked
                # above x (by probability, ties by token id) plus v * p(x) is uniform.
                mass = original[chosen.token]
                ids = torch.arange(original.numel(), device=original.device)
                above = (original > mass) | ((original == mass) & (ids < chosen.token))
                # TV(p_d, p_R) of the verified node bounds a readout child's acceptance: 1 - TV.
                tvs = [0.5 * (original - current.dists[d]).abs().sum()
                       if d in current.dists else torch.full_like(mass, -1)
                       for d in (self.d1, self.d2)]
                self.sampling_check.append(
                    torch.stack((mass, original[above].sum(), original.square().sum(), *tvs))
                )
            for sibling in current.children.values():
                if sibling is not chosen:
                    removed.update(self._descendants(sibling))
            accepted.append(current)
            self.cursor = chosen
            self.output.append(chosen.token)
            self.stats["committed_decode_tokens"] += 1
            self.stats["commit_sources"][source] += 1
            origins = self.stats["commit_origins"]
            origins[chosen.origin] = origins.get(chosen.origin, 0) + 1
            self._trace(
                "commit",
                token=chosen.token,
                position=chosen.position,
                source=source,
                origin=chosen.origin,
                depth=chosen.depth,
            )
            if chosen.token in self.eos or len(self.output) >= self.max_new:
                stopped = True
                break
            if source == "bonus":
                self._queue(current, chosen, R, "bonus")
        if accepted:
            self._drop(removed)
            self._commit(accepted)
            self.prefix_tokens.extend(n.token for n in accepted)
            self.cursor.parent = None
            for node in accepted:
                self._retire(self.nodes.pop(node.node_id))
            self.blocks = {
                key: ids for key, ids in self.blocks.items() if key in self.nodes
            }
        return stopped

    def _draft_events(self):
        plans, requests = [], []
        events, self.events = self.events, []
        seen = set()
        for event in events:
            head = event.head
            if head.node_id not in self.nodes or not head.growing:
                continue
            if head.node_id in seen or head.children:
                raise RuntimeError("An anchor was scheduled to draft twice")
            seen.add(head.node_id)
            width = min(self.k, self.output_limit - head.position)
            self.blocks[head.node_id] = [head.node_id]
            by_depth = self.stats["draft_events_by_depth"]
            by_depth[str(event.depth)] += 1
            if width <= 1:
                continue
            path = self._path(head)
            if (
                len(self.prefix_tokens) + len(path) != head.position + 1
                or event.parent_position + 1 != head.position
            ):
                raise RuntimeError(
                    "Draft anchor and conditioning parent positions differ"
                )
            requests.append(self._context_request(event, width, path))
            plans.append(head)
        if self.node_state and self.loop_context:
            # One gather for the loop-d spans of every request of the batch.
            spans = [r for r in requests if r["span_features"] is not None]
            if spans:
                features = self.runner.span_features(
                    [r["span_features"] for r in spans]
                )
                for request, span in zip(spans, features):
                    request["span_features"] = span
        if requests:
            results = self.drafter.draft_context_many(requests)
            if len(results) != len(requests):
                raise RuntimeError("Drafter batch result count differs")
            sizes = [r.token_ids.numel() for r in results]
            if any(n != r["k"] - 1 for n, r in zip(sizes, requests)):
                raise RuntimeError("Drafter returned an invalid block width")
            token_ids = (
                results[0].token_ids if len(results) == 1
                else torch.cat([r.token_ids for r in results])
            )
            tokens = token_ids.tolist()
            if self.draft_confidence:
                if any(r.confidence is None for r in results):
                    raise RuntimeError("draft_confidence needs a drafter with a confidence head")
                low = (torch.cat([r.confidence for r in results]) < self.draft_confidence).tolist()
                kept, offset = [], 0
                for size in sizes:
                    block = low[offset : offset + size]
                    keep = block.index(True) if True in block else size
                    kept.append(tokens[offset : offset + keep])
                    offset += size
                self.stats["draft_slots_cut"] += sum(sizes) - sum(len(x) for x in kept)
                tokens, sizes = [t for block in kept for t in block], [len(x) for x in kept]
            offset = 0
            for head, size in zip(plans, sizes):
                parent = head
                for token in tokens[offset : offset + size]:
                    parent = self._child(parent, token)
                    self.blocks[head.node_id].append(parent.node_id)
                offset += size
            self.stats["drafter_batches"] += 1
            self.stats["drafter_prefix_rows"] += self.runner.prefix_length
            self.stats["drafter_calls"] += len(requests)
            self.stats["draft_slots"] += sum(sizes)
            self.stats["drafter_readout_rows"] += sum(sizes)
        self._register_pending()

    def _context_request(self, event, width, path=None):
        """Verified loop-R prefix plus the loop-d span from the cursor to the parent.

        ``path`` is the head's path when the caller has it. With a runner that
        keeps node features the span is returned as (depth, node ids) for
        ``span_features``."""
        prefix_len = self.runner.prefix_length
        if event.depth == self.runner.R:
            if event.parent_position + 1 != prefix_len:
                raise RuntimeError("A loop-R parent was drafted before its commit")
            span = None
        else:
            path = (self._path(event.head) if path is None else path)[:-1]
            if path[0].position != prefix_len or path[-1].position != event.parent_position:
                raise RuntimeError("Loop-context span does not start at the cursor")
            if self.node_state:
                if any(n.depth < event.depth for n in path):
                    raise RuntimeError("Loop-context span lost a loop feature")
                span = (event.depth, [n.node_id for n in path])
            else:
                if any(event.depth not in n.loop_features for n in path):
                    raise RuntimeError("Loop-context span lost a loop feature")
                span = torch.stack([n.loop_features[event.depth] for n in path])
        return dict(
            prefix_features=self.prefix_features,
            prefix_len=prefix_len,
            span_features=span,
            head=event.head.token,
            k=width,
        )

    def _advance_spans(self, active):
        """One target call through per-node hidden states and results. Returns the
        rows of ``active`` at a readout depth and their logits."""
        fresh = [n for n in active if n.depth == 0]
        embeddings = (
            self.runner.embed(
                _token_ids_tensor([n.token for n in fresh], self.runner.device)
            )
            if fresh
            else []
        )
        for node, hidden in zip(fresh, embeddings):
            node.hidden = hidden
        spans = [_TreeSpan(n.node_id, n.depth, n.hidden) for n in active]
        if self.fused_readout:
            results, logits, readout_ids = self.runner.advance_readout(spans)
        else:
            results, logits = self.runner.advance(spans), None
        hidden = torch.cat([r.hidden for r in results])
        features = torch.cat([r.features for r in results])
        for i, node in enumerate(active):
            node.hidden, node.features = hidden[i], features[i]
            node.depth += 1
            if node.depth in self.context_depths:
                node.loop_features[node.depth] = features[i]
        rows = [i for i, n in enumerate(active) if n.depth in self.readout_depths]
        if self.fused_readout and (
            list(readout_ids) != [active[row].node_id for row in rows]
            or (rows and (logits is None or logits.shape[0] != len(rows)))
        ):
            raise RuntimeError("Runner readout rows differ from the schedule")
        if rows and not self.fused_readout:
            read_hidden = (
                hidden
                if len(rows) == len(active)
                else hidden.index_select(0, _token_ids_tensor(rows, self.runner.device))
            )
            logits = self.runner.logits(read_hidden)
        return rows, logits

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new, eos_token_ids, seed=None):
        if (
            not prompt_ids
            or max_new < 1
            or len(prompt_ids) + max_new > self.runner.max_len
        ):
            raise ValueError("Invalid prompt or generation capacity")
        if self.sampling:
            seed = 0 if seed is None else int(seed)
            self.generator = torch.Generator(device=self.runner.device)
            self.generator.manual_seed(seed)
            self.rng = random.Random(seed)
        self._reset_candidates()
        self.runner.reset()
        if self.drafter is not None:
            self.drafter.reset()
        self.max_new, self.eos = max_new, set(eos_token_ids)
        self.output_limit = len(prompt_ids) + max_new
        self.stats = dict(
            calls=0,
            cells=0,
            loop_groups=0,
            max_live_nodes=0,
            committed_decode_tokens=0,
            accepted_candidates=0,
            bonus_tokens=0,
            stage1_blocks=0,
            stage1_progress_tokens=0,
            stage1_pass_lengths={str(i): 0 for i in range(1, self.k + 1)},
            stage2_reviews=0,
            stage2_disagreements=0,
            frozen_created=0,
            frozen_tokens=0,
            promotions=0,
            promotion_reused_tokens=0,
            promoted_heads_committed=0,
            previously_frozen_tokens_committed=0,
            forks=0,
            pruned_nodes=0,
            drafter_batches=0,
            drafter_calls=0,
            draft_slots=0,
            draft_slots_cut=0,
            target_readout_calls=0,
            target_readout_rows=0,
            drafter_readout_rows=0,
            prefix_rows=0,
            loop_prefix_rows={str(d): 0 for d in range(1, self.runner.R + 1)},
            tree_rows_bound=0,
            drafter_prefix_rows=0,
            attention_history_pairs=0,
            attention_tree_pairs=0,
            draft_events_by_depth={str(d): 0 for d in range(1, self.runner.R + 1)},
            loop_cells={str(d): 0 for d in range(1, self.runner.R + 1)},
            commit_sources={name: 0 for name in ("main", "frozen", "fork", "bonus")},
            commit_origins={},
            per_call=[],
        )
        self.trace = []
        self.sampling_check = []
        self.nucleus_calls = self.nucleus_sorts = 0
        _sync(self.runner.device)
        started = time.perf_counter()
        pref = self.runner.prefill(_token_ids_tensor(prompt_ids, self.runner.device))
        logits = self.runner.logits(pref.hidden[-1:])
        if self.sampling:
            first = int(self._draw(self._distribution(logits, self.top_p)).item())
        else:
            first = int(logits.argmax(-1).item())
        self.prefix_tokens = list(prompt_ids)
        self.prefix_features = torch.empty(
            (self.runner.max_len, pref.features.shape[-1]),
            device=pref.features.device,
            dtype=pref.features.dtype,
        )
        self.prefix_features[: len(prompt_ids)].copy_(pref.features)
        self.output = [first]
        self.cursor = self._child(
            None, first, origin="initial", proposal_depth=self.runner.R
        )
        _sync(self.runner.device)
        prefill_done = time.perf_counter()
        stopped = first in self.eos or max_new == 1
        if not stopped:
            self.events.append(
                DraftEvent(
                    self.cursor,
                    len(prompt_ids) - 1,
                    pref.features[-1],
                    self.runner.R,
                    "initial",
                )
            )
            self._draft_events()
        needed = self.readout_depths
        R, stats = self.runner.R, self.stats
        loop_names = [str(d) for d in range(1, R + 1)]
        while not stopped:
            active = [n for n in self.nodes.values() if n.depth < R]
            if not active:
                raise RuntimeError("Finalized tree has no executable cursor")
            active.sort(key=ROW_ORDER)
            depths = [n.depth for n in active]
            cells, prefix_length = len(active), self.runner.prefix_length
            stats["calls"] += 1
            stats["cells"] += cells
            # Verified-prefix rows behind this call's KV reads, per executed loop; tree rows are bounded
            # by the live nodes each executed loop can see.
            loops = readouts = 0
            for depth, name in enumerate(loop_names):
                count = depths.count(depth)
                if count:
                    loops += 1
                    stats["loop_cells"][name] += count
                    stats["loop_prefix_rows"][name] += prefix_length
                    if depth + 1 in needed:
                        readouts += count
            stats["loop_groups"] += loops
            stats["prefix_rows"] += prefix_length
            stats["tree_rows_bound"] += len(self.nodes) * loops
            # Attention keys of this call's queries (one per cell, any loop): every verified prefix
            # position (shared, band or native rows alike) and the cell's own tree path from the
            # first speculative position, itself included; position + 1 keys in all.
            stats["attention_history_pairs"] += cells * prefix_length
            stats["attention_tree_pairs"] += (
                sum([n.position for n in active]) - cells * (prefix_length - 1)
            )
            if self.node_state:
                rows, argmax, logits = self.runner.advance_nodes(
                    [n.node_id for n in active], depths
                )
                for node in active:
                    node.depth += 1
                # The runner has checked its loops against ``depths``, so its rows
                # are those of the schedule when their number is.
                if len(rows) != readouts:
                    raise RuntimeError("Runner readout rows differ from the schedule")
            else:
                rows, logits = self._advance_spans(active)
                argmax = None
            if rows:
                self.stats["target_readout_calls"] += 1
                self.stats["target_readout_rows"] += len(rows)
                if self.sampling:
                    self._readout(logits, [active[row] for row in rows])
                else:
                    if argmax is None:
                        argmax = logits.argmax(-1)
                    for row, prediction in zip(rows, argmax.tolist()):
                        node = active[row]
                        node.readouts[node.depth] = int(prediction)
            before = len(self.output)
            if self.sampling:
                stopped = self._verify_sampled()
                if not stopped:
                    self._stage2_sampled(active)
                    self._stage1_sampled()
                    self._draft_events()
            else:
                stopped = self._verify()
                if not stopped:
                    self._stage2(active)
                    self._stage1()
                    self._draft_events()
            growing = sum([n.growing for n in self.nodes.values()])
            stats["per_call"].append(
                dict(
                    cells=cells,
                    commits=len(self.output) - before,
                    live_nodes=len(self.nodes),
                    growing_nodes=growing,
                    frozen_nodes=len(self.nodes) - growing,
                )
            )
        _sync(self.runner.device)
        finished = time.perf_counter()
        if self.sampling and self.nucleus_width and self.top_p < 1:
            # Calls whose target rows were cut from their most probable tokens, and those of them
            # that went to the sort of the vocabulary.
            self.stats.update(nucleus_calls=self.nucleus_calls, nucleus_sorts=self.nucleus_sorts)
        self.stats["S_count"] = (
            self.runner.R * (len(self.output) - 1) / max(self.stats["calls"], 1)
        )
        self.stats["stage1_mean_progress"] = self.stats["stage1_progress_tokens"] / max(
            self.stats["stage1_blocks"], 1
        )
        result = dict(
            output_ids=self.output,
            prefill_seconds=prefill_done - started,
            decode_seconds=finished - prefill_done,
            seconds=finished - started,
            stats=self.stats,
        )
        if self.record_trace:
            result["scheduler_trace"] = self.trace
        if self.record_sampling_check:
            rows = torch.stack(self.sampling_check).tolist() if self.sampling_check else []
            result["sampling_check"] = dict(
                zip(("token_probability", "more_probable_mass", "sum_squares", "tv_d1", "tv_d2"),
                    zip(*rows))
            ) if rows else {}
        self._reset_candidates()
        return result


def run_ouro_finalized(
    worker,
    requests,
    drafter_export=None,
    k=4,
    stage="stage2b",
    d1=1,
    d2=2,
    max_model_len=8192,
    initial_capacity=128,
    use_cuda_graph=True,
    result_callback=None,
    warmup=0,
    use_fused_norm=True,
    use_fused_rope=True,
    prewarm_tree_graphs=True,
    record_trace=False,
    context_cuda_graph=None,
    context_graph_min_calls=8,
    record_shapes=False,
    context_graph_grid=True,
    temperature=0.0,
    record_sampling_check=False,
    stage2_gate="argmax",
    top_p=1.0,
    proposal_top_p=None,
    compact_shared_prefix=True,
):
    """Worker-side 2DSD generation for Ouro targets (batch size one).

    ``requests`` is a list of dicts with ``prompt_ids``, ``max_new``, optional
    ``eos_token_ids`` and ``seed``. Completed requests are returned, or passed
    to ``result_callback`` without retention.

    record_shapes adds per-request call counts by target cell count and by drafter leaf layout.
    context_graph_grid captures the padded loop-context drafting grid (drafting_grid) at startup.
    temperature > 0 samples; each request's "seed" fixes its draws, and top_p < 1 applies
    vLLM's nucleus mask to the target (first token and loop-R readouts) and, unless
    proposal_top_p sets another mass, to the shallow proposal readouts. record_sampling_check
    adds, per committed token, its loop-R probability, the mass of more probable tokens
    and sum p^2 (an exactness check, one extra reduction per commit).
    """
    import json
    from pathlib import Path

    from vllm.model_executor.models.looped_kv import check_tree_window
    from vllm.model_executor.models.ouro_finalized import NativeOuroFinalizedRunner
    from vllm.v1.spec_decode.looped_dflash_context import (
        ContextDFlashAdapter,
        drafting_grid,
    )

    if not requests:
        return []
    if warmup < 0:
        raise ValueError("Warmup must be nonnegative")
    if k > 1 and not drafter_export:
        raise ValueError("Blocks larger than one require a drafter export")
    layers, input_format = None, None
    if drafter_export:
        config = json.loads((Path(drafter_export) / "config.json").read_text())
        layers = config["loopedtf_drafter"]["feature_layers_per_loop"]
        input_format = config["loopedtf_drafter"].get("input_format")
        if input_format != "loop-context":
            raise ValueError("2DSD drafters use the loop-context input format")
    model_config = worker.model_runner.get_model().config
    kv_policy = getattr(model_config, "loopedtf_kv_policy", None) or "full"
    check_tree_window(model_config, k)
    runner = NativeOuroFinalizedRunner(
        worker.model_runner.get_model(),
        max_model_len,
        max_nodes=initial_capacity,
        feat_layers=layers,
        use_cuda_graph=use_cuda_graph,
        max_graph_shapes=128,
        use_fused_norm=use_fused_norm,
        use_fused_rope=use_fused_rope,
        compact_shared_prefix=compact_shared_prefix,
    )
    adapter = (
        None
        if k == 1
        else ContextDFlashAdapter.from_export(
            drafter_export,
            embed=runner.embed,
            lm_head=runner.logits,
            max_len=max_model_len,
            device=runner.device,
            dtype=runner.dtype,
            use_cuda_graph=(use_cuda_graph if context_cuda_graph is None
                            else use_cuda_graph and context_cuda_graph),
            batched_graph_min_calls=context_graph_min_calls,
        )
    )
    executor = FinalizedTreeExecutor(
        runner, adapter, k, stage, d1, d2, record_trace, temperature=temperature,
        record_sampling_check=record_sampling_check, stage2_gate=stage2_gate, top_p=top_p,
        proposal_top_p=proposal_top_p,
    )

    def generate(request):
        return executor.generate(
            request["prompt_ids"],
            request["max_new"],
            request.get("eos_token_ids", []),
            seed=request.get("seed"),
        )

    try:
        prewarm = runner.prewarm_graphs(
            min(initial_capacity, 16) if prewarm_tree_graphs else 0
        )
        prewarm["scope"] = "once_per_rpc_before_measured_requests"
        drafter_prewarm = (
            adapter.prewarm_grid(k, *drafting_grid(runner.R, k))
            # The padded grid reads the masks after the anchor; a DSpark drafter
            # whose anchor slot predicts drafts on the layout path.
            if context_graph_grid and adapter is not None and adapter.first_slot == 1
            else {"enabled": False}
        )
        for _ in range(warmup):
            request = dict(requests[0])
            request["max_new"] = min(64, request["max_new"])
            generate(request)
        results = []
        for request in requests:
            before = dict(runner.stats)
            draft_before = dict(adapter.stats) if adapter is not None else {}
            cells_before = dict(runner.cell_counts)
            layouts_before = dict(getattr(adapter, "layout_counts", {}))
            if runner.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runner.device)
            result = generate(request)
            result["target_runtime_stats"] = {
                name: value - before.get(name, 0)
                for name, value in runner.stats.items()
                if name not in ("capacity", "tree_kv_bytes", "peak_live_nodes")
            }
            result["target_runtime_state"] = {
                "prefix_kv_bytes": runner.prefix.kv_bytes,
                "prefix_kv_storage": runner.prefix.kv_storage,
                "physical_capacity": runner.max_nodes,
                "path_width": runner.path_width,
                "tree_kv_bytes": runner.stats.get("tree_kv_bytes", 0),
                "lifetime_peak_live_nodes": runner.stats.get("peak_live_nodes", 0),
            }
            result["drafter_runtime_stats"] = (
                {
                    name: value - draft_before.get(name, 0)
                    for name, value in adapter.stats.items()
                }
                if adapter is not None
                else {}
            )
            if record_shapes:
                result["target_cell_counts"] = {
                    str(n): c - cells_before.get(n, 0)
                    for n, c in sorted(runner.cell_counts.items())
                    if c > cells_before.get(n, 0)
                }
                result["drafter_layout_counts"] = [
                    [[list(leaf) for leaf in layout], c - layouts_before.get(layout, 0)]
                    for layout, c in getattr(adapter, "layout_counts", {}).items()
                    if c > layouts_before.get(layout, 0)
                ]
            result["startup_target_graph_prewarm"] = prewarm
            result["startup_drafter_graph_prewarm"] = drafter_prewarm
            result["settings"] = dict(
                execution="2dsd-sampling" if temperature > 0 else "2dsd-greedy",
                **({"temperature": temperature, "seed": request.get("seed"),
                    "stage2_gate": stage2_gate} if temperature > 0 else {}),
                **({"top_p": top_p} if top_p < 1 else {}),
                **({"proposal_top_p": proposal_top_p}
                   if proposal_top_p is not None and proposal_top_p != top_p else {}),
                stage=stage,
                d1=d1,
                d2=d2 if stage != "stage1" else None,
                block_size=k,
                initial_capacity=initial_capacity,
                physical_capacity=runner.max_nodes,
                max_graph_shapes=runner.max_graph_shapes,
                kv_policy=kv_policy,
                kv_storage=runner.prefix.kv_storage,
                kv_window=getattr(model_config, "loopedtf_kv_window", None)
                if kv_policy == "shared_last"
                else None,
                target_fa_version=getattr(runner.prefix, "fa_version", None),
                drafter_fa_version=getattr(adapter, "fa_version", None),
                drafter_input_format=input_format if k > 1 else None,
                cuda_graph=runner.use_cuda_graph,
                context_cuda_graph=bool(adapter and adapter.use_cuda_graph),
                context_graph_min_calls=context_graph_min_calls if adapter is not None else None,
                context_graph_grid=drafter_prewarm.get("enabled", False),
                fused_norm=use_fused_norm,
                fused_rope=use_fused_rope,
                prewarm_tree_graphs=prewarm_tree_graphs,
            )
            if runner.device.type == "cuda":
                result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    runner.device
                )
                result["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                    runner.device
                )
            if result_callback is None:
                results.append(result)
            else:
                result_callback(result)
        return results
    finally:
        executor._reset_candidates()
        runner.reset()
        if adapter is not None:
            adapter.reset()
