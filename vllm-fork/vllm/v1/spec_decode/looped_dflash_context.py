# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""DFlash drafting with the in-flight span as target context (loop-context drafters).

Positions below ``prefix_len`` have finished loop R and enter as loop-R features,
projected once into the incremental prefix cache. The in-flight span from the
cursor to the drafting parent enters as that parent's loop-d features, projected
per request. The block holds the anchor token embedding and k - 1 mask
embeddings and attends to the prefix, its span and itself bidirectionally. A
parent that has finished loop R has an empty span: standard DFlash.

An export with DSpark heads (loopedtf_drafter.dspark) decides the drafted tokens left to right after the
one backbone pass, each with the Markov head's bias from the token before it, and every drafted token comes
with the confidence head's acceptance probability. Its block is the same anchor and k - 1 masks unless the
export sets sample_from_anchor (the DSpark paper's block): then the anchor's slot predicts the first
drafted token and a request for k - 1 drafted tokens runs k - 1 slots, the anchor and k - 2 masks. The
padded grid graphs cover only the former.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import accumulate
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .looped_dflash import DraftResult, LoopedDFlashAdapter


@dataclass
class _ContextInputs:
    heads: torch.Tensor
    starts: torch.Tensor
    positions: torch.Tensor
    span_features: torch.Tensor
    span_positions: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    prefix_cu: torch.Tensor
    keep: torch.Tensor
    layouts: tuple[tuple[int, int], ...]
    prefix_len: int

    @property
    def query_rows(self):
        return sum(k for _, k in self.layouts)


@dataclass
class _ContextGraph:
    inputs: _ContextInputs
    graph: Any
    logits: torch.Tensor
    cache_bytes: int


@dataclass
class _GridInputs:
    """One call padded to a grid shape: L leaves, S span rows and K block rows per leaf.

    Leaf i owns span rows [i*S, i*S + S) and block rows [i*K, i*K + K); dummy leaves and
    padded rows hold finite placeholders. Local keys are gathered leaf after leaf as each leaf's
    real block rows, then its real span rows; ``cu_k`` bounds them and padding follows the last leaf.
    """

    heads: torch.Tensor           # [L] anchor tokens
    positions: torch.Tensor       # [L*K] block query positions
    span_features: torch.Tensor   # [L*S, F]
    span_positions: torch.Tensor  # [L*S]
    gather: torch.Tensor          # [L*(S+K)] rows of cat(span keys, block keys)
    cu_k: torch.Tensor            # [L+1] int32 local key bounds per leaf
    prefix_cu: torch.Tensor       # [2] int32


@dataclass
class _GridGraph:
    inputs: _GridInputs
    graph: Any
    logits: torch.Tensor          # [L*(K-1), V]


@dataclass(frozen=True)
class _FastGrid:
    """Opt-in execution of the padded grid (ContextDFlashAdapter.configure_fast_grid)."""

    prefix_rows: int                   # newly verified positions projected inside the graph
    fused_kernels: bool                # norm + RoPE + placement and residual + norm as single kernels
    joint_attention: bool              # one-leaf calls read prefix, span and block keys in one attention
    merge_projection: bool             # prefix rows and spans share one fc GEMM
    head: torch.Tensor | None          # [V', H] logits weight, or None for the adapter's lm_head callable
    head_scale: torch.Tensor | None    # [V'] scale applied to the logits after the GEMM
    vocab_ids: torch.Tensor | None     # [V'] token ID of each head row; None for the whole vocabulary


@dataclass
class _FastGraph:
    """One fast grid shape. ``meta`` is the only integer input: every index vector is a view of it.
    ``hidden`` holds the mask embedding in every block row; a call rewrites the anchor rows only.

    Context rows are the ``prefix_rows`` slots for newly verified positions followed by each leaf's S
    span rows. ``context_rows`` names the cache row of the slots (and of the span rows when ``joint``);
    padded entries name the scratch row behind the last token position.
    """

    count: int
    rows: int
    joint: bool
    meta: torch.Tensor                 # [M] int64
    fields: dict[str, slice]           # views of meta: heads, positions, context_positions, context_rows, ...
    features: torch.Tensor             # [prefix_rows + L*S, F]
    hidden: torch.Tensor               # [L*K, H]
    identity: torch.Tensor             # [L*(S+K)] row numbers, for kernels that place rows in order
    graph: Any = None
    logits: torch.Tensor | None = None  # [L*(K-1), V']
    ids: torch.Tensor | None = None     # [L*(K-1)] token IDs


def _states(hidden):
    """Stands in for the logits callable of an adapter whose heads read the slots' backbone states."""
    return hidden


@dataclass
class _DSparkHeads:
    """speculators' DSpark heads of an export: the sequential (Markov) head, whose bias on a slot's logits
    comes from the token before it, and the confidence head."""

    kind: str                     # "vanilla", "gated" or "rnn"
    w1: torch.Tensor              # [V, r] embedding of the previous token
    w2: torch.Tensor              # [V, r] projection of the rank-r latent to logits
    projection: tuple | None      # (weight, bias) of the gate ("gated") or the joint recurrence ("rnn")
    confidence: tuple             # (weight [H + r], bias) over [backbone state, previous token embedding]
    wide_base: bool = False       # the target head returns FP32 logits and training summed them as such

    @classmethod
    def load(cls, path, settings, device, dtype):
        from safetensors import safe_open

        kind = settings["markov_head_type"]
        with safe_open(Path(path) / "model.safetensors", framework="pt") as handle:
            get = lambda name: handle.get_tensor(name).to(device=device, dtype=dtype)
            projection = {"gated": "gate_proj", "rnn": "joint_proj"}.get(kind)
            heads = cls(kind, get("markov_head.markov_w1.weight"), get("markov_head.markov_w2.weight"),
                        projection and (get(f"markov_head.{projection}.weight"), get(f"markov_head.{projection}.bias")),
                        (get("confidence_head.proj.weight")[0], get("confidence_head.proj.bias")[0]))
        if kind not in ("vanilla", "gated", "rnn") or heads.w1.shape[1] != settings["markov_rank"]:
            raise ValueError("Unsupported DSpark head")
        return heads

    def draft_base(self, base):
        """The head's logits as a greedy draft adds the Markov bias to them: in the heads' dtype. On a GPU
        that is BF16, the arithmetic of training (BF16 logits plus a BF16 bias), which also keeps the slot
        loop's sums and argmax over the vocabulary in half the bytes. Drafted tokens are the same as with an
        FP32 sum except at near-ties, and acceptance does not move (measured on 128 prompts).

        ``wide_base`` keeps the head's own dtype: a head that scales its logits in FP32 after the GEMM
        (Dory's) gave training FP32 logits, and rounding them to BF16 first drafts another token at about one
        slot in a hundred."""
        return base if self.wide_base else base.to(self.w2.dtype)

    def bias(self, hidden, previous, state, w2=None):
        """One slot of n blocks: its backbone states [n, H], the tokens before it [n] and the recurrent
        state [n, r] -> the logit bias [n, V] and the next state. ``w2`` (rows of the Markov projection)
        gives the bias of those tokens only."""
        embedding = latent = self.w1[previous]
        if self.kind == "gated":
            latent = torch.sigmoid(F.linear(torch.cat((hidden, embedding), -1), *self.projection)) * embedding
        elif self.kind == "rnn":
            gate, candidate, latent = F.linear(torch.cat((state, embedding, hidden), -1), *self.projection).chunk(3, -1)
            gate = torch.sigmoid(gate)
            state, latent = gate * state + (1.0 - gate) * torch.tanh(candidate), torch.tanh(latent)
        return latent @ (self.w2 if w2 is None else w2).T, state

    def confidence_logits(self, hidden, previous):
        """Confidence logits [...] of slots from their backbone states [..., H] and the tokens before them
        [...]. They do not enter the next slot, so a loop over slots can take them for all slots at once."""
        weight, bias = self.confidence
        return torch.cat((hidden, self.w1[previous]), -1) @ weight + bias

    def step(self, hidden, previous, state, w2=None):
        """``bias`` and ``confidence_logits`` of one slot: the logit bias [n, V], confidence logits [n] and
        the next state."""
        bias, state = self.bias(hidden, previous, state, w2)
        return bias, self.confidence_logits(hidden, previous), state


@dataclass
class _DSparkLoop:
    """The slot loop of ``count`` blocks of the same length as one CUDA graph: the backbone states of the
    blocks' slots and the token before each block go in, the drafted tokens and confidences come out."""

    hidden: torch.Tensor                     # [count * slots, H], blocks in order
    head: torch.Tensor                       # [count]
    tokens: torch.Tensor | None = None       # [count, slots]
    confidence: torch.Tensor | None = None   # [count, slots]
    graph: Any = None


def drafting_grid(loops: int, block: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Default padded drafting grid: leaves per call and span rows per leaf.

    A finalized schedule keeps at most (R - 1) * k in-flight positions under a drafting parent, so spans
    run from 0 to that bound in steps of k / 2. Up to eight leaves are drafted together.
    """
    step = max(1, block // 2)
    return (1, 2, 3, 4, 6, 8), tuple(range(0, (loops - 1) * block + 1, step))


class ContextDFlashAdapter(LoopedDFlashAdapter):
    """Batched loop-context drafting sharing one prefix cache across requests.

    Each request maps ``prefix_features`` (loop-R features of every verified
    position, shared by all requests), ``prefix_len``, ``span_features``
    ([n, F] loop-d features of positions prefix_len .. prefix_len + n - 1, the
    last being the parent; n = 0 when the parent is verified), ``head`` and
    ``k`` (block length including the head).
    """

    input_format = "loop-context"

    def __init__(self, *args, fast_grid: Mapping[str, Any] | bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_graphs: dict[tuple, _ContextGraph] = {}
        self._context_shape_counts: OrderedDict = OrderedDict()
        self._context_rejections: OrderedDict = OrderedDict()
        self._context_graph_bytes = 0
        self._context_stream = None
        # Calls per leaf layout ((span length, block length) per drafted leaf), graph or eager.
        self.layout_counts: dict[tuple, int] = {}
        # Padded shape grid (prewarm_grid): (leaves, span rows) -> graph; empty until prewarmed.
        self._grid_graphs: dict[tuple[int, int], _GridGraph] = {}
        self._grid_fixed: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        self._grid_leaves: tuple[int, ...] = ()
        self._grid_spans: tuple[int, ...] = ()
        self._grid_block = 0
        self._grid_pool = None
        self.grid_eager = False  # run grid calls with the same padded layout but no graph (bitwise checks)
        self._fast: _FastGrid | None = None  # configure_fast_grid
        self.dspark: _DSparkHeads | None = None  # from_export; the query passes then return backbone states
        self.first_slot = 1  # the first block row that predicts: 0 when the anchor's slot drafts (DSpark)
        self.readout: Callable | None = None     # with DSpark heads: the logits callable
        self._dspark_loops: dict[tuple[int, int], _DSparkLoop] = {}
        self._draft_vocabulary: tuple | None = None  # configure_draft_vocabulary: (ids, readout, w2 rows)
        self._fast_host = self._fast_upload = None
        self.stats.update({
            "context_graph_captures": 0, "context_graph_replays": 0,
            "context_graph_cache_bytes": 0, "context_graph_cold_fallbacks": 0,
            "context_graph_oversize_fallbacks": 0, "context_graph_budget_fallbacks": 0,
            "grid_graph_replays": 0, "grid_outside_calls": 0, "grid_padded_rows": 0,
        })
        # Work per drafting call for FLOPs accounting: rows through the query layers, span rows
        # through the context projection, readout rows, and attention keys (query-key pairs) against
        # the verified prefix and each leaf's own [span | block]; "_padded" as executed by a grid graph.
        for name in ("query_rows", "span_rows", "readout_rows", "prefix_pairs", "local_pairs"):
            self.stats[f"flops_{name}"] = 0
            self.stats[f"flops_{name}_padded"] = 0
        if fast_grid:
            self.configure_fast_grid(**({} if fast_grid is True else fast_grid))

    @classmethod
    def from_export(cls, path, *, fast_grid: Mapping[str, Any] | bool | None = None,
                    **kwargs) -> "ContextDFlashAdapter":
        adapter = super().from_export(path, **kwargs)
        drafter = adapter.config.get("loopedtf_drafter", {})
        if drafter.get("input_format") != "loop-context":
            raise ValueError("This export was not trained with loop-context inputs")
        if drafter.get("dspark"):
            adapter.dspark = _DSparkHeads.load(path, drafter["dspark"], adapter.device, adapter.dtype)
            adapter.dspark.wide_base = drafter.get("target_head", {}).get("kind") == "dory-scaled"
            adapter.first_slot = 0 if drafter["dspark"]["sample_from_anchor"] else 1
            # The heads read the slots' backbone states, so the query passes return those.
            adapter.readout, adapter.lm_head = adapter.lm_head, _states
        if fast_grid:
            adapter.configure_fast_grid(**({} if fast_grid is True else fast_grid))
        return adapter

    def configure_fast_grid(self, *, prefix_rows: int | None = None, fused_kernels: bool = True,
                            joint_attention: bool = False, merge_projection: bool = False,
                            head_weight: torch.Tensor | None = None, head_scale: torch.Tensor | None = None,
                            vocab_ids: torch.Tensor | None = None) -> None:
        """Run grid calls through the fast graphs. Off unless called; call it before prewarm_grid.

        A fast graph never reads a key-less sequence, so FlashAttention runs without the empty-row masks;
        the argmax is inside the graph, and one upload carries every index of a call. With only that
        (``fused_kernels=False``) the logits are bitwise those of the default grid. The other options
        keep the model and change BF16 rounding at most:

        prefix_rows      Newly verified positions, up to this many per call (default 8), are projected
                         inside the graph instead of in an eager pass before it; more than that in one
                         call take the eager pass. Up to 8 rows the fc GEMM rounds as the eager pass
                         does (cuBLAS switches kernels above), so the cache stays bitwise the same.
        fused_kernels    Query/key norm, RoPE and key/value placement run as one kernel per projection,
                         each residual addition with the norm that follows it as one kernel, and a
                         leaf's own [span | block] attention with its merge into the prefix attention
                         as one kernel. They follow the arithmetic of the kernels they replace; about
                         one element in 10^4 lands on the neighbouring BF16 value.
        joint_attention  Off by default. A one-leaf call writes its span and block keys behind the
                         verified prefix in the cache and reads prefix, span and block in one attention
                         instead of two merged ones: one softmax, so every row is rounded anew.
        merge_projection Off by default. The prefix rows share the spans' fc GEMM instead of running
                         their own; cuBLAS then rounds them as a larger batch, unlike the eager pass.
        head_weight      [V, H] logits weight, with ``head_scale`` [V] multiplied in FP32 after the GEMM:
                         the tensors behind the lm_head callable. The scale is then applied inside the
                         argmax (same token IDs) and the scaled logits are formed only for context_logits.
        vocab_ids        With head_weight: sorted token IDs the drafter may propose. Logits of other tokens
                         are not computed (context_logits reports -inf), which changes drafts outside it.

        Calls outside the grid keep the layout path. Without native operators and CUDA graphs there is
        no grid and these options have no effect.
        """
        prefix_rows = 8 if prefix_rows is None else int(prefix_rows)
        if prefix_rows < 0:
            raise ValueError("Invalid number of in-graph prefix rows")
        if head_weight is None and (head_scale is not None or vocab_ids is not None):
            raise ValueError("A head scale or vocabulary needs the head weight")
        if head_weight is not None and self.dspark is not None:
            raise ValueError("DSpark heads read the slots' backbone states; keep the logits outside the graph")
        head = scale = ids = None
        if head_weight is not None:
            if head_weight.ndim != 2 or head_weight.shape[1] != self.hidden_size:
                raise ValueError("Invalid head weight")
            head = head_weight.detach().to(device=self.device, dtype=self.dtype)
            if head_scale is not None:
                scale = head_scale.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
                if scale.shape[0] != head.shape[0]:
                    raise ValueError("Head scale and weight differ in vocabulary size")
            if vocab_ids is not None:
                ids = torch.as_tensor(vocab_ids, dtype=torch.long, device=self.device).reshape(-1)
                if (not ids.numel() or int(ids.min()) < 0 or int(ids.max()) >= head.shape[0]
                        or bool((ids[1:] <= ids[:-1]).any())):
                    raise ValueError("Vocabulary IDs must be sorted, unique and inside the head")
                head = head.index_select(0, ids)
                scale = None if scale is None else scale.index_select(0, ids)
            head = head.contiguous()
        self._fast = _FastGrid(prefix_rows, bool(fused_kernels), bool(joint_attention), bool(merge_projection),
                               head, scale, ids)
        self._fast_full_vocab = None if head_weight is None else head_weight.shape[0]
        self._grid_graphs.clear()  # captured for the previous setting
        # Fast grid calls whose new prefix rows went through the graph, or through the eager pass.
        self.stats.setdefault("grid_prefix_graph_calls", 0)
        self.stats.setdefault("grid_prefix_eager_calls", 0)

    def _extend_prefix(self, prefix_features: torch.Tensor, prefix_len: int) -> None:
        if not 1 <= prefix_len <= min(len(prefix_features), self.max_len):
            raise ValueError("Invalid verified prefix length")
        self.truncate(prefix_len)
        start = self.cached_prefix
        if start == prefix_len:
            return
        keys, values = self._project_context(
            prefix_features[start:prefix_len], self.positions[start:prefix_len]
        )
        self.k_cache[:, start:prefix_len] = keys
        self.v_cache[:, start:prefix_len] = values
        self.cached_prefix = prefix_len
        self.context_tokens_projected += prefix_len - start
        self.stats["context_tokens_projected"] += prefix_len - start

    def _local_attention_lse(self, q, keys, values, cu_q, cu_k, max_q, max_k):
        """Each request's block attends to its own [span | block] keys."""
        if self.native:
            return self._flash_attention(
                q,
                keys,
                values,
                max_seqlen_q=max_q,
                cu_seqlens_q=cu_q,
                max_seqlen_k=max_k,
                cu_seqlens_k=cu_k,
                causal=False,
                return_softmax_lse=True,
                fa_version=self.fa_version,
            )
        out = torch.empty_like(q)
        lse = torch.empty(
            self.num_heads, q.shape[0], device=self.device, dtype=torch.float32
        )
        bounds_q, bounds_k = cu_q.tolist(), cu_k.tolist()
        for i in range(len(bounds_q) - 1):
            rows = slice(bounds_q[i], bounds_q[i + 1])
            cols = slice(bounds_k[i], bounds_k[i + 1])
            out[rows], lse[:, rows] = self._attention_lse(
                q[rows], keys[cols], values[cols]
            )
        return out, lse

    @torch.inference_mode()
    def draft_context_many(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[DraftResult]:
        """Greedy drafts: k - 1 token IDs per request, none for k = 1."""
        if self.dspark is not None:
            return [DraftResult(tokens, confidence) for tokens, confidence, _ in self._dspark_outputs(requests)]
        # Argmax allocates owned IDs outside the graph; avoid copying full-vocab
        # logits just to consume them immediately. A fast grid graph returns the IDs themselves.
        outputs, tokens = self._context_outputs(requests, owned=False, tokens=True)
        return [
            DraftResult(
                torch.empty(0, device=self.device, dtype=torch.long)
                if x is None
                else x if tokens else x.argmax(-1),
                None,
            )
            for x in outputs
        ]

    def _dspark_outputs(self, requests, logits=False):
        """DSpark drafting: one backbone pass, then the slots left to right. A slot's logits are the head's
        logits of its backbone state plus the Markov head's bias from the token before it (the given token
        for the first slot); its confidence is the confidence head's acceptance probability. Per request:
        token IDs [k - 1], confidences [k - 1] and, with ``logits``, the corrected logits [k - 1, V].

        The logits callable returns logits before any monotone cap, which the argmax does not need."""
        states, _ = self._context_outputs(requests, owned=False)
        heads = self.dspark
        live = [i for i, x in enumerate(states) if x is not None]
        sizes = [states[i].shape[0] for i in live]
        if (live and not logits and self.use_cuda_graph and self.device.type == "cuda"
                and sizes[0] > 0 and all(size == sizes[0] for size in sizes)):
            # Blocks of one length: the readout and the slots replay from a graph, with no host round trip
            # between slots. Blocks of several lengths and the logits take the loop below.
            loop = self._dspark_loop(len(live), sizes[0])
            torch.cat([states[i] for i in live], out=loop.hidden)
            loop.head.copy_(torch.tensor([int(requests[i]["head"]) for i in live], device=self.device))
            loop.graph.replay()
            self.stats["graph_replays"] += 1
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            found = {i: n for n, i in enumerate(live)}
            # Copies: the graph's outputs are rewritten by the next call of this shape.
            return [(loop.tokens[found[i]].clone(), loop.confidence[found[i]].clone(), None) if i in found
                    else (empty, None, None) for i in range(len(requests))]
        tokens = {i: [] for i in live}
        confidences = {i: [] for i in live}
        rows = {i: [] for i in live}
        if live:
            hidden = torch.cat([states[i] for i in live])
            ids, readout, w2 = (None, self.readout, None) if logits or not self._draft_vocabulary \
                else self._draft_vocabulary
            base = readout(hidden)  # every slot in one pass
            if not logits:
                base = heads.draft_base(base)
            starts = [0, *accumulate(sizes)]
            previous = torch.tensor([int(requests[i]["head"]) for i in live], device=self.device)
            state = hidden.new_zeros(len(live), heads.w1.shape[1])
            for slot in range(max(sizes)):
                keep = [n for n, size in enumerate(sizes) if size > slot]
                index = torch.tensor(keep, device=self.device)
                at = torch.tensor([starts[n] + slot for n in keep], device=self.device)
                bias, confidence, state[index] = heads.step(hidden[at], previous[index], state[index], w2)
                corrected = base[at] + bias.to(base.dtype)
                chosen = corrected.argmax(-1)
                if ids is not None:
                    chosen = ids[chosen]
                previous[index] = chosen
                confidence = torch.sigmoid(confidence.float())
                for row, n in enumerate(keep):
                    tokens[live[n]].append(chosen[row])
                    confidences[live[n]].append(confidence[row])
                    if logits:
                        rows[live[n]].append(corrected[row])
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        return [(torch.stack(tokens[i]), torch.stack(confidences[i]), torch.stack(rows[i]) if logits else None)
                if i in tokens else (empty, None, None) for i in range(len(requests))]

    def _dspark_slots(self, loop, count, slots):
        """The slot loop of ``_dspark_outputs`` for ``count`` blocks of ``slots`` slots: tokens and
        confidences [count, slots]."""
        heads = self.dspark
        ids, readout, w2 = self._draft_vocabulary or (None, self.readout, None)
        hidden = loop.hidden.view(count, slots, -1)
        base = heads.draft_base(readout(loop.hidden)).view(count, slots, -1)
        previous = loop.head
        state = loop.hidden.new_zeros(count, heads.w1.shape[1])
        tokens, before = [], []
        for slot in range(slots):
            before.append(previous)
            bias, state = heads.bias(hidden[:, slot], previous, state, w2)
            previous = (base[:, slot] + bias.to(base.dtype)).argmax(-1)
            if ids is not None:
                previous = ids[previous]
            tokens.append(previous)
        # The confidence head after the loop, for every slot in one product: it is no input of a later slot.
        confidence = heads.confidence_logits(hidden, torch.stack(before, 1))
        return torch.stack(tokens, 1), torch.sigmoid(confidence.float())

    def configure_draft_vocabulary(self, ids, readout):
        """Restrict a DSpark drafter's drafts to the sorted token IDs ``ids``.

        ``readout`` maps backbone states [n, H] to the logits [n, len(ids)] of those tokens, before any
        cap. The slot loop then reads only those rows of the head and of the Markov projection: a token
        outside ``ids`` is never drafted (the target still produces it). ``context_logits`` keeps the
        whole vocabulary. Off unless called.
        """
        if self.dspark is None:
            raise ValueError("A drafting vocabulary needs DSpark heads")
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if (ids.ndim != 1 or ids.numel() == 0 or bool((ids[1:] <= ids[:-1]).any()) or int(ids[0]) < 0
                or int(ids[-1]) >= self.dspark.w2.shape[0]):
            raise ValueError("The drafting vocabulary is a sorted list of distinct token IDs of the head")
        self._draft_vocabulary = (ids, readout, self.dspark.w2[ids].contiguous())
        self._dspark_loops.clear()

    def _dspark_loop(self, count, slots):
        """The graph of the slot loop for this many blocks of this length, captured on first use."""
        loop = self._dspark_loops.get((count, slots))
        if loop is None:
            started = time.perf_counter()
            loop = self._dspark_loops[count, slots] = _DSparkLoop(
                torch.zeros(count * slots, self.hidden_size, dtype=self.dtype, device=self.device),
                torch.zeros(count, dtype=torch.long, device=self.device))
            for _ in range(2):  # also compiles the kernels of this shape
                self._dspark_slots(loop, count, slots)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                loop.tokens, loop.confidence = self._dspark_slots(loop, count, slots)
            loop.graph = graph
            self.stats["graph_captures"] += 1
            self.stats["capture_seconds"] += time.perf_counter() - started
        return loop

    @torch.inference_mode()
    def context_logits(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[torch.Tensor | None]:
        """Draft-slot logits [k - 1, V] per request; one shared weight pass. With DSpark heads: the
        corrected logits along the greedy draft, before any cap of the target's head."""
        if self.dspark is not None:
            return [rows for _, _, rows in self._dspark_outputs(requests, logits=True)]
        return self._context_logits(requests, owned=True)

    def _context_logits(self, requests, *, owned):
        return self._context_outputs(requests, owned=owned)[0]

    def _context_outputs(self, requests, *, owned, tokens=False):
        """Per request draft-slot logits, or token IDs when ``tokens`` and a fast grid graph ran (flag)."""
        if not requests:
            return [], False
        prefix_features = requests[0]["prefix_features"]
        prefix_len = int(requests[0]["prefix_len"])
        for r in requests:
            if (
                r["prefix_features"].data_ptr() != prefix_features.data_ptr()
                or int(r["prefix_len"]) != prefix_len
            ):
                raise ValueError("Batched drafts require one shared verified prefix")
            # block_size counts the slots of a block; k the anchor and the drafted tokens.
            if not 1 <= r["k"] <= self.block_size + 1 - self.first_slot:
                raise ValueError("Invalid drafter block length")
        results: list[torch.Tensor | None] = [None] * len(requests)
        active = [(index, r) for index, r in enumerate(requests) if r["k"] > 1]
        if not active:
            return results, False
        # A fast grid graph projects the new prefix rows itself; other calls project them here.
        fast = self._fast if self._grid_graphs else None
        if fast is None or not fast.prefix_rows:
            self._extend_prefix(prefix_features, prefix_len)
        elif not 1 <= prefix_len <= min(len(prefix_features), self.max_len):
            raise ValueError("Invalid verified prefix length")

        spans = [
            r["span_features"].reshape(-1, self.fc.shape[1])
            if r.get("span_features") is not None
            else prefix_features[:0]
            for _, r in active
        ]
        lengths = [span.shape[0] for span in spans]
        parents = [prefix_len - 1 + n for n in lengths]
        if any(p + r["k"] + 1 > self.max_len for p, (_, r) in zip(parents, active)):
            raise ValueError("Drafter token capacity exceeded")
        block_lengths = [r["k"] - 1 + self.first_slot for _, r in active]  # slots (rows) of each block
        layout = tuple(zip(lengths, block_lengths))
        self.layout_counts[layout] = self.layout_counts.get(layout, 0) + 1
        shape = self._grid_shape(lengths, block_lengths)
        self._count_work(lengths, block_lengths, prefix_len, shape)
        if shape is not None:
            if fast is not None:
                logits, ids = self._run_fast(shape, active, spans, lengths, parents, block_lengths,
                                             prefix_features, prefix_len, eager=self.grid_eager)
                # One owned copy for the call: IDs, or the logits over the whole vocabulary.
                rows = ids.clone() if tokens else self._fast_logits(logits, owned)
                step = self._grid_block - 1
                drafts = [rows[i * step: i * step + k - 1] for i, k in enumerate(block_lengths)]
            else:
                drafts = self._run_grid(shape, active, spans, lengths, parents, block_lengths, prefix_len,
                                        eager=self.grid_eager)
            if sum(lengths):
                self.stats["context_weight_passes"] += 1
                self.stats["context_projection_rows"] += sum(lengths)
            self.stats["leaf_drafts"] += len(active)
            self.stats["query_weight_passes"] += 1
            self.stats["query_rows"] += shape[0] * self._grid_block
            if fast is not None:
                for (index, _), rows in zip(active, drafts):
                    results[index] = rows
                return results, tokens
            for (index, _), logits in zip(active, drafts):
                results[index] = logits.clone() if owned else logits
            return results, False
        if fast is not None and fast.prefix_rows:
            self._extend_prefix(prefix_features, prefix_len)
        if self._grid_graphs:
            self.stats["grid_outside_calls"] += 1
        if sum(lengths):
            span_positions = torch.cat(
                [
                    self.positions[prefix_len : prefix_len + n]
                    for n in lengths
                    if n
                ]
            )
            span_features = torch.cat([span for span in spans if span.shape[0]])
            self.stats["context_weight_passes"] += 1
            self.stats["context_projection_rows"] += sum(lengths)
        else:
            span_features = prefix_features[:0]
            span_positions = self.positions[:0]
        positions = torch.cat(
            [self.positions[p + 1 : p + 1 + k] for p, k in zip(parents, block_lengths)]
        )
        bounds = [0, *accumulate(block_lengths)]
        local_lengths = [n + k for n, k in zip(lengths, block_lengths)]
        vectors = ([int(r["head"]) for _, r in active], bounds[:-1],
                   [row for start, k in zip(bounds, block_lengths)
                    for row in range(start + self.first_slot, start + k)])
        sizes = [len(v) for v in vectors]
        cu_vectors = (bounds, [0, *accumulate(local_lengths)], [0, prefix_len])
        metadata = torch.tensor(
            [x for v in (*vectors, *cu_vectors) for x in v], dtype=torch.long,
            pin_memory=self.device.type == "cuda",
        ).to(self.device, non_blocking=True)
        heads, starts, keep = metadata[:sum(sizes)].split(sizes)
        cu_q, cu_k, prefix_cu = metadata[sum(sizes):].to(torch.int32).split(
            [len(v) for v in cu_vectors])
        # Prime the fixed Q bound before capture. No device -> host reads occur
        # in the query forward, and prefix length is a live device input.
        self._cu(sum(block_lengths))
        inputs = _ContextInputs(heads, starts, positions, span_features, span_positions, cu_q, cu_k,
                                prefix_cu, keep, tuple(zip(lengths, block_lengths)), prefix_len)
        logits = self._run_context_queries(inputs)
        if owned:
            logits = logits.clone()
        self.stats["leaf_drafts"] += len(active)
        self.stats["query_weight_passes"] += 1
        self.stats["query_rows"] += inputs.query_rows
        offset = 0
        for index, r in active:
            end = offset + r["k"] - 1
            results[index] = logits[offset:end]
            offset = end
        return results, False

    def _count_work(self, lengths, block_lengths, prefix_len, shape):
        """FLOPs counters of one drafting call (host lists only, no device work)."""
        actual = {
            "query_rows": sum(block_lengths),
            "span_rows": sum(lengths),
            "readout_rows": sum(k - self.first_slot for k in block_lengths),
            "prefix_pairs": sum(block_lengths) * prefix_len,
            "local_pairs": sum(k * (n + k) for n, k in zip(lengths, block_lengths)),
        }
        if shape is None:
            padded = actual  # the layout path runs exactly these rows
        else:
            count, rows = shape
            block = self._grid_block
            dummies = count - len(lengths)  # placeholder leaves: one block key, no span
            padded = {
                "query_rows": count * block,
                "span_rows": count * rows,
                "readout_rows": count * (block - 1),
                "prefix_pairs": count * block * prefix_len,
                "local_pairs": sum(block * (n + k) for n, k in zip(lengths, block_lengths)) + dummies * block,
            }
        for name, value in actual.items():
            self.stats[f"flops_{name}"] += value
            self.stats[f"flops_{name}_padded"] += padded[name]

    def _context_query_forward(self, inputs):
        positions = inputs.positions
        hidden = self.mask_embedding.expand(inputs.query_rows, -1).clone()
        hidden.index_copy_(0, inputs.starts, self.embed(inputs.heads).to(self.dtype))
        lengths, block_lengths = zip(*inputs.layouts)
        prefix_len = inputs.prefix_len
        if inputs.span_features.shape[0]:
            span_keys, span_values = self._project_context_forward(
                inputs.span_features, inputs.span_positions)
        for i, layer in enumerate(self.layers):
            normalized = self._rms(hidden, layer.input_norm)
            q, key, value = F.linear(normalized, layer.qkv).split(
                (self.q_size, self.kv_size, self.kv_size), dim=-1
            )
            q = self._rms(q.view(-1, self.num_heads, self.head_dim), layer.q_norm)
            key = self._rms(key.view(-1, self.num_kv_heads, self.head_dim), layer.k_norm)
            value = value.view(-1, self.num_kv_heads, self.head_dim)
            if self.native:
                self._ops.rotary_embedding(
                    positions,
                    q.view(-1, self.q_size),
                    key.view(-1, self.kv_size),
                    self.head_dim,
                    self.rope_cache,
                    True,
                )
                prefix_out, prefix_lse = self._flash_attention(
                    q,
                    self.k_cache[i],
                    self.v_cache[i],
                    max_seqlen_q=q.shape[0],
                    cu_seqlens_q=self._cu(q.shape[0]),
                    max_seqlen_k=self.max_len,
                    cu_seqlens_k=inputs.prefix_cu,
                    causal=False,
                    return_softmax_lse=True,
                    fa_version=self.fa_version,
                )
            else:
                q, key = self._rotate(q, positions), self._rotate(key, positions)
                prefix_out, prefix_lse = self._attention_lse(
                    q, self.k_cache[i, :prefix_len], self.v_cache[i, :prefix_len]
                )
            # Per request: its span's projected context, then its own block.
            local_keys, local_values, offset, span_offset = [], [], 0, 0
            for n, k in zip(lengths, block_lengths):
                if n:
                    local_keys.append(span_keys[i, span_offset : span_offset + n])
                    local_values.append(span_values[i, span_offset : span_offset + n])
                    span_offset += n
                local_keys.append(key[offset : offset + k])
                local_values.append(value[offset : offset + k])
                offset += k
            local_out, local_lse = self._local_attention_lse(
                q,
                torch.cat(local_keys),
                torch.cat(local_values),
                inputs.cu_q,
                inputs.cu_k,
                max(block_lengths),
                max(n + k for n, k in inputs.layouts),
            )
            if self.native:
                attention = torch.empty_like(q)
                self._ops.merge_attn_states(
                    attention, prefix_out, prefix_lse, local_out, local_lse
                )
            else:
                total = torch.logaddexp(prefix_lse, local_lse)
                attention = (
                    prefix_out.float() * (prefix_lse - total).exp().T[..., None]
                    + local_out.float() * (local_lse - total).exp().T[..., None]
                ).to(q.dtype)
            hidden = hidden + F.linear(attention.reshape(-1, self.q_size), layer.o_proj)
            gate_up = F.linear(self._rms(hidden, layer.post_norm), layer.gate_up)
            if self.native:
                activated = torch.empty(
                    gate_up.shape[0],
                    gate_up.shape[1] // 2,
                    device=self.device,
                    dtype=self.dtype,
                )
                torch.ops._C.silu_and_mul(activated, gate_up)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                activated = F.silu(gate) * up
            hidden = hidden + F.linear(activated, layer.down)
        return self.lm_head(self._rms(hidden[inputs.keep], self.norm))

    def _context_graph_estimate(self, inputs):
        estimate = self._many_graph_estimate([(0, n, k) for n, k in inputs.layouts])
        return estimate + sum(t.numel() * t.element_size()
                              for t in (inputs.span_features, inputs.span_positions))

    def _capture_context_queries(self, source, estimate):
        started = time.perf_counter()
        inputs = _ContextInputs(**{
            f.name: (getattr(source, f.name).clone()
                     if isinstance(getattr(source, f.name), torch.Tensor)
                     else getattr(source, f.name)) for f in fields(source)
        })
        if self._context_stream is None:
            self._context_stream = torch.cuda.Stream(device=self.device)
        stream = self._context_stream
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._context_query_forward(inputs)
        current.wait_stream(stream)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = self._context_query_forward(inputs)
        current.wait_stream(stream)
        charge = max(estimate, torch.cuda.memory_allocated(self.device) - allocated,
                     torch.cuda.memory_reserved(self.device) - reserved)
        self.stats["graph_captures"] += 1
        self.stats["context_graph_captures"] += 1
        self.stats["query_weight_passes"] += 3
        self.stats["query_rows"] += 3 * source.query_rows
        if source.span_features.shape[0]:
            self.stats["context_weight_passes"] += 3
            self.stats["context_projection_rows"] += 3 * source.span_features.shape[0]
        self.stats["capture_seconds"] += time.perf_counter() - started
        return _ContextGraph(inputs, graph, logits, charge)

    def _run_context_queries(self, inputs):
        if not self.use_cuda_graph:
            return self._context_query_forward(inputs)
        key = inputs.layouts  # prefix length and token values are live inputs
        self._context_shape_counts[key] = self._context_shape_counts.get(key, 0) + 1
        self._context_shape_counts.move_to_end(key)
        if len(self._context_shape_counts) > 4096:
            self._context_shape_counts.popitem(last=False)
        captured = self._context_graphs.get(key)
        if captured is None:
            if self._context_shape_counts[key] < self.batched_graph_min_calls:
                self.stats["context_graph_cold_fallbacks"] += 1
                return self._context_query_forward(inputs)
            estimate = self._context_graph_estimate(inputs)
            rejection = self._context_rejections.get(key)
            if estimate > self.batched_graph_max_shape_bytes:
                rejection = "oversize"
            elif (len(self._context_graphs) >= self.batched_graph_max_shapes or
                  self._context_graph_bytes + estimate > self.batched_graph_max_bytes):
                rejection = "budget"
            if rejection:
                self.stats[f"context_graph_{rejection}_fallbacks"] += 1
                return self._context_query_forward(inputs)
            captured = self._capture_context_queries(inputs, estimate)
            if (captured.cache_bytes > self.batched_graph_max_shape_bytes or
                    self._context_graph_bytes + captured.cache_bytes > self.batched_graph_max_bytes):
                reason = "oversize" if captured.cache_bytes > self.batched_graph_max_shape_bytes else "budget"
                self._context_rejections[key] = reason
                if len(self._context_rejections) > 4096:
                    self._context_rejections.popitem(last=False)
                self.stats[f"context_graph_{reason}_fallbacks"] += 1
                del captured
                return self._context_query_forward(inputs)
            self._context_graphs[key] = captured
            self._context_graph_bytes += captured.cache_bytes
            self.stats["context_graph_cache_bytes"] = self._context_graph_bytes
        else:
            for name in ("heads", "positions", "span_features", "span_positions", "prefix_cu"):
                getattr(captured.inputs, name).copy_(getattr(inputs, name))
        captured.graph.replay()
        self.stats["graph_replays"] += 1
        self.stats["context_graph_replays"] += 1
        return captured.logits

    # Padded shape grid. Every grid graph is captured at startup, so drafting never captures or
    # falls back to eager for calls inside the grid, whatever their leaf/span combination.

    def prewarm_grid(self, block: int, leaves: Sequence[int], spans: Sequence[int]) -> dict[str, Any]:
        """Capture the drafting graph of every (leaves, span rows) grid shape for blocks of ``block`` rows.

        A call with at most max(leaves) drafted leaves, spans of at most max(spans) positions and blocks
        of at most ``block`` rows is padded to the smallest covering shape and replays its graph; other
        calls keep the layout path. All grid graphs share one memory pool, so a returned logits tensor is
        valid until the next drafting call, as for the layout graphs.
        """
        leaves, spans = tuple(leaves), tuple(spans)
        report = {"enabled": False, "block": block, "leaves": list(leaves), "spans": list(spans),
                  "graphs": 0, "seconds": 0.0}
        if not (self.native and self.use_cuda_graph) or block < 2:
            return report
        if self.first_slot != 1:
            raise NotImplementedError("The padded grid graphs read out the masks after the anchor; a drafter "
                                      "whose anchor slot predicts runs the layout path")
        if (not leaves or not spans or list(leaves) != sorted(set(leaves)) or list(spans) != sorted(set(spans))
                or leaves[0] < 1 or spans[0] < 0 or block > self.block_size):
            raise ValueError("Invalid drafting graph grid")
        if block != self._grid_block:
            self._grid_graphs.clear()
            self._grid_fixed.clear()
        self._grid_leaves, self._grid_spans, self._grid_block = leaves, spans, block
        if self._grid_pool is None:
            self._grid_pool = torch.cuda.graph_pool_handle()
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        started = time.perf_counter()
        fast = self._fast
        if fast is not None:
            self._prepare_fast(leaves, spans, block)
            report["fast"] = {"prefix_rows": fast.prefix_rows, "fused_kernels": fast.fused_kernels,
                              "joint_attention": fast.joint_attention,
                              "merge_projection": fast.merge_projection,
                              "head": "callable" if fast.head is None else "weight",
                              "vocabulary": None if fast.head is None else fast.head.shape[0]}
        capture = self._capture_grid if fast is None else self._capture_fast
        for count in leaves:
            for rows in spans:
                if (count, rows) not in self._grid_graphs:
                    self._grid_graphs[(count, rows)] = capture(count, rows, block)
        torch.cuda.synchronize(self.device)
        report.update(enabled=True, graphs=len(self._grid_graphs), seconds=time.perf_counter() - started,
                      retained_bytes=torch.cuda.memory_allocated(self.device) - allocated)
        return report

    def _grid_constants(self, count, rows, block):
        key = (count, rows)
        if key not in self._grid_fixed:
            starts = torch.arange(0, count * block, block, device=self.device, dtype=torch.long)
            cu_q = torch.arange(0, (count + 1) * block, block, device=self.device, dtype=torch.int32)
            keep = (starts[:, None] + torch.arange(1, block, device=self.device)[None, :]).flatten()
            self._grid_fixed[key] = (starts, cu_q, keep)
        return self._grid_fixed[key]

    def _grid_forward(self, inputs, count, rows, block):
        """Drafter layers for a padded grid call; graph capture and eager share this code."""
        from .looped_dflash_ops import gather_context_kv

        starts, cu_q, keep = self._grid_constants(count, rows, block)
        hidden = self.mask_embedding.expand(count * block, -1).clone()
        hidden.index_copy_(0, starts, self.embed(inputs.heads).to(self.dtype))
        if rows:
            span_keys, span_values = self._project_context_forward(inputs.span_features, inputs.span_positions)
        for i, layer in enumerate(self.layers):
            normalized = self._rms(hidden, layer.input_norm)
            q, key, value = F.linear(normalized, layer.qkv).split(
                (self.q_size, self.kv_size, self.kv_size), dim=-1
            )
            q = self._rms(q.view(-1, self.num_heads, self.head_dim), layer.q_norm)
            key = self._rms(key.view(-1, self.num_kv_heads, self.head_dim), layer.k_norm)
            value = value.view(-1, self.num_kv_heads, self.head_dim)
            self._ops.rotary_embedding(inputs.positions, q.view(-1, self.q_size), key.view(-1, self.kv_size),
                                       self.head_dim, self.rope_cache, True)
            prefix_out, prefix_lse = self._flash_attention(
                q, self.k_cache[i], self.v_cache[i], max_seqlen_q=q.shape[0], cu_seqlens_q=self._cu(q.shape[0]),
                max_seqlen_k=self.max_len, cu_seqlens_k=inputs.prefix_cu, causal=False,
                return_softmax_lse=True, fa_version=self.fa_version,
            )
            keys, values = gather_context_kv(
                span_keys[i] if rows else None, span_values[i] if rows else None,
                key, value, inputs.gather,
            )
            local_out, local_lse = self._flash_attention(
                q, keys, values,
                max_seqlen_q=block, cu_seqlens_q=cu_q, max_seqlen_k=rows + block, cu_seqlens_k=inputs.cu_k,
                causal=False, return_softmax_lse=True, fa_version=self.fa_version,
            )
            attention = torch.empty_like(q)
            self._ops.merge_attn_states(attention, prefix_out, prefix_lse, local_out, local_lse)
            hidden = hidden + F.linear(attention.reshape(-1, self.q_size), layer.o_proj)
            gate_up = F.linear(self._rms(hidden, layer.post_norm), layer.gate_up)
            activated = torch.empty(gate_up.shape[0], gate_up.shape[1] // 2, device=self.device, dtype=self.dtype)
            torch.ops._C.silu_and_mul(activated, gate_up)
            hidden = hidden + F.linear(activated, layer.down)
        return self.lm_head(self._rms(hidden[keep], self.norm))

    def _grid_buffers(self, count, rows, block):
        dev = self.device
        return _GridInputs(
            heads=torch.zeros(count, device=dev, dtype=torch.long),
            positions=torch.arange(count * block, device=dev, dtype=torch.long) % block,
            span_features=torch.zeros(count * rows, self.fc.shape[1], device=dev, dtype=self.dtype),
            span_positions=torch.zeros(count * rows, device=dev, dtype=torch.long),
            gather=torch.zeros(count * (rows + block), device=dev, dtype=torch.long),
            cu_k=torch.arange(count + 1, device=dev, dtype=torch.int32),
            prefix_cu=torch.tensor([0, 1], device=dev, dtype=torch.int32),
        )

    def _capture_grid(self, count, rows, block):
        inputs = self._grid_buffers(count, rows, block)
        self._grid_constants(count, rows, block)
        self._cu(count * block)  # fixed Q bound, created outside the capture
        if self._context_stream is None:
            self._context_stream = torch.cuda.Stream(device=self.device)
        stream = self._context_stream
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._grid_forward(inputs, count, rows, block)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._grid_pool, stream=stream):
            logits = self._grid_forward(inputs, count, rows, block)
        current.wait_stream(stream)
        self.stats["graph_captures"] += 1
        return _GridGraph(inputs, graph, logits)

    def _grid_shape(self, lengths, block_lengths):
        if (not self._grid_graphs or len(lengths) > self._grid_leaves[-1] or max(lengths) > self._grid_spans[-1]
                or max(block_lengths) > self._grid_block):
            return None
        count = next(n for n in self._grid_leaves if n >= len(lengths))
        rows = next(n for n in self._grid_spans if n >= max(lengths))
        return count, rows

    def _run_grid(self, shape, active, spans, lengths, parents, block_lengths, prefix_len, *, eager=False):
        """Pad one call to ``shape`` and replay its graph (or run the same code eagerly); per-leaf logits."""
        count, rows = shape
        block = self._grid_block
        n = len(lengths)
        heads = [int(r["head"]) for _, r in active] + [0] * (count - n)
        positions, span_positions, span_rows, gather, cu_k = [], [], [], [], [0]
        for i in range(count):
            real = i < n
            k, m = (block_lengths[i], lengths[i]) if real else (1, 0)
            positions += ([min(parents[i] + 1 + j, self.max_len - 1) for j in range(block)] if real
                          else list(range(block)))
            span_positions += [prefix_len + j for j in range(m)] + [0] * (rows - m)
            span_rows += [i * rows + j for j in range(m)]
            gather += [count * rows + i * block + j for j in range(k)] + [i * rows + j for j in range(m)]
            cu_k.append(len(gather))
        gather += [gather[0]] * (count * (rows + block) - len(gather))  # past cu_k[-1]: never attended
        vectors = (heads, positions, span_positions, gather, span_rows, cu_k, [0, prefix_len])
        metadata = torch.tensor([x for v in vectors for x in v], dtype=torch.long,
                                pin_memory=True).to(self.device, non_blocking=True)
        heads_t, positions_t, span_positions_t, gather_t, rows_t, cu_k_t, prefix_t = metadata.split(
            [len(v) for v in vectors])
        graph = None if eager else self._grid_graphs.get(shape)
        inputs = graph.inputs if graph is not None else self._grid_buffers(count, rows, block)
        inputs.heads.copy_(heads_t)
        inputs.positions.copy_(positions_t)
        inputs.span_positions.copy_(span_positions_t)
        inputs.gather.copy_(gather_t)
        inputs.cu_k.copy_(cu_k_t)
        inputs.prefix_cu.copy_(prefix_t)
        if rows:
            inputs.span_features.zero_()
            if span_rows:
                inputs.span_features.index_copy_(
                    0, rows_t, torch.cat([s for s in spans if s.shape[0]]).to(self.dtype))
        if graph is not None:
            graph.graph.replay()
            logits = graph.logits
            self.stats["graph_replays"] += 1
            self.stats["grid_graph_replays"] += 1
        else:
            logits = self._grid_forward(inputs, count, rows, block)
        self.stats["grid_padded_rows"] += count * (rows + block) - sum(lengths) - sum(block_lengths)
        return [logits[i * (block - 1): i * (block - 1) + k - 1] for i, k in enumerate(block_lengths)]

    # Fast grid (configure_fast_grid): the same padded shapes with fewer kernels and host operations.

    def _fast_fields(self, count, rows, block):
        """Views of a fast graph's integer input, in order, and whether the shape reads one joint sequence."""
        slots = self._fast.prefix_rows
        joint = self._fast.joint_attention and count == 1
        sizes = [("heads", count), ("positions", count * block), ("context_positions", slots + count * rows),
                 ("context_rows", slots + rows if joint else slots)]
        sizes += ([("block_rows", block), ("key_bounds", 2)] if joint
                  else [("gather", count * (rows + block)), ("cu_k", count + 1), ("prefix_cu", 2)])
        fields, offset = {}, 0
        for name, size in sizes:
            fields[name] = slice(offset, offset + size)
            offset += size
        return joint, fields, offset

    def _prepare_fast(self, leaves, spans, block):
        """Scratch cache row for padded writes and the host staging buffer of the integer inputs."""
        if self.k_cache.shape[1] == self.max_len:
            pad = torch.zeros_like(self.k_cache[:, :1])
            self.k_cache = torch.cat((self.k_cache, pad), dim=1)
            self.v_cache = torch.cat((self.v_cache, pad), dim=1)
            # Graphs captured before this point read the previous cache tensors.
            self._context_graphs.clear()
            self._context_graph_bytes = 0
        elif self.k_cache.shape[1] != self.max_len + 1:
            raise ValueError("The fast grid needs the single-request prefix cache")
        size = max(self._fast_fields(count, rows, block)[2] for count in leaves for rows in spans)
        if self._fast_host is None or self._fast_host.shape[0] < size:
            self._fast_host = np.zeros(size, dtype=np.int64)
            self._fast_upload = torch.from_numpy(self._fast_host)
        self._fast_steps = np.arange(max(block, self._fast.prefix_rows, spans[-1]) + 1, dtype=np.int64)

    def _fast_buffers(self, count, rows, block):
        joint, fields, size = self._fast_fields(count, rows, block)
        meta = torch.zeros(size, device=self.device, dtype=torch.long)
        # Placeholders: one prefix key, every padded write on the scratch row.
        meta[fields["positions"]] = torch.arange(count * block, device=self.device) % block
        meta[fields["context_rows"]] = self.max_len
        if joint:
            meta[fields["block_rows"]] = self.max_len
            meta[fields["key_bounds"].stop - 1] = 1
        else:
            meta[fields["cu_k"]] = torch.arange(count + 1, device=self.device)
            meta[fields["prefix_cu"].stop - 1] = 1
        features = torch.zeros(self._fast.prefix_rows + count * rows, self.fc.shape[1], device=self.device,
                               dtype=self.dtype)
        hidden = self.mask_embedding.expand(count * block, -1).clone()
        identity = torch.arange(count * (rows + block), device=self.device)
        return _FastGraph(count, rows, joint, meta, fields, features, hidden, identity)

    def _fast_forward(self, g, block):
        """Drafter layers and argmax of a fast grid call; graph capture and eager share this code."""
        from vllm.vllm_flash_attn import flash_attn_varlen_func as flash

        from .looped_dflash_ops import (
            add_rms_norm,
            gather_context_kv,
            local_merge_attention,
            norm_rope_place,
            scaled_argmax,
            scatter_context_kv,
        )

        fast, count, rows, meta = self._fast, g.count, g.rows, g.meta
        fused, joint, slots = fast.fused_kernels, g.joint, fast.prefix_rows
        heads, kv_heads, dim = self.num_heads, self.num_kv_heads, self.head_dim
        field = {name: meta[part] for name, part in g.fields.items()}
        starts, cu_q, keep = self._grid_constants(count, rows, block)
        positions = field["positions"]
        hidden = g.hidden  # only read below: the mask rows stay as they are
        hidden.index_copy_(0, starts, self.embed(field["heads"]).to(self.dtype))
        placed = field["context_rows"].shape[0]  # new prefix rows, and a joint call's span, live in the cache
        span_keys = span_values = None
        if fused and not joint:
            # Local keys of every layer: all span rows, then all block rows (the order ``gather`` indexes).
            local_keys = torch.empty(self.num_layers, count * (rows + block), kv_heads, dim, device=self.device,
                                     dtype=self.dtype)
            local_values = torch.empty_like(local_keys)
            block_rows = g.identity[count * rows:]
            bounds = field["cu_k"]
        # The fc GEMM picks its kernel, and with it the BF16 rounding, by row count: the prefix slots and
        # the spans keep the row counts of the eager projection and of the default grid unless merged.
        parts = [g.features] if fast.merge_projection else [g.features[:slots], g.features[slots:]]
        parts = [part for part in parts if part.shape[0]]
        if parts and fused:
            context = [F.linear(part, self.fc) for part in parts]
            context = self._rms(context[0] if len(context) == 1 else torch.cat(context), self.hidden_norm)
            kv = F.linear(context, self.context_kv).view(-1, self.num_layers, 2 * self.kv_size)
            if placed:
                norm_rope_place(kv[:placed], None, self.context_k_norm, field["context_positions"],
                                field["context_rows"], self.rope_cache, None, self.k_cache, self.v_cache,
                                0, kv_heads, dim, self.eps)
            if not joint and rows:  # into the leaves' local keys, in front of the block rows
                norm_rope_place(kv[slots:], None, self.context_k_norm, field["context_positions"][slots:],
                                g.identity, self.rope_cache, None, local_keys, local_values, 0, kv_heads,
                                dim, self.eps)
        elif len(parts) == 1 and slots and rows:  # merged
            keys, values = self._project_context_forward(parts[0], field["context_positions"])
            scatter_context_kv(keys[:, :placed], values[:, :placed], self.k_cache, self.v_cache,
                               field["context_rows"])
            span_keys, span_values = keys[:, slots:], values[:, slots:]
        elif parts:
            if slots:
                keys, values = self._project_context_forward(parts[0], field["context_positions"][:slots])
                scatter_context_kv(keys, values, self.k_cache, self.v_cache, field["context_rows"][:slots])
            if rows:
                span_keys, span_values = self._project_context_forward(
                    parts[-1], field["context_positions"][slots:])
                if joint:
                    scatter_context_kv(span_keys, span_values, self.k_cache, self.v_cache,
                                       field["context_rows"][slots:])
        if joint:
            key_bounds = field["key_bounds"].to(torch.int32)
        else:
            prefix_cu = field["prefix_cu"].to(torch.int32)
            cu_k = None if fused else field["cu_k"].to(torch.int32)
        normalized = self._rms(hidden, self.layers[0].input_norm)
        for i, layer in enumerate(self.layers):
            qkv = F.linear(normalized, layer.qkv)
            if fused:
                q = torch.empty(qkv.shape[0], heads, dim, device=self.device, dtype=self.dtype)
                if joint:
                    key_out, value_out, key_rows = self.k_cache[i], self.v_cache[i], field["block_rows"]
                else:
                    key_out, value_out, key_rows = local_keys[i], local_values[i], block_rows
                norm_rope_place(qkv[:, None], layer.q_norm, layer.k_norm, positions, key_rows, self.rope_cache,
                                q, key_out, value_out, heads, kv_heads, dim, self.eps)
            else:
                q, key, value = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
                q = self._rms(q.view(-1, heads, dim), layer.q_norm)
                key = self._rms(key.view(-1, kv_heads, dim), layer.k_norm)
                value = value.view(-1, kv_heads, dim)
                self._ops.rotary_embedding(positions, q.view(-1, self.q_size), key.view(-1, self.kv_size),
                                           dim, self.rope_cache, True)
                if joint:
                    scatter_context_kv(key[None], value[None], self.k_cache[i:i + 1], self.v_cache[i:i + 1],
                                       field["block_rows"])
            if joint:
                # Cache rows are token positions: the verified prefix, then the span, then this block.
                attention = flash(
                    q, self.k_cache[i], self.v_cache[i], max_seqlen_q=q.shape[0],
                    cu_seqlens_q=self._cu(q.shape[0]), max_seqlen_k=self.max_len, cu_seqlens_k=key_bounds,
                    causal=False, fa_version=self.fa_version,
                )
            else:
                # Every sequence has a key (the prefix is not empty, a leaf holds its block): no masks.
                prefix_out, prefix_lse = flash(
                    q, self.k_cache[i], self.v_cache[i], max_seqlen_q=q.shape[0],
                    cu_seqlens_q=self._cu(q.shape[0]), max_seqlen_k=self.max_len, cu_seqlens_k=prefix_cu,
                    causal=False, return_softmax_lse=True, fa_version=self.fa_version,
                )
            if fused and not joint:
                attention = local_merge_attention(q, local_keys[i], local_values[i], field["gather"], bounds,
                                                  prefix_out, prefix_lse, block, rows + block)
            elif not joint:
                keys_i, values_i = gather_context_kv(
                    span_keys[i] if rows else None, span_values[i] if rows else None,
                    key, value, field["gather"],
                )
                local_out, local_lse = flash(
                    q, keys_i, values_i,
                    max_seqlen_q=block, cu_seqlens_q=cu_q, max_seqlen_k=rows + block, cu_seqlens_k=cu_k,
                    causal=False, return_softmax_lse=True, fa_version=self.fa_version,
                )
                attention = torch.empty_like(q)
                self._ops.merge_attn_states(attention, prefix_out, prefix_lse, local_out, local_lse)
            update = F.linear(attention.reshape(-1, self.q_size), layer.o_proj)
            if fused:
                hidden, normalized = add_rms_norm(hidden, update, layer.post_norm, self.eps)
            else:
                hidden = hidden + update
                normalized = self._rms(hidden, layer.post_norm)
            gate_up = F.linear(normalized, layer.gate_up)
            activated = torch.empty(gate_up.shape[0], gate_up.shape[1] // 2, device=self.device, dtype=self.dtype)
            torch.ops._C.silu_and_mul(activated, gate_up)
            update = F.linear(activated, layer.down)
            last = i + 1 == self.num_layers
            if fused:  # with the next layer's input norm, or the output norm of every row
                hidden, normalized = add_rms_norm(
                    hidden, update, self.norm if last else self.layers[i + 1].input_norm, self.eps)
            else:
                hidden = hidden + update
                if not last:
                    normalized = self._rms(hidden, self.layers[i + 1].input_norm)
        if fused:
            hidden = normalized[1:block] if count == 1 else normalized[keep]
        else:
            hidden = self._rms(hidden[1:block] if count == 1 else hidden[keep], self.norm)
        if fast.head is None:
            logits = self.lm_head(hidden)
            ids = scaled_argmax(logits)
        else:
            logits = F.linear(hidden, fast.head)  # unscaled; the argmax applies the scale
            ids = scaled_argmax(logits, fast.head_scale)
        if fast.vocab_ids is not None:
            ids = fast.vocab_ids.index_select(0, ids)
        return logits, ids

    def _capture_fast(self, count, rows, block):
        g = self._fast_buffers(count, rows, block)
        self._grid_constants(count, rows, block)
        self._cu(count * block)  # fixed Q bound, created outside the capture
        if self._context_stream is None:
            self._context_stream = torch.cuda.Stream(device=self.device)
        stream = self._context_stream
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._fast_forward(g, block)
        current.wait_stream(stream)
        g.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g.graph, pool=self._grid_pool, stream=stream):
            g.logits, g.ids = self._fast_forward(g, block)
        current.wait_stream(stream)
        self.stats["graph_captures"] += 1
        return g

    def _fast_logits(self, logits, owned):
        """Graph logits as the public [rows, V] layout: -inf for tokens outside a reduced vocabulary."""
        fast = self._fast
        if fast.head_scale is not None:
            logits = torch.mul(logits, fast.head_scale)  # FP32, as the lm_head callable returns
        elif owned and fast.vocab_ids is None:
            logits = logits.clone()
        if fast.vocab_ids is None:
            return logits
        full = torch.full((logits.shape[0], self._fast_full_vocab), float("-inf"), device=self.device,
                          dtype=logits.dtype)
        return full.index_copy_(1, fast.vocab_ids, logits)

    def _run_fast(self, shape, active, spans, lengths, parents, block_lengths, prefix_features, prefix_len, *,
                  eager=False):
        """Pad one call to ``shape`` and replay its fast graph: (logits, token IDs) of all L*(K-1) slots."""
        count, rows = shape
        block, slots, scratch, steps = self._grid_block, self._fast.prefix_rows, self.max_len, self._fast_steps
        g = self._grid_graphs[shape]
        n, at = len(lengths), {name: part.start for name, part in g.fields.items()}
        self.truncate(prefix_len)
        start = self.cached_prefix
        new = prefix_len - start
        if new > slots:  # a prompt, or more commits than slots: the eager projection
            self._extend_prefix(prefix_features, prefix_len)
            start, new = prefix_len, 0
            self.stats["grid_prefix_eager_calls"] += 1
        host = self._fast_host
        a = at["heads"]
        host[a: a + n] = [int(r["head"]) for _, r in active]
        host[a + n: a + count] = 0
        a = at["positions"]
        for i in range(count):
            host[a + i * block: a + (i + 1) * block] = (
                np.minimum(parents[i] + 1 + steps[:block], self.max_len - 1) if i < n else steps[:block])
        a, b = at["context_positions"], at["context_rows"]
        host[a: a + new] = host[b: b + new] = start + steps[:new]
        host[a + new: a + slots] = 0
        host[b + new: b + slots] = scratch
        a += slots
        for i in range(count):
            m = lengths[i] if i < n else 0
            host[a + i * rows: a + i * rows + m] = prefix_len + steps[:m]
            host[a + i * rows + m: a + (i + 1) * rows] = 0
        if g.joint:
            m, k = lengths[0], block_lengths[0]
            b += slots
            host[b: b + m] = prefix_len + steps[:m]
            host[b + m: b + rows] = scratch
            a = at["block_rows"]
            host[a: a + k] = prefix_len + m + steps[:k]
            host[a + k: a + block] = scratch
            host[at["key_bounds"]: at["key_bounds"] + 2] = 0, prefix_len + m + k
        else:
            gather, cu_k = [], [0]
            for i in range(count):
                k, m = (block_lengths[i], lengths[i]) if i < n else (1, 0)
                gather += [count * rows + i * block + j for j in range(k)] + [i * rows + j for j in range(m)]
                cu_k.append(len(gather))
            a = at["gather"]
            host[a: a + len(gather)] = gather
            host[a + len(gather): a + count * (rows + block)] = gather[0]  # past cu_k[-1]: never attended
            host[at["cu_k"]: at["cu_k"] + count + 1] = cu_k
            host[at["prefix_cu"]: at["prefix_cu"] + 2] = 0, prefix_len
        g.meta.copy_(self._fast_upload[: g.meta.shape[0]])  # blocking, so the buffer is free for the next call
        # Padded feature rows keep earlier (finite) values: they are projected and never read.
        if new:
            g.features[:new].copy_(prefix_features[start:prefix_len])
        for i, span in enumerate(spans):
            if lengths[i]:
                g.features[slots + i * rows: slots + i * rows + lengths[i]].copy_(span)
        if eager:
            logits, ids = self._fast_forward(g, block)
        else:
            g.graph.replay()
            logits, ids = g.logits, g.ids
            self.stats["graph_replays"] += 1
            self.stats["grid_graph_replays"] += 1
        if new:
            self.stats["grid_prefix_graph_calls"] += 1
            self.cached_prefix = prefix_len
            self.context_tokens_projected += new
            self.stats["context_tokens_projected"] += new
            self.stats["context_projection_rows"] += new
            if not (self._fast.merge_projection and sum(lengths)):
                self.stats["context_weight_passes"] += 1  # merged with spans: the caller counts that pass
        self.stats["grid_padded_rows"] += count * (rows + block) - sum(lengths) - sum(block_lengths)
        return logits, ids
