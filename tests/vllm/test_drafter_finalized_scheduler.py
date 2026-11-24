"""2DSD scheduling semantics against exact recurrent and event oracles."""

import collections
import itertools
import random
import unittest
from types import SimpleNamespace

import torch
from helpers import FakeTreeRunner, FusedReadoutRunner, NodeStateRunner, oracle_next
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor


class GrowingRunner(FakeTreeRunner):
    def ensure_capacity(self, required):
        while required > self.max_nodes:
            self.max_nodes *= 2
        return self.max_nodes


class Draft:
    """Oracle loop-context drafter: the exact continuation of each request's path.

    A loop-context request carries features only. ``attach`` records, for every
    request the executor builds, the draft event behind it (verified prefix,
    path tokens, conditioning position and depth); the drafter checks that the
    request matches the event and proposes the exact greedy continuation of the
    path, optionally corrupted.
    """

    input_format = "loop-context"
    R = 4  # the attached executor's loop count

    def __init__(self, causal=True, corrupt=False):
        self.causal, self.corrupt = causal, corrupt
        self.stats = {}
        self.events = []
        self.reset()

    def reset(self):
        self.requests = []
        self.batches = []

    def attach(self, executor):
        build = executor._context_request

        def recorded(event, width, path=None):
            request = build(event, width, path)
            self.events.append(
                dict(
                    tokens=executor.prefix_tokens
                    + [n.token for n in executor._path(event.head)],
                    verified_end=executor.runner.prefix_length - 1,
                    anchor_pos=event.parent_position,
                    anchor_depth=event.depth,
                    head=event.head.token,
                    k=width,
                )
            )
            return request

        executor._context_request = recorded
        self.R = executor.runner.R
        return executor

    def draft_context_many(self, requests):
        assert len(self.events) == len(requests)
        events, self.events = self.events, []
        self.batches.append(len(requests))
        results = []
        for request, event in zip(requests, events):
            self.requests.append(event)
            self.assert_boundary(event)
            self.assert_matches(request, event)
            sequence = list(event["tokens"])
            slots = []
            for i in range(event["k"] - 1):
                token = self.propose(sequence, i)
                slots.append(token)
                sequence.append(token)
            results.append(
                SimpleNamespace(token_ids=torch.tensor(slots, dtype=torch.long))
            )
        return results

    def propose(self, sequence, slot):
        token = oracle_next(sequence, self.causal, self.R)
        if self.corrupt and (sum(sequence) + slot) % 3 == 0:
            token = (token + 7) % 97
        return token

    def assert_boundary(self, event):
        a, v = event["anchor_pos"], event["verified_end"]
        assert len(event["tokens"]) == a + 2
        assert event["tokens"][-1] == event["head"]
        assert v <= a
        if v == a:
            assert event["anchor_depth"] == self.R

    def assert_matches(self, request, event):
        """The loop-context request carries exactly the event's boundaries."""
        assert request["prefix_len"] == event["verified_end"] + 1
        assert request["head"] == event["head"] and request["k"] == event["k"]
        assert len(request["prefix_features"]) >= request["prefix_len"]
        span = request["span_features"]
        if event["anchor_depth"] == self.R:
            assert span is None
        else:
            assert len(span) == event["anchor_pos"] - event["verified_end"]


def executor(runner, draft, k, stage, **kwargs):
    result = FinalizedTreeExecutor(runner, draft, k, stage, **kwargs)
    return draft.attach(result) if draft is not None else result


class ControlledRunner(GrowingRunner):
    """Deterministic readout disagreements; causal token identity is explicit."""

    def __init__(self, kind="promotion", loops=4):
        self.kind = kind
        super().__init__(8, causal=False, loops=loops)

    def prefill(self, tokens):
        result = super().prefill(tokens)
        result.hidden = tokens.to(self.dtype).reshape(-1, 1) + 1000 * self.R
        result.features = result.hidden.clone()
        return result

    def advance(self, spans):
        results = super().advance(spans)
        for span, result in zip(spans, results):
            value = self.nodes[span.node_id]["token"] + (span.loop + 1) * 1000
            result.hidden = torch.tensor([[value]], dtype=self.dtype)
            result.features = result.hidden.clone()
        return results

    def logits(self, hidden):
        values = hidden[:, 0].long()
        tokens, depths = values % 1000, values // 1000
        predictions = (tokens + 1) % 97
        if self.kind == "wide":
            predictions = torch.where(depths == 2, (tokens + 7) % 97, predictions)
        else:
            predictions = torch.where((tokens == 12) & (depths == 1), 50, predictions)
            if self.kind == "fork":
                predictions = torch.where(
                    (tokens == 12) & (depths >= 2), 60, predictions
                )
            if self.kind == "nested":
                predictions = torch.where(
                    (tokens == 13) & (depths == 2), 70, predictions
                )
        logits = torch.full((len(hidden), 97), -10.0, dtype=self.dtype)
        logits.scatter_(1, predictions[:, None], 10.0)
        return logits


def expected(prompt, count, causal=True, eos=(), loops=4):
    sequence, output = list(prompt), []
    for _ in range(count):
        token = oracle_next(sequence, causal, loops)
        output.append(token)
        sequence.append(token)
        if token in eos:
            break
    return output


class FinalizedSchedulerTests(unittest.TestCase):
    def test_all_stages_and_widths_match_exact_causal_ar(self):
        for stage in ("stage1", "stage2a", "stage2b"):
            for k in (1, 4, 8, 16):
                for corrupt in (False, True):
                    with self.subTest(stage=stage, k=k, corrupt=corrupt):
                        runner = GrowingRunner(8)
                        draft = Draft(corrupt=corrupt) if k > 1 else None
                        result = executor(runner, draft, k, stage).generate(
                            [3, 4, 5], 37, []
                        )
                        self.assertEqual(result["output_ids"], expected([3, 4, 5], 37))
                        self.assertEqual(
                            sum(result["stats"]["commit_sources"].values()), 36
                        )
                        self.assertEqual(
                            sum(x["commits"] for x in result["stats"]["per_call"]), 36
                        )

    def test_loop_spec_d1_and_dflash_boundary_call_counts(self):
        for k, d1, calls in ((1, 1, 67), (4, 1, 19), (4, 4, 64)):
            with self.subTest(k=k, d1=d1):
                runner = GrowingRunner(8, causal=False)
                draft = Draft(causal=False) if k > 1 else None
                result = executor(runner, draft, k, "stage1", d1=d1).generate(
                    [10], 65, []
                )
                self.assertEqual(result["output_ids"], expected([10], 65, False))
                self.assertEqual(result["stats"]["calls"], calls)
                if k == 1:
                    self.assertEqual(result["stats"]["drafter_batches"], 0)
                if d1 == 4:
                    self.assertEqual({r["anchor_depth"] for r in draft.requests}, {4})

    def test_frozen_is_retained_until_final_and_can_win_without_promotion(self):
        runner, draft = ControlledRunner(), Draft(causal=False)
        result = executor(runner, draft, 4, "stage1", record_trace=True).generate(
            [10], 20, []
        )
        self.assertEqual(result["output_ids"], list(range(11, 31)))
        self.assertGreater(result["stats"]["frozen_created"], 0)
        self.assertGreater(result["stats"]["commit_sources"]["frozen"], 0)
        self.assertEqual(result["stats"]["promotions"], 0)
        freezes = [t for t in result["scheduler_trace"] if t["event"] == "freeze"]
        self.assertEqual(freezes[0]["tokens"], [13, 14])

    def test_promotion_reuses_depth_keeps_role_and_drafts_immediately(self):
        runner, draft = ControlledRunner(), Draft(causal=False)
        result = executor(runner, draft, 4, "stage2a", record_trace=True).generate(
            [10], 20, []
        )
        self.assertEqual(result["output_ids"], list(range(11, 31)))
        self.assertGreater(result["stats"]["promotions"], 0)
        self.assertGreater(result["stats"]["promoted_heads_committed"], 0)
        self.assertEqual(result["stats"]["forks"], 0)
        trace = result["scheduler_trace"]
        promotion = next(t for t in trace if t["event"] == "promotion")
        self.assertTrue(
            any(
                t["event"] == "anchor"
                and t["cause"].startswith("promotion")
                and t["call"] == promotion["call"]
                for t in trace
            )
        )
        chosen = next(t for t in trace if t["event"] == "commit" and t["token"] == 13)
        self.assertEqual(chosen["source"], "frozen")
        self.assertEqual(chosen["depth"], 4)
        self.assertIn(2, [r["anchor_depth"] for r in draft.requests])

    def test_d2_fork_is_separate_from_promotion_and_preserves_target_output(self):
        times = {}
        for stage in ("stage1", "stage2a", "stage2b"):
            runner, draft = ControlledRunner("fork"), Draft(causal=False)
            result = executor(runner, draft, 4, stage, record_trace=True).generate(
                [10], 20, []
            )
            self.assertEqual(result["output_ids"], [11, 12] + list(range(60, 78)))
            self.assertEqual(result["stats"]["promotions"], 0)
            times[stage] = result["stats"]["calls"]
            if stage == "stage2b":
                self.assertGreater(result["stats"]["commit_sources"]["fork"], 0)
                self.assertIn(2, [r["anchor_depth"] for r in draft.requests])
            else:
                self.assertEqual(result["stats"]["forks"], 0)
        self.assertLess(times["stage2b"], times["stage2a"])

    def test_no_hidden_64_node_pruning_and_all_nodes_advance(self):
        runner, draft = ControlledRunner("wide"), Draft(causal=False)
        result = executor(runner, draft, 16, "stage2b").generate([10], 80, [])
        self.assertEqual(result["output_ids"], expected([10], 80, False))
        self.assertGreater(result["stats"]["max_live_nodes"], 64)
        self.assertGreater(runner.max_nodes, 64)
        self.assertTrue(all(depth == 4 for _, _, depth in runner.release_roots))

    def test_promoted_suffix_rejection_is_frozen_again_without_second_d2_event(self):
        runner, draft = ControlledRunner("nested"), Draft(causal=False)
        result = executor(runner, draft, 4, "stage2b", record_trace=True).generate(
            [10], 20, []
        )
        self.assertEqual(result["output_ids"], list(range(11, 31)))
        trace = result["scheduler_trace"]
        nested = [
            x
            for x in trace
            if x["event"] == "anchor" and x["cause"] == "promotion_reject"
        ]
        self.assertEqual(len(nested), 1)
        self.assertEqual(nested[0]["token"], 70)
        fourteen = next(x for x in trace if x["event"] == "commit" and x["token"] == 14)
        self.assertEqual(fourteen["source"], "frozen")
        self.assertEqual(fourteen["depth"], 4)
        self.assertEqual(result["stats"]["forks"], 0)

    def test_eos_partial_blocks_and_request_reuse(self):
        runner, draft = GrowingRunner(8, causal=False), Draft(causal=False)
        reused = executor(runner, draft, 16, "stage2b")
        for count, eos in ((1, []), (2, []), (3, []), (30, [11]), (30, [13]), (40, [])):
            with self.subTest(count=count, eos=eos):
                result = reused.generate([10], count, eos)
                self.assertEqual(
                    result["output_ids"], expected([10], count, False, eos)
                )
                self.assertFalse(reused.nodes)
                self.assertFalse(reused.events)

    def test_kv_row_counters_match_each_call_prefix_and_executed_loops(self):
        class Recording(GrowingRunner):
            def advance(self, spans):
                self.seen.append((self.prefix_length, {s.loop + 1 for s in spans}))
                return super().advance(spans)

        for stage, k, d1, d2 in (("stage1", 1, 1, 2), ("stage1", 16, 2, 2), ("stage2b", 1, 1, 2),
                                 ("stage2a", 8, 1, 3), ("stage2b", 16, 2, 3)):
            with self.subTest(stage=stage, k=k, d1=d1, d2=d2):
                runner, draft = Recording(8), Draft() if k > 1 else None
                runner.seen = []
                stats = executor(runner, draft, k, stage, d1=d1, d2=d2).generate(
                    [3, 4, 5], 41, [])["stats"]
                self.assertEqual(stats["prefix_rows"], sum(n for n, _ in runner.seen))
                self.assertEqual(stats["loop_prefix_rows"], {
                    str(d): sum(n for n, loops in runner.seen if d in loops) for d in range(1, 5)})
                self.assertEqual(sum(stats["loop_prefix_rows"].values()),
                                 sum(n * len(loops) for n, loops in runner.seen))
                self.assertGreaterEqual(stats["tree_rows_bound"], sum(len(loops) for _, loops in runner.seen))
                verified = [r["verified_end"] + 1 for r in draft.requests] if draft else []
                self.assertEqual(stats["drafter_prefix_rows"] > 0, k > 1)
                if draft:
                    # One batch per drafting step; every request of a batch sees the same verified prefix.
                    starts = [sum(draft.batches[:i]) for i in range(len(draft.batches))]
                    self.assertEqual(stats["drafter_prefix_rows"], sum(verified[i] for i in starts))


class DrafterContractTests(unittest.TestCase):
    def test_drafter_must_use_the_loop_context_input_format(self):
        class Legacy:
            def reset(self):
                pass

        class Inflight(Legacy):
            input_format = "inflight-embedding"

        for drafter in (Legacy(), Inflight()):
            with self.subTest(drafter=type(drafter).__name__), \
                    self.assertRaisesRegex(ValueError, "loop-context"):
                FinalizedTreeExecutor(GrowingRunner(8), drafter, 4, "stage2b")
        with self.assertRaisesRegex(ValueError, "require a drafter"):
            FinalizedTreeExecutor(GrowingRunner(8), None, 4, "stage2b")
        for stage in ("stage1", "stage2b"):
            control = FinalizedTreeExecutor(GrowingRunner(8), None, 1, stage)
            self.assertFalse(control.loop_context)
            self.assertFalse(control.context_depths)
        drafted = FinalizedTreeExecutor(GrowingRunner(8), Draft(), 4, "stage2b")
        self.assertTrue(drafted.loop_context)
        self.assertEqual(drafted.context_depths, {1, 2})


class ContextDraft(Draft):
    """Loop-context drafter that rebuilds its sequence from features alone.

    ControlledRunner features are token + 1000 * loop, so the verified prefix
    must decode at loop R and the span at the event depth along the live path
    from the cursor to the head's parent.
    """

    def __init__(self):
        super().__init__(causal=False)
        self.executor = None

    def reset(self):
        super().reset()
        self.depths, self.spans = [], []

    def draft_context_many(self, requests):
        executor = self.executor
        results = []
        for request in requests:
            n = request["prefix_len"]
            prefix = request["prefix_features"][:n, 0].long()
            assert n == executor.runner.prefix_length
            assert (prefix // 1000 == executor.runner.R).all()
            sequence = (prefix % 1000).tolist()
            assert sequence == executor.prefix_tokens
            span, depth, parent = request["span_features"], executor.runner.R, None
            if span is not None:
                values = span[:, 0].long()
                depths = set((values // 1000).tolist())
                assert len(depths) == 1, depths
                depth = depths.pop()
                tokens = (values % 1000).tolist()
                node = executor.cursor
                for i, token in enumerate(tokens):
                    assert node is not None and node.token == token
                    assert node.depth >= depth
                    parent = node
                    node = node.children.get(tokens[i + 1]) if i + 1 < len(tokens) else None
                sequence += tokens
                assert request["head"] in parent.children
            self.depths.append(depth)
            self.spans.append(0 if span is None else len(span))
            sequence.append(request["head"])
            slots = []
            for _ in range(request["k"] - 1):
                token = oracle_next(sequence, False)
                slots.append(token)
                sequence.append(token)
            results.append(
                SimpleNamespace(token_ids=torch.tensor(slots, dtype=torch.long))
            )
        self.batches.append(len(requests))
        return results


class LoopContextSchedulerTests(unittest.TestCase):
    """Features alone reconstruct the token path of every draft event."""

    @staticmethod
    def run_context(kind, k, stage, count):
        draft = ContextDraft()
        context = FinalizedTreeExecutor(ControlledRunner(kind), draft, k, stage)
        draft.executor = context
        return context, draft

    def test_spans_carry_event_depth_features_and_match_token_drafting(self):
        for kind in ("promotion", "fork", "nested", "wide"):
            for stage in ("stage1", "stage2a", "stage2b"):
                for k in (4, 16):
                    with self.subTest(kind=kind, stage=stage, k=k):
                        count = 80 if kind == "wide" else 20
                        token_draft = Draft(causal=False)
                        token = executor(
                            ControlledRunner(kind), token_draft, k, stage
                        ).generate([10], count, [])
                        context_executor, draft = self.run_context(kind, k, stage, count)
                        context = context_executor.generate([10], count, [])
                        self.assertTrue(context_executor.loop_context)
                        self.assertEqual(context["output_ids"], token["output_ids"])
                        for name in ("calls", "commit_sources", "promotions", "forks",
                                     "frozen_created", "draft_slots"):
                            self.assertEqual(context["stats"][name], token["stats"][name])
                        self.assertEqual(draft.depths, [r["anchor_depth"] for r in token_draft.requests])
                        self.assertEqual(
                            draft.spans,
                            [r["anchor_pos"] - r["verified_end"] for r in token_draft.requests],
                        )
                        self.assertIn(1, draft.depths)
                        self.assertGreater(max(draft.spans), 1)
                        if stage == "stage2b":
                            self.assertIn(2, draft.depths)

    def test_nodes_keep_drafting_depth_features_until_commit(self):
        for stage, kept in (("stage1", {1}), ("stage2b", {1, 2})):
            context, draft = self.run_context("fork", 8, stage, 30)
            calls = []
            verify = context._verify

            def checked():
                for node in context.nodes.values():
                    self.assertEqual(set(node.loop_features), {d for d in kept if node.depth >= d})
                calls.append(len(context.nodes))
                return verify()

            context._verify = checked
            context.generate([10], 30, [])
            self.assertTrue(calls)


class FlopsCounterTests(unittest.TestCase):
    """Attention key counters equal their reconstruction from the per-call record."""

    def test_history_and_tree_pairs(self):
        for stage, k in (("stage1", 4), ("stage2b", 16), ("stage2b", 1)):
            with self.subTest(stage=stage, k=k):
                prompt = [3, 4, 5]
                result = executor(
                    GrowingRunner(8), Draft() if k > 1 else None, k, stage
                ).generate(prompt, 37, [])
                st = result["stats"]
                prefix, history = len(prompt), 0
                for call in st["per_call"]:
                    history += call["cells"] * prefix
                    prefix += call["commits"]
                self.assertEqual(st["attention_history_pairs"], history)
                self.assertEqual(st["attention_history_pairs"], sum(
                    c["cells"] * p for c, p in zip(st["per_call"], _prefixes(len(prompt), st["per_call"]))))
                # Every cell reads at least its own tree row and at most the deepest path.
                self.assertGreaterEqual(st["attention_tree_pairs"], st["cells"])
                self.assertLessEqual(st["attention_tree_pairs"], st["cells"] * (k * 4 + 4))


def _prefixes(prompt, calls):
    out, prefix = [], prompt
    for call in calls:
        out.append(prefix)
        prefix += call["commits"]
    return out


# Runner protocols of the executor: spans with head(hidden) readouts (Ouro,
# Nanbeige), spans with fused readouts and node state with fused readouts (Dory;
# the two an input/looped/output-block target such as Inkling2 can use).
PROTOCOLS = (None, FusedReadoutRunner, NodeStateRunner)


def scheduled(runner, draft, k, stage, d1=1, protocol=None, **kwargs):
    """An executor on the span protocol or behind a fused-readout protocol."""
    if protocol is not None:
        runner = protocol.for_schedule(runner, stage, d1)
    return executor(runner, draft, k, stage, d1=d1, **kwargs)


def longest_span(draft):
    """In-flight positions behind a drafting parent, over the last request."""
    return max(r["anchor_pos"] - r["verified_end"] for r in draft.requests)


class TableRunner(ControlledRunner):
    """Loop d of token t reads out table[d][t]: any agreement between loops."""

    def __init__(self, table):
        self.table = table
        super().__init__("table", max(table))

    def logits(self, hidden):
        values = hidden[:, 0].long().tolist()
        predictions = torch.tensor([self.table[v // 1000][v % 1000] for v in values])
        logits = torch.full((len(hidden), 97), -10.0, dtype=self.dtype)
        logits.scatter_(1, predictions[:, None], 10.0)
        return logits


class TableDraft(Draft):
    """Drafts table[t] after token t, whatever the context before it."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def propose(self, sequence, slot):
        return self.table[sequence[-1]]


class ThreeLoopSchedulerTests(unittest.TestCase):
    """R = 3, the loop count of Inkling2.

    Stage 1 reviews at d1 = 1, 2 or 3 (3 is the final review: plain block
    drafting). Stage 2 has the single schedule (d1, d2) = (1, 2), so the deeper
    review runs one loop before the final one. Schedules run on every runner
    protocol. A drafting parent has at most (R - 1) * k = 2k in-flight
    positions behind it (a path of at most 3k), the sizes behind the drafter's
    span grid and a runner's static path width.
    """

    SCHEDULES = (
        ("stage1", 1), ("stage1", 2), ("stage1", 3), ("stage2a", 1), ("stage2b", 1),
    )
    BLOCKS = (1, 8, 16)

    CASES = [
        (stage, d1, k, protocol)
        for (stage, d1), k, protocol in itertools.product(SCHEDULES, BLOCKS, PROTOCOLS)
    ]

    def test_every_schedule_and_block_equals_exact_causal_ar(self):
        """Token-for-token equality with three-loop AR, for exact and corrupted
        drafts and prompts of three tokens and of one. Loops 1 and 2 almost never
        agree with loop 3 here, so a node that ran a wrong number of loops or was
        read at a wrong depth changes the output."""
        self.assertNotEqual(expected([7], 37, loops=3), expected([7], 37, loops=4))
        for (stage, d1, k, protocol), prompt in itertools.product(
            self.CASES, ([3, 4, 5], [7])
        ):
            for corrupt in (False, True)[: 1 + (k > 1)]:
                with self.subTest(stage=stage, d1=d1, k=k, protocol=protocol,
                                  prompt=prompt, corrupt=corrupt):
                    runner = GrowingRunner(8, loops=3)
                    draft = Draft(corrupt=corrupt) if k > 1 else None
                    result = scheduled(runner, draft, k, stage, d1, protocol).generate(
                        prompt, 37, []
                    )
                    self.assertEqual(result["output_ids"], expected(prompt, 37, loops=3))
                    stats = result["stats"]
                    self.assertEqual(sum(stats["commit_sources"].values()), 36)
                    # The per-loop counters hold loops 1 to 3 and what the runner ran.
                    ran = collections.Counter(
                        loop + 1 for call in runner.trace for _, loop, _ in call
                    )
                    self.assertEqual(
                        stats["loop_cells"], {str(d): ran[d] for d in (1, 2, 3)}
                    )
                    for name in ("loop_prefix_rows", "draft_events_by_depth"):
                        self.assertEqual(set(stats[name]), {"1", "2", "3"})
                    self.assertLessEqual(runner.stats["longest_path"], 3 * k)
                    if draft is not None:
                        self.assertLessEqual(longest_span(draft), 2 * k)

    def test_call_counts_are_those_of_a_three_deep_pipeline(self):
        """When every readout agrees a block starts every d1 calls and the last
        block needs three, so n tokens take d1 * ((n - 2) // k) + 3 calls and
        2 // d1 blocks are in flight behind a drafting parent: with its own
        block a candidate lies up to 3k tokens from the cursor, the path width
        a runner must hold. Guards the pipeline depth, and S_count, against a
        loop count other than 3."""
        n = 65
        for stage, d1, k, protocol in self.CASES:
            with self.subTest(stage=stage, d1=d1, k=k, protocol=protocol):
                runner = GrowingRunner(8, causal=False, loops=3)
                draft = Draft(causal=False) if k > 1 else None
                result = scheduled(runner, draft, k, stage, d1, protocol).generate(
                    [10], n, []
                )
                self.assertEqual(result["output_ids"], expected([10], n, False))
                stats = result["stats"]
                self.assertEqual(stats["calls"], d1 * ((n - 2) // k) + 3)
                self.assertEqual(stats["S_count"], 3 * (n - 1) / stats["calls"])
                self.assertEqual(stats["frozen_created"] + stats["forks"], 0)
                self.assertEqual(runner.stats["longest_path"], (2 // d1 + 1) * k)
                if draft is None:
                    self.assertEqual(stats["drafter_batches"], 0)
                else:  # the first block is drafted from the prompt, the others at d1
                    depths = {r["anchor_depth"] for r in draft.requests}
                    self.assertEqual(depths, {d1, 3})
                    self.assertEqual(longest_span(draft), (2 // d1) * k)

    def test_eos_and_budget_cut_a_block_that_commits_in_one_call(self):
        """Every readout agrees, so a whole block reaches loop 3 in one call. EOS
        in its middle, EOS as the first token and budgets of one to three tokens
        must cut the output exactly and leave the reused executor empty."""
        requests = (
            (1, []), (2, []), (3, []), (30, [11]), (30, [13]), (30, [20]), (40, []),
        )
        for stage, d1, k, protocol in self.CASES:
            runner = GrowingRunner(8, causal=False, loops=3)
            draft = Draft(causal=False) if k > 1 else None
            reused = scheduled(runner, draft, k, stage, d1, protocol)
            for count, eos in requests:
                with self.subTest(stage=stage, d1=d1, k=k, protocol=protocol,
                                  count=count, eos=eos):
                    result = reused.generate([10], count, eos)
                    self.assertEqual(
                        result["output_ids"], expected([10], count, False, eos)
                    )
                    self.assertFalse(reused.nodes)
                    self.assertFalse(reused.events)
                    stats = result["stats"]
                    if eos == [11]:  # the first token: no target call at all
                        self.assertEqual(stats["calls"], 0)
                    if eos == [20] and k == 16:
                        self.assertGreater(stats["per_call"][-1]["commits"], 1)

    def test_request_ending_at_the_last_context_position_fits(self):
        """No draft, anchor, fork or bonus token may lie at or beyond prompt +
        max_new: a runner has no row there once the request fills its context
        (the fake refuses such a candidate). The budgets cut a block short, end
        it exactly and run one or two tokens into the next, with agreeing
        readouts, where a frontier anchor would follow the cut, and with
        disagreeing ones."""
        for (stage, d1, k, protocol), causal, count in itertools.product(
            self.CASES, (False, True), (1, 2, 3, 15, 16, 17, 18, 33)
        ):
            with self.subTest(stage=stage, d1=d1, k=k, protocol=protocol,
                              causal=causal, count=count):
                runner = GrowingRunner(8, causal=causal, loops=3)
                runner.max_len = 2 + count
                draft = Draft(causal, corrupt=causal) if k > 1 else None
                result = scheduled(runner, draft, k, stage, d1, protocol).generate(
                    [10, 20], count, []
                )
                self.assertEqual(
                    result["output_ids"], expected([10, 20], count, causal, loops=3)
                )

    def test_unverified_eos_candidates_never_stop_decoding(self):
        """Every id that three-loop AR does not emit is an EOS id, so rejected
        drafts, early-readout anchors and forks carry EOS. Decoding must stop
        only where AR emits EOS: at the first token, the second, or later."""
        full = expected([10], 30, loops=3)
        never = set(range(97)) - set(full)
        self.assertEqual(len(set(full[:13])), 13)
        for stage, d1, k, protocol in self.CASES:
            with self.subTest(stage=stage, d1=d1, k=k, protocol=protocol):
                runner = GrowingRunner(8, loops=3)
                draft = Draft(corrupt=True) if k > 1 else None
                reused = scheduled(runner, draft, k, stage, d1, protocol)
                for stop in (None, 0, 1, 12, None):  # the last: clean after a stop
                    eos = never if stop is None else never | {full[stop]}
                    result = reused.generate([10], 30, eos)
                    self.assertEqual(
                        result["output_ids"], full if stop is None else full[: stop + 1]
                    )
                    self.assertFalse(reused.nodes)
                # k = 1 reviewed at loop 3 is plain AR: the one case without candidates.
                candidates = any(t in never for _, _, t in runner.registrations)
                self.assertEqual(candidates, k > 1 or d1 < 3)

    def test_random_readout_agreement_equals_the_loop_3_chain(self):
        """Loops 1 and 2 and the drafter agree with loop 3 on a random part of a
        seven-token alphabet, so d1 rejections, frozen wins, promotions, forks,
        bonus restarts and EOS candidates overlap in one tree: the interplay
        that single events do not reach, with d2 one loop before the final."""
        totals = collections.Counter()
        for seed in range(12):
            rng = random.Random(seed)
            agree = (0.3, 0.6, 0.85)[seed % 3]
            final = [rng.randrange(7) for _ in range(7)]

            def noisy():
                return [t if rng.random() < agree else rng.randrange(7) for t in final]

            table, draft = {1: noisy(), 2: noisy(), 3: final}, noisy()
            eos = [rng.randrange(7)] if seed % 2 else []
            prompt = [rng.randrange(7)]
            want = []
            while len(want) < 30 and not (want and want[-1] in eos):
                want.append(final[want[-1] if want else prompt[0]])
            for stage, d1, k, protocol in self.CASES:
                with self.subTest(seed=seed, stage=stage, d1=d1, k=k, protocol=protocol):
                    drafter = TableDraft(draft) if k > 1 else None
                    runner = TableRunner(table)
                    result = scheduled(
                        runner, drafter, k, stage, d1, protocol
                    ).generate(prompt, 30, eos)
                    self.assertEqual(result["output_ids"], want)
                    self.assertLessEqual(runner.stats["longest_path"], 3 * k)
                    if drafter is not None and drafter.requests:
                        self.assertLessEqual(longest_span(drafter), 2 * k)
                    stats = result["stats"]
                    for name in ("promotions", "promoted_heads_committed", "bonus_tokens"):
                        totals[name] += stats[name]
                    for name in ("frozen", "fork"):
                        totals[name] += stats["commit_sources"][name]
        self.assertEqual(len(totals), 5)
        self.assertGreater(min(totals.values()), 0, totals)

    def test_frozen_suffix_runs_to_loop_3_and_wins_without_promotion(self):
        """Stage 1 never promotes: a suffix rejected at loop 1 must still run all
        three loops and win the final review as the frozen candidate."""
        for protocol in PROTOCOLS:
            with self.subTest(protocol=protocol):
                result = scheduled(
                    ControlledRunner("promotion", 3), Draft(causal=False), 4, "stage1",
                    protocol=protocol, record_trace=True,
                ).generate([10], 20, [])
                self.assertEqual(result["output_ids"], list(range(11, 31)))
                self.assertEqual(result["stats"]["promotions"], 0)
                trace = result["scheduler_trace"]
                [freeze] = [t for t in trace if t["event"] == "freeze"]
                self.assertEqual(freeze["tokens"], [13, 14])
                frozen = [
                    (t["token"], t["depth"])
                    for t in trace
                    if t["event"] == "commit" and t["source"] == "frozen"
                ]
                self.assertEqual(frozen, [(13, 3), (14, 3)])

    def test_promotion_at_loop_2_keeps_the_loops_of_the_frozen_suffix(self):
        """With d2 = R - 1 a promoted head is verified in the next call. It must
        commit as the frozen node with its three loops, not restarted; its
        continuation is drafted at depth 2 in the call of the promotion; and a
        promoted suffix that fails the loop-2 review is frozen again."""
        facts = {  # kind: (anchor after the promotion, tokens committed frozen)
            "promotion": (("promotion_frontier", 15), [13]),
            "nested": (("promotion_reject", 70), [13, 14]),
        }
        for kind, stage, protocol in itertools.product(
            facts, ("stage2a", "stage2b"), PROTOCOLS
        ):
            with self.subTest(kind=kind, stage=stage, protocol=protocol):
                draft = Draft(causal=False)
                result = scheduled(
                    ControlledRunner(kind, 3), draft, 4, stage,
                    protocol=protocol, record_trace=True,
                ).generate([10], 20, [])
                self.assertEqual(result["output_ids"], list(range(11, 31)))
                stats, trace = result["stats"], result["scheduler_trace"]
                self.assertEqual((stats["promotions"], stats["forks"]), (1, 0))
                [promotion] = [t for t in trace if t["event"] == "promotion"]
                [anchor] = [
                    t for t in trace
                    if t["event"] == "anchor" and t["cause"].startswith("promotion")
                ]
                self.assertEqual((anchor["cause"], anchor["token"]), facts[kind][0])
                self.assertEqual(
                    (anchor["call"], anchor["depth"]), (promotion["call"], 2)
                )
                self.assertIn(2, [r["anchor_depth"] for r in draft.requests])
                frozen = [
                    (t["token"], t["depth"])
                    for t in trace
                    if t["event"] == "commit" and t["source"] == "frozen"
                ]
                self.assertEqual(frozen, [(token, 3) for token in facts[kind][1]])
                self.assertEqual(stats["promoted_heads_committed"], 1)

    def test_fork_at_loop_2_is_verified_one_call_later(self):
        """A d2 fork starts one call before its parent's final review. It must
        advance with the rest of the tree, win that review as the fork after a
        single loop, and save the call a bonus restart costs."""
        for protocol in PROTOCOLS:
            calls = {}
            for stage in ("stage1", "stage2a", "stage2b"):
                with self.subTest(stage=stage, protocol=protocol):
                    draft = Draft(causal=False)
                    result = scheduled(
                        ControlledRunner("fork", 3), draft, 4, stage,
                        protocol=protocol, record_trace=True,
                    ).generate([10], 20, [])
                    self.assertEqual(
                        result["output_ids"], [11, 12] + list(range(60, 78))
                    )
                    stats = result["stats"]
                    calls[stage] = stats["calls"]
                    [commit] = [
                        t for t in result["scheduler_trace"]
                        if t["event"] == "commit" and t["token"] == 60
                    ]
                    if stage == "stage2b":
                        self.assertEqual((commit["source"], commit["depth"]), ("fork", 1))
                        self.assertIn(2, [r["anchor_depth"] for r in draft.requests])
                    else:
                        self.assertEqual((commit["source"], stats["forks"]), ("bonus", 0))
            self.assertEqual(calls["stage2b"] + 1, calls["stage2a"])

    def test_wide_fork_tree_is_not_pruned_before_loop_3(self):
        """Every loop-2 readout disagrees, so each node forks a 16-token block.
        All candidates must stay live and advance, past 64 nodes; none is
        released before its parent's final review."""
        for protocol in PROTOCOLS:
            with self.subTest(protocol=protocol):
                runner = ControlledRunner("wide", 3)
                result = scheduled(
                    runner, Draft(causal=False), 16, "stage2b", protocol=protocol
                ).generate([10], 80, [])
                self.assertEqual(result["output_ids"], expected([10], 80, False))
                self.assertGreater(result["stats"]["max_live_nodes"], 64)
                self.assertGreater(runner.max_nodes, 64)
                self.assertTrue(runner.release_roots)
                self.assertTrue(all(depth == 3 for _, _, depth in runner.release_roots))

    def test_drafter_reads_loop_3_prefix_and_event_depth_spans(self):
        """Features alone rebuild every draft context: the verified prefix at
        loop 3 and the in-flight span at the drafting depth, 1 or (after a
        promotion or a fork) 2. Guards the depth of the features a loop-context
        drafter is conditioned on, on each runner protocol."""
        for kind, stage, protocol in itertools.product(
            ("promotion", "fork", "nested", "wide"),
            ("stage1", "stage2a", "stage2b"),
            PROTOCOLS,
        ):
            with self.subTest(kind=kind, stage=stage, protocol=protocol):
                count = 80 if kind == "wide" else 20
                token_draft = Draft(causal=False)
                token = scheduled(
                    ControlledRunner(kind, 3), token_draft, 8, stage
                ).generate([10], count, [])
                runner, draft = ControlledRunner(kind, 3), ContextDraft()
                if protocol is not None:
                    runner = protocol.for_schedule(runner, stage)
                draft.executor = FinalizedTreeExecutor(runner, draft, 8, stage)
                context = draft.executor.generate([10], count, [])
                self.assertEqual(context["output_ids"], token["output_ids"])
                self.assertEqual(context["stats"]["calls"], token["stats"]["calls"])
                requests = token_draft.requests
                self.assertEqual(draft.depths, [r["anchor_depth"] for r in requests])
                self.assertEqual(
                    draft.spans, [r["anchor_pos"] - r["verified_end"] for r in requests]
                )
                self.assertGreater(max(draft.spans), 1)
                events = context["stats"]["promotions"] + context["stats"]["forks"]
                self.assertEqual(set(draft.depths), {1, 2, 3} if events else {1, 3})
                if stage == "stage2b":
                    self.assertGreater(events, 0)

    def test_stage_2_admits_only_the_schedule_1_2(self):
        """With three loops d2 = 3 would be the final review itself and d1 >= d2
        leaves no deeper review; both are rejected for either Stage 2 variant.
        Stage 1 takes d1 up to 3 and ignores d2."""

        def build(stage, d1, d2):
            runner = GrowingRunner(8, loops=3)
            return FinalizedTreeExecutor(runner, None, 1, stage, d1=d1, d2=d2)

        rejected = ((1, 3), (2, 3), (3, 3), (2, 2), (1, 1), (2, 1), (3, 2), (1, 4), (1, 0))
        for stage in ("stage2a", "stage2b"):
            for d1, d2 in rejected:
                with self.subTest(stage=stage, d1=d1, d2=d2), \
                        self.assertRaisesRegex(ValueError, "d1 < d2 < R"):
                    build(stage, d1, d2)
            self.assertEqual(build(stage, 1, 2).readout_depths, {1, 2, 3})
        for stage in ("stage1", "stage2a", "stage2b"):
            for d1 in (0, 4):
                with self.subTest(stage=stage, d1=d1), \
                        self.assertRaisesRegex(ValueError, "1 <= d1 <= R"):
                    build(stage, d1, 2)
        for d1, d2 in itertools.product((1, 2, 3), (1, 2, 3)):
            self.assertEqual(build("stage1", d1, d2).readout_depths, {d1, 3})

    def test_fused_runner_must_read_out_the_schedule_depths(self):
        """A fused-readout runner has logits only at the depths it was built for
        (an output block with its own KV per depth); a schedule that reviews at
        another depth must be refused before decoding."""
        for protocol in (FusedReadoutRunner, NodeStateRunner):
            runner = protocol(GrowingRunner(8, loops=3), {1, 3})
            FinalizedTreeExecutor(runner, None, 1, "stage1", d1=1)
            for stage, d1 in (("stage1", 2), ("stage1", 3), ("stage2a", 1), ("stage2b", 1)):
                with self.subTest(protocol=protocol, stage=stage, d1=d1), \
                        self.assertRaisesRegex(ValueError, "readout depths"):
                    FinalizedTreeExecutor(runner, None, 1, stage, d1=d1)


class TwoLoopSchedulerTests(unittest.TestCase):
    """R = 2 (Nanbeige) on the shared fakes: their loop-count parameter keeps
    the Stage 1 path of that family exact, and Stage 2 undefined."""

    def test_stage_1_equals_two_loop_ar(self):
        """The schedule run_nanbeige_finalized builds: Stage 1, d1 = 1, the
        default d2 (2, here the last loop) and k in {1, 4, 8, 16}."""
        self.assertNotEqual(expected([7], 37, loops=2), expected([7], 37, loops=3))
        for k in (1, 4, 8, 16):
            for corrupt in (False, True)[: 1 + (k > 1)]:
                with self.subTest(k=k, corrupt=corrupt):
                    runner = GrowingRunner(8, loops=2)
                    draft = Draft(corrupt=corrupt) if k > 1 else None
                    result = executor(runner, draft, k, "stage1", d1=1).generate(
                        [3, 4, 5], 37, []
                    )
                    self.assertEqual(
                        result["output_ids"], expected([3, 4, 5], 37, loops=2)
                    )
                    self.assertEqual(set(result["stats"]["loop_cells"]), {"1", "2"})

    def test_call_counts_are_those_of_a_two_deep_pipeline(self):
        """d1 * ((n - 2) // k) + 2 calls when every readout agrees; d1 = 2 is the
        final review."""
        for k, d1 in ((1, 1), (16, 1), (16, 2)):
            with self.subTest(k=k, d1=d1):
                runner = GrowingRunner(8, causal=False, loops=2)
                draft = Draft(causal=False) if k > 1 else None
                result = executor(runner, draft, k, "stage1", d1=d1).generate(
                    [10], 65, []
                )
                self.assertEqual(result["output_ids"], expected([10], 65, False))
                self.assertEqual(result["stats"]["calls"], d1 * (63 // k) + 2)

    def test_two_loops_admit_no_stage_2_schedule(self):
        """d1 < d2 < 2 has no solution, so Stage 2 stays undefined at R = 2."""
        for stage in ("stage2a", "stage2b"):
            for d1, d2 in ((1, 2), (1, 1), (2, 2)):
                with self.subTest(stage=stage, d1=d1, d2=d2), \
                        self.assertRaisesRegex(ValueError, "d1 < d2 < R"):
                    FinalizedTreeExecutor(
                        GrowingRunner(8, loops=2), None, 1, stage, d1=d1, d2=d2
                    )


if __name__ == "__main__":
    unittest.main()
