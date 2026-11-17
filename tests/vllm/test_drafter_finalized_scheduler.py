"""2DSD scheduling semantics against exact recurrent and event oracles."""

import unittest
from types import SimpleNamespace

import torch
from helpers import FakeTreeRunner, oracle_next
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
                token = oracle_next(sequence, self.causal)
                if self.corrupt and (sum(sequence) + i) % 3 == 0:
                    token = (token + 7) % 97
                slots.append(token)
                sequence.append(token)
            results.append(
                SimpleNamespace(token_ids=torch.tensor(slots, dtype=torch.long))
            )
        return results

    @staticmethod
    def assert_boundary(event):
        a, v = event["anchor_pos"], event["verified_end"]
        assert len(event["tokens"]) == a + 2
        assert event["tokens"][-1] == event["head"]
        assert v <= a
        if v == a:
            assert event["anchor_depth"] == 4

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

    def __init__(self, kind="promotion"):
        self.kind = kind
        super().__init__(8, causal=False)

    def prefill(self, tokens):
        result = super().prefill(tokens)
        result.hidden = tokens.to(self.dtype).reshape(-1, 1) + 4000
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


def expected(prompt, count, causal=True, eos=()):
    sequence, output = list(prompt), []
    for _ in range(count):
        token = oracle_next(sequence, causal)
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
            assert (prefix // 1000 == 4).all()
            sequence = (prefix % 1000).tolist()
            assert sequence == executor.prefix_tokens
            span, depth, parent = request["span_features"], 4, None
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


if __name__ == "__main__":
    unittest.main()
