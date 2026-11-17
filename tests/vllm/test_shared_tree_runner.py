"""Shared-last KV in the native tree runner: 2D schedules compute the shared AR target.

A query t in an early loop reads final-loop KV at j <= t - W and its own loop
above; the final loop and layer-1/loop-1 stay native. Prefix tokens are
committed only after their own final loop, so in-flight paths within W find
every final-loop KV they need. The CPU test compares, in float64 so batching
cannot hide a rule difference, a pipelined schedule with committing one token
at a time; the CUDA test compares the FlashAttention path with the FP32 SDPA
reference.
"""

import copy
import unittest
from types import SimpleNamespace

import torch
from helpers import _CausalLM, _TinyLM, cuda_test
from vllm.model_executor.models.looped_kv import check_tree_window
from vllm.model_executor.models.ouro_finalized import (
    NativeOuroFinalizedRunner,
    TreeSpan,
)
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor


def shared(model, window):
    model = copy.deepcopy(model)
    model.config.loopedtf_kv_policy = "shared_last"
    model.config.loopedtf_kv_window = window
    return model


class ReferenceDraft:
    """Loop-context drafter proposing a known continuation, wrong every third slot."""

    input_format = "loop-context"

    def __init__(self, prompt, reference):
        self.prompt, self.reference, self.slots = len(prompt), reference, 0

    def reset(self):
        pass

    def draft_context_many(self, requests):
        results = []
        for request in requests:
            span = request["span_features"]
            head = request["prefix_len"] + (0 if span is None else len(span))
            tokens = []
            for i in range(request["k"] - 1):
                index = head + 1 + i - self.prompt
                token = self.reference[index] if index < len(self.reference) else 0
                self.slots += 1
                tokens.append((token + 5) % 97 if self.slots % 3 == 0 else token)
            results.append(SimpleNamespace(token_ids=torch.tensor(tokens)))
        return results


def run_chain(runner, tokens, pipelined):
    """Teacher-forced chain; returns every token's final-loop hidden and features."""
    R = runner.R
    hidden, done = {}, {}
    if not pipelined:
        for i, token in enumerate(tokens):
            runner.register(i, None, token)
            hidden[i] = runner.embed(torch.tensor([token], device=runner.device))[0]
            for loop in range(R):
                result = runner.advance([TreeSpan(i, loop, hidden[i])])[0]
                hidden[i] = result.hidden[0].clone()
            done[i] = (result.hidden[0].clone(), result.features[0].clone())
            runner.commit([i])
        return done
    # Token i runs loop c - i at call c: R tokens in flight at staggered depths.
    for call in range(len(tokens) + R - 1):
        cells = [(i, call - i) for i in range(len(tokens)) if 0 <= call - i < R]
        for i, loop in cells:
            if loop == 0:
                runner.register(i, i - 1 if i - 1 in hidden and i - 1 not in done else None, tokens[i])
                hidden[i] = runner.embed(torch.tensor([tokens[i]], device=runner.device))[0]
        results = runner.advance([TreeSpan(i, loop, hidden[i]) for i, loop in cells])
        for (i, loop), result in zip(cells, results):
            hidden[i] = result.hidden[0].clone()
            if loop == R - 1:
                done[i] = (result.hidden[0].clone(), result.features[0].clone())
                runner.commit([i])
    return done


def assert_live_prefix_equal(dense, compact):
    """Compare every retained KV value with the independent full-bank layout."""
    store = compact.prefix.compact_prefix
    end = compact.prefix_length
    assert end == dense.prefix_length
    for layer in range(compact.L):
        for loop in range(compact.R):
            full = loop == compact.R - 1 or (layer == 0 and loop == 0)
            start = 0 if full else max(0, end - store.ring_size)
            positions = torch.arange(start, end, device=compact.device)
            rows = store.rows(layer, loop, positions)
            for old, new in ((dense.prefix.k, store.k), (dense.prefix.v, store.v)):
                expected = old[layer].view(
                    compact.R, -1, compact.kv_heads, compact.head_dim
                )[loop, positions]
                actual = new.view(-1, compact.kv_heads, compact.head_dim)[rows]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def compare_forked_commits(dense, compact, repeats):
    """Exercise prune, multi-token commit, retained descendants and ring wraps."""
    for batch in range(repeats):
        finals = []
        for runner in (dense, compact):
            hidden = {}
            for node, parent in ((0, None), (1, 0), (2, 0), (3, 1)):
                token = (3 * batch + node) % 20
                runner.register(node, parent, token)
                hidden[node] = runner.embed(
                    torch.tensor([token], device=runner.device)
                )[0]
            for loop in range(runner.R):
                results = runner.advance(
                    [TreeSpan(node, loop, hidden[node]) for node in hidden]
                )
                hidden = {
                    node: result.hidden[0].clone()
                    for node, result in zip(hidden, results)
                }
            finals.append([(r.hidden.clone(), r.features.clone()) for r in results])
            runner.release([2])
            runner.commit([0, 1])
        for actual, expected in zip(finals[1], finals[0]):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert_live_prefix_equal(dense, compact)
        for runner in (dense, compact):
            runner.commit([3])
        assert_live_prefix_equal(dense, compact)


class SharedTreeCpu(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.base = _CausalLM()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def runner(self, model, prompt):
        runner = NativeOuroFinalizedRunner(
            model, max_len=128, max_nodes=8, feat_layers=[0, 1], attention_backend="sdpa"
        )
        runner.prefill(torch.tensor(prompt))
        return runner

    def test_pipelined_schedule_equals_one_token_at_a_time(self):
        tokens = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        for window, prompt in ((4, [3, 4, 5, 6, 7, 8, 9, 10]), (4, [3, 4]), (6, [3, 4, 5, 6, 7, 8, 9, 10, 20])):
            model = shared(self.base, window).double()
            ar = run_chain(self.runner(model, prompt), tokens, pipelined=False)
            pipe = run_chain(self.runner(model, prompt), tokens, pipelined=True)
            for i in range(len(tokens)):
                torch.testing.assert_close(pipe[i][0], ar[i][0], atol=1e-12, rtol=0)
                torch.testing.assert_close(pipe[i][1], ar[i][1], atol=1e-12, rtol=0)

    def test_shared_rule_changes_the_computation(self):
        """The fixture is sensitive: the native read rule gives different states."""
        tokens = [11, 12, 13, 14, 15, 16]
        prompt = [3, 4, 5, 6, 7, 8, 9, 10]
        native = run_chain(self.runner(self.base.double(), prompt), tokens, pipelined=False)
        sharing = run_chain(self.runner(shared(self.base, 4).double(), prompt), tokens, pipelined=False)
        self.assertGreater(max((native[i][0] - sharing[i][0]).abs().max().item() for i in range(len(tokens))), 1e-4)

    def test_paths_beyond_the_window_are_rejected(self):
        runner = self.runner(shared(self.base, 4), [3, 4, 5])
        for i in range(4):
            runner.register(i, i - 1 if i else None, 11 + i)
        with self.assertRaises(ValueError):
            runner.register(4, 3, 15)

    def test_2dsd_at_the_largest_admitted_block_commits_shared_ar_output(self):
        """check_tree_window admits R * k == W; 2DSD on that shared-last target,
        with frozen suffixes and Stage 2 events, commits its greedy AR output."""
        prompt, count = [3, 4, 5, 6, 7], 40
        for k in (2, 4):
            model = shared(self.base, 4 * k).double()
            check_tree_window(model.config, k)
            reference = FinalizedTreeExecutor(
                self.runner(model, prompt), None, 1, "stage1", d1=4
            ).generate(prompt, count, [])["output_ids"]
            for stage in ("stage1", "stage2b"):
                with self.subTest(k=k, stage=stage):
                    result = FinalizedTreeExecutor(
                        self.runner(model, prompt), ReferenceDraft(prompt, reference), k, stage
                    ).generate(prompt, count, [])
                    stats = result["stats"]
                    self.assertEqual(result["output_ids"], reference)
                    self.assertGreater(stats["frozen_created"], 0)
                    if stage == "stage2b":
                        self.assertGreater(stats["promotions"] + stats["forks"], 0)

    def test_compact_prefix_matches_full_banks_across_wraps_and_reset(self):
        for window in (4, 64):
            model = shared(self.base, window).double()
            runners = [
                NativeOuroFinalizedRunner(
                    model, 256, max_nodes=2, feat_layers=[0, 1],
                    attention_backend="sdpa", compact_shared_prefix=compact,
                )
                for compact in (False, True)
            ]
            dense, compact = runners
            for prompt_length in (window - 1, 3):
                prompt = torch.arange(prompt_length) % 20
                prefill = [r.prefill(prompt) for r in runners]
                torch.testing.assert_close(
                    prefill[0].hidden, prefill[1].hidden, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    prefill[0].features, prefill[1].features, atol=0, rtol=0
                )
                assert_live_prefix_equal(dense, compact)
                compare_forked_commits(dense, compact, 2 * window // 3 + 2)
            self.assertLess(compact.prefix.kv_bytes, dense.prefix.kv_bytes)
            with self.assertRaisesRegex(ValueError, "cannot rewind"):
                compact.prefix.truncate(compact.prefix_length - 1)


@cuda_test
class SharedTreeCuda(unittest.TestCase):
    def setUp(self):
        import vllm.kernels  # noqa: F401
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

        self.enterContext(
            set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"])))
        )
        torch.manual_seed(71)
        self.model = _TinyLM().to(device="cuda", dtype=torch.bfloat16)

    def runner(self, model, prompt, graph, reference):
        model = copy.deepcopy(model).float() if reference else model
        runner = NativeOuroFinalizedRunner(
            model, 128, max_nodes=8, feat_layers=[0, 1], use_cuda_graph=graph,
            attention_backend="sdpa" if reference else "flash_attn", max_graph_shapes=16,
        )
        runner.prefill(torch.tensor(prompt, device="cuda"))
        return runner

    def test_flash_and_graph_match_fp32_sdpa_reference(self):
        tokens = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for window, prompt in ((4, [0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20]), (6, [0, 1])):
            model = shared(self.model, window)
            graph = run_chain(self.runner(model, prompt, True, False), tokens, pipelined=True)
            eager = run_chain(self.runner(model, prompt, False, False), tokens, pipelined=True)
            reference = run_chain(self.runner(model, prompt, False, True), tokens, pipelined=True)
            for i in range(len(tokens)):
                torch.testing.assert_close(graph[i][0], eager[i][0], atol=0, rtol=0)
                torch.testing.assert_close(graph[i][0].float(), reference[i][0], atol=0.06, rtol=0.01)
                torch.testing.assert_close(graph[i][1].float(), reference[i][1], atol=0.06, rtol=0.01)

    def test_compact_prefix_matches_full_banks_in_cuda_graphs(self):
        for window in (4, 64):
            model = shared(self.model, window)
            runners = [
                NativeOuroFinalizedRunner(
                    model, 128, max_nodes=2, feat_layers=[0, 1],
                    use_cuda_graph=True, attention_backend="flash_attn",
                    compact_shared_prefix=compact,
                )
                for compact in (False, True)
            ]
            dense, compact = runners
            for length in (window - 1, 3):
                prompt = torch.arange(length, device="cuda") % 20
                prefill = [r.prefill(prompt) for r in runners]
                for field in ("hidden", "features"):
                    torch.testing.assert_close(
                        getattr(prefill[0], field), getattr(prefill[1], field),
                        atol=0, rtol=0,
                    )
                assert_live_prefix_equal(dense, compact)
                compare_forked_commits(dense, compact, 21)


if __name__ == "__main__":
    unittest.main()
