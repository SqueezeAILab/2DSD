"""Installed CUDA target paths, storage growth and 2DSD greedy scheduling.

The native-shaped fixture is a correctness check, not a checkpoint quality or
speed benchmark.
"""

import copy
import unittest
from types import SimpleNamespace

import torch
from helpers import _TinyLM, cuda_test
from vllm.model_executor.models.ouro_2d import NativeOuro2DRunner
from vllm.model_executor.models.ouro_finalized import (
    NativeOuroFinalizedRunner,
    TreeSpan,
)
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor


class _WrongDraft:
    """Force early rejects and unverified EOS while keeping a real target."""

    input_format = "loop-context"

    def reset(self):
        pass

    def draft_context_many(self, requests):
        return [
            SimpleNamespace(
                token_ids=torch.tensor(
                    [31] * (request["k"] - 1), device="cuda", dtype=torch.long
                )
            )
            for request in requests
        ]


@cuda_test
class FinalizedTargetCuda(unittest.TestCase):
    def setUp(self):
        import vllm.kernels  # noqa: F401
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

        self.enterContext(
            set_current_vllm_config(
                VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))
            )
        )
        torch.manual_seed(71)
        self.model = _TinyLM().to(device="cuda", dtype=torch.bfloat16)
        self.prompt = [0, 1, 2]

    def runner(self, graph=True, capacity=4, reference=False):
        model = copy.deepcopy(self.model).float() if reference else self.model
        runner = NativeOuroFinalizedRunner(
            model,
            128,
            max_nodes=capacity,
            feat_layers=[0, 1],
            use_cuda_graph=graph,
            attention_backend="sdpa" if reference else "flash_attn",
            max_graph_shapes=16,
        )
        runner.prefill(torch.tensor(self.prompt, device="cuda"))
        if not reference:
            self.assertIsNotNone(runner.prefix._flash)
            self.assertEqual(runner.use_cuda_graph, graph)
        return runner

    @staticmethod
    def wave(runner, hidden, cells):
        result = runner.advance(
            [TreeSpan(node, loop, hidden[node]) for node, loop in cells]
        )
        for (node, _), value in zip(cells, result):
            hidden[node] = value.hidden[0].clone()
        return result

    def assert_readout_margin(self, actual_runner, hidden, reference_runner, reference):
        actual = actual_runner.logits(hidden).float()
        expected = reference_runner.logits(reference).float()
        error = (actual - expected).abs().amax(-1)
        top = expected.topk(2, dim=-1).values
        margin = top[:, 0] - top[:, 1]
        # A near tie makes this fixture unsuitable as an exact greedy gate. Keep
        # canonical argmax and require the reference margin to exceed drift.
        self.assertTrue(torch.all(margin > 4 * error + 0.1).item())
        torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.01)
        self.assertEqual(actual.argmax(-1).tolist(), expected.argmax(-1).tolist())

    def test_graph_bound_grows_when_a_loop_partition_becomes_unbalanced(self):
        runners = [self.runner(), self.runner(False)]
        for depths in ((0, 1, 2, 3), (0, 0, 0, 0), (0, 1, 2, 3)):
            outputs = []
            for runner in runners:
                hidden = {}
                for node, depth in enumerate(depths):
                    runner.register(node, None, node + 3)
                    state = runner.embed(torch.tensor([node + 3], device="cuda"))[0]
                    for loop in range(depth):
                        result = runner.advance([TreeSpan(node, loop, state)])[0]
                        state = result.hidden[0].clone()
                    hidden[node] = state
                results = runner.advance(
                    [TreeSpan(node, depth, hidden[node])
                     for node, depth in enumerate(depths)]
                )
                outputs.append(torch.cat([r.hidden for r in results]))
                runner.release(list(range(4)))
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)

    def test_commit_copies_fragmented_slots_exactly_without_touching_other_rows(self):
        for heads, kv_heads, dim in ((4, 2, 16), (4, 4, 80)):
            with self.subTest(heads=heads, kv_heads=kv_heads, dim=dim):
                self.model = _TinyLM(heads, kv_heads, dim).to(
                    device="cuda", dtype=torch.bfloat16
                )
                runner = self.runner(capacity=16)
                hidden = {}
                for node in range(4):
                    runner.register(100 + node, None, 20 + node)
                    runner.register(node, None if node == 0 else node - 1, node + 3)
                    hidden[node] = runner.embed(
                        torch.tensor([node + 3], device="cuda")
                    )[0]
                for loop in range(runner.R):
                    self.wave(runner, hidden, [(node, loop) for node in range(4)])
                runner.release([100 + node for node in range(4)])
                slots = [runner._nodes[node].slot for node in range(4)]
                self.assertEqual(slots, [1, 3, 5, 7])
                begin = runner.prefix_length
                expected = []
                for prefix, tree in ((runner.prefix.k, runner.k_tree),
                                     (runner.prefix.v, runner.v_tree)):
                    cache = prefix.view(
                        runner.L, runner.R, runner.prefix.padded_len,
                        runner.kv_heads, runner.head_dim,
                    ).clone()
                    bank = tree.view(
                        runner.L, runner.R, runner.padded_nodes,
                        runner.kv_heads, runner.head_dim,
                    )
                    for index, slot in enumerate(slots):
                        cache[:, :, begin + index].copy_(bank[:, :, slot])
                    expected.append(cache)
                runner.commit(list(range(4)))
                self.assertEqual(runner.prefix_length, begin + 4)
                for actual, wanted in zip((runner.prefix.k, runner.prefix.v), expected):
                    self.assertTrue(torch.equal(
                        actual.reshape_as(wanted).view(torch.int16),
                        wanted.view(torch.int16),
                    ))

    def test_flash_graph_and_eager_match_segmented_fp32_sdpa_paths(self):
        runners = [
            self.runner(),
            self.runner(False),
            self.runner(False, reference=True),
        ]
        states = []
        for runner in runners:
            hidden = {}
            for node, parent, token in (
                (0, None, 3),
                (1, 0, 4),
                (2, 0, 9),
                (3, 1, 5),
                (4, 3, 6),
            ):
                runner.register(node, parent, token)
                hidden[node] = runner.embed(torch.tensor([token], device="cuda"))[0]
            states.append(hidden)
        for cells in (
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1)],
            [(0, 2), (2, 1), (1, 2)],
            [(0, 3)],
            [(1, 3), (2, 2), (3, 1), (4, 0)],
            [(2, 3), (3, 2), (4, 1)],
            [(3, 3), (4, 2)],
            [(4, 3)],
        ):
            values = [self.wave(r, h, cells) for r, h in zip(runners, states)]
            for graph, eager, reference in zip(*values):
                torch.testing.assert_close(graph.hidden, eager.hidden, atol=0, rtol=0)
                torch.testing.assert_close(
                    graph.features, eager.features, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    graph.hidden.float(), reference.hidden, atol=0.06, rtol=0.01
                )
                self.assert_readout_margin(
                    runners[0], graph.hidden, runners[2], reference.hidden
                )
        self.assertEqual(runners[0].stats["graph_replays"], 8)

    def test_capture_growth_commit_rebase_and_reused_slot_match_eager(self):
        runners, states = [self.runner(), self.runner(False)], []
        for runner in runners:
            runner.register(0, None, 3)
            runner.register(1, 0, 4)
            hidden = {
                node: runner.embed(torch.tensor([token], device="cuda"))[0]
                for node, token in ((0, 3), (1, 4))
            }
            self.wave(runner, hidden, [(0, 0), (1, 0)])
            self.wave(runner, hidden, [(0, 1)])
            states.append(hidden)
        self.assertEqual(runners[0].stats["captures"], 2)
        slots = [r._nodes[0].slot for r in runners]
        for runner in runners:
            runner.ensure_capacity(70)
            self.assertFalse(runner._graphs)
        self.assertEqual(runners[0].stats["invalidated_graphs"], 2)
        for cells in ([(0, 2), (1, 1)], [(0, 3), (1, 2)], [(1, 3)]):
            first, second = (self.wave(r, h, cells) for r, h in zip(runners, states))
            for graph, eager in zip(first, second):
                torch.testing.assert_close(graph.hidden, eager.hidden, atol=0, rtol=0)
        for runner, hidden, slot in zip(runners, states, slots):
            runner.commit([0])
            self.assertIsNone(runner._nodes[1].parent)
            runner.register(99, None, 9)
            self.assertEqual(runner._nodes[99].slot, slot)
            hidden[99] = runner.embed(torch.tensor([9], device="cuda"))[0]
        for loop in range(4):
            first, second = (
                self.wave(r, h, [(99, loop)]) for r, h in zip(runners, states)
            )
            torch.testing.assert_close(
                first[0].hidden, second[0].hidden, atol=0, rtol=0
            )
        for runner in runners:
            runner.release([99])
            runner.commit([1])
            self.assertEqual(runner.prefix_length, 5)
        self.assertGreater(runners[0].stats["captures"], 2)

    def test_more_than_64_cells_use_one_packed_projection_per_layer(self):
        runner = self.runner(False)
        hidden = {}
        runner.ensure_capacity(70)
        for node in range(70):
            token = node % 32
            runner.register(node, None if node == 0 else node - 1, token)
            hidden[node] = runner.embed(torch.tensor([token], device="cuda"))[0]
        before = [layer.self_attn.qkv_proj.calls for layer in self.model.model.layers]
        result = self.wave(runner, hidden, [(i, 0) for i in range(70)])
        self.assertEqual(len(result), 70)
        self.assertEqual(runner.stats["calls"], 1)
        self.assertEqual(runner.stats["cells"], 70)
        self.assertEqual(runner.path_width, 128)
        self.assertEqual(
            [
                layer.self_attn.qkv_proj.calls - count
                for layer, count in zip(self.model.model.layers, before)
            ],
            [1, 1],
        )
        self.assertTrue(
            torch.isfinite(torch.cat([r.hidden for r in result])).all().item()
        )

    def test_single_cell_replay_uses_correct_loop_prefix_for_gqa_and_ouro_mha(self):
        for heads, kv_heads, dim in ((4, 2, 16), (16, 16, 128)):
            with self.subTest(heads=heads, kv_heads=kv_heads, dim=dim):
                self.model = _TinyLM(heads, kv_heads, dim).to(
                    device="cuda", dtype=torch.bfloat16
                )
                runners = [
                    self.runner(),
                    self.runner(False),
                    self.runner(False, reference=True),
                ]
                states = []
                for runner in runners:
                    runner.register(0, None, 3)
                    states.append(
                        {0: runner.embed(torch.tensor([3], device="cuda"))[0]}
                    )
                for loop in range(4):
                    graph, eager, reference = (
                        self.wave(r, h, [(0, loop)])[0] for r, h in zip(runners, states)
                    )
                    torch.testing.assert_close(
                        graph.hidden, eager.hidden, atol=0, rtol=0
                    )
                    self.assert_readout_margin(
                        runners[0], graph.hidden, runners[2], reference.hidden
                    )
                self.assertEqual(runners[0].stats["captures"], 1)
                self.assertEqual(runners[0].stats["graph_replays"], 4)

    def test_all_stages_match_high_margin_ar_with_unverified_eos_drafts(self):
        reference = NativeOuro2DRunner(
            copy.deepcopy(self.model).float(),
            128,
            feat_layers=[0, 1],
            attention_backend="sdpa",
            use_cuda_graph=False,
        )
        pref = reference.prefill(torch.tensor(self.prompt, device="cuda"))
        hidden = pref.hidden[-1:]
        expected = []
        for i in range(18):
            logits = reference.logits(hidden)
            top = logits.topk(2, dim=-1).values
            self.assertGreater((top[:, 0] - top[:, 1]).item(), 1)
            token = int(logits.argmax(-1).item())
            expected.append(token)
            if i < 17:
                hidden = reference.embed(torch.tensor([token], device="cuda"))
                for loop in range(4):
                    hidden = reference.run(loop, len(self.prompt) + i, hidden).hidden
        for stage in ("stage1", "stage2a", "stage2b"):
            for k in (1, 4, 8, 16):
                with self.subTest(stage=stage, k=k):
                    runner = self.runner()
                    executor = FinalizedTreeExecutor(
                        runner, None if k == 1 else _WrongDraft(), k, stage
                    )
                    result = executor.generate(self.prompt, 18, [31])
                    self.assertEqual(result["output_ids"], expected)
                    self.assertEqual(result["stats"]["calls"], runner.stats["calls"])
                    self.assertEqual(len(result["output_ids"]), 18)
                    self.assertFalse(executor.nodes)
                    if k == 1:
                        self.assertEqual(result["stats"]["drafter_batches"], 0)
                    else:
                        self.assertGreater(result["stats"]["frozen_created"], 0)


if __name__ == "__main__":
    unittest.main()
