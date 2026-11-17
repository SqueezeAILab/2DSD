"""Dynamic tree capacity preserves causal state and physical call accounting."""

import unittest
from unittest.mock import patch

import torch
from helpers import _CausalLM
from vllm.model_executor.models.ouro_2d import NativeOuro2DRunner
from vllm.model_executor.models.ouro_finalized import NativeOuroFinalizedRunner
from vllm.model_executor.models.ouro_tree import NativeOuroTreeRunner, TreeSpan


class FinalizedTreeCapacity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = _CausalLM()
        self.prompt = [3, 4, 5]
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def runner(self, capacity=2, max_len=128, runner_type=NativeOuroFinalizedRunner):
        runner = runner_type(
            self.model,
            max_len=max_len,
            max_nodes=capacity,
            feat_layers=[0, 1],
            attention_backend="sdpa",
        )
        runner.prefill(torch.tensor(self.prompt))
        return runner

    @staticmethod
    def advance(runner, hidden, wave):
        results = runner.advance(
            [TreeSpan(node, loop, hidden[node]) for node, loop in wave]
        )
        for (node, _), result in zip(wave, results):
            hidden[node] = result.hidden[0].clone()
        return results

    def test_growth_preserves_mixed_loop_kv_results_and_commit(self):
        dynamic = self.runner()
        fixed = self.runner(128, runner_type=NativeOuroTreeRunner)
        states = []
        for runner in (dynamic, fixed):
            runner.register(0, None, 11)
            runner.register(1, 0, 12)
            hidden = {
                node: runner.embed(torch.tensor([token]))[0]
                for node, token in ((0, 11), (1, 12))
            }
            self.advance(runner, hidden, [(0, 0), (1, 0)])
            self.advance(runner, hidden, [(0, 1)])
            states.append(hidden)
        snapshots = {
            (node.node_id, loop): (
                dynamic.k_tree[:, loop * dynamic.padded_nodes + node.slot].clone(),
                dynamic.v_tree[:, loop * dynamic.padded_nodes + node.slot].clone(),
            )
            for node in dynamic._nodes.values()
            for loop in range(node.depth)
        }
        old_slots = {node: value.slot for node, value in dynamic._nodes.items()}
        dynamic._graphs[2] = object()
        self.assertEqual(dynamic.ensure_capacity(70), 128)
        self.assertFalse(dynamic._graphs)
        self.assertEqual(dynamic.stats["invalidated_graphs"], 1)
        self.assertEqual(dynamic.stats["kv_reallocations"], 1)
        self.assertEqual(dynamic.path_width, 2)
        for (node_id, loop), (key, value) in snapshots.items():
            node = dynamic._nodes[node_id]
            self.assertEqual(node.slot, old_slots[node_id])
            index = loop * dynamic.padded_nodes + node.slot
            torch.testing.assert_close(dynamic.k_tree[:, index], key, atol=0, rtol=0)
            torch.testing.assert_close(dynamic.v_tree[:, index], value, atol=0, rtol=0)
        for runner, hidden in zip((dynamic, fixed), states):
            runner.register(2, 1, 14)
            hidden[2] = runner.embed(torch.tensor([14]))[0]
        for wave in (
            [(2, 0), (0, 2), (1, 1)],
            [(2, 1), (0, 3), (1, 2)],
            [(2, 2), (1, 3)],
            [(2, 3)],
        ):
            actual, expected = (
                self.advance(runner, hidden, wave)
                for runner, hidden in zip((dynamic, fixed), states)
            )
            for first, second in zip(actual, expected):
                torch.testing.assert_close(first.hidden, second.hidden, atol=0, rtol=0)
                torch.testing.assert_close(
                    first.features, second.features, atol=0, rtol=0
                )
        dynamic.commit([0, 1, 2])
        fixed.commit([0, 1, 2])
        for first, second in (
            (dynamic.prefix.k, fixed.prefix.k),
            (dynamic.prefix.v, fixed.prefix.v),
        ):
            for loop in range(dynamic.R):
                begin = loop * dynamic.prefix.padded_len
                torch.testing.assert_close(
                    first.view(dynamic.L, -1, dynamic.kv_heads, dynamic.head_dim)[
                        :, begin : begin + 6
                    ],
                    second.view(fixed.L, -1, fixed.kv_heads, fixed.head_dim)[
                        :, begin : begin + 6
                    ],
                    atol=0,
                    rtol=0,
                )
        self.assertEqual(dynamic.prefix_length, 6)
        self.assertEqual(dynamic.stats["calls"], fixed.stats["calls"])
        self.assertEqual(dynamic.stats["cells"], fixed.stats["cells"])
        dynamic.register(99, None, 21)
        self.assertEqual(dynamic._nodes[99].slot, old_slots[0])
        self.assertEqual(dynamic._nodes[99].depth, 0)

    def test_more_than_64_nodes_advance_in_one_projection_wave(self):
        runner = self.runner(4)
        tokens = [(11 + i) % 97 for i in range(70)]
        hidden = {}
        runner.ensure_capacity(len(tokens))
        for node, token in enumerate(tokens):
            runner.register(node, None if node == 0 else node - 1, token)
            hidden[node] = runner.embed(torch.tensor([token]))[0]
        self.assertEqual(len(runner._nodes), 70)
        self.assertEqual(runner.stats["peak_live_nodes"], 70)
        self.assertEqual(runner.path_width, 128)
        for loop in range(runner.R):
            before = [
                layer.self_attn.qkv_proj.calls for layer in self.model.model.layers
            ]
            actual = self.advance(runner, hidden, [(i, loop) for i in range(70)])
            self.assertEqual(
                [
                    layer.self_attn.qkv_proj.calls - count
                    for layer, count in zip(self.model.model.layers, before)
                ],
                [1, 1],
            )
        reference = NativeOuro2DRunner(
            self.model, max_len=128, feat_layers=[0, 1], attention_backend="sdpa"
        ).prefill(torch.tensor([*self.prompt, *tokens]))
        torch.testing.assert_close(
            actual[-1].hidden[0], reference.hidden[-1], atol=2e-6, rtol=2e-6
        )
        self.assertEqual(runner.stats["calls"], 4)
        self.assertEqual(runner.stats["cells"], 280)
        self.assertTrue(all(node.depth == runner.R for node in runner._nodes.values()))

    def test_node_pool_growth_does_not_inflate_ancestor_path_work(self):
        initial = self.runner(256)
        self.assertEqual(initial.max_nodes, 256)
        self.assertEqual(initial.path_width, 64)
        runner = self.runner()
        runner.ensure_capacity(80)
        runner.register(0, None, 11)
        for node in range(1, 71):
            runner.register(node, 0, node)
        self.assertEqual(runner.path_width, 2)
        self.assertEqual(runner.stats["path_width_growths"], 0)
        banks = runner.k_tree, runner.v_tree
        runner._graphs[71] = object()
        runner.register(99, 1, 21)
        self.assertEqual(runner.path_width, 4)
        self.assertEqual(runner.stats["path_width_growths"], 1)
        self.assertFalse(runner._graphs)
        self.assertIs(runner.k_tree, banks[0])
        self.assertIs(runner.v_tree, banks[1])
        runner.release([99])
        growths = runner.stats["capacity_growths"]
        runner.register(100, 1, 22)
        self.assertEqual(runner.stats["capacity_growths"], growths)
        self.assertEqual(len(runner._nodes), 72)

    def test_failed_allocation_keeps_existing_state_and_graphs_usable(self):
        runner = self.runner()
        runner.register(0, None, 11)
        hidden = {0: runner.embed(torch.tensor([11]))[0]}
        self.advance(runner, hidden, [(0, 0)])
        banks, free = (runner.k_tree, runner.v_tree), list(runner._free)
        stats = dict(runner.stats)
        graph = runner._graphs[1] = object()
        with (
            patch("torch.zeros_like", side_effect=torch.OutOfMemoryError("test OOM")),
            self.assertRaisesRegex(torch.OutOfMemoryError, "test OOM"),
        ):
            runner.ensure_capacity(70)
        self.assertIs(runner.k_tree, banks[0])
        self.assertIs(runner.v_tree, banks[1])
        self.assertIs(runner._graphs[1], graph)
        self.assertEqual(runner._free, free)
        self.assertEqual(runner.stats, stats)
        self.assertEqual(runner._nodes[0].depth, 1)
        for loop in range(1, runner.R):
            self.advance(runner, hidden, [(0, loop)])
        runner.ensure_capacity(70)
        self.assertEqual(runner.stats["capacity_growths"], 1)
        self.assertEqual(runner._nodes[0].depth, runner.R)

    def test_invalid_registration_does_not_allocate_or_change_capacity(self):
        runner = self.runner(1, max_len=4)
        runner.register(0, None, 11)
        stats = dict(runner.stats)
        for args, message in (
            ((1, 99, 12), "Parent"),
            ((1, None, 11), "Duplicate"),
            ((1, 0, 12), "token capacity"),
        ):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, message):
                runner.register(*args)
            self.assertEqual(runner.stats, stats)
        for required in (-1, True, 1.5):
            with self.subTest(required=required), self.assertRaises(ValueError):
                runner.ensure_capacity(required)
        self.assertEqual(runner.ensure_capacity(0), 1)
        self.assertEqual(len(runner._nodes), 1)

    def test_single_query_prefix_selects_its_loop_and_preserves_tree_visibility(self):
        runner = self.runner()
        runner.register(0, None, 11)
        hidden = runner.embed(torch.tensor([11]))[0]
        inputs = runner._make_inputs([TreeSpan(0, 0, hidden)])
        runner._single_prefix_cu = torch.tensor([0, 1], dtype=torch.int32)
        with patch.object(NativeOuroTreeRunner, "_forward", side_effect=lambda x: x):
            for loop in range(runner.R):
                with self.subTest(loop=loop):
                    inputs.loops[0] = loop
                    selected = runner._forward(inputs)
                    self.assertEqual(selected.cu_query.tolist(), [0, 1])
                    self.assertEqual(selected.sequence_lengths.numel(), 1)
                    torch.testing.assert_close(
                        selected.block_table,
                        runner.prefix._block_table[loop : loop + 1],
                    )
                    self.assertIs(selected.loops, inputs.loops)
                    self.assertIs(selected.paths, inputs.paths)
                    self.assertIs(selected.slots, inputs.slots)
                    self.assertEqual(inputs.block_table.shape[0], runner.R)
            runner.register(1, 0, 12)
            packed = runner._make_inputs(
                [TreeSpan(0, 0, hidden), TreeSpan(1, 0, hidden)]
            )
            self.assertIs(runner._forward(packed), packed)


if __name__ == "__main__":
    unittest.main()
