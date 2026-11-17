"""Token-tree visibility, shared node KV, and accepted-prefix promotion."""

import gc
import unittest
import weakref
from unittest.mock import patch

import torch
from helpers import _CausalLM, cuda_test
from torch import nn
from vllm.model_executor.models.ouro_2d import NativeOuro2DRunner
from vllm.model_executor.models.ouro_tree import NativeOuroTreeRunner, TreeSpan


class TreeTargetBehavior(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = _CausalLM()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.prompt = [3, 4, 5]

    def tree(self):
        runner = NativeOuroTreeRunner(
            self.model,
            max_len=128,
            max_nodes=8,
            feat_layers=[0, 1],
            attention_backend="sdpa",
        )
        runner.prefill(torch.tensor(self.prompt))
        return runner

    def reference(self, tokens):
        runner = NativeOuro2DRunner(
            self.model, max_len=128, feat_layers=[0, 1], attention_backend="sdpa"
        )
        return runner.prefill(torch.tensor([*self.prompt, *tokens]))

    def compute_tree(self, sibling=13):
        runner = self.tree()
        tokens = {0: 11, 1: 12, 2: sibling, 3: 14, 4: 15}
        parents = {0: None, 1: 0, 2: 0, 3: 1, 4: 2}
        for node_id, token in tokens.items():
            runner.register(node_id, parents[node_id], token)
        hidden = {
            node_id: runner.embed(torch.tensor([token]))[0]
            for node_id, token in tokens.items()
        }
        final = {}
        waves = (
            [(i, 0) for i in range(5)],
            [(0, 1), (1, 1)],
            [(0, 2), (2, 1), (1, 2), (3, 1), (4, 1)],
            [(0, 3), (2, 2), (1, 3), (3, 2), (4, 2)],
            [(2, 3), (3, 3), (4, 3)],
        )
        for wave in waves:
            spans = [TreeSpan(node, loop, hidden[node]) for node, loop in wave]
            results = runner.advance(spans)
            for (node, loop), result in zip(wave, results):
                hidden[node] = result.hidden[0].clone()
                if loop == runner.R - 1:
                    final[node] = result
        return runner, final

    def test_same_wave_ancestors_and_mixed_loops_match_full_causal_paths(self):
        tree, results = self.compute_tree()
        for node, path in (
            (0, [11]),
            (1, [11, 12]),
            (2, [11, 13]),
            (3, [11, 12, 14]),
            (4, [11, 13, 15]),
        ):
            with self.subTest(node=node):
                expected = self.reference(path)
                torch.testing.assert_close(
                    results[node].hidden[0], expected.hidden[-1], atol=2e-6, rtol=2e-6
                )
                torch.testing.assert_close(
                    results[node].features[0],
                    expected.features[-1],
                    atol=2e-6,
                    rtol=2e-6,
                )
        self.assertEqual(tree.stats["calls"], 5)
        self.assertEqual(tree.stats["cells"], 20)

    def test_position_capacity_respects_rope_cache_before_kv_allocation(self):
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        self.enterContext(
            set_current_vllm_config(
                VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))
            )
        )
        # The real Ouro cache supports 64K, including both requested budgets.
        rope = RotaryEmbedding(4, 4, 65536, 1e6, True, torch.float32)
        for layer in self.model.model.layers:
            layer.self_attn.rotary_emb = rope
        for capacity in (8192, 16384):
            with self.subTest(capacity=capacity):
                runner = NativeOuro2DRunner(
                    self.model,
                    capacity,
                    feat_layers=[0, 1],
                    attention_backend="sdpa",
                )
                result = runner.prefill(torch.tensor(self.prompt))
                self.assertTrue(torch.isfinite(result.hidden).all())
        # Reject unsupported positions before reserving a private KV bank.
        with (
            patch("torch.empty", side_effect=AssertionError("KV allocated")),
            patch("torch.zeros", side_effect=AssertionError("KV allocated")),
            self.assertRaisesRegex(ValueError, "capacity.*RoPE cache"),
        ):
            NativeOuro2DRunner(
                self.model,
                65537,
                feat_layers=[0, 1],
                attention_backend="sdpa",
            )

    def test_sibling_token_changes_cannot_affect_another_branch(self):
        _, first = self.compute_tree(sibling=13)
        _, changed = self.compute_tree(sibling=78)
        torch.testing.assert_close(first[3].hidden, changed[3].hidden, atol=0, rtol=0)
        self.assertFalse(torch.equal(first[4].hidden, changed[4].hidden))

    def test_a_shared_parent_wave_uses_one_physical_projection_call_per_layer(self):
        tree = self.tree()
        for node, parent, token in ((0, None, 11), (1, 0, 12), (2, 0, 13)):
            tree.register(node, parent, token)
        before = [layer.self_attn.qkv_proj.calls for layer in self.model.model.layers]
        tree.advance(
            [
                TreeSpan(node, 0, tree.embed(torch.tensor([token]))[0])
                for node, token in ((0, 11), (1, 12), (2, 13))
            ]
        )
        self.assertEqual(
            [
                layer.self_attn.qkv_proj.calls - count
                for layer, count in zip(self.model.model.layers, before)
            ],
            [1, 1],
        )

    def test_commit_promotes_all_loop_kv_and_recycled_slots_do_not_leak(self):
        tree, _ = self.compute_tree()
        tree.release([2, 4])
        tree.commit([0])
        self.assertEqual(tree.prefix_length, 4)
        self.assertEqual(tree.register(1, None, 12), 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            tree.register(99, None, 12)
        tree.register(99, None, 21)
        hidden = tree.embed(torch.tensor([21]))[0]
        for loop in range(tree.R):
            result = tree.advance([TreeSpan(99, loop, hidden)])[0]
            hidden = result.hidden[0]
        expected = self.reference([11, 21])
        torch.testing.assert_close(
            result.hidden[0], expected.hidden[-1], atol=2e-6, rtol=2e-6
        )
        tree.release([99])
        tree.commit([1, 3])
        self.assertEqual(tree.prefix_length, 6)
        reference = NativeOuro2DRunner(
            self.model, max_len=128, feat_layers=[0, 1], attention_backend="sdpa"
        )
        reference.prefill(torch.tensor([*self.prompt, 11, 12, 14]))
        for loop in range(tree.R):
            begin = loop * tree.prefix.padded_len
            actual = tree.prefix.k.view(tree.L, -1, tree.kv_heads, tree.head_dim)
            expected = reference.k.view(tree.L, -1, tree.kv_heads, tree.head_dim)
            torch.testing.assert_close(
                actual[:, begin : begin + 6],
                expected[:, begin : begin + 6],
                atol=2e-6,
                rtol=2e-6,
            )

    def test_ancestor_readiness_and_subtree_pruning_fail_before_state_changes(self):
        tree = self.tree()
        tree.register(0, None, 11)
        tree.register(1, 0, 12)
        with self.assertRaisesRegex(ValueError, "Ancestor KV"):
            tree.advance([TreeSpan(1, 0, tree.embed(torch.tensor([12]))[0])])
        with self.assertRaisesRegex(ValueError, "every descendant"):
            tree.release([0])
        self.assertEqual(tree.stats["calls"], 0)
        tree.release([0, 1])
        tree.register(2, None, 13)
        self.assertEqual(tree.stats["registered_nodes"], 3)

    def disable_cyclic_gc(self):
        self.addCleanup(gc.enable if gc.isenabled() else gc.disable)
        gc.disable()

    def test_release_reclaims_pruned_nodes_without_cyclic_gc(self):
        self.disable_cyclic_gc()
        tree = self.tree()
        tree.register(0, None, 11)
        tree.register(1, 0, 12)
        tree.register(2, None, 13)
        retired = [weakref.ref(tree._nodes[i]) for i in (0, 1)]

        tree.release([0, 1])

        self.assertTrue(all(ref() is None for ref in retired))
        self.assertEqual(tree._nodes[2].path, (tree._nodes[2],))
        self.assertIsNone(tree._nodes[2].parent)

    def test_commit_reclaims_ancestors_after_rebasing_without_cyclic_gc(self):
        self.disable_cyclic_gc()
        tree, _ = self.compute_tree()
        tree.release([2, 4])
        retired = [weakref.ref(tree._nodes[i]) for i in (0, 1)]

        tree.commit([0, 1])

        self.assertTrue(all(ref() is None for ref in retired))
        self.assertEqual(tree.prefix_length, 5)
        self.assertEqual(tree._nodes[3].path, (tree._nodes[3],))
        self.assertIsNone(tree._nodes[3].parent)
        self.assertEqual(tree.register(3, None, 14), 3)

    def test_reset_reclaims_all_nodes_without_cyclic_gc(self):
        self.disable_cyclic_gc()
        tree = self.tree()
        tree.register(0, None, 11)
        tree.register(1, 0, 12)
        retired = [weakref.ref(node) for node in tree._nodes.values()]

        tree.reset()

        self.assertTrue(all(ref() is None for ref in retired))
        self.assertEqual(tree.prefix_length, 0)
        self.assertFalse(tree._nodes)
        tree.prefill(torch.tensor(self.prompt))
        tree.register(0, None, 13)
        self.assertEqual(tree._nodes[0].path, (tree._nodes[0],))


@cuda_test
class TreeAttentionCudaBehavior(unittest.TestCase):
    def test_fused_norm_scope_restores_priority_and_preserves_fp32_results(self):
        import vllm.kernels  # noqa: F401
        from vllm import ir

        class IrNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(8))

            def forward(self, x, residual=None):
                if residual is None:
                    return ir.ops.rms_norm(x, self.weight, 1e-6, None)
                return ir.ops.fused_add_rms_norm.maybe_inplace(
                    x, residual, self.weight, 1e-6, None
                )

        torch.manual_seed(37)
        model = _CausalLM()
        for layer in model.model.layers:
            for name in (
                "input_layernorm",
                "input_layernorm_2",
                "post_attention_layernorm",
                "post_attention_layernorm_2",
            ):
                setattr(layer, name, IrNorm())
        model.model.norm = IrNorm()
        model = model.cuda()
        prompt = torch.tensor([3, 4, 5], device="cuda")
        original = (
            ir.ops.rms_norm.get_priority(),
            ir.ops.fused_add_rms_norm.get_priority(),
        )
        with (
            ir.ops.rms_norm.set_priority(["native"]),
            ir.ops.fused_add_rms_norm.set_priority(["native"]),
        ):
            plain = NativeOuro2DRunner(
                model,
                128,
                feat_layers=[0, 1],
                attention_backend="sdpa",
                use_fused_norm=False,
                use_fused_rope=False,
            )
            fused = NativeOuro2DRunner(
                model,
                128,
                feat_layers=[0, 1],
                attention_backend="sdpa",
                use_fused_norm=True,
                use_fused_rope=False,
            )
            expected, actual = plain.prefill(prompt), fused.prefill(prompt)
            torch.testing.assert_close(
                actual.hidden, expected.hidden, atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                actual.features, expected.features, atol=1e-5, rtol=1e-5
            )
            x = torch.randn(7, 2048, device="cuda", dtype=torch.float32)
            weight = torch.randn(2048, device="cuda", dtype=torch.float32)
            residual = torch.randn_like(x)
            norm = x * (x.square().mean(-1, True) + 1e-6).rsqrt() * weight
            summed = x + residual
            add_norm = summed * (summed.square().mean(-1, True) + 1e-6).rsqrt() * weight
            with fused._norm_context():
                self.assertEqual(ir.ops.rms_norm.get_priority()[0], "vllm_c")
                out = ir.ops.rms_norm(x, weight, 1e-6, None)
                add, out_residual = ir.ops.fused_add_rms_norm.maybe_inplace(
                    x.clone(), residual.clone(), weight, 1e-6, None
                )
            torch.testing.assert_close(out, norm, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(add, add_norm, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(out_residual, summed, atol=0, rtol=0)
            self.assertEqual(ir.ops.rms_norm.get_priority(), ["native"])
            self.assertEqual(ir.ops.fused_add_rms_norm.get_priority(), ["native"])
            with (
                self.assertRaisesRegex(RuntimeError, "scope failure"),
                fused._norm_context(),
            ):
                raise RuntimeError("scope failure")
            self.assertEqual(ir.ops.rms_norm.get_priority(), ["native"])
            self.assertEqual(ir.ops.fused_add_rms_norm.get_priority(), ["native"])
        self.assertEqual(
            (ir.ops.rms_norm.get_priority(), ir.ops.fused_add_rms_norm.get_priority()),
            original,
        )

    def test_fused_rope_preserves_packed_kv_and_dynamic_graph_inputs(self):
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        self.enterContext(
            set_current_vllm_config(
                VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))
            )
        )
        torch.manual_seed(53)
        rows, heads, kv_heads, dim = 5, 4, 2, 128
        positions = torch.tensor(
            [0, 2047, 8192, 16383, 65535], device="cuda", dtype=torch.long
        )
        runner = NativeOuro2DRunner.__new__(NativeOuro2DRunner)
        runner.device = torch.device("cuda")
        runner.use_fused_rope = True
        query_size, key_size = heads * dim, kv_heads * dim

        def split(x):
            return x.split([query_size, key_size, key_size], -1)

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                rope = RotaryEmbedding(dim, dim, 65536, 1e6, True, dtype)
                rope = rope.cuda()
                inherited_dispatch = rope._forward_method
                packed = torch.randn(
                    rows,
                    (heads + 2 * kv_heads) * dim,
                    device="cuda",
                    dtype=dtype,
                )

                def reference(x, rotary=rope):
                    query, key, _ = split(x)
                    return rotary.forward_native(positions, query, key)

                expected_q, expected_k = reference(packed)
                staging = packed.clone()
                q, k, v = split(staging)
                q_result, k_result = runner._apply_rope(rope, positions, q, k)
                self.assertEqual(q_result.data_ptr(), q.data_ptr())
                self.assertEqual(k_result.data_ptr(), k.data_ptr())
                self.assertNotEqual(q.stride(0), query_size)
                tolerance = 1e-6 if dtype == torch.float32 else 0.032
                relative = 1e-6 if dtype == torch.float32 else 0.008
                torch.testing.assert_close(
                    q_result, expected_q, atol=tolerance, rtol=relative
                )
                torch.testing.assert_close(
                    k_result, expected_k, atol=tolerance, rtol=relative
                )
                torch.testing.assert_close(v, split(packed)[2], atol=0, rtol=0)
                self.assertEqual(rope._forward_method, inherited_dispatch)

                # The fused kernel mutates Q/K. Replay receives fresh QKV and
                # changed positions, just as each runner layer projection does.
                staging.copy_(packed)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    runner._apply_rope(rope, positions, q, k)
                for offset in (1, 3):
                    positions.copy_(
                        torch.tensor(
                            [offset, 2047 + offset, 8192 + offset, 16383, 65535],
                            device="cuda",
                        )
                    )
                    fresh = packed * (1 + offset / 4)
                    staging.copy_(fresh)
                    expected_q, expected_k = reference(fresh)
                    graph.replay()
                    torch.testing.assert_close(
                        q, expected_q, atol=tolerance, rtol=relative
                    )
                    torch.testing.assert_close(
                        k, expected_k, atol=tolerance, rtol=relative
                    )
                    torch.testing.assert_close(v, split(fresh)[2], atol=0, rtol=0)
                runner.use_fused_rope = False
                expected_q, expected_k = rope(positions, *split(packed.clone())[:2])
                actual_q, actual_k = runner._apply_rope(
                    rope, positions, *split(packed.clone())[:2]
                )
                torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
                torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
                self.assertEqual(rope._forward_method, inherited_dispatch)
                runner.use_fused_rope = True

    def test_packed_qkv_query_strides_match_contiguous_queries(self):
        from vllm.model_executor.models.ouro_tree import _ancestor_attention_merge

        torch.manual_seed(19)
        rows, heads, kv_heads, dim, pool, width = 5, 4, 2, 128, 16, 4
        packed = torch.randn(
            rows, (heads + 2 * kv_heads) * dim, device="cuda", dtype=torch.bfloat16
        )
        query = packed[:, : heads * dim].view(rows, heads, dim)
        self.assertNotEqual(query.stride(0), heads * dim)
        keys = torch.randn(pool, kv_heads, dim, device="cuda", dtype=query.dtype)
        values = torch.randn_like(keys)
        paths = torch.tensor(
            [
                [0, -1, -1, -1],
                [0, 1, -1, -1],
                [0, 2, -1, -1],
                [0, 1, 3, -1],
                [0, 2, 4, -1],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        loops = torch.zeros(rows, device="cuda", dtype=torch.int64)
        prefix = torch.randn_like(query)
        lse = torch.randn(heads, rows, device="cuda", dtype=torch.float32)

        def merge(q):
            output = torch.empty(rows, heads, dim, device="cuda", dtype=q.dtype)
            _ancestor_attention_merge[(rows, heads)](
                q,
                keys,
                values,
                paths,
                loops,
                prefix,
                lse,
                output,
                heads,
                kv_heads,
                dim,
                pool,
                width,
                dim**-0.5,
                lse.stride(0),
                lse.stride(1),
                dim,
                QUERY_ROW_STRIDE=q.stride(0),
                QUERY_HEAD_STRIDE=q.stride(1),
                num_warps=4,
            )
            return output

        actual = merge(query)
        expected = merge(query.contiguous())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for row in range(rows):
            ancestors = paths[row][paths[row] >= 0].long()
            k = keys[ancestors].float().repeat_interleave(heads // kv_heads, dim=1)
            v = values[ancestors].float().repeat_interleave(heads // kv_heads, dim=1)
            scores = torch.einsum("hd,khd->hk", query[row].float(), k) * dim**-0.5
            all_scores = torch.cat((lse[:, row, None], scores), dim=-1)
            probabilities = all_scores.softmax(-1)
            reference = probabilities[:, :1] * prefix[row].float()
            reference += torch.einsum("hk,khd->hd", probabilities[:, 1:], v)
            torch.testing.assert_close(
                actual[row].float(), reference, atol=0.015, rtol=0.008
            )


if __name__ == "__main__":
    unittest.main()
