"""Independent Nanbeige R2 residual/GQA/window and packed scheduling contracts.

Tiny fixtures are numerical/runtime tests, not checkpoint quality checks.
"""

import copy
import unittest
from types import SimpleNamespace

import torch
from helpers import cuda_test
from torch import nn
from vllm.model_executor.models.nanbeige_2d import (
    NativeNanbeigeTreeRunner,
    TreeSpan,
)
from vllm.v1.spec_decode import nanbeige_2d as rpc
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor


class _Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width) * 0.1 + 1)

    def forward(self, hidden, residual=None):
        raw = hidden if residual is None else hidden + residual
        normalized = raw * torch.rsqrt(raw.square().mean(-1, True) + 1e-6)
        normalized = normalized * self.weight
        return normalized if residual is None else (normalized, raw)


class _Linear(nn.Linear):
    calls = 0

    def forward(self, hidden):
        self.calls += 1
        return super().forward(hidden), None


class _MLP(nn.Module):
    calls = 0

    def __init__(self, width):
        super().__init__()
        self.up = nn.Linear(width, width * 4, bias=False)
        self.down = nn.Linear(width * 2, width, bias=False)

    def forward(self, hidden):
        self.calls += 1
        gate, up = self.up(hidden).chunk(2, -1)
        return self.down(torch.nn.functional.silu(gate) * up)


class _RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, positions, query, key):
        def rotate(value):
            value = value.reshape(len(value), -1, self.dim)
            theta = positions[:, None, None].to(value.dtype) * (
                1e6 ** (-torch.arange(0, self.dim, 2, device=value.device) / self.dim)
            )[None, None].to(value.dtype)
            a, b = value[..., : self.dim // 2], value[..., self.dim // 2 :]
            return torch.cat((a * theta.cos() - b * theta.sin(),
                              b * theta.cos() + a * theta.sin()), -1).flatten(1)
        return rotate(query), rotate(key)


class _TinyNanbeige(nn.Module):
    def __init__(
        self, window=None, heads=6, kv_heads=2, dim=4, layers=3, hidden_width=None
    ):
        super().__init__()
        query_width, kv_width = heads * dim, kv_heads * dim
        width = query_width if hidden_width is None else hidden_width
        self.config = SimpleNamespace(
            hidden_size=width, num_hidden_layers=layers, num_attention_heads=heads,
            num_key_value_heads=kv_heads, num_loops=2,
            loopedtf_kv_policy="full" if window is None else "shared_last",
            loopedtf_kv_window=window,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(97, width)
        blocks = []
        for _ in range(layers):
            layer = nn.Module()
            attn = layer.self_attn = nn.Module()
            attn.num_heads, attn.num_kv_heads, attn.head_dim = heads, kv_heads, dim
            attn.q_size, attn.kv_size, attn.scaling = query_width, kv_width, dim**-0.5
            attn.qkv_proj = _Linear(width, query_width + 2 * kv_width, bias=False)
            attn.o_proj = _Linear(query_width, width, bias=False)
            attn.qk_norm = True
            attn.q_layernorm, attn.k_layernorm = _Norm(dim), _Norm(dim)
            attn.rotary_emb = _RoPE(dim)
            attn.loop_kv = None
            attn.attn = [SimpleNamespace(impl=SimpleNamespace(sliding_window=None))] * 2
            layer.input_layernorm = _Norm(width)
            layer.post_attention_layernorm = _Norm(width)
            layer.mlp = _MLP(width)
            blocks.append(layer)
        self.model.layers = nn.ModuleList(blocks)
        self.model.norm = _Norm(width)
        self.head = nn.Linear(width, 97, bias=False)

    def embed_input_ids(self, tokens):
        return self.model.embed_tokens(tokens)

    def compute_logits(self, hidden):
        return self.head(hidden)


def _serial_loop(model, hidden, positions, loop, cache, *, prefill=False):
    """Explicit float64 attention, separate from runner SDPA/cache metadata."""
    width = model.config.hidden_size
    features = []
    for index, layer in enumerate(model.model.layers):
        residual = hidden
        attn = layer.self_attn
        qkv = nn.functional.linear(layer.input_layernorm(hidden), attn.qkv_proj.weight)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
        if attn.qk_norm:
            q = attn.q_layernorm(q.reshape(-1, attn.num_heads, attn.head_dim)).flatten(1)
            k = attn.k_layernorm(k.reshape(-1, attn.num_kv_heads, attn.head_dim)).flatten(1)
        q, k = attn.rotary_emb(positions, q, k)
        q, k, v = (
            q.reshape(-1, attn.num_heads, attn.head_dim),
            k.reshape(-1, attn.num_kv_heads, attn.head_dim),
            v.reshape(-1, attn.num_kv_heads, attn.head_dim),
        )
        for row, position in enumerate(positions.tolist()):
            cache[index, loop, position] = (k[row], v[row])
        outputs = []
        for row, position in enumerate(positions.tolist()):
            keys, values = [], []
            for previous in range(position + 1):
                plane = loop
                window = model.config.loopedtf_kv_window
                if window is not None and not prefill and loop == 0 and index != 0:
                    plane = 1 if previous <= position - window else 0
                key, value = cache[index, plane, previous]
                keys.append(key)
                values.append(value)
            keys, values = torch.stack(keys), torch.stack(values)
            repeat = attn.num_heads // attn.num_kv_heads
            keys = keys.repeat_interleave(repeat, 1)
            values = values.repeat_interleave(repeat, 1)
            scores = torch.einsum("hd,shd->hs", q[row], keys) * attn.scaling
            outputs.append(torch.einsum("hs,shd->hd", scores.softmax(-1), values))
        attention = torch.stack(outputs).reshape(-1, attn.q_size)
        hidden = residual + nn.functional.linear(attention, attn.o_proj.weight)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        features.append(hidden)
    return SimpleNamespace(hidden=model.model.norm(hidden), features=torch.cat(features, -1))


def _serial(model, prompt, tail):
    cache = {}
    hidden = model.embed_input_ids(torch.tensor(prompt, device=next(model.parameters()).device))
    for loop in range(2):
        value = _serial_loop(model, hidden, torch.arange(len(prompt), device=hidden.device),
                             loop, cache, prefill=True)
        hidden = value.hidden
    output = []
    for position, token in enumerate(tail, len(prompt)):
        hidden = model.embed_input_ids(torch.tensor([token], device=hidden.device))
        for loop in range(2):
            value = _serial_loop(model, hidden, torch.tensor([position], device=hidden.device),
                                 loop, cache)
            hidden = value.hidden
        output.append(value)
    return output


def _greedy_reference(model, prompt, count):
    cache = {}
    hidden = model.embed_input_ids(torch.tensor(prompt))
    for loop in range(2):
        value = _serial_loop(model, hidden, torch.arange(len(prompt)), loop,
                             cache, prefill=True)
        hidden = value.hidden
    result = [int(model.compute_logits(hidden[-1:]).argmax(-1))]
    for position in range(len(prompt), len(prompt) + count - 1):
        hidden = model.embed_input_ids(torch.tensor([result[-1]]))
        for loop in range(2):
            value = _serial_loop(model, hidden, torch.tensor([position]), loop, cache)
            hidden = value.hidden
        result.append(int(model.compute_logits(hidden).argmax(-1)))
    return result


class _WrongDraft:
    input_format = "loop-context"

    def reset(self):
        pass

    def draft_context_many(self, requests):
        return [SimpleNamespace(token_ids=torch.full((request["k"] - 1,), 92))
                for request in requests]


class NanbeigeCpuBehavior(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(301)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.prompt, self.tail = [3, 8, 13], [21, 34, 55, 89]

    def runner(self, model, capacity=4):
        runner = NativeNanbeigeTreeRunner(
            model, 128, max_nodes=capacity, feat_layers=list(range(3)),
            attention_backend="sdpa",
        )
        runner.prefill(torch.tensor(self.prompt))
        return runner

    def test_float64_mixed_loop_pipeline_matches_independent_serial(self):
        for window in (None, 64, 4):
            with self.subTest(window=window):
                model = _TinyNanbeige(window).double()
                expected = _serial(model, self.prompt, self.tail)
                runner = self.runner(model, capacity=2)
                hidden, final = {}, {}
                for step in range(len(self.tail) + 1):
                    cells = []
                    if step:
                        cells.append((step - 1, 1))
                    if step < len(self.tail):
                        runner.register(step, None if step == 0 else step - 1, self.tail[step])
                        hidden[step] = runner.embed(torch.tensor([self.tail[step]]))[0]
                        cells.append((step, 0))
                    before = [layer.self_attn.qkv_proj.calls for layer in model.model.layers]
                    mlp_before = [layer.mlp.calls for layer in model.model.layers]
                    result = runner.advance([TreeSpan(node, loop, hidden[node])
                                             for node, loop in cells])
                    for (node, loop), value in zip(cells, result):
                        hidden[node] = value.hidden[0].clone()
                        if loop == 1:
                            final[node] = value
                    self.assertEqual([layer.self_attn.qkv_proj.calls - count
                                      for layer, count in zip(model.model.layers, before)], [1] * 3)
                    self.assertEqual([layer.mlp.calls - count
                                      for layer, count in zip(model.model.layers, mlp_before)], [1] * 3)
                for node, value in final.items():
                    torch.testing.assert_close(value.hidden, expected[node].hidden,
                                               atol=2e-12, rtol=2e-12)
                    torch.testing.assert_close(value.features, expected[node].features,
                                               atol=2e-12, rtol=2e-12)
                self.assertEqual(runner.stats["calls"], 5)
                self.assertEqual(runner.stats["cells"], 8)
                self.assertEqual(runner.stats["capacity_growths"], 1)

    def test_window_boundary_changes_only_early_nonlexical_attention(self):
        model = _TinyNanbeige(2).double()
        runner = self.runner(model)
        for node, token in enumerate(self.tail):
            hidden = runner.embed(torch.tensor([token]))[0]
            runner.register(node, None, token)
            actual = []
            for loop in range(2):
                value = runner.advance([TreeSpan(node, loop, hidden)])[0]
                actual.append(value)
                hidden = value.hidden[0]
            expected = _serial(model, self.prompt, self.tail[:node + 1])[-1]
            torch.testing.assert_close(actual[-1].hidden, expected.hidden,
                                       atol=2e-12, rtol=2e-12)
            runner.commit([node])
        changed = copy.deepcopy(model)
        changed.config.loopedtf_kv_window = 64
        expected_small = _serial(model, self.prompt, self.tail)
        expected_wide = _serial(changed, self.prompt, self.tail)
        self.assertFalse(torch.allclose(expected_small[-1].hidden,
                                        expected_wide[-1].hidden, atol=1e-7, rtol=1e-7))

    def test_shared_path_guard_and_prefill_only_span_guard_fail_explicitly(self):
        model = _TinyNanbeige(2).double()
        runner = self.runner(model)
        runner.register(0, None, 21)
        runner.register(1, 0, 34)
        with self.assertRaisesRegex(ValueError, "window"):
            runner.register(2, 1, 55)
        with self.assertRaisesRegex(ValueError, "tree runner"):
            runner.prefix.run(0, len(self.prompt), runner.embed(torch.tensor([21])))
        model.config.skip_loop_final_norm = True
        with self.assertRaisesRegex(ValueError, "loop boundary"):
            self.runner(model)

    def test_stage1_all_blocks_match_serial_greedy_without_pruning(self):
        for window in (None, 64):
            model = _TinyNanbeige(window).double()
            expected = _greedy_reference(model, self.prompt, 32)
            for k in (1, 4, 8, 16):
                with self.subTest(window=window, k=k):
                    runner = self.runner(model, capacity=2)
                    executor = FinalizedTreeExecutor(
                        runner, None if k == 1 else _WrongDraft(),
                        k=k, stage="stage1", d1=1,
                    )
                    result = executor.generate(self.prompt, 32, [])
                    self.assertEqual(result["output_ids"], expected)
                    self.assertEqual(result["stats"]["calls"], runner.stats["calls"])
                    self.assertEqual(runner.R, 2)
                    self.assertFalse(hasattr(model.config, "total_ut_steps"))

    def test_rpc_rejects_undefined_stage2_and_unqualified_block_sizes(self):
        for stage in ("stage2a", "stage2b"):
            with self.assertRaisesRegex(ValueError, "undefined"):
                rpc.validate_nanbeige_schedule(4, stage, 1)
        for k, d1 in ((3, 1), (32, 1), (4, 2), (True, 1)):
            with self.assertRaises(ValueError):
                rpc.validate_nanbeige_schedule(k, "stage1", d1)


@cuda_test
class NanbeigeCudaBehavior(unittest.TestCase):
    def setUp(self):
        import vllm.kernels  # noqa: F401
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

        self.enterContext(set_current_vllm_config(VllmConfig(
            compilation_config=CompilationConfig(custom_ops=["none"])
        )))
        torch.manual_seed(303)

    @staticmethod
    def model(window, heads=6, kv_heads=2, dim=16, layers=3, hidden_width=None):
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        model = _TinyNanbeige(window, heads, kv_heads, dim, layers, hidden_width)
        norms = [model.model]
        for layer in model.model.layers:
            norms.extend((layer, layer.self_attn))
            layer.self_attn.rotary_emb = RotaryEmbedding(
                dim, dim, 256, 1e6, True, torch.bfloat16
            )
        with torch.no_grad():
            for owner in norms:
                for name, child in list(owner.named_children()):
                    if isinstance(child, _Norm):
                        native = RMSNorm(len(child.weight), eps=1e-6)
                        native.weight.copy_(child.weight)
                        setattr(owner, name, native)
        return model.to(device="cuda", dtype=torch.bfloat16)

    @staticmethod
    def runner(model, prompt, *, graph, reference=False):
        runner = NativeNanbeigeTreeRunner(
            copy.deepcopy(model).float() if reference else model,
            256, max_nodes=8,
            feat_layers=list(range(model.config.num_hidden_layers)),
            use_cuda_graph=graph, attention_backend="sdpa" if reference else "flash_attn",
            max_graph_shapes=8,
        )
        runner.prefill(torch.tensor(prompt, device="cuda"))
        return runner

    @staticmethod
    def chain(runner):
        hidden, results = {}, []
        for node, token in enumerate((21, 34, 55)):
            runner.register(node, None if node == 0 else node - 1, token)
            hidden[node] = runner.embed(torch.tensor([token], device="cuda"))[0]
        for cells in (((0, 0),), ((0, 1), (1, 0)), ((1, 1), (2, 0)), ((2, 1),)):
            wave = runner.advance([TreeSpan(node, loop, hidden[node]) for node, loop in cells])
            results.extend((value.hidden.clone(), value.features.clone()) for value in wave)
            for (node, _), value in zip(cells, wave):
                hidden[node] = value.hidden[0].clone()
        return results

    def test_flash_graph_q1_and_mixed_loops_match_sdpa_at_w64_boundary(self):
        for window in (None, 64):
            for length in (3, 70):
                with self.subTest(window=window, prefix=length):
                    model = self.model(window)
                    prompt = [i % 97 for i in range(length)]
                    graph = self.runner(model, prompt, graph=True)
                    eager = self.runner(model, prompt, graph=False)
                    reference = self.runner(model, prompt, graph=False, reference=True)
                    actual, expected, fp32 = self.chain(graph), self.chain(eager), self.chain(reference)
                    for got, same, exact in zip(actual, expected, fp32):
                        for value, eager_value, reference_value in zip(got, same, exact):
                            torch.testing.assert_close(value, eager_value, atol=0, rtol=0)
                            torch.testing.assert_close(value.float(), reference_value,
                                                       atol=0.045, rtol=0.035)
                    graph.commit([0, 1, 2])
                    eager.commit([0, 1, 2])
                    again_graph, again_eager = self.chain(graph), self.chain(eager)
                    for got, same in zip(again_graph, again_eager):
                        for value, eager_value in zip(got, same):
                            torch.testing.assert_close(value, eager_value, atol=0, rtol=0)
                    self.assertEqual(graph.stats["captures"], 2)
                    self.assertEqual(graph.stats["calls"], 8)
                    self.assertEqual(graph.stats["cells"], 12)

    def test_real_head_geometry_48_query_8_kv_heads_preserves_packed_results(self):
        model = self.model(
            64, heads=48, kv_heads=8, dim=128, layers=1, hidden_width=3072
        )
        graph = self.runner(model, [3, 8, 13], graph=True)
        eager = self.runner(model, [3, 8, 13], graph=False)
        reference = self.runner(model, [3, 8, 13], graph=False, reference=True)
        for got, same, fp32 in zip(self.chain(graph), self.chain(eager), self.chain(reference)):
            for value, eager_value, reference_value in zip(got, same, fp32):
                torch.testing.assert_close(value, eager_value, atol=0, rtol=0)
                torch.testing.assert_close(value.float(), reference_value,
                                           atol=0.045, rtol=0.035)
        self.assertEqual(
            (graph.H, graph.R, graph.heads, graph.kv_heads, graph.head_dim),
            (3072, 2, 48, 8, 128),
        )

    def test_stage1_blocks_match_well_separated_native_greedy_readouts(self):
        model = self.model(64)
        width = model.config.hidden_size
        with torch.no_grad():
            model.model.embed_tokens.weight.zero_()
            model.model.embed_tokens.weight[:32, :32].copy_(torch.eye(32, device="cuda"))
            model.head.weight.zero_()
            for token in range(32):
                model.head.weight[(token + 1) % 32, token] = 1
            for layer in model.model.layers:
                layer.self_attn.qkv_proj.weight.mul_(0.002)
                layer.self_attn.o_proj.weight.mul_(0.002)
                for parameter in layer.mlp.parameters():
                    parameter.mul_(0.002)
        self.assertGreaterEqual(width, 32)
        for k in (1, 4, 8, 16):
            runner = self.runner(model, [0, 1, 2], graph=True)
            result = FinalizedTreeExecutor(
                runner, None if k == 1 else _WrongDraft(), k=k, stage="stage1", d1=1
            ).generate([0, 1, 2], 32, [])
            self.assertEqual(result["output_ids"], [(i + 3) % 32 for i in range(32)])


if __name__ == "__main__":
    unittest.main()
