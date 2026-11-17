"""Shared-rule masks, packing, K2 gradients, exact replay and the KD loss (CPU semantic checks,
not checkpoint quality measurements)."""
import copy
import unittest
from unittest.mock import patch

import torch

from training.hf_models import model_classes
from training.kd.data import best_fit_packs, pack_tensors
from training.kd.loss import chunked_full_kd_sum, full_forward_kl
from training.sharekv import AttentionPlan, Layout, exact_document, exact_documents, k2_forward, loop_count, native_forward


def tiny():
    config_cls, model_cls = model_classes("ouro")
    torch.manual_seed(7)
    config = config_cls(vocab_size=96, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, total_ut_steps=4,
                        layer_types=["full_attention"] * 2, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    return model_cls(config).eval()


def doc(ids, p):
    return {"input_ids": ids, "prompt_length": p}


class MaskTests(unittest.TestCase):
    def test_partition_prompt_document_and_future(self):
        docs = [doc([1, 3, 4, 5, 6, 2], 2), doc([1, 7, 8, 2], 2)]
        _, layout, queries, _ = pack_tensors(docs, 12, 2)
        for window in (1, 2, 64):
            plan = AttentionPlan(layout, window, "dense")
            allowed = plan.shared_mask[0, 0]
            for q in range(10):
                folded = allowed[q, :12].int() + allowed[q, 12:].int()
                expected = ((layout.documents[0] == layout.documents[0, q]) &
                            (layout.positions[0] <= layout.positions[0, q])).int()
                torch.testing.assert_close(folded, expected)
                if layout.is_prompt[0, q]:
                    self.assertFalse(allowed[q, :12].any())
            self.assertEqual(allowed[10].sum().item(), 1)
        self.assertEqual(queries.tolist(), [1, 2, 3, 4, 7, 8])

    def test_best_fit_is_complete_deterministic(self):
        docs = [doc([1] * n, 1) for n in (5, 7, 3, 2, 6)]
        packs = best_fit_packs(docs, 10, 42)
        self.assertEqual(packs, best_fit_packs(docs, 10, 42))
        self.assertEqual(sorted(i for p in packs for i in p["documents"]), list(range(5)))
        self.assertTrue(all(p["input_tokens"] <= 10 for p in packs))


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny()
        self.ids, self.layout, _, _ = pack_tensors([doc([1, 5, 9, 7, 8, 3, 6, 2], 3)], 8, 2)

    def test_native_matches_pinned_adapter(self):
        with torch.no_grad():
            expected = self.model.model(self.ids, position_ids=self.layout.positions,
                                        use_cache=False)[1][-1]
            actual = native_forward(self.model, self.ids, self.layout, backend="dense")[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_prompt_exact_and_large_window_native(self):
        native, _, _ = native_forward(self.model, self.ids, self.layout, backend="dense")
        for window in (1, 3, 64):
            student, _ = k2_forward(self.model, self.ids, self.layout, window=window,
                                    backend="dense", checkpoint_layers=0)
            torch.testing.assert_close(student[:, :3], native[:, :3], atol=2e-6, rtol=1e-5)
            if window >= self.ids.shape[1]:
                torch.testing.assert_close(student, native, atol=2e-6, rtol=1e-5)

    def test_packing_matches_unpacked_and_no_future_leak(self):
        documents = [doc([1, 5, 9, 7, 8, 2], 3), doc([1, 4, 3, 6, 2], 2)]
        ids, layout, _, slices = pack_tensors(documents, 14, 2)
        packed = k2_forward(self.model, ids, layout, backend="dense", checkpoint_layers=0)[0]
        for document, (start, end, _) in zip(documents, slices):
            ii, ll, _, _ = pack_tensors([document], len(document["input_ids"]), 2)
            separate = k2_forward(self.model, ii, ll, backend="dense", checkpoint_layers=0)[0]
            torch.testing.assert_close(packed[:, start:end], separate, atol=2e-6, rtol=1e-5)
        changed = self.ids.clone()
        changed[0, -2] = 20
        one = k2_forward(self.model, self.ids, self.layout, backend="dense", checkpoint_layers=0)[0]
        two = k2_forward(self.model, changed, self.layout, backend="dense", checkpoint_layers=0)[0]
        torch.testing.assert_close(one[:, :-2], two[:, :-2], atol=2e-6, rtol=1e-5)

    def test_detach_and_checkpoint_gradients(self):
        with torch.no_grad():
            _, memory, _ = native_forward(self.model, self.ids, self.layout, backend="dense")
        self.assertTrue(all(not x.requires_grad for pair in memory.values() for x in pair))
        gradients = []
        for checkpoint_layers in (0, 2):
            self.model.zero_grad(set_to_none=True)
            h, _ = k2_forward(self.model, self.ids, self.layout, backend="dense",
                              checkpoint_layers=checkpoint_layers, memory=memory)
            self.model.lm_head(h[:, 2:-1]).square().mean().backward()
            gradients.append({name: p.grad.clone() for name, p in self.model.named_parameters()
                              if p.grad is not None})
        self.assertGreater(gradients[0]["model.layers.0.self_attn.k_proj.weight"].norm().item(), 0)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-6, rtol=1e-4)

    def test_exact_chunk_invariance(self):
        for window in (1, 3, 64):
            a = exact_document(self.model, self.ids, 3, window, 1)
            b = exact_document(self.model, self.ids, 3, window, min(window, 3))
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)

    def test_exact_variable_length_batch(self):
        documents = [doc([1, 5, 9, 7, 8, 3, 6, 2], 3), doc([1, 3, 4, 5, 2], 2), doc([1, 4, 3, 6, 5, 7, 2], 4)]
        for window in (1, 3, 64):
            batch = exact_documents(self.model, documents, window)
            for document, actual in zip(documents, batch):
                ids = torch.tensor([document["input_ids"]])
                expected = exact_document(self.model, ids, document["prompt_length"], window)[0]
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_exact_features_and_loop_states(self):
        documents = [doc([1, 5, 9, 7, 8, 3, 6, 2], 3), doc([1, 3, 4, 5, 2], 2), doc([1, 4, 3, 6, 5, 7, 2], 4)]
        loops = loop_count(self.model)  # subclasses run this test on targets with other loop counts
        for window in (1, 3, 64):
            plain = exact_documents(self.model, documents, window)
            rich = exact_documents(self.model, documents, window, feature_layers=(1, 0), loop_states=True)
            for document, hidden, result in zip(documents, plain, rich):
                length = len(document["input_ids"])
                self.assertEqual(tuple(result["features"].shape), (loops, length, 64))
                self.assertEqual(tuple(result["loops"].shape), (loops, length, 32))
                torch.testing.assert_close(result["hidden"], hidden, atol=0, rtol=0)
                torch.testing.assert_close(result["loops"][-1], hidden, atol=0, rtol=0)
        # A band wider than every document never reads shared KV, so the replay is the native pass.
        for document, result in zip(documents, rich):
            ids = torch.tensor([document["input_ids"]])
            positions = torch.arange(ids.shape[1])[None]
            layout = Layout(torch.zeros_like(ids), positions, positions < document["prompt_length"])
            taps = {}
            _, _, states = native_forward(self.model, ids, layout, backend="dense", capture="none", return_loops=True,
                                          layer_hook=lambda loop, index, h: taps.__setitem__((loop, index), h[0]))
            features = torch.stack([torch.cat([taps[(loop, 1)], taps[(loop, 0)]], -1) for loop in range(loops)])
            torch.testing.assert_close(result["features"], features, atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(result["loops"], torch.stack([s[0] for s in states]), atol=2e-6, rtol=1e-5)


class LossTests(unittest.TestCase):
    def test_chunked_full_kd_has_same_gradient_and_no_teacher_gradient(self):
        torch.manual_seed(17)
        head = torch.nn.Linear(8, 96, bias=False)
        reference = copy.deepcopy(head)
        teacher = torch.nn.Linear(8, 96, bias=False)
        hidden = torch.randn(13, 8, requires_grad=True)
        hidden_ref = hidden.detach().clone().requires_grad_()
        target_hidden = torch.randn(13, 8)
        expected = full_forward_kl(reference(hidden_ref), teacher(target_hidden)).sum()
        actual = chunked_full_kd_sum(head, hidden, teacher, target_hidden, chunk_size=5)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(hidden.grad, hidden_ref.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(head.weight.grad, reference.weight.grad, atol=1e-6, rtol=1e-5)
        self.assertIsNone(teacher.weight.grad)

    def test_forward_kl_is_zero_for_equal_and_positive_otherwise(self):
        torch.manual_seed(13)
        teacher, student = torch.randn(12, 96), torch.randn(12, 96)
        torch.testing.assert_close(full_forward_kl(teacher, teacher), torch.zeros(12), atol=1e-6, rtol=0)
        self.assertTrue((full_forward_kl(student, teacher) > 0).all())


class EvaluationTests(unittest.TestCase):
    def test_packed_evaluation_counts_and_native_zero_kl(self):
        from training.kd.evaluate import evaluate

        class Store:
            index = [{"id": "a", "input_tokens": 6, "domain": "math"},
                     {"id": "b", "input_tokens": 7, "domain": "code"}]

            def document(self, index):
                return {**self.index[index], **[doc([1, 8, 7, 4, 3, 2], 2),
                                                doc([1, 5, 3, 6, 7, 8, 2], 3)][index]}

        teacher = tiny()
        student = copy.deepcopy(teacher)
        with patch("training.kd.evaluate.native_forward", lambda *a, **kw: native_forward(*a, backend="dense", **kw)), \
             patch("training.kd.evaluate.k2_forward", lambda *a, **kw: k2_forward(*a, backend="dense", **kw)):
            result = evaluate(student, teacher, Store(), {"documents_per_domain": 4,
                              "max_document_tokens": 16, "train_window": 64})
        self.assertEqual(result["teacher_to_k2_full_kl"]["tokens"], 8)
        self.assertLess(abs(result["teacher_to_k2_full_kl"]["mean_kl"]), 1e-6)
        self.assertEqual(sorted(r["id"] for r in result["documents"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
