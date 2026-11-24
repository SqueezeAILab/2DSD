"""The loop-context adapter computes the trained drafter's block logits (CPU, FP32).

Training reads a packed document through drafter.blocks' loop-context mask; the
adapter reads a verified-prefix cache plus a per-request loop-d span. Both must
give the same logits for every depth, in-flight length and block length, also
when several requests share one weight pass. A DSpark drafter decides its drafted
tokens left to right with its heads, in the same block or in the DSpark paper's,
whose anchor slot drafts the first token; the adapter's drafts, logits and
confidences are the training module's.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_mask

from vllm.v1.spec_decode.looped_dflash_context import ContextDFlashAdapter

from training.drafter.blocks import Blocks, Packed, build_context_inputs, context_mask_mod
from training.drafter.model import DepthDFlash, drafter_config

R, K, T, V, H = 4, 8, 48, 32, 16


class ContextAdapter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        target = SimpleNamespace(hidden_size=H, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=8, rms_norm_eps=1e-6, rope_theta=10000.0)
        cfg = drafter_config(2, target)
        cfg._attn_implementation = "sdpa"  # noqa: SLF001  dense reference of the flex mask
        self.module = DepthDFlash(cfg, 2).eval()
        with torch.no_grad():
            self.module.mask_emb.normal_()
            for parameter in self.module.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        self.embedding, self.head = nn.Embedding(V, H), nn.Linear(H, V, bias=False)
        ids = torch.randint(0, V, (T,))
        self.packed = Packed(ids, torch.zeros(T, dtype=torch.long), torch.arange(T), [(0, 6, T, "math")],
                             torch.randn(R, T, 2 * H), torch.randint(0, V, (T,)))
        config = {"hidden_size": H, "intermediate_size": 32, "num_hidden_layers": 2, "vocab_size": V,
                  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "rms_norm_eps": 1e-6,
                  "rope_theta": 10000.0, "block_size": K, "layer_types": ["full_attention"] * 2,
                  "loopedtf_drafter": {"input_format": "loop-context"}}
        self.adapter = ContextDFlashAdapter(
            config, self.module.state_dict(), self.module.mask_emb.detach(), embed=self.embedding,
            lm_head=self.head, max_len=64, device="cpu", dtype=torch.float32, use_native_ops=False)

    def trained_logits(self, blocks):
        """[(d, a, L_if, k)] -> per block [k - 1, V] logits of the training forward."""
        column = lambda j: torch.tensor([x[j] for x in blocks])
        b = Blocks(a=column(1), d=column(0), lif=column(2), k=column(3), doc=torch.zeros(len(blocks), dtype=torch.long),
                   valid=torch.ones(len(blocks), dtype=torch.bool))
        inp = build_context_inputs(self.packed, b, R, K)
        mask_mod, q_len, kv_len = context_mask_mod(self.packed, b, R, K)
        mask = create_mask(mask_mod, None, None, q_len, kv_len, "cpu")
        with torch.no_grad():
            hidden = self.module(self.packed.feats.reshape(-1, 2 * H)[None], self.embedding(inp["q_tok"])[None],
                                 inp["q_is_mask"], torch.cat([self.packed.pos.repeat(R), inp["q_pos"]])[None], mask)[0]
            logits = self.head(hidden[inp["q_index"]])
        return [logits[i, : x[3] - 1] for i, x in enumerate(blocks)]

    def request(self, d, a, lif, k):
        prefix_len = a + 1 if d == R else a - lif
        return {"prefix_features": self.packed.feats[R - 1], "prefix_len": prefix_len,
                "span_features": None if d == R else self.packed.feats[d - 1, prefix_len : a + 1],
                "head": int(self.packed.input_ids[a + 1]), "k": k}

    def test_every_depth_and_in_flight_length_matches_training(self):
        blocks = [(1, 20, 5, 8), (2, 25, 0, 6), (R, 30, 0, 8), (3, 33, 10, 4), (1, 40, 12, 8), (2, 12, 3, 2)]
        expected = self.trained_logits(blocks)
        for block, reference in zip(blocks, expected):
            self.adapter.reset()
            actual = self.adapter.context_logits([self.request(*block)])[0]
            torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)

    def test_requests_sharing_a_prefix_match_separate_passes(self):
        # Same verified prefix (a - L_if = 20), different depths, spans and block lengths.
        blocks = [(1, 26, 6, 8), (2, 23, 3, 5), (3, 20, 0, 8), (1, 21, 1, 1)]
        expected = self.trained_logits(blocks)
        self.adapter.reset()
        batched = self.adapter.context_logits([self.request(*block) for block in blocks])
        for block, actual, reference in zip(blocks, batched, expected):
            if block[3] == 1:
                self.assertIsNone(actual)
                continue
            torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)

    def test_prefix_cache_follows_growth_and_truncation(self):
        first, later = (1, 20, 5, 8), (2, 30, 4, 8)
        self.adapter.reset()
        self.adapter.context_logits([self.request(*first)])
        grown = self.adapter.context_logits([self.request(*later)])[0]
        self.adapter.truncate(10)
        regrown = self.adapter.context_logits([self.request(*later)])[0]
        reference = self.trained_logits([later])[0]
        torch.testing.assert_close(grown, reference, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(regrown, reference, atol=1e-4, rtol=1e-4)

    def test_exports_of_other_input_formats_are_rejected(self):
        import json
        import tempfile

        from safetensors.torch import save_file

        weights = {k: v.contiguous() for k, v in self.module.state_dict().items() if k != "mask_emb"}
        for fmt, ok in (("loop-context", True), ("inflight-embedding", False)):
            with tempfile.TemporaryDirectory() as directory:
                config = dict(self.adapter.config, dflash_config={"mask_token_id": 3},
                              loopedtf_drafter={"input_format": fmt})
                Path(directory, "config.json").write_text(json.dumps(config))
                save_file(weights, str(Path(directory, "model.safetensors")))
                load = lambda: ContextDFlashAdapter.from_export(
                    directory, embed=self.embedding, lm_head=self.head, max_len=64, device="cpu",
                    dtype=torch.float32, use_native_ops=False)
                if ok:
                    self.assertIsInstance(load(), ContextDFlashAdapter)
                else:
                    with self.assertRaises(ValueError):
                        load()


class DSparkContextAdapter(ContextAdapter):
    """An exported DSpark drafter. ANCHOR_PREDICTS: K - 1 slots per block, the anchor's slot drafting the
    first token, instead of the anchor and K - 1 masks."""

    RANK = 6
    ANCHOR_PREDICTS = False

    def setUp(self):
        import json
        import tempfile

        from safetensors.torch import save_file

        super().setUp()
        torch.manual_seed(7)
        self.module = DepthDFlash(self.module.cfg, 2, heads={"vocab": V, "rank": self.RANK, "type": self.KIND}).eval()
        with torch.no_grad():
            self.module.mask_emb.normal_()
            for parameter in self.module.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
            for weight in (self.module.markov_head.markov_w1.weight, self.module.markov_head.markov_w2.weight):
                weight.normal_()  # a bias large enough to change drafts
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = dict(self.adapter.config, block_size=K - self.ANCHOR_PREDICTS, dflash_config={"mask_token_id": 3},
                      loopedtf_drafter={"input_format": "loop-context",
                                        "dspark": {"markov_rank": self.RANK, "markov_head_type": self.KIND,
                                                   "confidence_head_with_markov": True,
                                                   "sample_from_anchor": self.ANCHOR_PREDICTS}})
        Path(self.directory.name, "config.json").write_text(json.dumps(config))
        save_file({k: v.contiguous() for k, v in self.module.state_dict().items() if k != "mask_emb"},
                  str(Path(self.directory.name, "model.safetensors")))
        torch.save({"mask_token_id": 3, "embedding": self.module.mask_emb.detach()},
                   Path(self.directory.name, "mask_embedding.pt"))
        self.adapter = ContextDFlashAdapter.from_export(
            self.directory.name, embed=self.embedding, lm_head=self.head, max_len=64, device="cpu",
            dtype=torch.float32, use_native_ops=False)

    def trained_states(self, blocks):
        """[(d, a, L_if, k)] -> per block the [k - 1, H] backbone states of the slots that draft its tokens
        in the training forward."""
        column = lambda j: torch.tensor([x[j] for x in blocks])
        b = Blocks(a=column(1), d=column(0), lif=column(2), k=column(3), doc=torch.zeros(len(blocks), dtype=torch.long),
                   valid=torch.ones(len(blocks), dtype=torch.bool))
        inp = build_context_inputs(self.packed, b, R, K, anchor_predicts=self.ANCHOR_PREDICTS)
        first = 0 if self.ANCHOR_PREDICTS else 1
        self.assertEqual((inp["S"], inp["q_index"][0].tolist()), (K - 1 + first, list(range(first, K - 1 + first))))
        mask_mod, q_len, kv_len = context_mask_mod(self.packed, b, R, K, anchor_predicts=self.ANCHOR_PREDICTS)
        mask = create_mask(mask_mod, None, None, q_len, kv_len, "cpu")
        with torch.no_grad():
            hidden = self.module(self.packed.feats.reshape(-1, 2 * H)[None], self.embedding(inp["q_tok"])[None],
                                 inp["q_is_mask"], torch.cat([self.packed.pos.repeat(R), inp["q_pos"]])[None], mask)[0]
        return [hidden[inp["q_index"][i, : x[3] - 1]] for i, x in enumerate(blocks)]

    def trained_logits(self, blocks):
        """The training heads decided left to right on those states: logits [k - 1, V] per block."""
        return [rows for rows, _, _ in self.trained_drafts(blocks)]

    def test_requests_sharing_a_prefix_match_separate_passes(self):
        blocks = [(1, 26, 6, 8), (2, 23, 3, 5), (3, 20, 0, 8), (1, 21, 1, 1)]
        self.adapter.reset()
        batched = self.adapter.context_logits([self.request(*block) for block in blocks])
        self.assertIsNone(batched[3])
        for actual, reference in zip(batched, self.trained_logits(blocks[:3])):
            torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)

    def trained_drafts(self, blocks):
        results = []
        for block, states in zip(blocks, self.trained_states(blocks)):
            tokens, rows, confidences = [], [], []
            for slot in range(states.shape[0]):
                previous = torch.tensor([[int(self.packed.input_ids[block[1] + 1]), *tokens]])
                with torch.no_grad():
                    logits, confidence = self.module.head_outputs(states[None, : slot + 1], previous, self.head)
                rows.append(logits[0, slot])
                tokens.append(int(logits[0, slot].argmax()))
                confidences.append(torch.sigmoid(confidence[0, slot]))
            results.append((torch.stack(rows), tokens, torch.stack(confidences)))
        return results

    def test_drafts_and_confidences_match_training(self):
        blocks = [(1, 26, 6, 8), (2, 23, 3, 5), (3, 20, 0, 8), (1, 21, 1, 2)]
        self.adapter.reset()
        drafts = self.adapter.draft_context_many([self.request(*block) for block in [*blocks, (1, 21, 1, 1)]])
        self.assertEqual((drafts[-1].token_ids.numel(), drafts[-1].confidence), (0, None))  # nothing to draft
        changed = []
        for draft, (rows, tokens, confidences), states in zip(drafts, self.trained_drafts(blocks),
                                                              self.trained_states(blocks)):
            self.assertEqual(draft.token_ids.tolist(), tokens)
            torch.testing.assert_close(draft.confidence, confidences, atol=1e-4, rtol=1e-4)
            changed.append(self.head(states).argmax(-1).tolist() != tokens)
        self.assertTrue(any(changed))  # the Markov bias changes drafts

    def test_a_drafting_vocabulary_restricts_the_drafts_and_nothing_else(self):
        """With every token in the vocabulary the drafts are those of the whole head. With a subset, a slot
        drafts the subset's best token given the tokens drafted before it (the training heads, restricted)."""
        blocks = [(1, 26, 6, 8), (2, 23, 3, 5), (3, 20, 0, 8)]
        requests = [self.request(*block) for block in blocks]
        self.adapter.reset()
        whole = self.adapter.draft_context_many(requests)
        self.adapter.configure_draft_vocabulary(torch.arange(V), self.adapter.readout)
        self.adapter.reset()
        for draft, reference in zip(self.adapter.draft_context_many(requests), whole):
            self.assertEqual(draft.token_ids.tolist(), reference.token_ids.tolist())
            torch.testing.assert_close(draft.confidence, reference.confidence)
        ids = torch.arange(0, V, 3)
        readout = self.adapter.readout
        self.adapter.configure_draft_vocabulary(ids, lambda hidden: readout(hidden)[:, ids])
        self.adapter.reset()
        drafts = self.adapter.draft_context_many(requests)
        self.assertNotEqual([d.token_ids.tolist() for d in drafts], [d.token_ids.tolist() for d in whole])
        for draft, block, states in zip(drafts, blocks, self.trained_states(blocks)):
            tokens = []
            for slot in range(states.shape[0]):
                previous = torch.tensor([[int(self.packed.input_ids[block[1] + 1]), *tokens]])
                with torch.no_grad():
                    logits, _ = self.module.head_outputs(states[None, : slot + 1], previous, self.head)
                tokens.append(int(ids[logits[0, slot][ids].argmax()]))
            self.assertEqual(draft.token_ids.tolist(), tokens)
        for bad in (torch.tensor([3, 1]), torch.tensor([0, V]), torch.empty(0, dtype=torch.long)):
            with self.assertRaisesRegex(ValueError, "sorted list"):
                self.adapter.configure_draft_vocabulary(bad, readout)

    def test_a_greedy_draft_sums_in_the_heads_dtype(self):
        """Training adds a BF16 bias to BF16 logits. Two tokens whose logits differ by less than BF16 keeps
        are then a tie, which the first one wins; an FP32 sum would draft the second."""
        from vllm.v1.spec_decode.looped_dflash_context import _DSparkHeads

        base = torch.tensor([[1.001953125, 1.00390625]])
        heads = _DSparkHeads("vanilla", torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.bfloat16), None, None)
        self.assertEqual(int(base.argmax(-1)), 1)
        drafted = heads.draft_base(base)
        self.assertEqual((drafted.dtype, drafted.tolist(), int(drafted.argmax(-1))), (torch.bfloat16, [[1.0, 1.0]], 0))
        self.assertIs(self.adapter.dspark.draft_base(base), base)  # FP32 heads on the CPU: unchanged
        # A head that scales its logits in FP32 (Dory's) trained on the FP32 sum: its logits stay as they are.
        heads.wide_base = True
        self.assertIs(heads.draft_base(base), base)

    def test_slots_per_block(self):
        """Blocks of 8 and 5 draft 7 and 4 tokens: with the anchor and that many masks, or with one slot
        per drafted token when the anchor's slot predicts. A block is at most the anchor and K - 1 tokens."""
        blocks = [(1, 26, 6, 8), (2, 23, 3, 5)]
        before = dict(self.adapter.stats)
        self.adapter.reset()
        self.adapter.draft_context_many([self.request(*block) for block in blocks])
        delta = {k: self.adapter.stats[k] - before.get(k, 0) for k in ("flops_query_rows", "flops_readout_rows")}
        rows = 7 + 4 if self.ANCHOR_PREDICTS else 8 + 5
        self.assertEqual(delta, {"flops_query_rows": rows, "flops_readout_rows": 7 + 4})
        with self.assertRaisesRegex(ValueError, "block length"):
            self.adapter.draft_context_many([self.request(1, 20, 5, K + 1)])
        self.adapter.draft_context_many([self.request(1, 20, 5, K)])

    def test_exports_of_other_input_formats_are_rejected(self):
        self.skipTest("covered by the DFlash export")


class VanillaDSparkContextAdapter(DSparkContextAdapter):
    KIND = "vanilla"


class RnnDSparkContextAdapter(DSparkContextAdapter):
    KIND = "rnn"


class AnchorSlotDSparkContextAdapter(DSparkContextAdapter):
    KIND, ANCHOR_PREDICTS = "vanilla", True


del DSparkContextAdapter  # abstract: its subclasses name the head


class ContextFlopsCounters(ContextAdapter):
    def test_counters_follow_the_request_layout(self):
        blocks = [(4, 20, 3, 8), (2, 30, 5, 4), (R, 25, 0, 8), (1, 40, 7, 1)]
        requests = [self.request(*b) for b in blocks]
        # One shared prefix per drafting call: use the first request's prefix for all.
        prefix_len = requests[0]["prefix_len"]
        requests = [dict(r, prefix_len=prefix_len,
                         span_features=self.packed.feats[d - 1, prefix_len : prefix_len + lif] if lif else None)
                    for r, (d, a, lif, k) in zip(requests, blocks)]
        before = dict(self.adapter.stats)
        self.adapter.context_logits(requests)
        delta = {k: self.adapter.stats[k] - before.get(k, 0) for k in self.adapter.stats if k.startswith("flops_")}
        drafted = [(lif, k) for (_, _, lif, k) in blocks if k > 1]
        self.assertEqual(delta["flops_query_rows"], sum(k for _, k in drafted))
        self.assertEqual(delta["flops_span_rows"], sum(n for n, _ in drafted))
        self.assertEqual(delta["flops_readout_rows"], sum(k - 1 for _, k in drafted))
        self.assertEqual(delta["flops_prefix_pairs"], sum(k for _, k in drafted) * prefix_len)
        self.assertEqual(delta["flops_local_pairs"], sum(k * (n + k) for n, k in drafted))
        for name in ("query_rows", "span_rows", "readout_rows", "prefix_pairs", "local_pairs"):
            self.assertEqual(delta[f"flops_{name}_padded"], delta[f"flops_{name}"])  # no grid on CPU


@unittest.skipUnless(
    os.environ.get("TWODSD_CUDA_TESTS") == "1" and torch.cuda.is_available(),
    "Requires TWODSD_CUDA_TESTS=1 and a GPU",
)
class ContextAdapterCUDA(unittest.TestCase):
    """BF16 FA2/merge path against the FP32 reference path with identical weights."""

    def test_native_flash_matches_fp32_reference(self):
        torch.manual_seed(7)
        hidden, vocab = 256, 512
        target = SimpleNamespace(hidden_size=hidden, intermediate_size=512, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6, rope_theta=10000.0)
        module = DepthDFlash(drafter_config(2, target), 2)
        with torch.no_grad():
            module.mask_emb.normal_()
            for parameter in module.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        weights = {k: v.to(torch.bfloat16) for k, v in module.state_dict().items()}
        embedding = nn.Embedding(vocab, hidden).to(device="cuda", dtype=torch.bfloat16)
        head = nn.Linear(hidden, vocab, bias=False).to(device="cuda", dtype=torch.bfloat16)
        config = {"hidden_size": hidden, "intermediate_size": 512, "num_hidden_layers": 2, "vocab_size": vocab,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "rms_norm_eps": 1e-6,
                  "rope_theta": 10000.0, "block_size": 16, "layer_types": ["full_attention"] * 2,
                  "loopedtf_drafter": {"input_format": "loop-context"}}
        native = ContextDFlashAdapter(config, weights, weights["mask_emb"], embed=embedding, lm_head=head,
                                      max_len=256, device="cuda", dtype=torch.bfloat16, use_native_ops=True,
                                      use_cuda_graph=False)
        embedding_fp32, head_fp32 = embedding.weight.float(), head.weight.float()
        reference = ContextDFlashAdapter(
            config, {k: v.float() for k, v in weights.items()}, weights["mask_emb"].float(),
            embed=lambda ids: F.embedding(ids, embedding_fp32), lm_head=lambda h: F.linear(h, head_fp32),
            max_len=256, device="cuda", dtype=torch.float32, use_native_ops=False, use_cuda_graph=False)
        features = torch.randn(200, 2 * hidden, device="cuda").to(torch.bfloat16)
        features32 = features.float()
        for prefix_len, shapes in ((70, ((0, 16), (5, 8), (23, 4), (63, 16), (1, 2))), (130, ((12, 16), (0, 3)))):
            requests = [
                {"prefix_features": features, "prefix_len": prefix_len,
                 "span_features": torch.randn(n, 2 * hidden, device="cuda").to(torch.bfloat16) if n else None,
                 "head": 7 + i, "k": k}
                for i, (n, k) in enumerate(shapes)
            ]
            fp32 = [dict(r, prefix_features=features32,
                         span_features=None if r["span_features"] is None else r["span_features"].float())
                    for r in requests]
            for actual, expected in zip(native.context_logits(requests), reference.context_logits(fp32)):
                # BF16 rounding stays below 0.015 here; a span shifted by one position moves logits by 0.04+.
                torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0)


@unittest.skipUnless(
    os.environ.get("TWODSD_CUDA_TESTS") == "1" and torch.cuda.is_available(),
    "Requires TWODSD_CUDA_TESTS=1 and a GPU",
)
class ContextGraphsCUDA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        h = 256
        target = SimpleNamespace(hidden_size=h, intermediate_size=512, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6, rope_theta=10000.)
        module = DepthDFlash(drafter_config(2, target), 2)
        with torch.no_grad():
            module.mask_emb.normal_()
        self.weights = {k: v.to(torch.bfloat16) for k, v in module.state_dict().items()}
        self.embedding = nn.Embedding(512, h).to(device="cuda", dtype=torch.bfloat16)
        self.head = nn.Linear(h, 512, bias=False).to(device="cuda", dtype=torch.bfloat16)
        self.config = dict(hidden_size=h, intermediate_size=512, num_hidden_layers=2, vocab_size=512,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6,
                           rope_theta=10000., block_size=16, layer_types=["full_attention"] * 2,
                           loopedtf_drafter={"input_format": "loop-context"})
        self.features = torch.randn(240, 2 * h, device="cuda", dtype=torch.bfloat16)
        self.eager = self.adapter(False)

    def adapter(self, graph, **kwargs):
        return ContextDFlashAdapter(
            self.config, self.weights, self.weights["mask_emb"], embed=self.embedding, lm_head=self.head,
            max_len=256, device="cuda", dtype=torch.bfloat16, use_native_ops=True,
            use_cuda_graph=graph, batched_graph_min_calls=2, **kwargs)

    def requests(self, prefix, shapes=((0, 16), (5, 8), (23, 4)), seed=0):
        return [dict(prefix_features=self.features, prefix_len=prefix,
                     span_features=self.features[seed:seed + n] if n else None, head=7 + i + seed, k=k)
                for i, (n, k) in enumerate(shapes)]

    def test_replay_is_bitwise_with_live_prefix_spans_and_heads(self):
        graph = self.adapter(True)
        retained = None
        for index, prefix in enumerate((70, 70, 130, 130, 31, 31, 100, 100)):
            if index == 4:
                graph.truncate(10)
                self.eager.truncate(10)
            if index == 6:
                graph.reset()
                self.eager.reset()
            req = self.requests(prefix, seed=index)
            expected, actual = self.eager.context_logits(req), graph.context_logits(req)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b), (index, (a - b).abs().max().item()))
            if retained:
                for a, b in zip(*retained):
                    self.assertTrue(torch.equal(a, b), "Replay overwrote a public logits result")
            retained = (actual, [a.clone() for a in actual])
        self.assertEqual(graph.stats["context_graph_captures"], 1)
        self.assertEqual(graph.stats["context_graph_replays"], 7)
        self.assertGreater(graph.stats["context_graph_cache_bytes"], 0)
        shapes = ((12, 8), (0, 4), (1, 1))
        for _ in range(3):
            req = self.requests(80, shapes)
            expected = self.eager.draft_context_many(req)
            actual = graph.draft_context_many(req)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a.token_ids, b.token_ids))
        self.assertEqual(graph.stats["context_graph_captures"], 2)
        self.assertEqual(graph.context_logits([]), [])
        self.assertEqual(graph.context_logits(self.requests(80, ((0, 1),))), [None])

    def test_shape_and_memory_budget_fallbacks_preserve_logits(self):
        for limits, reason in ((dict(batched_graph_max_shape_bytes=1), "oversize"),
                               (dict(batched_graph_max_bytes=1), "budget")):
            graph = self.adapter(True, **limits)
            for _ in range(3):
                req = self.requests(70)
                for a, b in zip(graph.context_logits(req), self.eager.context_logits(req)):
                    self.assertTrue(torch.equal(a, b))
            self.assertEqual(graph.stats["context_graph_captures"], 0)
            self.assertEqual(graph.stats[f"context_graph_{reason}_fallbacks"], 2)


@unittest.skipUnless(
    os.environ.get("TWODSD_CUDA_TESTS") == "1" and torch.cuda.is_available(),
    "Requires TWODSD_CUDA_TESTS=1 and a GPU",
)
class ContextGridCUDA(unittest.TestCase):
    """Startup drafting grid: padded calls replay prewarmed graphs instead of layout-keyed ones."""

    setUp = ContextGraphsCUDA.setUp
    adapter = ContextGraphsCUDA.adapter
    requests = ContextGraphsCUDA.requests
    layouts = (((0, 16),), ((5, 8), (23, 4)), ((3, 16), (0, 16), (17, 9)), ((24, 16),), ((1, 2), (8, 16)))

    def grid(self, **kwargs):
        adapter = self.adapter(True, **kwargs)
        report = adapter.prewarm_grid(16, (1, 2, 4), (0, 8, 24))
        self.assertTrue(report["enabled"])
        self.assertEqual(report["graphs"], 9)
        return adapter

    def test_grid_replay_is_bitwise_with_the_padded_eager_layout(self):
        graph, eager = self.grid(), self.grid()
        eager.grid_eager = True
        retained = None
        for index, (prefix, shapes) in enumerate(zip((70, 70, 130, 31, 100), self.layouts)):
            req = self.requests(prefix, shapes, seed=index)
            actual = graph.context_logits(req)
            for a, b in zip(actual, eager.context_logits(req)):
                self.assertTrue(torch.equal(a, b), (index, (a.float() - b.float()).abs().max().item()))
            if retained:
                for a, b in zip(*retained):
                    self.assertTrue(torch.equal(a, b), "Replay overwrote a public logits result")
            retained = (actual, [a.clone() for a in actual])
        self.assertEqual(graph.stats["grid_graph_replays"], len(self.layouts))
        self.assertEqual(graph.stats["context_graph_captures"], 0)
        self.assertEqual(graph.stats["grid_outside_calls"], 0)

    def test_grid_matches_the_unpadded_layout_path(self):
        graph = self.grid()
        for index, (prefix, shapes) in enumerate(zip((70, 70, 130, 31, 100), self.layouts)):
            req = self.requests(prefix, shapes, seed=index)
            for a, b in zip(graph.context_logits(req), self.eager.context_logits(req)):
                # Padding changes GEMM shapes and key order only; BF16 rounding, as against FP32 above.
                torch.testing.assert_close(a.float(), b.float(), atol=0.025, rtol=0)
                self.assertTrue(torch.equal(a.argmax(-1), b.argmax(-1)))

    def test_calls_outside_the_grid_take_the_layout_path(self):
        graph = self.grid()
        for shapes in (((25, 16),), ((0, 16),) * 5):
            req = self.requests(70, shapes)
            for a, b in zip(graph.context_logits(req), self.eager.context_logits(req)):
                self.assertTrue(torch.equal(a, b))
        self.assertEqual(graph.stats["grid_outside_calls"], 2)
        self.assertEqual(graph.stats["grid_graph_replays"], 0)


if __name__ == "__main__":
    unittest.main()
