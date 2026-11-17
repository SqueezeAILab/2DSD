"""Drafter block semantics: labels, chain mask, loop-context queries and attention visibility."""
import random
import unittest

import torch
from torch.nn.attention.flex_attention import create_mask

from training.drafter.blocks import Blocks, Packed, block_labels, build_context_inputs, context_mask_mod, sample_blocks

R, K = 4, 4
T = 40


def toy():
    # doc 0: tokens 0..24 (prompt 0..4), doc 1: 25..39 (prompt 25..27); the trace is greedy everywhere.
    ids = torch.arange(100, 100 + T)
    feats = torch.zeros(R, T, 8, dtype=torch.bfloat16)
    feats[:, :, 0] = torch.arange(R)[:, None].to(feats.dtype) + 1
    feats[:, :, 1] = torch.arange(T)[None, :].to(feats.dtype)
    p = Packed(ids, torch.tensor([0] * 25 + [1] * 15), torch.cat([torch.arange(25), torch.arange(15)]),
               [(0, 5, 25, "math"), (25, 28, 40, "code")], feats, torch.roll(ids, -1))
    # block 0: d=1, a=15, L_if=5, k=4; block 1: d=R, a=30, k=3; block 2: padding
    b = Blocks(a=torch.tensor([15, 30, 0]), d=torch.tensor([1, 4, 4]), lif=torch.tensor([5, 0, 0]),
               k=torch.tensor([4, 3, 0]), doc=torch.tensor([0, 1, -2]), valid=torch.tensor([True, True, False]))
    return p, b


class DrafterBlocks(unittest.TestCase):
    def setUp(self):
        self.p, self.b = toy()

    def copy(self):
        return Packed(*(getattr(self.p, f) for f in ("input_ids", "doc_ids", "pos", "docs", "feats")),
                      self.p.final_argmax.clone())

    def test_labels_are_the_next_argmax_of_each_drafted_slot(self):
        labels, valid = block_labels(self.p, self.b, K)
        self.assertEqual(labels[0].tolist(), [117, 118, 119])
        self.assertEqual(valid.tolist(), [[True, True, True], [True, True, False], [False, False, False]])

    def test_chain_mask_drops_slots_after_a_trace_label_disagreement(self):
        p = self.copy()
        p.final_argmax[16] = 0
        self.assertEqual(block_labels(p, self.b, K)[1][0].tolist(), [True, False, False])

    def test_sampled_trace_chain_starts_after_the_head(self):
        p = self.copy()
        p.final_argmax[15] = 0  # the head at a+1 = 16 is not the argmax of position 15
        self.assertEqual(block_labels(p, self.b, K)[1][0].tolist(), [False, False, False])
        p.doc_sampled = torch.tensor([True, False])
        self.assertEqual(block_labels(p, self.b, K)[1][0].tolist(), [True, True, True])
        p.final_argmax[16] = 0  # a later disagreement still ends the chain in a sampled trace
        self.assertEqual(block_labels(p, self.b, K)[1][0].tolist(), [True, False, False])

    def test_loop_context_reads_loop_r_prefix_loop_d_span_and_block(self):
        mask_mod, q_len, kv_len = context_mask_mod(self.p, self.b, R, K)
        self.assertEqual((q_len, kv_len), (3 * K, R * T + 3 * K))
        mask = create_mask(mask_mod, None, None, q_len, kv_len, "cpu")[0, 0]

        def sees(q):
            row = mask[q].nonzero().flatten().tolist()
            loops = {}
            for k in row:
                if k < R * T:
                    loops.setdefault(k // T + 1, []).append(k % T)
            return loops, [k - R * T for k in row if k >= R * T]

        # d=1, a=15, L_if=5: loop R up to v=9, loop 1 over the in-flight span 10..15 (parent included).
        for q in (0, 3):
            self.assertEqual(sees(q), ({1: list(range(10, 16)), R: list(range(10))}, [0, 1, 2, 3]))
        # d=R: the verified parent is the last loop-R position and there is no loop-d span.
        self.assertEqual(sees(K), ({R: list(range(25, 31))}, [K, K + 1, K + 2]))
        self.assertEqual(sees(K + 3), ({}, [K + 3]))
        self.assertEqual(sees(2 * K), ({}, [2 * K]))

    def test_loop_context_inputs_share_labels(self):
        inp = build_context_inputs(self.p, self.b, R, K)
        self.assertEqual(inp["q_tok"].view(3, K)[0].tolist(), [116, 117, 118, 119])
        self.assertEqual(inp["q_is_mask"].view(3, K)[0].tolist(), [False, True, True, True])
        self.assertEqual(inp["q_pos"].view(3, K)[0].tolist(), [16, 17, 18, 19])
        self.assertEqual(inp["q_index"][0].tolist(), [1, 2, 3])
        self.assertEqual(inp["q_valid"].view(3, K).tolist(),
                         [[True] * 4, [True, True, True, False], [False] * 4])
        labels, valid = block_labels(self.p, self.b, K)
        torch.testing.assert_close(inp["labels"], labels)
        torch.testing.assert_close(inp["lab_valid"], valid)

    def test_sampled_blocks_respect_depth_inflight_and_document_bounds(self):
        b = sample_blocks(self.p, 64, R, K, {1: 0.5, 2: 0.125, 3: 0.125, 4: 0.25}, random.Random(3))
        self.assertTrue(b.valid.all())
        self.assertTrue((b.k == K).all())
        self.assertEqual(b.a.tolist(), sorted(b.a.tolist()))
        for a, d, lif, doc in zip(b.a.tolist(), b.d.tolist(), b.lif.tolist(), b.doc.tolist()):
            start, rs, end, _ = self.p.docs[doc]
            self.assertTrue((rs - 1 if d == R else rs) <= a <= end - 3)
            self.assertTrue(0 <= lif <= max(0, min((R - 1) * K, a - rs)))  # a = rs - 1 only for d = R
            if d == R:
                self.assertEqual(lif, 0)


if __name__ == "__main__":
    unittest.main()
