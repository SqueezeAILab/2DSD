"""Exact-replay grouping in packed_stream: groups are counted in packs, so DDP ranks replay at the same steps."""
import unittest

import torch

from training.drafter.data import packed_stream


class ReplayTarget:
    """Records every batched replay; features and loop states are zeros of the right shape."""

    R, H, feat_layers = 2, 4, [0]

    def __init__(self):
        self.calls = []

    def features_many(self, docs):
        self.calls.append(len(docs))
        return [(torch.zeros(self.R, len(d["ids"]), self.H), torch.zeros(self.R, len(d["ids"]), self.H))
                for d in docs]

    def readout_argmax(self, loop_out):
        return torch.zeros(loop_out.shape[0], dtype=torch.long)


def replays_seen(lengths, group_tokens, T=16):
    """Replay calls made by the time each pack is yielded, for one rank holding one document per pack."""
    docs = [{"ids": list(range(n)), "resp_start": 1, "domain": "math"} for n in lengths]
    target = ReplayTarget()
    seen = [len(target.calls) for _ in packed_stream(target, docs, [[i] for i in range(len(docs))], T, "cpu",
                                                     group_tokens)]
    return seen, target.calls


class ReplayGroups(unittest.TestCase):
    def test_ranks_with_different_document_lengths_replay_at_the_same_packs(self):
        long_rank, long_calls = replays_seen([16] * 8, group_tokens=64)
        short_rank, short_calls = replays_seen([3] * 8, group_tokens=64)
        self.assertEqual(long_rank, [1, 1, 1, 1, 2, 2, 2, 2])
        self.assertEqual(short_rank, long_rank)
        self.assertEqual(long_calls, [4, 4])
        self.assertEqual(short_calls, [4, 4])

    def test_last_group_is_partial_and_groups_hold_at_most_group_tokens(self):
        seen, calls = replays_seen([16] * 10, group_tokens=64)
        self.assertEqual(seen, [1] * 4 + [2] * 4 + [3] * 2)
        self.assertEqual(calls, [4, 4, 2])
        self.assertTrue(all(n * 16 <= 64 for n in calls))


if __name__ == "__main__":
    unittest.main()
