"""Trace generation helpers (CPU): online loop stopping, KV-policy selection and admission."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from training.drafter.traces import LOOP_STRIDE, LoopWatch, loop_onset, reserved_blocks, target_policy


def online(ids, rng):
    """Feed ids in random-sized deltas, as engine steps do, and return the first reported onset."""
    watch, i = LoopWatch(), 0
    while i < len(ids):
        step = rng.randint(1, 40)
        onset = watch.extend(ids[i:i + step])
        i += step
        if onset is not None:
            return onset
    return None


class LoopStopTests(unittest.TestCase):
    def test_no_loop_in_varied_text(self):
        rng = random.Random(0)
        ids = [rng.randrange(50000) for _ in range(9000)]
        self.assertIsNone(loop_onset(ids))
        self.assertIsNone(online(ids, rng))

    def test_paragraph_loop_is_cut_near_its_start(self):
        rng = random.Random(1)
        paragraph = [rng.randrange(50000) for _ in range(250)]
        ids = [rng.randrange(50000) for _ in range(3000)] + paragraph * 16
        onset = loop_onset(ids)
        self.assertIsNotNone(onset)
        self.assertEqual(onset % LOOP_STRIDE, 0)
        self.assertGreaterEqual(onset, 3000 - 512)
        self.assertLessEqual(onset, 3000 + LOOP_STRIDE)
        self.assertEqual(online(ids, rng), onset)


class PolicyTests(unittest.TestCase):
    def checkpoint(self, root, name, **keys):
        path = Path(root) / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "ouro", **keys}))
        (path / "model.safetensors").write_bytes(b"")
        return path

    def test_checkpoint_keys_are_used_without_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            shared = self.checkpoint(root, "w64", loopedtf_kv_policy="shared_last", loopedtf_kv_window=64)
            original = self.checkpoint(root, "original")
            policy = {"loopedtf_kv_policy": "shared_last", "loopedtf_kv_window": 64}
            self.assertEqual(target_policy(shared, "w64"), (policy, None))
            self.assertEqual(target_policy(original, "original"), ({"loopedtf_kv_policy": "full"}, None))
            with self.assertRaisesRegex(ValueError, "fix_config"):
                target_policy(original, "w64")
            with self.assertRaisesRegex(ValueError, "needs"):
                target_policy(shared, "original")
            with self.assertRaisesRegex(ValueError, "needs"):
                target_policy(shared, "w64", window=32)

    def test_forced_policy_overrides_only_a_checkpoint_without_keys(self):
        with tempfile.TemporaryDirectory() as root:
            shared = self.checkpoint(root, "w64", loopedtf_kv_policy="shared_last", loopedtf_kv_window=64)
            old = self.checkpoint(root, "old-export", loopedtf_kd_policy={"train_window": 64})
            policy = {"loopedtf_kv_policy": "shared_last", "loopedtf_kv_window": 64}
            self.assertEqual(target_policy(old, "w64", "shared_last"), (policy, policy))
            self.assertEqual(target_policy(shared, "w64", "shared_last"), (policy, None))
            with self.assertRaisesRegex(ValueError, "contradicts"):
                target_policy(shared, "w64", "full")
            w32 = self.checkpoint(root, "w32", loopedtf_kv_policy="shared_last", loopedtf_kv_window=32)
            with self.assertRaisesRegex(ValueError, "refusing"):
                target_policy(w32, "w64", "shared_last")

    def test_full_budget_admission_counts_every_cache_group(self):
        layout = [{"kind": "FullAttentionSpec", "block_size": 16, "window": None, "layers": 6},
                  {"kind": "SlidingWindowSpec", "block_size": 16, "window": 64, "layers": 18}]
        request = {"prompt_tokens": 100, "max_new_tokens": 900}
        self.assertEqual(reserved_blocks(request, layout, 16), 63 + max(7, 4 + 2))
        with self.assertRaises(ValueError):
            reserved_blocks(request, [{"kind": "MambaSpec", "block_size": 16, "window": None, "layers": 1}], 16)


if __name__ == "__main__":
    unittest.main()
