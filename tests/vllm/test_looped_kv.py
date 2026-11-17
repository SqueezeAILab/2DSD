"""KV policy options of looped models and the 2DSD in-flight window bound."""

import unittest
from types import SimpleNamespace

from vllm.model_executor.models.looped_kv import (
    PolicyOptions,
    check_tree_window,
    loop_count,
)


def ouro(policy=None, window=None, **extra):
    config = SimpleNamespace(total_ut_steps=4, **extra)
    if policy is not None:
        config.loopedtf_kv_policy = policy
    if window is not None:
        config.loopedtf_kv_window = window
    return config


class PolicyOptionTests(unittest.TestCase):
    def test_defaults_and_known_fields(self):
        self.assertEqual(PolicyOptions.from_hf(ouro()), PolicyOptions("full", None))
        options = PolicyOptions.from_hf(ouro("shared_last", 64))
        self.assertEqual((options.policy, options.window), ("shared_last", 64))
        options.validate()
        PolicyOptions.from_hf(ouro("full")).validate()

    def test_unknown_kv_fields_are_rejected(self):
        for name in ("loopedtf_kv_anchor", "loopedtf_kv_bits", "loopedtf_kv_"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                PolicyOptions.from_hf(ouro("shared_last", 64, **{name: 4}))
        # Other loopedtf_* fields are not KV options.
        PolicyOptions.from_hf(ouro("shared_last", 64, loopedtf_drafter={}))

    def test_share_layers_scope(self):
        PolicyOptions("shared_last", 64, "full_attention").validate()
        PolicyOptions("shared_last", 64, "all").validate()
        with self.assertRaisesRegex(ValueError, "Unknown loopedtf_kv_share_layers"):
            PolicyOptions("shared_last", 64, "sliding").validate()
        with self.assertRaisesRegex(ValueError, "applies to shared_last only"):
            PolicyOptions("full", None, "full_attention").validate()

    def test_loop_count_of_each_family(self):
        self.assertEqual(loop_count(SimpleNamespace(total_ut_steps=4)), 4)
        self.assertEqual(loop_count(SimpleNamespace(num_loops=2)), 2)
        self.assertEqual(loop_count(SimpleNamespace(n_recurrent_loops=4)), 4)

    def test_shared_last_requires_an_integer_window_of_at_least_one(self):
        for window in (None, 0, -1, 1.0, 64.0, "64", True):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window"):
                PolicyOptions("shared_last", window).validate()
        for window in (1, 64):
            PolicyOptions("shared_last", window).validate()
        with self.assertRaisesRegex(ValueError, "policy"):
            PolicyOptions("per_loop").validate()

    def test_loop_count_reads_ouro_and_nanbeige_configs(self):
        self.assertEqual(loop_count(ouro()), 4)
        self.assertEqual(loop_count(SimpleNamespace(num_loops=2)), 2)


class TreeWindowTests(unittest.TestCase):
    def test_block_fills_the_window_exactly(self):
        for loops, window, k in ((4, 64, 16), (4, 16, 4), (4, 4, 1), (2, 8, 4)):
            with self.subTest(loops=loops, window=window, k=k):
                config = SimpleNamespace(
                    num_loops=loops, loopedtf_kv_policy="shared_last",
                    loopedtf_kv_window=window,
                )
                check_tree_window(config, k)
                for smaller in range(1, k):
                    check_tree_window(config, smaller)
                with self.assertRaisesRegex(ValueError, r"R \* k <= W"):
                    check_tree_window(config, k + 1)

    def test_ouro_blocks_beyond_the_window_are_rejected(self):
        config = ouro("shared_last", 32)
        check_tree_window(config, 8)
        for k in (9, 16):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "4 \\* %d > 32" % k):
                check_tree_window(config, k)

    def test_full_policy_has_no_window_bound(self):
        for config in (ouro(), ouro("full"), ouro("full", 4)):
            for k in (1, 4, 16, 1024):
                check_tree_window(config, k)

    def test_invalid_options_fail_before_the_bound(self):
        with self.assertRaisesRegex(ValueError, "window"):
            check_tree_window(ouro("shared_last"), 1)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            check_tree_window(ouro("full", loopedtf_kv_residual_bits=2), 1)


if __name__ == "__main__":
    unittest.main()
