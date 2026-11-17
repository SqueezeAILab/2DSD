"""Nanbeige residual, GQA, packing and detached K2 semantic checks."""
import unittest

import torch

import test_kd
from training.hf_models import model_classes, validate_config
from training.kd.data import pack_tensors
from training.sharekv import exact_document, k2_forward, native_forward


def tiny_nanbeige():
    config_cls, model_cls = model_classes("nanbeige")
    torch.manual_seed(11)
    config = config_cls(vocab_size=96, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, num_loops=2, qk_layernorm=True,
                        pad_token_id=0, eos_token_id=2, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    return model_cls(config).eval()


class NanbeigeForwardTests(test_kd.ForwardTests):
    def setUp(self):
        self.model = tiny_nanbeige()
        self.ids, self.layout, _, _ = pack_tensors([test_kd.doc([1, 5, 9, 7, 8, 3, 6, 2], 3)], 8, 0)

    def test_native_matches_pinned_adapter(self):
        with torch.no_grad():
            expected = self.model.model(self.ids, position_ids=self.layout.positions,
                                       use_cache=False, return_dict=True).last_hidden_state
            actual = native_forward(self.model, self.ids, self.layout, backend="dense")[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_converged_jacobi_equals_sequential_shared(self):
        with torch.no_grad():
            _, initial, _ = native_forward(self.model, self.ids, self.layout, backend="dense")
            for window in (1, 3):
                memory = initial
                for _ in range(self.ids.shape[1]):
                    hidden, _, memory = k2_forward(self.model, self.ids, self.layout, window=window,
                         backend="dense", checkpoint_layers=0, memory=memory, return_memory=True)
                exact = exact_document(self.model, self.ids, 3, window, 1)
                torch.testing.assert_close(hidden, exact, atol=2e-6, rtol=1e-5)

    def test_optional_architecture_changes_are_rejected(self):
        validate_config(self.model.config, "nanbeige")
        for flag in ("enable_double_loop_split", "loop_share_kv", "enable_hyper_connection",
                     "enable_depth_attention", "skip_loop_final_norm"):
            setattr(self.model.config, flag, True)
            with self.assertRaisesRegex(ValueError, "two complete loops"):
                validate_config(self.model.config, "nanbeige")
            setattr(self.model.config, flag, False)


if __name__ == "__main__":
    unittest.main()
