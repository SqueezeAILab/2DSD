"""Native target feature/readout contract, including Nanbeige's two loops."""
import unittest
from unittest.mock import patch

import torch

from training.drafter.targets import NanbeigeTarget, OuroTarget, default_feature_layers, native_target
from training.hf_models import model_classes
from training.sharekv import exact_documents


def tiny_target(family):
    config_cls, model_cls = model_classes(family)
    torch.manual_seed(117)
    loops = ({"num_loops": 2, "qk_layernorm": True} if family == "nanbeige"
             else {"total_ut_steps": 4, "layer_types": ["full_attention"] * 2})
    config = config_cls(vocab_size=96, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, pad_token_id=0,
                        eos_token_id=2, attention_dropout=0.0, **loops)
    config._attn_implementation = "sdpa"
    return model_cls(config).eval()


class NativeTargetTests(unittest.TestCase):
    def test_native_features_states_and_logits_match_exact_full_window(self):
        for family, cls in (("nanbeige", NanbeigeTarget), ("ouro", OuroTarget)):
            with self.subTest(family=family):
                model = tiny_target(family)
                target = cls.from_model(model, [1, 0])
                for tokens, prompt in (([1, 7, 8, 9, 11, 4, 2], 3),
                                       ([1, 5, 6, 12, 2], 1)):
                    ids = torch.tensor(tokens)
                    features, residuals = target.features(ids)
                    with torch.no_grad():
                        reference = exact_documents(
                            model, [{"input_ids": tokens, "prompt_length": prompt}],
                            window=len(tokens), chunk_size=2, feature_layers=[1, 0],
                            loop_states=True, attention="sdpa", cuda_graph=False,
                        )[0]
                        states = target.final_norm(residuals)
                    self.assertEqual(features.dtype, torch.float32)
                    self.assertEqual(residuals.dtype, torch.float32)
                    self.assertEqual(target._calls, target.R * target.L)
                    torch.testing.assert_close(features, reference["features"],
                                               rtol=1e-5, atol=2e-6)
                    torch.testing.assert_close(states, reference["loops"],
                                               rtol=1e-5, atol=2e-6)
                    torch.testing.assert_close(residuals, features[:, :, :target.H],
                                               rtol=0, atol=0)
                    torch.testing.assert_close(target.readout_logits(residuals),
                                               model.lm_head(reference["loops"]),
                                               rtol=1e-5, atol=2e-6)
                    self.assertTrue(torch.equal(target.readout_argmax(residuals[-1], chunk=2),
                                                model.lm_head(reference["loops"][-1]).argmax(-1)))
                    self.assertIsNone(target._buf)
                    self.assertIsNone(target._out)
                    self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_native_family_factory(self):
        for name, constructor in (("nanbeige4.2-3b", "NanbeigeTarget"),
                                  ("ouro-1.4b-thinking", "OuroTarget"),
                                  ("ouro-2.6b-thinking", "OuroTarget")):
            with self.subTest(name=name), patch("training.drafter.targets." + constructor) as cls:
                self.assertIs(native_target(name, [1], "cpu", torch.float32, path="/local/target"),
                              cls.return_value)
                cls.assert_called_once_with("/local/target", [1], "cpu", torch.float32)
        with self.assertRaises(ValueError):
            native_target("unknown", [1], "cpu")

    def test_unsupported_nanbeige_recurrence_is_rejected(self):
        for flag in ("enable_double_loop_split", "skip_loop_final_norm",
                     "enable_depth_attention", "enable_hyper_connection"):
            model = tiny_target("nanbeige")
            setattr(model.config, flag, True)
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "two complete loops"):
                NanbeigeTarget.from_model(model, [0])

    def test_bad_feature_indices_are_rejected(self):
        for layers in ([], [0, 0], [-1], [2]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                NanbeigeTarget.from_model(tiny_target("nanbeige"), layers)
        self.assertEqual(default_feature_layers(22), [1, 6, 10, 14, 19])

    def test_capture_buffer_released_after_failed_forward(self):
        target = NanbeigeTarget.from_model(tiny_target("nanbeige"), [0])
        with patch.object(target.model.model, "forward", side_effect=RuntimeError("probe")):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                target.features(torch.tensor([1, 7, 2]))
        self.assertIsNone(target._buf)
        self.assertIsNone(target._out)
        self.assertEqual(target.features(torch.tensor([1, 7, 2]))[0].shape, (2, 3, 32))


if __name__ == "__main__":
    unittest.main()
