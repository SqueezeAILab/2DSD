"""KD checkpoints declare the shared-KV rule to the vLLM fork (config.json keys), and fix_config adds
them to older exports (CPU)."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

import test_kd
from training.kd import fix_config
from training.kd.optim import ShardedAdamW
from training.kd.train import save_checkpoint
from training.sharekv import kd_policy

OPTIONS = {"beta1": .9, "beta2": .95, "eps": 1e-8, "weight_decay": .01, "max_grad_norm": 1.0}


class FixConfigTests(unittest.TestCase):
    def export(self, root, **config):
        path = Path(root) / "export"
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        return path

    def test_adds_keys_from_the_training_window_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.export(root, model_type="ouro", loopedtf_kd_policy=kd_policy(64))
            self.assertTrue(fix_config.fix(path)["changed"])
            config = json.loads((path / "config.json").read_text())
            self.assertEqual((config["loopedtf_kv_policy"], config["loopedtf_kv_window"]), ("shared_last", 64))
            self.assertEqual(config["loopedtf_kd_policy"], kd_policy(64))
            before = (path / "config.json").read_text()
            self.assertFalse(fix_config.fix(path)["changed"])
            self.assertEqual((path / "config.json").read_text(), before)

    def test_refuses_conflicts_and_untrained_configs(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.export(root, model_type="ouro")
            with self.assertRaisesRegex(ValueError, "not a KD export"):
                fix_config.fix(path)
            config = {"loopedtf_kd_policy": kd_policy(64), "loopedtf_kv_policy": "full"}
            with self.assertRaisesRegex(ValueError, "already sets"):
                fix_config.fixed_config(config)
            with self.assertRaisesRegex(ValueError, "training window"):
                fix_config.fixed_config({"loopedtf_kd_policy": kd_policy(64)}, window=32)
            self.assertEqual(fix_config.fixed_config(config, force=True)["loopedtf_kv_policy"], "shared_last")
            (Path(root) / "dry").mkdir()
            dry = self.export(Path(root) / "dry", loopedtf_kd_policy=kd_policy(64))
            self.assertTrue(fix_config.fix(dry, dry_run=True)["changed"])
            self.assertNotIn("loopedtf_kv_policy", json.loads((dry / "config.json").read_text()))


class CheckpointTests(unittest.TestCase):
    def test_saved_model_config_enables_the_shared_rule(self):
        model = test_kd.tiny()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            original = root / "original"
            original.mkdir()
            (original / "config.json").write_text(json.dumps({"model_type": "ouro"}))
            (original / "tokenizer.json").write_text("{}")
            (original / "generation_config.json").write_text("{}")
            optimizer = ShardedAdamW(model, OPTIONS)
            identity = {"config": {"window": 64}, "model_policy": kd_policy(64)}
            state = {"updates": 1, "input_tokens": 123, "target_tokens": 100, "next_pack": 1, "milestones_saved": []}
            final = save_checkpoint(model, optimizer, root / "run", state, identity, 0, 1, original)
            config = json.loads((final / "model" / "config.json").read_text())
            self.assertEqual(config["loopedtf_kv_policy"], "shared_last")
            self.assertEqual(config["loopedtf_kv_window"], 64)
            self.assertEqual(config["loopedtf_kd_policy"]["train_window"], 64)
            for name in ("tokenizer.json", "generation_config.json", "modeling_ouro.py", "configuration_ouro.py"):
                self.assertTrue((final / "model" / name).is_file(), name)
            self.assertTrue((root / "run" / "last").resolve() == final.resolve())
            # The ported Transformers forward warns once that it computes native KV for such a config.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with torch.no_grad():
                    model(torch.tensor([[1, 5, 9]]), use_cache=False)
                    model(torch.tensor([[1, 5, 9]]), use_cache=False)
            messages = [str(w.message) for w in caught if "native per-loop KV" in str(w.message)]
            self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
