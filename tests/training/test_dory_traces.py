"""Dory trace generation (CPU): registry targets, engine settings, prompt rendering and the request plan."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.common import registry
from training.common.chat import load_tokenizer
from training.drafter import traces

W64 = {"n_recurrent_loops": 4, "recurrent_kv_cache_mode": "per_loop", "loopedtf_kv_policy": "shared_last",
       "loopedtf_kv_window": 64, "loopedtf_kv_share_layers": "full_attention"}
NATIVE = {"n_recurrent_loops": 4, "recurrent_kv_cache_mode": "per_loop", "loopedtf_kv_policy": "full"}


def dory_checkpoint(root):
    path = Path(root) / "dory"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "dory", "n_recurrent_loops": 2}))
    (path / "model.safetensors").write_bytes(b"")
    return path


class FakeTokenizer:
    """Records the chat-template calls; token IDs are the characters of the rendered text."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant><think>\n"

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return [ord(c) for c in text]


class DoryTargets(unittest.TestCase):
    def test_both_variants_are_the_pinned_checkpoint(self):
        original = registry.pinned("dory1.0-4b")
        self.assertEqual(original, ("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
                                    "f6c832e6963ddb74f62c75813a3e8c74b1af67b1"))
        self.assertEqual(registry.pinned("dory1.0-4b", "w64"), original)
        self.assertEqual(registry.label("dory1.0-4b", "w64"), "@".join(original))
        self.assertEqual(registry.family("dory1.0-4b"), "dory")
        # Adapted W64 targets are their own repositories, given as plain repo ids.
        self.assertEqual(registry.pinned("ouro-1.4b-thinking", "w64"),
                         ("minseo25/2dsd-ouro-1.4b-thinking-shared-kv", None))
        self.assertEqual(registry.label("ouro-1.4b-thinking", "w64"), "minseo25/2dsd-ouro-1.4b-thinking-shared-kv")

    def test_kv_policy_is_passed_as_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            path = dory_checkpoint(root)
            self.assertEqual(traces.target_policy(path, "w64"), (W64, W64))
            self.assertEqual(traces.target_policy(path, "w64", "shared_last"), (W64, W64))
            self.assertEqual(traces.target_policy(path, "original"), (NATIVE, NATIVE))
            with self.assertRaisesRegex(ValueError, "W = 64"):
                traces.target_policy(path, "w64", window=32)
            with self.assertRaisesRegex(ValueError, "contradicts"):
                traces.target_policy(path, "w64", "full")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                traces.target_policy(path, "original", "shared_last")

    def test_engine_compiles_only_for_dory(self):
        self.assertEqual(traces.compilation_config("dory"), {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"})
        self.assertEqual(traces.compilation_config("ouro"), {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
        options = traces.runtime_options(16384, 64, W64, traces.compilation_config("dory"))
        self.assertEqual((options["hf_overrides"], options["compilation_config"]["mode"]), (W64, 3))
        self.assertFalse(options["disable_hybrid_kv_cache_manager"])
        self.assertEqual(traces.runtime_options(16384, 64)["compilation_config"]["mode"], 0)
        self.assertEqual(traces.SAMPLING["dory"], [{"temperature": 0.6, "top_p": 0.95, "top_k": 0}])
        self.assertEqual(traces.MAX_SEQS["dory1.0-4b"], 64)

    def test_tokenizer_loads_through_the_fork_config(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with tempfile.TemporaryDirectory() as root:
            vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}
            tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(Path(root) / "tokenizer.json"))
            (Path(root) / "tokenizer_config.json").write_text(json.dumps({
                "tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "</s>", "unk_token": "<unk>"}))
            layer_types = ["mlp" if i % 2 else "sliding_attention" for i in range(72)]
            (Path(root) / "config.json").write_text(json.dumps({
                "model_type": "dory", "num_hidden_layers": 72, "n_input_layers": 16, "n_recurrent_layers": 40,
                "n_output_layers": 16, "layer_types": layer_types}))
            self.assertEqual(load_tokenizer(root).encode("hello world", add_special_tokens=False), [3, 4])


class DoryRequestPlan(unittest.TestCase):
    def test_native_template_without_system_turn_and_sampled_quarter(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path = dory_checkpoint(root)
            prompts = root / "prompts"
            prompts.mkdir()
            rows = [{"id": f"p{i}", "prompt": f"question {i}", "pool": "train", "source": "fixture",
                     "domain": "math", "origin": {}} for i in range(80)]
            (prompts / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            settings = {"model": "dory1.0-4b", "variant": "w64", "target_path": str(path), "target": "dory-w64",
                        "prompts": str(prompts), "budget": 16384, "pools": ["train"], "sampled_fraction": 0.25,
                        "kv_policy": "auto", "window": 64, "ouro_tokenizer": "unused", "gpu_lock_dir": ""}
            tokenizer = FakeTokenizer()
            with patch.object(traces, "load_tokenizer", return_value=tokenizer):
                traces.init(root / "traces", settings)
            setup = json.loads((root / "traces/queue/init.json").read_text())
            items = [json.loads(line) for line in (root / "traces/queue/pending.jsonl").read_text().splitlines()]
        self.assertEqual(setup["rendering"]["system"], [])
        self.assertEqual(setup["sampling_settings"], traces.SAMPLING["dory"])
        for messages, kwargs in tokenizer.calls:
            self.assertEqual([m["role"] for m in messages], ["user"])
            self.assertTrue(kwargs["enable_thinking"] and kwargs["add_generation_prompt"])
        greedy = [i for i in items if i["sampling"] is None]
        sampled = [i for i in items if i["sampling"] is not None]
        self.assertEqual([i["id"] for i in greedy], [r["id"] for r in rows])
        self.assertEqual({i["prompt_id"] for i in sampled},
                         {r["id"] for r in rows if traces.unit(f"sampled:{r['id']}") < 0.25})
        for item in sampled:
            self.assertEqual(item["id"], item["prompt_id"] + "#sampled")
            self.assertEqual(item["sampling"], {"temperature": 0.6, "top_p": 0.95, "top_k": 0,
                                                "seed": int(traces.unit(item["prompt_id"]) * 2**31)})
        self.assertEqual("".join(map(chr, greedy[3]["prompt_ids"])), "<user>question 3<assistant><think>\n")


if __name__ == "__main__":
    unittest.main()
