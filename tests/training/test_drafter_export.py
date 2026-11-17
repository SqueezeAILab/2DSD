"""vLLM DFlash export: default mask token and path-free metadata (CPU, fake target files)."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from training.drafter import export


def tokenizer_json(vocab, added):
    return {"model": {"type": "BPE", "vocab": vocab}, "added_tokens": [{"id": i, "content": c} for i, c in added]}


class MaskTokenDefault(unittest.TestCase):
    def test_first_unused_embedding_row_past_the_tokenizer(self):
        # Nanbeige-like: ids 0..106 used, embedding table of 144 rows, last added token a real special token.
        vocab = {f"t{i}": i for i in range(101)}
        added = [(101, "<|im_end|>"), (102, "<|endoftext|>"), (103, "<think>"), (104, "</think>"),
                 (105, "<tool_call>"), (106, "</tool_call>")]
        config = {"vocab_size": 144, "bos_token_id": None, "eos_token_id": 101, "pad_token_id": 0}
        self.assertEqual(export.default_mask_token_id(tokenizer_json(vocab, added), config),
                         (107, "first-unused-embedding-row"))

    def test_unigram_vocabulary_list(self):
        tokenizer = {"model": {"type": "Unigram", "vocab": [[f"t{i}", -1.0] for i in range(50)]},
                     "added_tokens": [{"id": 50, "content": "<s>"}]}
        self.assertEqual(export.default_mask_token_id(tokenizer, {"vocab_size": 64}), (51, "first-unused-embedding-row"))

    def test_no_spare_row_falls_back_to_a_reserved_added_token(self):
        # Ouro-like: every embedding row is a token; bos/eos and chat/thinking tokens are never chosen.
        added = [(0, "<|endoftext|>"), (1, "<|im_start|>"), (2, "<|im_end|>"), (3, "<think>"),
                 (15, "<jupyter_script>"), (16, "<empty_output>"), (17, "</think>")]
        vocab = {f"t{i}": i for i in range(18, 64)}
        config = {"vocab_size": 64, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": None}
        self.assertEqual(export.default_mask_token_id(tokenizer_json(vocab, added), config),
                         (16, "last-reserved-added-token"))
        with self.assertRaises(ValueError):
            export.default_mask_token_id(tokenizer_json(vocab, [(1, "<|im_start|>")]), config)


class ExportFiles(unittest.TestCase):
    def test_export_writes_no_paths_and_a_consistent_mask(self):
        H, V, L = 8, 48, 3
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            target = root / "target"
            target.mkdir()
            (target / "config.json").write_text(json.dumps({
                "model_type": "nanbeige", "num_hidden_layers": L, "num_loops": 2, "vocab_size": V,
                "eos_token_id": 40, "rope_theta": 7e7, "max_position_embeddings": 4096}))
            vocab = {f"t{i}": i for i in range(40)}
            (target / "tokenizer.json").write_text(json.dumps(tokenizer_json(vocab, [(40, "<|im_end|>"), (41, "</tool_call>")])))
            head = torch.randn(V, H)
            save_file({"lm_head.weight": head, "model.embed_tokens.weight": torch.randn(V, H)},
                      target / "model.safetensors")
            mask = torch.randn(H)
            checkpoint = root / "best.pt"
            torch.save({"model": {"fc.weight": torch.randn(H, 2 * H), "mask_emb": mask,
                                  "rotary.inv_freq": torch.ones(2)},
                        "cfg": {"hidden_size": H, "intermediate_size": 16, "head_dim": 4, "num_attention_heads": 2,
                                "num_key_value_heads": 1, "num_hidden_layers": 1, "rms_norm_eps": 1e-6},
                        "feat_layers": [0, 2], "d_weights": {1: 0.5, 2: 0.5}, "learn_mask": True,
                        "input_format": "loop-context", "target": "nanbeige4.2-3b", "target_variant": "w64",
                        "target_spec": str(target), "step": 7}, checkpoint)
            out = root / "export"
            with self.assertRaises(SystemExit):  # a local target needs an explicit public name
                export.main(["--checkpoint", str(checkpoint), "--out", str(out)])
            export.main(["--checkpoint", str(checkpoint), "--out", str(out), "--target-name", "org/target-w64"])
            config = json.loads((out / "config.json").read_text())
            self.assertNotIn(str(root), (out / "config.json").read_text())
            self.assertEqual(config["dflash_config"], {"mask_token_id": 42, "target_layer_ids": [3, 5]})
            self.assertEqual(config["num_target_layers"], 2 * L)
            self.assertEqual(config["loopedtf_drafter"]["target_checkpoint"], "org/target-w64")
            self.assertEqual(config["loopedtf_drafter"]["feature_layers_per_loop"], [0, 2])
            self.assertEqual(config["loopedtf_drafter"]["input_format"], "loop-context")
            saved = torch.load(out / "mask_embedding.pt", weights_only=True)
            self.assertEqual(saved["mask_token_id"], 42)
            torch.testing.assert_close(saved["embedding"], mask.to(torch.bfloat16))
            with safe_open(out / "model.safetensors", "pt") as handle:
                self.assertEqual(sorted(handle.keys()), ["fc.weight", "lm_head.weight"])
                torch.testing.assert_close(handle.get_tensor("lm_head.weight"), head.to(torch.bfloat16))
            export.main(["--checkpoint", str(checkpoint), "--out", str(out), "--target-name", "org/target-w64",
                         "--mask-token-id", "41"])
            self.assertEqual(json.loads((out / "config.json").read_text())["dflash_config"]["mask_token_id"], 41)


if __name__ == "__main__":
    unittest.main()
