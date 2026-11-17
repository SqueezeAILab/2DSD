"""The token-array KD data layout described in training/README.md (CPU, synthetic arrays)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.common import registry
from training.common.io import digest, file_sha
from training.kd.data import ArrayDocumentStore, open_store

MODEL = "ouro-1.4b-thinking"


def write_split(root, split, documents, tokenizer_dir):
    """documents: [(record id, prompt length, total length)]."""
    directory = root / "tokenized" / split
    directory.mkdir(parents=True)
    tokens, mask, offsets = [], [], [0]
    for _, prompt, total in documents:
        tokens += list(range(3, 3 + total - 1)) + [2]
        mask += [0] * prompt + [1] * (total - prompt)
        offsets.append(len(tokens))
    np.save(directory / "tokens.npy", np.array(tokens, dtype=np.int32))
    np.save(directory / "offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(directory / "loss_mask.npy", np.array(mask, dtype=np.uint8))
    payload = {"split": split, "documents": len(documents)}
    meta = {"ordered_record_ids": [d[0] for d in documents], "tokens": len(tokens), "target_tokens": sum(mask),
            "template_kwargs": {"enable_thinking": False}, "model_revision": registry.revision(MODEL),
            "files": {name: file_sha(directory / name) for name in ("tokens.npy", "offsets.npy", "loss_mask.npy")},
            "data_hash_payload": payload, "data_hash": digest(payload),
            "tokenizer_fingerprint_payload": {"asset_sha256": {"tokenizer.json": file_sha(tokenizer_dir / "tokenizer.json")}}}
    (directory / "meta.json").write_text(json.dumps(meta))
    return meta["data_hash"]


class ArrayStoreTests(unittest.TestCase):
    def test_documented_layout_opens_packs_and_verifies(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            (tokenizer_dir / "tokenizer.json").write_text("{}")
            data = root / "data"
            hashes = {"train": write_split(data, "train", [("a", 3, 10), ("b", 5, 30), ("c", 2, 25)], tokenizer_dir),
                      "dev": write_split(data, "dev", [("d", 4, 12)], tokenizer_dir)}
            (data / "manifest.json").write_text(json.dumps({"record_index": {"a": {"domain": "math"}}}))
            (data / "ready.json").write_text(json.dumps({"data_ready": True, "data_hashes": hashes}))
            store = open_store(data, "train", MODEL, 40, 20260930, tokenizer_dir)
            self.assertIsInstance(store, ArrayDocumentStore)
            store.verify()
            self.assertEqual(sorted(i for p in store.packs for i in p["documents"]), [0, 1, 2])
            self.assertTrue(all(p["input_tokens"] <= 40 for p in store.packs))
            self.assertEqual(sum(p["targets"] for p in store.packs), 7 + 25 + 23)
            first = store.document(0)
            self.assertEqual((first["id"], first["prompt_length"], first["domain"]), ("a", 3, "math"))
            self.assertEqual(first["input_ids"][-1], 2)
            self.assertEqual(store.manifest["model_tokenizer"], MODEL)
            (tokenizer_dir / "tokenizer.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "tokenizer asset differs"):
                ArrayDocumentStore(data, "train", MODEL, 40, 20260930, tokenizer_dir)
            with self.assertRaisesRegex(ValueError, "exceeds configured pack length"):
                ArrayDocumentStore(data, "train", MODEL, 20, 20260930)


if __name__ == "__main__":
    unittest.main()
