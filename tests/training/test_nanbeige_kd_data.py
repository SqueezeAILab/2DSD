"""Nanbeige KD data preserves native prompts and excludes whole documents explicitly."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.common.io import file_sha
from training.kd import prepare_nanbeige as data
from training.kd.data import DocumentStore


class Tokenizer:
    eos_token_id = 1000

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        prompt = "PROMPT:" + messages[0]["content"] + "<think>\n"
        return prompt if add_generation_prompt else prompt + messages[-1]["content"] + "<EOS>\n"

    def encode(self, text, **kwargs):
        tokens = []
        for i, part in enumerate(text.split("<EOS>")):
            if i:
                tokens.append(self.eos_token_id)
            tokens += [ord(c) for c in part]
        return tokens


class NanbeigeDataTests(unittest.TestCase):
    def row(self, identity, response):
        return {"record_id": identity, "prompt_history_sha256": identity, "domain": "math",
                "messages": [{"role": "user", "content": identity}, {"role": "assistant", "content": response}]}

    def test_prompt_tokens_and_terminal_eos_are_preserved(self):
        tokenizer = Tokenizer()
        row = self.row("a", "reasoning\n</think>\nanswer")
        doc = data.native_document(row, tokenizer, {})
        prefix = tokenizer.encode(tokenizer.apply_chat_template(row["messages"][:-1], add_generation_prompt=True))
        self.assertEqual(doc["input_ids"][:doc["prompt_length"]], prefix)
        self.assertEqual(doc["input_ids"][-1], tokenizer.eos_token_id)
        self.assertEqual(doc["id"], "a")

    def test_retokenization_seals_split_and_whole_document_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            (parent / "shards").mkdir(parents=True)
            model = root / "models" / data.MODEL
            model.mkdir(parents=True)
            (model / "tokenizer_config.json").write_text("pinned tokenizer")
            pin = {"revision": "pinned", "asset_sha256": {"tokenizer_config.json": file_sha(model / "tokenizer_config.json")}}
            rows = [self.row("dev", "ok"), self.row("a", "ok"), self.row("b", "x"*200), self.row("c", "x"*50)]
            shard = parent / "shards/source.jsonl.gz"
            with gzip.open(shard, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row)+"\n")
            manifest = {"config": {"tokenizers": {data.MODEL: pin}},
                "ordered_record_ids": {"dev": ["dev"], "train": ["a", "b", "c"]},
                "shards": [{"path": "source.jsonl.gz", "record_ids": [r["record_id"] for r in rows],
                            "records": len(rows), "sha256": file_sha(shard)}]}
            (parent / "manifest.json").write_text(json.dumps(manifest))
            (parent / "ready.json").write_text(json.dumps({"data_ready": True, "manifest_sha256": file_sha(parent / "manifest.json")}))
            out = root / "derived"
            with self.assertRaisesRegex(ValueError, "revision differs"):
                data.prepare(parent, root / "other", tokenizer_dir=model, revision="unpinned")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()):
                result = data.prepare(parent, out, tokenizer_dir=model, revision="pinned", budget=40, length=100)
            store = DocumentStore(out, "train")
            store.verify()
            self.assertEqual(store.document(0)["id"], "a")
            self.assertEqual(result["splits"]["train"]["ordered_record_ids"], ["a"])
            self.assertEqual(result["splits"]["dev"]["ordered_record_ids"], ["dev"])
            reasons = {r["reason"] for r in json.loads((out / "exclusions.json").read_text())}
            self.assertEqual(reasons, {"complete-document-over-32K", "train-input-token-budget"})
            self.assertEqual(result["truncated_documents"], 0)
            with gzip.open(shard, "at") as stream:
                stream.write("modified\n")
            with self.assertRaisesRegex(ValueError, "shard hash"):
                data.read_messages(parent)


if __name__ == "__main__":
    unittest.main()
