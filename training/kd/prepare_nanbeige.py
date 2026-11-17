"""Retokenize the D_OT3 KD messages for Nanbeige (CPU only; no response generation).

Reads the normalized D_OT3 message shards (training/README.md) and renders each conversation with
Nanbeige's native template (enable_thinking, preserve_thinking), then writes a DocumentStore:
<split>.jsonl, <split>.index.json, <split>.packs.json, exclusions.json and manifest.json.

  python -m training.kd.prepare_nanbeige --parent D_OT3_DIR --out DATA_DIR [--tokenizer PATH]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

from ..common import registry
from ..common.io import atomic_json, digest, file_sha
from .data import best_fit_packs, tokenizer_fingerprint

MODEL = "nanbeige4.2-3b"
TEMPLATE = {"enable_thinking": True, "preserve_thinking": True}


def native_document(row, tokenizer, source):
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("missing-final-assistant-response")
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                                add_generation_prompt=True, **TEMPLATE)
    # Remove only the template's whitespace after the terminal EOS.
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False, **TEMPLATE).rstrip()
    if not full_text.startswith(prefix_text):
        raise ValueError("native-template-prefix-mismatch")
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    if ids[:len(prefix)] != prefix:
        raise ValueError("native-token-prefix-mismatch")
    if not prefix or len(prefix) >= len(ids) or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("missing-response-or-terminal-EOS")
    return {"id": row["record_id"], "input_ids": ids, "prompt_length": len(prefix),
            "sha256": digest({"input_ids": ids, "prompt_length": len(prefix)}),
            "prompt_sha256": digest(prefix), "domain": row.get("domain", "unknown"),
            "source": {**source, "original_response_source": row.get("source"),
                       "teacher": row.get("teacher")}}


def read_messages(parent):
    manifest = json.loads((parent / "manifest.json").read_text())
    ready = json.loads((parent / "ready.json").read_text())
    if not ready.get("data_ready") or ready["manifest_sha256"] != file_sha(parent / "manifest.json"):
        raise ValueError("parent D_OT3 manifest is not sealed ready")
    rows = {}
    for shard in manifest["shards"]:
        path = parent / "shards" / Path(shard["path"]).name
        if file_sha(path) != shard["sha256"]:
            raise ValueError(f"normalized shard hash mismatch: {path.name}")
        identities = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = row["record_id"]
                if identity in rows:
                    raise ValueError("duplicate source record ID")
                rows[identity] = row
                identities.append(identity)
        if identities != shard["record_ids"] or len(identities) != shard["records"]:
            raise ValueError("normalized shard order/count mismatch")
    ordered = manifest["ordered_record_ids"]
    identities = [identity for split in ("dev", "train") for identity in ordered[split]]
    if len(identities) != len(set(identities)) or set(identities) != set(rows):
        raise ValueError("source split inventory differs from normalized shards")
    train_groups = {rows[i]["prompt_history_sha256"] for i in ordered["train"]}
    dev_groups = {rows[i]["prompt_history_sha256"] for i in ordered["dev"]}
    if train_groups & dev_groups:
        raise ValueError("source train/dev prompt groups overlap")
    return manifest, rows


def prepare(parent, out, *, tokenizer_dir, revision, budget=128_000_000, length=32768, seed=20260930):
    """tokenizer_dir: Nanbeige tokenizer files; revision: the pinned model revision they belong to."""
    from transformers import AutoTokenizer

    parent, out, tokenizer_dir = Path(parent).resolve(), Path(out).resolve(), Path(tokenizer_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a new empty derived dataset path")
    source_manifest, rows = read_messages(parent)
    pin = source_manifest["config"]["tokenizers"][MODEL]
    if pin["revision"] != revision:
        raise ValueError("Nanbeige tokenizer revision differs from the pinned model")
    for name, expected in pin["asset_sha256"].items():
        if file_sha(tokenizer_dir / name) != expected:
            raise ValueError(f"pinned Nanbeige tokenizer file differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    out.mkdir(parents=True, exist_ok=True)
    source = {"parent_manifest_sha256": file_sha(parent / "manifest.json"),
              "responses": "preexisting-normalized-messages-no-new-generation"}
    reports, exclusions, dev_prompts = {}, [], set()
    started = time.monotonic()
    for split in ("dev", "train"):
        indices, kept_ids, tokens, targets, domains = [], [], 0, 0, {}
        with (out / f"{split}.jsonl").open("wb") as handle:
            for number, identity in enumerate(source_manifest["ordered_record_ids"][split], 1):
                row = rows[identity]
                try:
                    doc = native_document(row, tokenizer, source)
                except ValueError as error:
                    exclusions.append({"split": split, "record_id": identity, "reason": str(error)})
                    continue
                n, p = len(doc["input_ids"]), doc["prompt_length"]
                reason = ("complete-document-over-32K" if n > length else
                          "train-input-token-budget" if split == "train" and tokens + n > budget else
                          "tokenized-dev-prompt-overlap" if split == "train" and doc["prompt_sha256"] in dev_prompts else None)
                if reason:
                    exclusions.append({"split": split, "record_id": identity, "reason": reason, "input_tokens": n})
                    continue
                if split == "dev":
                    dev_prompts.add(doc["prompt_sha256"])
                indices.append({"offset": handle.tell(), "id": identity, "sha256": doc["sha256"],
                                "input_tokens": n, "targets": n-p, "domain": doc["domain"]})
                handle.write((json.dumps(doc, ensure_ascii=False, separators=(",", ":"))+"\n").encode())
                kept_ids.append(identity)
                tokens += n
                targets += n-p
                domains[doc["domain"]] = domains.get(doc["domain"], 0) + 1
                if number % 100 == 0:
                    print(json.dumps({"split": split, "processed": number, "input_tokens": tokens,
                                      "excluded": len(exclusions), "seconds": time.monotonic()-started}), flush=True)
        if not indices:
            raise ValueError(f"no usable {split} documents; inspect native template exclusions")
        packs = best_fit_packs(indices, length, seed)
        atomic_json(out / f"{split}.index.json", indices)
        atomic_json(out / f"{split}.packs.json", packs)
        reports[split] = {"documents": len(indices), "input_tokens": tokens, "target_tokens": targets,
                          "packs": len(packs), "domains": domains, "ordered_record_ids": kept_ids}
    atomic_json(out / "exclusions.json", exclusions)
    files = {p.name: file_sha(p) for p in out.iterdir() if p.is_file()}
    manifest = {"schema_version": 1, "model_tokenizer": MODEL, "model_revision": revision,
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer_dir), "template_kwargs": TEMPLATE,
                "sequence_length": length, "seed": seed, "input_token_budget": budget,
                "source": source, "splits": reports, "files": files,
                "positions": "reset-within-document", "packing": "seeded-best-fit-complete-documents",
                "prompt_rule": "native-single-pass-thinking-prompt",
                "loss_mask": "final-response-suffix-through-EOS",
                "excluded_documents": len(exclusions), "truncated_documents": 0,
                "preparation_script_sha256": file_sha(__file__)}
    manifest["sha256"] = digest(manifest)
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"data_sha256": manifest["sha256"], "splits": {
        split: {k:v for k,v in report.items() if k != "ordered_record_ids"} for split,report in reports.items()},
        "excluded_documents": len(exclusions), "seconds": time.monotonic()-started}), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, required=True, help="normalized D_OT3 message shards")
    parser.add_argument("--out", type=Path, required=True, help="new, empty output directory")
    parser.add_argument("--tokenizer", default=MODEL,
                        help="local directory or HF repo id with the pinned Nanbeige tokenizer (default: registry)")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_dir = registry.resolve(args.tokenizer, allow_patterns=registry.TOKENIZER_FILES)
    prepare(args.parent, args.out, tokenizer_dir=tokenizer_dir, revision=registry.revision(MODEL))


if __name__ == "__main__":
    main()
