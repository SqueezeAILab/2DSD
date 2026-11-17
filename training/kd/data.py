"""KD documents and deterministic document-isolated packing into fixed-length (32K) rows.

The trainer reads fixed, already available responses; it never generates teacher or student text.
Two on-disk formats are accepted (training/README.md):

  ArrayDocumentStore  tokenized/<split>/{tokens,offsets,loss_mask}.npy + meta.json (D_OT3 for Ouro)
  DocumentStore       <split>.jsonl + <split>.index.json + <split>.packs.json + manifest.json
                      (written by training.kd.prepare_nanbeige)
"""
from __future__ import annotations

import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..common import registry
from ..common.io import digest, file_sha
from ..sharekv.rule import Layout


def tokenizer_fingerprint(model_path):
    model_path = Path(model_path)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    return digest({name: file_sha(model_path / name) for name in names if (model_path / name).exists()})


def best_fit_packs(documents, length, seed):
    """Seeded document order, best-fit bins, no splitting/cross-document targets."""
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    available, packs = [], []
    for index in order:
        item = documents[index]
        size = item["input_tokens"] if "input_tokens" in item else len(item["input_ids"])
        if size > length:
            raise ValueError("oversized documents must be explicitly excluded before packing")
        offset = bisect.bisect_left(available, (size, -1))
        if offset == len(available):
            pack_index, remaining = len(packs), length
            packs.append([])
        else:
            remaining, pack_index = available.pop(offset)
        packs[pack_index].append(index)
        remaining -= size
        if remaining:
            bisect.insort(available, (remaining, pack_index))
    def size(i):
        d = documents[i]
        return d["input_tokens"] if "input_tokens" in d else len(d["input_ids"])

    return [{"documents": pack, "input_tokens": sum(size(i) for i in pack),
             "targets": sum(documents[i].get("targets", size(i) - documents[i].get("prompt_length", 0)) for i in pack)}
            for pack in packs]


def pack_tensors(documents, length, pad_token_id, device="cpu"):
    ids = torch.full((1, length), pad_token_id, dtype=torch.long, device=device)
    doc_ids = torch.full_like(ids, -1)
    positions = torch.zeros_like(ids)
    is_prompt = torch.zeros_like(ids, dtype=torch.bool)
    query_indices, slices, start = [], [], 0
    for index, doc in enumerate(documents):
        n, p = len(doc["input_ids"]), doc["prompt_length"]
        if start + n > length:
            raise ValueError("pack overflows sequence length")
        ids[0, start:start + n] = torch.tensor(doc["input_ids"], device=device)
        doc_ids[0, start:start + n] = index
        positions[0, start:start + n] = torch.arange(n, device=device)
        is_prompt[0, start:start + p] = True
        # Predict the first response token at the last prompt query, and EOS
        # at the previous response query. Never predict across documents.
        query_indices.extend(range(start + p - 1, start + n - 1))
        slices.append((start, start + n, p))
        start += n
    return ids, Layout(doc_ids, positions, is_prompt), torch.tensor(query_indices, dtype=torch.long,
                                                                  device=device), slices


class DocumentStore:
    def __init__(self, directory, split):
        self.directory, self.split = Path(directory), split
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.index = json.loads((self.directory / f"{split}.index.json").read_text())
        self.packs = json.loads((self.directory / f"{split}.packs.json").read_text())
        self.path = self.directory / f"{split}.jsonl"

    def document(self, index):
        with self.path.open("rb") as f:
            f.seek(self.index[index]["offset"])
            return json.loads(f.readline())

    def pack(self, index):
        return [self.document(i) for i in self.packs[index]["documents"]]

    def verify(self):
        for name, expected in self.manifest["files"].items():
            if file_sha(self.directory / name) != expected:
                raise ValueError(f"prepared data hash mismatch: {name}")


class ArrayDocumentStore:
    """Read sealed tokens/offsets/loss_mask directly; never rewrite the arrays.

    The arrays' tokenizer must be the model's: meta.json's model revision is checked against the
    registry pin of `model` (when it is a registry name) and its tokenizer file hashes against
    tokenizer_dir (when given; the trainer passes the teacher's checkpoint directory).
    """
    def __init__(self, directory, split, model, length=32768, seed=20260930, tokenizer_dir=None):
        self.directory, self.split = Path(directory), split
        self.source_manifest = json.loads((self.directory / "manifest.json").read_text())
        self.ready = json.loads((self.directory / "ready.json").read_text())
        if not self.ready.get("data_ready"):
            raise ValueError("data is not sealed ready")
        self.meta = json.loads((self.directory / "tokenized" / split / "meta.json").read_text())
        self.array_path = self.directory / "tokenized" / split
        self.tokens = np.load(self.array_path / "tokens.npy", mmap_mode="r")
        self.offsets = np.load(self.array_path / "offsets.npy", mmap_mode="r")
        self.loss_mask = np.load(self.array_path / "loss_mask.npy", mmap_mode="r")
        if self.tokens.ndim != 1 or self.loss_mask.shape != self.tokens.shape or self.offsets.ndim != 1:
            raise ValueError("invalid array dimensions")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.tokens) or np.any(np.diff(self.offsets) <= 0):
            raise ValueError("document offsets do not partition the token array")
        pinned = registry.revision(model)
        if pinned is not None and self.meta["model_revision"] != pinned:
            raise ValueError("sealed tokenizer/model revision does not match the target model")
        if tokenizer_dir is not None:
            for name, expected in self.meta["tokenizer_fingerprint_payload"]["asset_sha256"].items():
                if file_sha(Path(tokenizer_dir) / name) != expected:
                    raise ValueError(f"sealed tokenizer asset differs: {name}")
        self.index = []
        record_ids = self.meta["ordered_record_ids"]
        if len(record_ids) != len(self.offsets) - 1:
            raise ValueError("record IDs and document boundaries differ")
        for index, identity in enumerate(record_ids):
            start, stop = map(int, self.offsets[index:index + 2])
            mask = self.loss_mask[start:stop]
            targets = np.flatnonzero(mask)
            if not len(targets) or targets[0] == 0 or not np.all(mask[targets[0]:] == 1) or np.any(mask[:targets[0]]):
                raise ValueError("KD requires one contiguous final response suffix per document")
            n, p = stop - start, int(targets[0])
            if n > length:
                raise ValueError("sealed document exceeds configured pack length")
            source = self.source_manifest.get("record_index", {}).get(identity, {})
            self.index.append({"id": identity, "start": start, "stop": stop, "input_tokens": n,
                               "prompt_length": p, "targets": len(targets), "domain": source.get("domain", "unknown"),
                               "sha256": digest({"sealed_data_hash": self.meta["data_hash"], "record_id": identity,
                                                 "document_index": index, "start": start, "stop": stop})})
        self.packs = best_fit_packs(self.index, length, seed)
        self.manifest = {"schema_version": 1, "format": "sealed-flat-arrays",
                         "source_manifest_sha256": file_sha(self.directory / "manifest.json"),
                         "data_hashes": self.ready["data_hashes"], "sequence_length": length,
                         "seed": seed, "model_tokenizer": model,
                         "template_kwargs": self.meta["template_kwargs"],
                         "loss_mask": "original-sealed-last-response-mask-unchanged"}
        self.manifest["sha256"] = digest(self.manifest)

    def document(self, index):
        item = self.index[index]
        return {**item, "input_ids": self.tokens[item["start"]:item["stop"]].astype(np.int64).tolist(),
                "source": {"sealed_data_hash": self.meta["data_hash"]}}

    def pack(self, index):
        return [self.document(i) for i in self.packs[index]["documents"]]

    def verify(self):
        for name, expected in self.meta["files"].items():
            if file_sha(self.array_path / name) != expected:
                raise ValueError(f"sealed array hash mismatch: {self.split}/{name}")
        if digest(self.meta["data_hash_payload"]) != self.meta["data_hash"]:
            raise ValueError("sealed array metadata hash mismatch")
        if self.ready["data_hashes"][self.split] != self.meta["data_hash"]:
            raise ValueError("ready.json and array metadata hash differ")
        if int(self.loss_mask.sum()) != self.meta["target_tokens"] or len(self.tokens) != self.meta["tokens"]:
            raise ValueError("sealed token counts differ")


def open_store(directory, split, model, length=32768, seed=20260930, tokenizer_dir=None):
    if (Path(directory) / "tokenized" / split / "tokens.npy").exists():
        return ArrayDocumentStore(directory, split, model, length, seed, tokenizer_dir)
    return DocumentStore(directory, split)
