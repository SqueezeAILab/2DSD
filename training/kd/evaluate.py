"""Fixed-prefix HF diagnostics of a KD student against its frozen teacher (no generation).

train.py calls evaluate() on the dev split: teacher->K2 full-vocabulary KL by query position,
per-loop readout KL and feature cosine, and optionally teacher->exact and exact->K2 KL of the
longest dev documents at several inference windows.

  python -m training.kd.evaluate --model ouro-1.4b-thinking --checkpoint RUN/last/model --data DATA \
      --out dev-kl.json --window 64 --exact
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..common import registry
from ..common.io import atomic_json, digest
from ..hf_models import load_model
from ..sharekv.replay import exact_document
from ..sharekv.rule import k2_forward, native_forward, pad_token
from .data import best_fit_packs, open_store, pack_tensors
from .loss import full_forward_kl


BINS = ((0, 512), (512, 2048), (2048, 4096), (4096, 8192), (8192, 32768))


def summarize(values, positions):
    values, positions = values.float().cpu(), positions.cpu()
    return {"tokens": len(values), "mean_kl": float(values.mean()),
            "p99_kl": float(values.quantile(.99)),
            "by_document_query_position": {f"{lo}-{hi}": {"tokens": int(mask.sum()),
                "mean_kl": float(values[mask].mean())} for lo, hi in BINS
                if (mask := (positions >= lo) & (positions < hi)).any()}}


@torch.no_grad()
def head_kl(student_head, student_hidden, teacher_head, teacher_hidden, chunk=128):
    return torch.cat([full_forward_kl(student_head(student_hidden[start:start + chunk]),
                                      teacher_head(teacher_hidden[start:start + chunk])).cpu()
                      for start in range(0, len(student_hidden), chunk)])


def select_documents(store, per_domain=4):
    counts, selected = {}, []
    for index in sorted(range(len(store.index)), key=lambda i: digest(store.index[i]["id"] if "id" in store.index[i]
                                                                     else store.index[i]["sha256"])):
        domain = store.index[index].get("domain", "unknown")
        if counts.get(domain, 0) < per_domain:
            counts[domain] = counts.get(domain, 0) + 1
            selected.append(index)
    return selected


@torch.no_grad()
def evaluate(student, teacher, store, config, *, exact=False, all_dev=False):
    started, rows = time.monotonic(), []
    selection = list(range(len(store.index))) if all_dev else select_documents(store, config["documents_per_domain"])
    max_length = config["max_document_tokens"]
    all_kl, all_positions = [], []
    documents = []
    for index in selection:
        document = store.document(index)
        if document["prompt_length"] >= max_length:
            continue
        documents.append({**document, "input_ids": document["input_ids"][:max_length],
                          "input_tokens": min(max_length, len(document["input_ids"]))})
    for pack in best_fit_packs(documents, max_length, 20260930):
        members = [documents[i] for i in pack["documents"]]
        ids, layout, queries, slices = pack_tensors(members, max_length, pad_token(student),
                                                   next(student.parameters()).device)
        teacher_hidden, _, teacher_loops = native_forward(teacher, ids, layout,
                                                          capture="none", return_loops=True)
        student_hidden, student_loops = k2_forward(student, ids, layout, window=config["train_window"],
                                                   checkpoint_layers=0, return_loops=True)
        values = head_kl(student.lm_head, student_hidden[0, queries], teacher.lm_head,
                         teacher_hidden[0, queries])
        loop_values, loop_cosines = [], []
        for loop, (native, shared) in enumerate(zip(teacher_loops, student_loops), 1):
            loop_values.append(values if loop == len(student_loops) else
                               head_kl(student.lm_head, shared[0, queries], teacher.lm_head, native[0, queries]))
            loop_cosines.append(F.cosine_similarity(shared[0, queries].float(), native[0, queries].float(), dim=-1).cpu())
        cursor = 0
        for document, (start, end, p) in zip(members, slices):
            n = end - start - p
            positions = torch.arange(p - 1, end - start - 1)
            row = {"id": document["id"], "domain": document.get("domain", "unknown"),
                   "prompt_length": p, "length": end - start,
                   "teacher_to_k2_full_kl": summarize(values[cursor:cursor + n], positions),
                   "loop_drift": [{"loop": loop + 1, "readout_kl": float(v[cursor:cursor + n].mean()),
                                   "feature_cosine": float(c[cursor:cursor + n].mean())}
                                  for loop, (v, c) in enumerate(zip(loop_values, loop_cosines))]}
            rows.append(row)
            cursor += n
        all_kl.append(values)
        all_positions.append(layout.positions[0, queries].cpu())
    if not rows:
        raise ValueError("dev selection has no response tokens below the context limit")
    output = {"scope": "fixed-prefix-HF-diagnostic-no-generation-no-benchmark-score",
              "teacher_to_k2_full_kl": summarize(torch.cat(all_kl), torch.cat(all_positions)),
              "documents": rows}
    if exact:
        selected = sorted(selection, key=lambda i: (-store.index[i]["input_tokens"], i))[:config["exact_documents"]]
        diagnostics = []
        for index in selected:
            document = store.document(index)
            p = document["prompt_length"]
            for window in config["inference_windows"]:
                # W1 is intentionally a short sequential diagnostic. W64 checks
                # the long-history deployment rule through >=4K positions.
                exact_max = config.get("exact_max_document_tokens", 8192)
                cap = exact_max if window == 64 else min(exact_max, p + config["exact_short_response_tokens"])
                if p >= cap:
                    continue
                d = {**document, "input_ids": document["input_ids"][:cap]}
                ids, layout, queries, _ = pack_tensors([d], exact_max, pad_token(student),
                                                       next(student.parameters()).device)
                t, _, _ = native_forward(teacher, ids, layout, capture="none")
                k2, _ = k2_forward(student, ids, layout, window=window, checkpoint_layers=0)
                ar = exact_document(student, ids[:, :len(d["input_ids"])], p, window)
                exact_kl = head_kl(student.lm_head, ar[0, queries], teacher.lm_head, t[0, queries])
                approx_kl = head_kl(student.lm_head, k2[0, queries], student.lm_head, ar[0, queries])
                diagnostics.append({"id": d["id"], "window": window, "length": len(d["input_ids"]),
                                    "teacher_to_exact_full_kl": summarize(exact_kl, queries),
                                    "exact_to_k2_full_kl": summarize(approx_kl, queries)})
        output["exact"] = diagnostics
    output["seconds"] = time.monotonic() - started
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="registry name of the original (teacher) model")
    parser.add_argument("--model-path", help="local directory or HF repo id of the teacher (default: registry)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="student checkpoint directory")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--all-dev", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    teacher_dir = registry.resolve(args.model_path or args.model)
    student = load_model(args.model, args.device, args.checkpoint, frozen=True)
    teacher = load_model(args.model, args.device, teacher_dir, frozen=True)
    store = open_store(args.data, "dev", args.model, tokenizer_dir=teacher_dir)
    store.verify()
    result = evaluate(student, teacher, store,
                      {"documents_per_domain": 4, "max_document_tokens": 32768,
                       "train_window": args.window, "exact_documents": 2,
                       "exact_max_document_tokens": 8192,
                       "inference_windows": [1, 16, 64, 128], "exact_short_response_tokens": 512},
                      exact=args.exact, all_dev=args.all_dev)
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("documents", "exact")}), flush=True)


if __name__ == "__main__":
    main()
