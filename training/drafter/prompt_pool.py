"""Prompt pools for drafter traces: OpenThoughts3-centered, decontaminated, optionally KD-disjoint.

Only prompts are chosen here; the target regenerates every response (training.drafter.traces).
train and dev mix OpenThoughts3 math/code/science with UltraChat and CodeAlpaca; eval holds an
optional earlier benchmark selection (--previous-eval) plus a held-out OpenThoughts3 split. A
candidate is dropped when it shares a normalized 13-gram with any benchmark or Spec-Bench prompt,
when its normalized prompt is in --exclude-prompts (the paper excluded every user turn of the KD
data), or when it renders to more than 1,024 tokens with the Ouro template. Evaluation
instructions and 13-grams that recur across benchmark prompts are boilerplate, not overlap.

  build         python -m training.drafter.prompt_pool build --out POOL --benchmark-cache DIR \\
                    --benchmark-extra FILE --spec-bench FILE [--previous-eval FILE] [--exclude-prompts FILE]
  hash-prompts  python -m training.drafter.prompt_pool hash-prompts --input KD_SHARDS... --out FILE
                (the exclusion list: SHA256 of each normalized user turn, one per line)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from ..common import registry
from ..common.chat import load_tokenizer, render_user_prompt
from ..common.io import file_sha, read_jsonl

OT3 = {"repo": "open-thoughts/OpenThoughts3-1.2M", "revision": "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"}
ULTRACHAT = {"repo": "HuggingFaceH4/ultrachat_200k", "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
             "split": "train_sft"}
CODEALPACA = {"repo": "sahil2801/CodeAlpaca-20k", "revision": "152bb5e9a29651266b018106053980070a0521a1",
              "split": "train"}
# Pinned long-benchmark sources (also used for the quality evaluation).
HARD = {
    "aime25": ("yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
               "data/train-00000-of-00001-243207c6c994e1bd.parquet",
               "b74b507c1a4de7c9fafd2c83dfe2ef7b80579bcf0e0a754a1765db0f21c205fc", "problem"),
    "aime26": ("MathArena/aime_2026", "d61f486805306e6e6b5913504c07763a99e1b112", "data/train-00000-of-00001.parquet",
               "ae72853349a46c5f08666feb42db2d09fb4ac4fd4768c726b29702df21c7768e", "problem"),
    "olympiadbench": ("Hothan/OlympiadBench", "91184b52131e7fc9455fef848035173aea8cc01a",
                      "OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet",
                      "7c17ca2662b3ea77cddcb66a64b04ad9b53d3d32fa372d444a4884739eb1638d", "question"),
}
SPEC_BENCH_COMMIT = "fd2c1cd7d2201ef71db4c5f4e455008f017967bf"
INSTRUCTIONS = ("Please reason step by step, and put your final answer within \\boxed{}.",)
OT3_DOMAINS = ("math", "code", "science")
# (pool, source) -> prompts; eval also keeps the earlier benchmark selection.
QUOTAS = {
    "train": {"ot3_math": 6500, "ot3_code": 1950, "ot3_science": 1300, "chat_ultrachat": 1950, "code_alpaca": 1300},
    "dev": {"ot3_math": 125, "ot3_code": 40, "ot3_science": 25, "chat_ultrachat": 40, "code_alpaca": 20},
    "eval": {"ot3_math": 100, "ot3_code": 50, "ot3_science": 50},
}
MAX_PROMPT_TOKENS = 1024
NGRAM = 13
BOILERPLATE_DF = 3
WORD = re.compile(r"\w+")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def words(text):
    for instruction in INSTRUCTIONS:
        text = text.replace(instruction, " ")
    return WORD.findall(unicodedata.normalize("NFKC", text).lower())


def key(text):
    return " ".join(words(text))


def key_sha(normalized):
    return hashlib.sha256(normalized.encode()).hexdigest()


def ngrams(text):
    w = words(text)
    return {tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def references(benchmark_cache=None, benchmark_extra=None, spec_bench=None):
    """Benchmark and Spec-Bench prompt texts, the provenance of each set and the sets not given."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    texts, provenance, missing = defaultdict(list), {}, []
    if benchmark_cache:
        manifest = json.loads((benchmark_cache / "manifest.json").read_text())
        for name, spec in manifest["evaluation_inputs"].items():
            path = benchmark_cache / spec["cache_path"]
            texts[name] = [r["task_prompt"] for r in read_jsonl(path)]
            provenance[name] = {"cache": spec["cache_path"], "sha256": file_sha(path)}
    else:
        missing.append("benchmark-cache")
    for name, (repo, revision, file, digest, column) in HARD.items():
        path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
        if file_sha(path) != digest:
            raise RuntimeError(f"{name} file hash differs from its pin")
        texts[name] = [str(t) for t in pq.read_table(path, columns=[column]).column(column).to_pylist()]
        provenance[name] = {"repo": repo, "revision": revision, "file": file, "sha256": digest}
    if benchmark_extra:
        for r in read_jsonl(benchmark_extra):
            texts[r["benchmark"] if r["benchmark"] == "gpqa_diamond" else r["benchmark"] + "_selected"].append(
                r["native_question"])
        provenance["gpqa_diamond"] = provenance["benchmark_extra_selected"] = {
            "file": benchmark_extra.name, "sha256": file_sha(benchmark_extra)}
    else:
        missing.append("benchmark-extra")
    if spec_bench:
        texts["spec_bench"] = [turn for q in read_jsonl(spec_bench) for turn in q["turns"]]
        provenance["spec_bench"] = {"commit": SPEC_BENCH_COMMIT, "sha256": file_sha(spec_bench)}
    else:
        missing.append("spec-bench")
    return texts, provenance, missing


def contamination_index(texts):
    exact, df = set(), Counter()
    for rows in texts.values():
        for text in rows:
            exact.add(key(text))
            df.update(ngrams(text))
    grams = {g for g, n in df.items() if n < BOILERPLATE_DF}
    return exact, grams, sum(n >= BOILERPLATE_DF for n in df.values())


def user_turns(record):
    """User turns of a conversation record (messages role/content, conversations from/value, or prompt)."""
    if "prompt" in record and "messages" not in record and "conversations" not in record:
        return [record["prompt"]]
    turns = record.get("messages", record.get("conversations", []))
    return [m.get("content", m.get("value")) for m in turns if m.get("role", m.get("from")) in ("user", "human")]


def load_exclusions(path):
    """SHA256 of normalized prompts to exclude: one hex digest per line, or JSON lines with
    key_sha256 or prompt."""
    hashes = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if SHA256.match(line):
            hashes.add(line)
            continue
        row = json.loads(line)
        hashes.add(row["key_sha256"] if "key_sha256" in row else key_sha(key(row["prompt"])))
    return hashes


def hash_prompts(inputs, out):
    hashes = sorted({key_sha(key(text)) for path in inputs for record in read_jsonl(path)
                     for text in user_turns(record) if text})
    Path(out).write_text("".join(h + "\n" for h in hashes))
    print(json.dumps({"out": str(out), "prompts": len(hashes)}))


def ot3_candidates():
    """First human turn of every OpenThoughts3 row, deduplicated by normalized prompt text."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(OT3["repo"], repo_type="dataset", revision=OT3["revision"],
                                  allow_patterns=["data/*.parquet"]))
    out, seen, rows = defaultdict(list), set(), 0
    for path in sorted((root / "data").glob("*.parquet")):
        file = str(path.relative_to(root))
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["difficulty", "source", "domain", "conversations"]):
            first = pc.list_element(batch.column("conversations"), 0)
            speakers = pc.struct_field(first, "from").to_pylist()
            values = pc.struct_field(first, "value").to_pylist()
            for i, (speaker, text, domain, source, difficulty) in enumerate(zip(
                    speakers, values, batch.column("domain").to_pylist(), batch.column("source").to_pylist(),
                    batch.column("difficulty").to_pylist())):
                rows += 1
                if speaker != "human" or not text or not text.strip() or domain not in OT3_DOMAINS:
                    continue
                k = key(text)
                if k in seen:
                    continue
                seen.add(k)
                out[f"ot3_{domain}"].append({"key": k, "prompt": text, "domain": domain, "source": f"ot3_{source}",
                                             "difficulty": difficulty, "origin_ref": {"file": file, "row_index": offset + i}})
            offset += batch.num_rows
    return out, rows


def chat_code_candidates():
    from datasets import load_dataset
    out = defaultdict(list)
    seen = set()
    chat = load_dataset(ULTRACHAT["repo"], split=ULTRACHAT["split"], revision=ULTRACHAT["revision"])
    for row in chat:
        message = row["messages"][0]
        if message["role"] == "user" and message["content"].strip() and key(message["content"]) not in seen:
            seen.add(key(message["content"]))
            out["chat_ultrachat"].append({"key": key(message["content"]), "prompt": message["content"],
                                          "domain": "chat", "source": "chat_ultrachat"})
    code = load_dataset(CODEALPACA["repo"], split=CODEALPACA["split"], revision=CODEALPACA["revision"])
    for row in code:
        text = row["instruction"] if not row.get("input") else f"{row['instruction']}\n\n{row['input']}"
        if text.strip() and key(text) not in seen:
            seen.add(key(text))
            out["code_alpaca"].append({"key": key(text), "prompt": text, "domain": "code", "source": "code_alpaca"})
    return out


def build(args):
    tokenizer_dir = registry.resolve(args.tokenizer, allow_patterns=registry.TOKENIZER_FILES)
    tokenizer = load_tokenizer(tokenizer_dir)
    check = (load_tokenizer(registry.resolve(args.check_tokenizer, allow_patterns=registry.TOKENIZER_FILES))
             if args.check_tokenizer else None)

    texts, provenance, missing = references(args.benchmark_cache, args.benchmark_extra, args.spec_bench)
    if missing:
        print(f"warning: decontamination without {', '.join(missing)}", file=sys.stderr, flush=True)
    exact, grams, boilerplate = contamination_index(texts)
    exclusions = load_exclusions(args.exclude_prompts) if args.exclude_prompts else set()
    candidates, ot3_rows = ot3_candidates()
    ot3_hashes = {key_sha(r["key"]) for rows in candidates.values() for r in rows}
    candidates.update(chat_code_candidates())
    unique = {source: len(rows) for source, rows in candidates.items()}

    rng = random.Random(args.seed)
    pools, reasons = defaultdict(list), defaultdict(Counter)
    for source in sorted(candidates):
        rows = sorted(candidates[source], key=lambda r: hashlib.sha256(r["key"].encode()).hexdigest())
        rng.shuffle(rows)
        need = [(pool, QUOTAS[pool].get(source, 0)) for pool in ("eval", "dev", "train")]
        stream = iter(rows)
        for pool, count in need:
            kept = 0
            while kept < count:
                row = next(stream, None)
                if row is None:
                    raise RuntimeError(f"{source} has too few usable prompts for {pool}")
                if exclusions and key_sha(row["key"]) in exclusions:
                    reasons[source]["kd_prompt"] += 1
                    continue
                if row["key"] in exact or ngrams(row["prompt"]) & grams:
                    reasons[source]["benchmark_overlap"] += 1
                    continue
                ids = render_user_prompt(tokenizer, row["prompt"])
                if len(ids) > MAX_PROMPT_TOKENS:
                    reasons[source]["too_long"] += 1
                    continue
                digest = hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
                record = {"id": f"{source}-{digest}", "source": row["source"], "domain": row["domain"], "pool": pool,
                          "origin": source.split("_")[0] if source.startswith("ot3") else source,
                          "prompt": row["prompt"], "prompt_ids": ids}
                for extra in ("difficulty", "origin_ref"):
                    if extra in row:
                        record[extra] = row[extra]
                pools[pool].append(record)
                reasons[source]["kept"] += 1
                kept += 1

    if check is not None:
        for record in pools["dev"][:3] + pools["train"][:3]:
            if render_user_prompt(check, record["prompt"]) != record["prompt_ids"]:
                raise RuntimeError("Prompt rendering differs between the two chat templates")
    if args.previous_eval:
        pools["eval"] = read_jsonl(args.previous_eval) + pools["eval"]
    else:
        print("warning: no --previous-eval; the eval pool holds only the OpenThoughts3 split", file=sys.stderr)
    ids = [r["id"] for rows in pools.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Prompt ids repeat across pools")
    keys = {name: {key(r["prompt"]) for r in rows if "prompt" in r} for name, rows in pools.items()}
    if keys["train"] & keys["dev"] or keys["train"] & keys["eval"] or keys["dev"] & keys["eval"]:
        raise RuntimeError("Prompt text overlaps across train/dev/eval")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed, "max_prompt_tokens": MAX_PROMPT_TOKENS, "quotas": QUOTAS,
        "tokenizer": {"model": registry.label(args.tokenizer) or Path(args.tokenizer).name,
                      "tokenizer_json_sha256": file_sha(tokenizer_dir / "tokenizer.json"), "enable_thinking": True},
        "sources": {"ot3": OT3 | {"rows": ot3_rows}, "ultrachat": ULTRACHAT, "codealpaca": CODEALPACA},
        "unique_candidates": unique,
        "decontamination": {"ngram": NGRAM, "ignored_boilerplate_ngrams": boilerplate,
                            "boilerplate_min_reference_count": BOILERPLATE_DF, "instructions_removed": INSTRUCTIONS,
                            "reference_texts": {name: len(rows) for name, rows in texts.items()},
                            "references": provenance, "missing_references": missing},
        "prompt_exclusion": None if not args.exclude_prompts else {
            "file": args.exclude_prompts.name, "sha256": file_sha(args.exclude_prompts),
            "prompts": len(exclusions), "ot3_prompts_found": len(exclusions & ot3_hashes)},
        "selection": {source: dict(counts) for source, counts in reasons.items()},
        "previous_eval": None if not args.previous_eval else {
            "file": args.previous_eval.name, "sha256": file_sha(args.previous_eval)},
        "pools": {},
    }
    for name, rows in pools.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        lengths = sorted(len(r["prompt_ids"]) for r in rows)
        difficulty = Counter(r["difficulty"] for r in rows if "difficulty" in r)
        summary["pools"][name] = {
            "prompts": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
            "by_domain": dict(Counter(r["domain"] for r in rows)),
            "ot3_difficulty": {str(d): difficulty[d] for d in sorted(difficulty, key=lambda d: (d is None, d or 0))},
            "prompt_tokens": {"median": lengths[len(lengths) // 2], "max": lengths[-1]}, "sha256": file_sha(path)}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({name: pool["prompts"] for name, pool in summary["pools"].items()}
                     | {"selection": summary["selection"]}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build", help="build train/dev/eval pools")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--benchmark-cache", type=Path,
                   help="directory with manifest.json (evaluation_inputs.<name>.cache_path) and the cached "
                        "benchmark rows (task_prompt) of GSM8K, MATH500, HumanEval+ and MBPP+")
    b.add_argument("--benchmark-extra", type=Path,
                   help="JSONL of further benchmark prompts (benchmark, native_question), e.g. GPQA-Diamond")
    b.add_argument("--spec-bench", type=Path, help=f"Spec-Bench question.jsonl (commit {SPEC_BENCH_COMMIT})")
    b.add_argument("--previous-eval", type=Path, help="earlier benchmark eval rows prepended to the eval pool")
    b.add_argument("--exclude-prompts", type=Path, help="normalized-prompt SHA256 list (see hash-prompts)")
    b.add_argument("--tokenizer", default="ouro-1.4b-thinking", help="renders prompt_ids and the length cap")
    b.add_argument("--check-tokenizer", default="ouro-2.6b-thinking",
                   help="must render identically ('' skips the check)")
    b.add_argument("--seed", type=int, default=20261027)
    h = commands.add_parser("hash-prompts", help="write an exclusion list from conversation JSONL(.gz) files")
    h.add_argument("--input", type=Path, nargs="+", required=True)
    h.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args)
    else:
        hash_prompts(args.input, args.out)


if __name__ == "__main__":
    main()
