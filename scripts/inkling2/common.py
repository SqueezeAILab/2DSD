# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Journal and throughput conventions shared by the Inkling 2 experiment scripts.

Standard library only: scripts/inkling2/native_baseline.py runs inside the official vLLM image, where this
repository is not installed, so nothing here imports the release code. The definitions are those of
scripts/benchmark.py:

  decode throughput    (output tokens - 1) / decode seconds, pooled over responses as the sum of tokens over
                       the sum of seconds; the first token belongs to prefill.
  repetition filter    a response that hit its output budget and whose final 2048 tokens have fewer than half
                       distinct 16-grams is left out of the repetition-filtered rate.
  journal              one JSON record per line, appended with fsync; only an interrupted final line may be
                       malformed, and it is dropped before appending.
  run identity         every record carries the settings of its run; a file holds one run only.

tests/test_inkling2_scripts.py checks these against scripts/benchmark.py, and a journal written with them is
also read by `scripts/benchmark.py --summarize`.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path

LOOP_TAIL, LOOP_NGRAM, LOOP_DISTINCT = 2048, 16, 0.5
TABLE_BENCHMARK = {"aime24": "aime24/25", "aime25": "aime24/25"}  # one column in the paper's tables


def request_seed(seed: int, row_id: str, repeat: int = 0) -> int:
    """Per-request sampling seed, the same for every method."""
    digest = hashlib.sha256(f"{seed}:{row_id}:{repeat}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def repetitive(output_ids: list[int], capped: bool) -> bool:
    tail = output_ids[-LOOP_TAIL:]
    grams = [tuple(tail[i : i + LOOP_NGRAM]) for i in range(len(tail) - LOOP_NGRAM + 1)]
    return bool(capped) and len(set(grams)) / max(len(grams), 1) < LOOP_DISTINCT


def capped(row: dict) -> bool:
    """Stopped by the output budget rather than by EOS."""
    return not row["stopped_by_eos"] and len(row["output_ids"]) >= row["max_new"]


def parse_jsonl(path: Path) -> tuple[list[dict], int | None]:
    """Records of a JSONL file and the byte offset of a torn final line (None without one).

    Only the final line may be incomplete (an interrupted run); any other malformed line raises.
    """
    lines = path.read_bytes().split(b"\n") if path.exists() else []
    rows, start = [], 0
    for number, line in enumerate(lines, 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                if any(rest.strip() for rest in lines[number:]):
                    raise ValueError(f"{path}: line {number} is not a valid JSON record "
                                     "(only an interrupted final line is skipped)") from None
                return rows, start
        start += len(line) + 1
    return rows, None


def read_jsonl(path: Path) -> list[dict]:
    return parse_jsonl(path)[0]


def prepare_append(path: Path) -> None:
    """Drop a torn final line and end the last record with a newline, so appended records stay readable."""
    if not path.exists():
        return
    torn = parse_jsonl(path)[1]
    with path.open("r+b") as handle:
        end = handle.truncate(torn) if torn is not None else handle.seek(0, os.SEEK_END)
        if end:
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                handle.write(b"\n")


def append_record(path: Path, record: dict) -> None:
    """Append one record and force it to disk before the next request starts."""
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity_mismatch(rows: list[dict], identity: dict) -> str | None:
    """Why result rows are not of the run with this identity, or None if they all are."""
    for row in rows:
        recorded = row.get("run_identity")
        if not isinstance(recorded, dict):
            return "rows without a run identity"
        changed = sorted(k for k in identity.keys() | recorded.keys() if recorded.get(k) != identity.get(k))
        if changed:
            return "different " + ", ".join(changed)
    return None


def decode_rate(rows: list[dict]) -> float:
    """Pooled decode throughput: tokens after the first one over the decode seconds."""
    tokens = sum(len(r["output_ids"]) - 1 for r in rows)
    seconds = sum(r["decode_seconds"] for r in rows)
    return tokens / seconds if seconds > 0 else float("nan")


def spec_counts(row: dict) -> dict | None:
    """Verify steps, accepted and proposed draft tokens of a response, or None without speculation.

    The engine's per-request record is preferred; the Prometheus counter difference around the request is
    the fallback of builds without it.
    """
    stats, counters = row.get("spec_decode"), row.get("spec_decode_counters")
    if stats:
        return {"steps": stats["num_spec_steps"], "accepted": stats["num_accepted_draft_tokens"],
                "drafted": stats["num_draft_tokens"]}
    if counters and counters.get("num_drafts"):
        return {"steps": counters["num_drafts"], "accepted": counters["num_accepted_tokens"],
                "drafted": counters["num_draft_tokens"]}
    return None


def acceptance(rows: list[dict]) -> dict | None:
    """Pooled speculative-decoding statistics of the responses that carry them.

    tokens_per_step     decode tokens committed per verify step, from the saved output.
    acceptance_length   1 + accepted draft tokens per verify step: the engine's "mean acceptance length",
                        which counts the bonus token and the tokens trimmed after EOS.
    acceptance_rate     accepted / proposed draft tokens.

    tokens_per_step falls below acceptance_length only by the trimmed tokens. If it is above, tokens came
    from outside the verify steps (a first step with more than one token, or steps without a draft) and
    the "first token belongs to prefill" convention needs a second look for that build.
    """
    counted = [(row, counts) for row in rows if (counts := spec_counts(row))]
    steps = sum(counts["steps"] for _, counts in counted)
    if not steps:
        return None
    accepted = sum(counts["accepted"] for _, counts in counted)
    drafted = sum(counts["drafted"] for _, counts in counted)
    return {"n": len(counted), "steps": steps,
            "tokens_per_step": sum(len(row["output_ids"]) - 1 for row, _ in counted) / steps,
            "acceptance_length": 1 + accepted / steps,
            "acceptance_rate": accepted / drafted if drafted else float("nan")}


def summary(rows: list[dict]) -> list[dict]:
    """Per-benchmark decode throughput over all responses and over the non-repetitive ones.

    AIME24 and AIME25 are reported together, as in the paper's tables; "all" pools every benchmark and
    comes last.
    """
    groups = collections.defaultdict(list)
    for row in rows:
        groups[TABLE_BENCHMARK.get(row["benchmark"], row["benchmark"])].append(row)
        groups["all"].append(row)
    result = []
    for name, items in sorted(groups.items(), key=lambda kv: (kv[0] == "all", kv[0])):
        clean = [r for r in items if not repetitive(r["output_ids"], capped(r))]
        result.append({"benchmark": name, "n": len(items), "tok_s": decode_rate(items), "clean_n": len(clean),
                       "clean_tok_s": decode_rate(clean), "acceptance": acceptance(items)})
    return result


def summarize(paths: list[Path]) -> None:
    """Print the table of `scripts/benchmark.py --summarize`, then the acceptance of speculative runs."""
    tables = [(path, summary(read_jsonl(path))) for path in paths]
    print(f"{'run':40s} {'benchmark':12s} {'n':>5s} {'tok/s':>9s} {'tok/s (rep.-filtered)':>22s}")
    for path, groups in tables:
        for group in groups:
            print(f"{path.name[:40]:40s} {group['benchmark']:12s} {group['n']:5d} {group['tok_s']:9.1f} "
                  f"{group['clean_tok_s']:22.1f}")
    if not any(group["acceptance"] for _, groups in tables for group in groups):
        return
    print()
    print(f"{'run':40s} {'benchmark':12s} {'n':>5s} {'steps':>9s} {'tok/step':>9s} {'accept len':>11s} "
          f"{'accept rate':>12s}")
    for path, groups in tables:
        for group in groups:
            stats = group["acceptance"]
            if stats:
                print(f"{path.name[:40]:40s} {group['benchmark']:12s} {stats['n']:5d} {stats['steps']:9d} "
                      f"{stats['tokens_per_step']:9.2f} {stats['acceptance_length']:11.2f} "
                      f"{stats['acceptance_rate']:12.3f}")
