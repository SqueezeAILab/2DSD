#!/usr/bin/env python3
"""Decode-only throughput of 2DSD (and the native AR reference) at batch size one.

Runs every prompt of a prompts file from scripts/build_prompts.py on one GPU and
appends one JSON line per request to --out. A rerun resumes: finished ids are
skipped, and --out must hold only results of the same run (model, revision,
drafter, method, sampling, 2DSD settings, KV policy and prompts file). Then
prints decode throughput per benchmark, as in the paper (App. D.2): tokens
after the first one (which prefill produces) over the decode time, excluding
prefill and including drafting, scheduling, verification and CUDA graph capture.
"repetition-filtered" excludes capped responses whose final 2048 tokens have
fewer than half distinct 16-grams.

Examples (the paper's 2DSD run and its native AR reference):
  python scripts/benchmark.py --model ouro-1.4b-thinking \\
      --prompts prompts/ouro-1.4b/core.prompts.jsonl --out runs/ouro-1.4b-core-2dsd.jsonl
  python scripts/benchmark.py --method ar --model ouro-1.4b-thinking --target original \\
      --prompts prompts/ouro-1.4b/core.prompts.jsonl --out runs/ouro-1.4b-core-ar.jsonl
  python scripts/benchmark.py --summarize runs/ouro-1.4b-core-2dsd.jsonl runs/ouro-1.4b-core-ar.jsonl

--model is a registry name of configs/models.json (ouro-1.4b-thinking,
ouro-2.6b-thinking, nanbeige4.2-3b, dory1.0-4b), a local checkpoint or a
Hugging Face id. A registry name runs one of its targets (--target):
  w64       the shared-KV target (W = 64), the paper's 2DSD target; 2DSD uses
            its drafter. Dory: its pinned base checkpoint with --kv-policy w64
            (training-free last-loop sharing).
  original  the base model at its pinned revision, the paper's native AR
            reference; 2DSD uses the base model's drafter. Dory: its pinned
            checkpoint with --kv-policy native (per-loop KV). One drafter
            serves both Dory targets.
Explicit flags override the registry: --revision sets the revision of the
selected target's repository, --drafter replaces the registry's drafter, and
--kv-policy (Dory only) the policy that --target sets. No drafter runs (and
none is recorded) for --method ar or with --block-size 1. Any other --model
runs as given, without --target: pass --drafter for 2DSD, --revision as needed,
and for a Dory checkpoint --kv-policy (w64 or native). The run identity records
the resolved repository, revision, drafter and KV policy, so a run given by
registry name resumes when given by those explicit values, and the other way
round.

Sampling (paper): --temperature 0.6 --top-p 0.95. Every request uses the seed
sha256(f"{seed}:{id}:0")[:4], the same for every method.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # training.common.registry reads configs/models.json
from training.common import registry  # noqa: E402

LOOP_TAIL, LOOP_NGRAM, LOOP_DISTINCT = 2048, 16, 0.5
# The settings that change a run's results; with the prompts file's sha256 they form its run identity.
IDENTITY = ("model", "revision", "drafter", "method", "temperature", "top_p", "seed", "stage", "block_size",
            "d1", "d2", "kv_policy")


def request_seed(seed: int, row_id: str, repeat: int = 0) -> int:
    """Per-request sampling seed, the same for every method."""
    digest = hashlib.sha256(f"{seed}:{row_id}:{repeat}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def repetitive(output_ids: list[int], capped: bool) -> bool:
    tail = output_ids[-LOOP_TAIL:]
    grams = [tuple(tail[i : i + LOOP_NGRAM]) for i in range(len(tail) - LOOP_NGRAM + 1)]
    return bool(capped) and len(set(grams)) / max(len(grams), 1) < LOOP_DISTINCT


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


def run_identity(args, prompts_bytes: bytes) -> dict:
    """The run identity recorded in every result row; --out resumes only rows of the same identity.

    `args` holds the resolved model, revision, drafter and KV policy (a registry name is recorded as its
    pinned repository and revision). Local model and drafter directories are recorded as absolute paths,
    so another spelling of the same directory resumes; Hugging Face ids are recorded as given."""
    identity = {key: getattr(args, key) for key in IDENTITY}
    for key in ("model", "drafter"):
        if identity[key] is not None and Path(identity[key]).exists():
            identity[key] = str(Path(identity[key]).resolve())
    identity["prompts_sha256"] = hashlib.sha256(prompts_bytes).hexdigest()
    return identity


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


def eos_ids(model: str, revision: str | None, tokenizer, config) -> list[int]:
    """Union of config.json, generation_config.json and tokenizer EOS ids."""
    from transformers import GenerationConfig

    values = [getattr(config, "eos_token_id", None), tokenizer.eos_token_id]
    try:
        values.append(GenerationConfig.from_pretrained(model, revision=revision).eos_token_id)
    except OSError:
        pass
    ids = set()
    for value in values:
        ids.update(value if isinstance(value, (list, tuple)) else [value])
    return sorted(v for v in ids if type(v) is int and v >= 0)


def run_2dsd(args, rows, out, identity):
    from vllm.v1.spec_decode.twodsd import TwoDSD

    engine = TwoDSD(
        args.model,
        args.drafter,
        max_model_len=max(r["max_total_tokens"] for r in rows),
        stage=args.stage,
        block_size=args.block_size,
        d1=args.d1,
        d2=args.d2,
        revision=args.revision,
        seed=args.seed,
        kv_policy=args.kv_policy,
    )
    sampling = args.temperature > 0
    # One worker call for the whole run, as in the paper's measurement: the runner,
    # drafter and CUDA graphs stay warm, and the worker appends every result to
    # --out as soon as it finishes (resumable).
    engine.generate(
        [r["prompt_ids"] for r in rows],
        max_new_tokens=[r["max_total_tokens"] - len(r["prompt_ids"]) for r in rows],
        temperature=args.temperature,
        top_p=args.top_p,
        seeds=[request_seed(args.seed, r["id"]) for r in rows] if sampling else None,
        metadata=[record_metadata(r, identity) for r in rows],
        warmup_requests=args.warmup,
        journal_path=str(out.resolve()),
    )


def run_ar(args, rows, out, identity):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.v1.spec_decode.twodsd import engine_options

    max_len = max(r["max_total_tokens"] for r in rows)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_len,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        disable_log_stats=False,
        model_impl="vllm",
        generation_config="vllm",
        **engine_options(args.model, args.revision, args.kv_policy),
    )
    config = llm.llm_engine.model_config.hf_config
    eos = eos_ids(args.model, args.revision, llm.get_tokenizer(), config)

    def generate(row, budget):
        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=request_seed(args.seed, row["id"]) if args.temperature > 0 else None,
            max_tokens=budget,
            stop_token_ids=eos,
            detokenize=False,
        )
        [output] = llm.generate(
            [{"prompt_token_ids": row["prompt_ids"]}], params, use_tqdm=False
        )
        metrics = output.metrics
        first, last = metrics.first_token_ts, metrics.last_token_ts
        ids = list(output.outputs[0].token_ids)
        return {
            "output_ids": ids,
            "prefill_seconds": first - metrics.scheduled_ts,
            "decode_seconds": last - first,
            "max_new": budget,
            "stopped_by_eos": bool(ids) and ids[-1] in eos,
        }

    for _ in range(args.warmup):
        generate(rows[0], min(64, rows[0]["max_total_tokens"] - len(rows[0]["prompt_ids"])))
    for row in rows:
        result = generate(row, row["max_total_tokens"] - len(row["prompt_ids"]))
        result.update(record_metadata(row, identity))
        with out.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def record_metadata(row: dict, identity: dict) -> dict:
    return {
        "id": row["id"],
        "benchmark": row.get("benchmark"),
        "method": identity["method"],
        "temperature": identity["temperature"],
        "top_p": identity["top_p"],
        "kv_policy": identity["kv_policy"],
        "prompt_tokens": len(row["prompt_ids"]),
        "run_identity": identity,
    }


def capped(row: dict) -> bool:
    """Stopped by the output budget rather than by EOS."""
    return not row["stopped_by_eos"] and len(row["output_ids"]) >= row["max_new"]


def summarize(paths: list[Path]) -> None:
    print(f"{'run':40s} {'benchmark':12s} {'n':>5s} {'tok/s':>9s} {'tok/s (rep.-filtered)':>22s}")
    for path in paths:
        rows = read_jsonl(path)
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row["benchmark"]].append(row)
            groups["all"].append(row)
        for name, items in sorted(groups.items(), key=lambda kv: (kv[0] == "all", kv[0])):
            def rate(selected):
                tokens = sum(len(r["output_ids"]) - 1 for r in selected)
                seconds = sum(r["decode_seconds"] for r in selected)
                return tokens / seconds if seconds > 0 else float("nan")

            clean = [r for r in items if not repetitive(r["output_ids"], capped(r))]
            print(f"{path.name[:40]:40s} {name:12s} {len(items):5d} {rate(items):9.1f} {rate(clean):22.1f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("2dsd", "ar"), default="2dsd")
    parser.add_argument("--model", help=f"registry name ({', '.join(registry.names())}), "
                        "local checkpoint or Hugging Face id")
    parser.add_argument("--target", choices=("w64", "original"),
                        help="registry names only: w64 shared-KV target (default) or original base model")
    parser.add_argument("--revision", help="revision of the target repository (replaces a base model's pin)")
    parser.add_argument("--drafter", help="drafter export (local path, Hugging Face id, or id@revision); "
                        "overrides a registry name's drafter")
    parser.add_argument("--prompts", type=Path, help="SUITE.prompts.jsonl from build_prompts.py")
    parser.add_argument("--out", type=Path, help="results JSONL (appended; finished ids of the same run are skipped)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="first N prompts of the shuffled order")
    parser.add_argument("--warmup", type=int, default=2, help="64-token warmup requests")
    parser.add_argument("--stage", choices=("stage1", "stage2a", "stage2b"),
                        help="2DSD variant: stage1 drafts at r1 only, stage2a adds deeper promotion at r2, stage2b "
                        "also forks at r2 (default: the paper's setting for the model)")
    parser.add_argument("--block-size", type=int, help="drafted block size k (default 16; 1 runs no drafter)")
    parser.add_argument("--d1", type=int, help="early readout loop r1 (default 1)")
    parser.add_argument("--d2", type=int, help="deeper readout loop r2 (default 2 when R = 4)")
    parser.add_argument("--kv-policy", choices=("w64", "native"),
                        help="Dory only: w64 last-loop sharing or native per-loop KV; set by --target for "
                        "dory1.0-4b, required for a Dory checkpoint given by path or id")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                        help="vLLM KV pool for --method ar (0.9 for Ouro-2.6B at 32K)")
    parser.add_argument("--summarize", nargs="+", type=Path, help="only print the tables of results files")
    args = parser.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return
    if not (args.model and args.prompts and args.out):
        parser.error("--model, --prompts and --out are required")
    try:
        resolved = registry.serving(args.model, args.target, revision=args.revision, drafter=args.drafter,
                                    kv_policy=args.kv_policy,
                                    with_drafter=args.method == "2dsd" and args.block_size != 1)
    except ValueError as error:
        parser.error(str(error))
    family = resolved.pop("family")
    vars(args).update(resolved)
    if args.method == "2dsd" and args.drafter is None and args.block_size != 1:
        parser.error("2DSD needs --drafter (or --block-size 1)")
    if family is None:  # outside the registry
        from vllm.v1.spec_decode.twodsd import model_type

        family = model_type(args.model, args.revision)
    if (family == "dory") != (args.kv_policy is not None):
        parser.error("--kv-policy is required for Dory (w64 or native) and applies to Dory only")
    print(f"target {args.model}" + (f"@{args.revision}" if args.revision else "")
          + f", drafter {args.drafter}, kv_policy {args.kv_policy or 'from config.json'}", file=sys.stderr)
    prompts_bytes = args.prompts.read_bytes()  # read once: the hash names the prompts that run
    rows = [json.loads(line) for line in prompts_bytes.decode().splitlines() if line.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    identity = run_identity(args, prompts_bytes)
    finished = read_jsonl(args.out)
    mismatch = identity_mismatch(finished, identity)
    if mismatch:
        parser.error(f"{args.out} holds results of another run ({mismatch}); choose a new --out")
    done = {r["id"] for r in finished}
    pending = [r for r in rows if r["id"] not in done]
    print(f"{len(pending)} of {len(rows)} prompts pending", file=sys.stderr)
    if pending:
        prepare_append(args.out)
        (run_2dsd if args.method == "2dsd" else run_ar)(args, pending, args.out, identity)
    summarize([args.out])


if __name__ == "__main__":
    main()
