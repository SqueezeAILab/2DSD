#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Native AR and native DSpark throughput of Inkling 2 on the official vLLM build, batch size one.

The 2DSD fork (vLLM v0.26.0) cannot serve Inkling 2, so the two vendor baselines are measured inside the
official image (vllm/vllm-openai:inkling-2-mini, Model Runner V2, torch 2.14, Python 3.14) with this
script. It needs only the standard library, vllm and torch, and scripts/inkling2/common.py next to it;
scripts/inkling2/serve_official.sh starts it in the pinned image on one GPU.

The conventions are those of scripts/benchmark.py: one request at a time (max_num_seqs 1), the prompt
token ids of a prompts file from scripts/build_prompts.py, natural EOS, a prompt + output budget of the
suite (8192 Core, 32768 Hard), greedy unless --temperature is given, --warmup short requests that are not
recorded, and the request order of random.Random(seed).shuffle, so --limit N selects the same questions
as scripts/benchmark.py. --shard I/N takes every N-th of them from the I-th on, for one engine per GPU; the
shards of a run share its run identity and their journals are concatenated.

  --method ar      plain decoding.
  --method dspark  the checkpoint's own DSpark draft (mtp.safetensors of the target repository):
                   speculative_config {"method": "dspark", "num_speculative_tokens": 7,
                   "draft_sample_method": greedy | probabilistic}. At temperature 0 both draft methods
                   accept a draft token exactly when it is the target argmax.

Timing comes from the engine's own record of the request, RequestOutput.metrics (RequestStateStats in
vllm/v1/metrics/stats.py, read at vLLM PR #69028 head b426b46e; the image reports another commit of that
work, so optional fields are probed, not assumed). Its timestamps are time.monotonic() values that the
engine core takes when it assembles the outputs of a step (EngineCoreOutputs.timestamp):

  prefill_seconds = first_token_ts - scheduled_ts   up to and including the step that yields token 1
  decode_seconds  = last_token_ts - first_token_ts  from token 1 to the last step of the request
  decode tok/s    = (len(output_ids) - 1) / decode_seconds

These are the fields and the formula of `scripts/benchmark.py --method ar` (the fork's vLLM AR reference),
and of vLLM's own decode_time and mean_time_per_output_token. Drafting, verification and scheduling between
the two timestamps are included, as for 2DSD; tokenization, queueing and detokenization (off) are not.
2DSD itself is timed inside its worker (time.perf_counter after a CUDA synchronisation, from the choice of
the first token to the end of the last call), so its interval lacks the engine-core scheduling that this
one contains. A speculative step yields several tokens at once: the last step counts in full even when
tokens after EOS are trimmed. vLLM fills RequestOutput.metrics only with disable_log_stats=False, which
this script sets.

Each result line of --out holds: id, benchmark, prompt_ids, output_ids, finish_reason, stop_reason,
stopped_by_eos, max_new, prefill_seconds, decode_seconds, decode_tokens, queue_seconds, preemptions,
spec_decode (the engine's per-request acceptance record, CompletionOutput.spec_decode_metrics, enabled
with per_request_spec_decode_metrics), spec_decode_counters (difference of the vllm:spec_decode_*
Prometheus counters around the request, for builds without the former) and run_identity. The lines are
readable by `scripts/benchmark.py --summarize`.

A rerun resumes: finished ids are skipped, and --out must hold only results of the same run (checkpoint,
method, draft sampling, sampling, budget cap, extra engine arguments, prompts file, vLLM and torch
versions, image digest and GPU model).

Examples (inside the image; see serve_official.sh for the wrapper):
  python3 native_baseline.py --model /hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/REV \\
      --prompts /prompts/core.prompts.jsonl --out /out/mini-core-ar.jsonl --limit 64
  python3 native_baseline.py --method dspark --model ... --prompts ... --out /out/mini-core-dspark.jsonl
  python3 native_baseline.py --summarize /out/mini-core-ar.jsonl /out/mini-core-dspark.jsonl

--summarize prints the table of `scripts/benchmark.py --summarize` (decode tok/s per benchmark over all
responses and repetition-filtered), then, for speculative runs, verify steps, tokens per step, mean
acceptance length and draft acceptance rate. It needs no GPU and no vllm.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path


def sibling(name: str):
    """A module of this directory, loaded by path: the scripts are not an installed package."""
    key = f"inkling2_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


common = sibling("common")

NUM_SPECULATIVE_TOKENS = 7  # the bundled draft's block: one anchor and seven drafted tokens
WARMUP_TOKENS = 64
SPEC_COUNTERS = "vllm:spec_decode_"
# The settings that change a run's results; with the checkpoint, the build and the prompts file's sha256
# they form its run identity.
IDENTITY = ("method", "draft_sample_method", "temperature", "top_p", "seed", "max_total_tokens", "engine_args")


def model_name(model: Path) -> str:
    """`org/name@revision` of a Hugging Face cache snapshot (the same on the host and in a container), else
    the resolved path."""
    path = model.resolve()
    cache = path.parent.parent.name
    if path.parent.name == "snapshots" and cache.startswith("models--"):
        return cache.removeprefix("models--").replace("--", "/") + "@" + path.name
    return str(path)


def gpu_name() -> str | None:
    """Model and compute capability of the visible GPU, from nvidia-smi so that CUDA stays uninitialised in
    this process."""
    command = ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.isdigit():
        command += ["-i", visible]
    try:
        listing = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return "; ".join(sorted({line.strip() for line in listing.splitlines() if line.strip()})) or None


def engine_identity() -> dict:
    """The build and device of the measurement. INKLING2_IMAGE is set by serve_official.sh."""
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {"vllm": version("vllm"), "torch": version("torch"), "python": platform.python_version(),
            "image": os.environ.get("INKLING2_IMAGE"), "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "gpu": gpu_name()}


def run_identity(args, prompts_bytes: bytes) -> dict:
    """The run identity recorded in every result row; --out resumes only rows of the same identity."""
    identity = {key: getattr(args, key) for key in IDENTITY}
    config = args.model / "config.json"
    identity["model"] = model_name(args.model)
    identity["model_config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None
    identity["num_speculative_tokens"] = NUM_SPECULATIVE_TOKENS if args.method == "dspark" else None
    identity["engine"] = engine_identity()
    identity["prompts_sha256"] = hashlib.sha256(prompts_bytes).hexdigest()
    return identity


def eos_ids(model: Path, tokenizer) -> list[int]:
    """Union of config.json, generation_config.json and tokenizer EOS ids, as in scripts/benchmark.py."""
    values = [getattr(tokenizer, "eos_token_id", None)]
    for name in ("config.json", "generation_config.json"):
        if (model / name).exists():
            values.append(json.loads((model / name).read_text()).get("eos_token_id"))
    ids = set()
    for value in values:
        ids.update(value if isinstance(value, (list, tuple)) else [value])
    return sorted(v for v in ids if type(v) is int and v >= 0)


def engine_accepts(field: str) -> bool:
    """Whether this build's engine arguments have the field."""
    from vllm.engine.arg_utils import EngineArgs

    return field in {f.name for f in dataclasses.fields(EngineArgs)}


def engine_options(args, max_len: int) -> dict:
    """Keyword arguments of vllm.LLM: batch size one, everything else the build's default unless
    --engine-args replaces it."""
    options = {
        "model": str(args.model),
        "tensor_parallel_size": 1,
        "max_model_len": max_len,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "seed": 0,
        "disable_log_stats": False,  # RequestOutput.metrics and the Prometheus counters need the stat loggers
        "generation_config": "vllm",
    }
    if args.method == "dspark":
        options["speculative_config"] = {"method": "dspark", "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                                         "draft_sample_method": args.draft_sample_method}
        if args.spec_metrics != "none" and engine_accepts("per_request_spec_decode_metrics"):
            options["per_request_spec_decode_metrics"] = args.spec_metrics
    options.update(args.engine_args)
    return options


def spec_counters(llm) -> dict:
    """The engine's cumulative speculative-decoding counters, by name without the vllm:spec_decode_ prefix;
    per-position counters become lists."""
    counters: dict = {}
    for metric in llm.get_metrics():
        if not metric.name.startswith(SPEC_COUNTERS):
            continue
        name = metric.name.removeprefix(SPEC_COUNTERS)
        if hasattr(metric, "values"):
            counters[name] = [int(v) for v in metric.values]
        elif "position" in metric.labels:
            values = counters.setdefault(name, [])
            position = int(metric.labels["position"])
            values.extend([0] * (position + 1 - len(values)))
            values[position] = int(metric.value)
        elif hasattr(metric, "value"):
            counters[name] = int(metric.value)
    return counters


def counter_delta(before: dict, after: dict) -> dict:
    """What one request added to the counters."""
    delta = {}
    for name, value in after.items():
        if isinstance(value, list):
            old = before.get(name, [])
            delta[name] = [v - (old[i] if i < len(old) else 0) for i, v in enumerate(value)]
        else:
            delta[name] = value - before.get(name, 0)
    return delta


def measure(output, eos: list[int], budget: int) -> dict:
    """Result fields of a finished request, from the engine's record of it (see the module docstring)."""
    metrics = output.metrics
    if metrics is None:
        raise RuntimeError("the engine returned no request metrics; vllm.LLM needs disable_log_stats=False")
    completion = output.outputs[0]
    ids = list(completion.token_ids)
    first, last = metrics.first_token_ts, metrics.last_token_ts
    spec = getattr(completion, "spec_decode_metrics", None)
    return {
        "output_ids": ids,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "stopped_by_eos": bool(ids) and ids[-1] in eos,
        "max_new": budget,
        "prefill_seconds": first - metrics.scheduled_ts,
        "decode_seconds": last - first,
        "decode_tokens": max(len(ids) - 1, 0),
        "queue_seconds": metrics.scheduled_ts - metrics.queued_ts,
        "preemptions": getattr(metrics, "num_preemptions", None),
        "spec_decode": spec.to_dict() if spec is not None else None,
    }


def record_metadata(row: dict, identity: dict) -> dict:
    return {
        "id": row["id"],
        "benchmark": row.get("benchmark"),
        "method": identity["method"],
        "temperature": identity["temperature"],
        "top_p": identity["top_p"],
        "prompt_tokens": len(row["prompt_ids"]),
        "prompt_ids": row["prompt_ids"],
        "run_identity": identity,
    }


def total_budget(row: dict, cap: int | None) -> int:
    """Prompt + output tokens of a request: the suite's budget, or --max-total-tokens if smaller."""
    return row["max_total_tokens"] if cap is None else min(row["max_total_tokens"], cap)


def output_budget(row: dict, cap: int | None) -> int:
    return total_budget(row, cap) - len(row["prompt_ids"])


def run(args, rows: list[dict], out: Path, identity: dict) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(**engine_options(args, max(total_budget(r, args.max_total_tokens) for r in rows)))
    eos = eos_ids(args.model, llm.get_tokenizer())
    speculative = args.method == "dspark"

    def generate(row, budget):
        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=common.request_seed(args.seed, row["id"]) if args.temperature > 0 else None,
            max_tokens=budget,
            stop_token_ids=eos,
            detokenize=False,
        )
        before = spec_counters(llm) if speculative else None
        [output] = llm.generate([{"prompt_token_ids": row["prompt_ids"]}], params, use_tqdm=False)
        result = measure(output, eos, budget)
        result["spec_decode_counters"] = counter_delta(before, spec_counters(llm)) if speculative else None
        return result

    for _ in range(args.warmup):
        generate(rows[0], min(WARMUP_TOKENS, output_budget(rows[0], args.max_total_tokens)))
    for row in rows:
        result = generate(row, output_budget(row, args.max_total_tokens))
        result.update(record_metadata(row, identity))
        common.append_record(out, result)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="local snapshot directory of the checkpoint (no download)")
    parser.add_argument("--method", choices=("ar", "dspark"), default="ar")
    parser.add_argument("--draft-sample-method", choices=("greedy", "probabilistic"),
                        help="dspark only: how the draft samples its tokens (default greedy; the vLLM recipe "
                        "serves probabilistic)")
    parser.add_argument("--prompts", type=Path, help="SUITE.prompts.jsonl from scripts/build_prompts.py")
    parser.add_argument("--out", type=Path, help="results JSONL (appended; finished ids of the same run are skipped)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="first N prompts of the shuffled order")
    parser.add_argument("--shard", help="I/N: only prompts I, I + N, ... of the shuffled (and limited) order, as "
                        "scripts/benchmark.py; give every shard its own --out (one engine per GPU)")
    parser.add_argument("--warmup", type=int, default=2, help=f"{WARMUP_TOKENS}-token warmup requests (not recorded)")
    parser.add_argument("--max-total-tokens", type=int,
                        help="cap the prompt + output budget below the prompts file's (part of the run identity)")
    parser.add_argument("--spec-metrics", choices=("summary", "detailed", "none"), default="summary",
                        help="dspark only: the engine's per-request acceptance record; detailed adds the "
                        "accepted, proposed and verified draft tokens of every verify step")
    parser.add_argument("--engine-args", type=json.loads, default={}, metavar="JSON",
                        help="extra vllm.LLM keyword arguments as a JSON object, e.g. "
                        '\'{"gpu_memory_utilization": 0.9}\' (part of the run identity)')
    parser.add_argument("--summarize", nargs="+", type=Path, help="only print the tables of results files")
    args = parser.parse_args(argv)
    if args.summarize:
        common.summarize(args.summarize)
        return
    if not (args.model and args.prompts and args.out):
        parser.error("--model, --prompts and --out are required")
    if not args.model.is_dir():
        parser.error(f"--model {args.model} is not a local snapshot directory")
    if not isinstance(args.engine_args, dict):
        parser.error("--engine-args takes a JSON object")
    if args.method == "ar" and args.draft_sample_method:
        parser.error("--draft-sample-method applies to --method dspark only")
    if args.method == "dspark":
        args.draft_sample_method = args.draft_sample_method or "greedy"
    prompts_bytes = args.prompts.read_bytes()  # read once: the hash names the prompts that run
    rows = [json.loads(line) for line in prompts_bytes.decode().splitlines() if line.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows
    if args.shard:
        try:
            shard, shards = (int(part) for part in args.shard.split("/"))
        except ValueError:
            parser.error("--shard is I/N")
        if not 0 <= shard < shards:
            parser.error("--shard I/N needs 0 <= I < N")
        rows = rows[shard::shards]
    for row in rows:
        if output_budget(row, args.max_total_tokens) < 1:
            parser.error(f"the prompt of {row['id']} ({len(row['prompt_ids'])} tokens) leaves no output budget")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    identity = run_identity(args, prompts_bytes)
    print("run identity " + json.dumps(identity, sort_keys=True), file=sys.stderr)
    finished = common.read_jsonl(args.out)
    mismatch = common.identity_mismatch(finished, identity)
    if mismatch:
        parser.error(f"{args.out} holds results of another run ({mismatch}); choose a new --out")
    done = {r["id"] for r in finished}
    pending = [r for r in rows if r["id"] not in done]
    print(f"{len(pending)} of {len(rows)} prompts pending", file=sys.stderr)
    if pending:
        common.prepare_append(args.out)
        run(args, pending, args.out, identity)
    common.summarize([args.out])


if __name__ == "__main__":
    main()
