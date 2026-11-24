#!/usr/bin/env python3
"""Decode-only throughput of 2DSD (and the native AR reference) at batch size one.

Runs every prompt of a prompts file from scripts/build_prompts.py on one GPU and
appends one JSON line per request to --out. A rerun resumes: finished ids are
skipped, and --out must hold only results of the same run (model, revision,
drafter, method, sampling, 2DSD settings, KV policy and prompts file). Then
prints decode throughput per benchmark, as in the paper (App. D.2): tokens
after the first one (which prefill produces) over the decode time, excluding
prefill and including drafting and verification. AIME24 and AIME25 are
reported together as aime24/25, as in the paper's tables. "repetition-filtered"
excludes capped responses whose final 2048 tokens have fewer than half distinct
16-grams.

Examples (the paper's 2DSD run and its native AR reference):
  python scripts/benchmark.py --model ouro-1.4b-thinking \\
      --prompts prompts/ouro-1.4b/core.prompts.jsonl --out runs/ouro-1.4b-core-2dsd.jsonl
  python scripts/benchmark.py --method ar --model ouro-1.4b-thinking --target original \\
      --prompts prompts/ouro-1.4b/core.prompts.jsonl --out runs/ouro-1.4b-core-ar.jsonl
  python scripts/benchmark.py --summarize runs/ouro-1.4b-core-2dsd.jsonl runs/ouro-1.4b-core-ar.jsonl

--model is a registry name of configs/models.json (ouro-1.4b-thinking,
ouro-2.6b-thinking, nanbeige4.2-3b, dory1.0-4b, inkling-2-mini, inkling-2-mini-varloop), a local
checkpoint or a Hugging Face id. A registry name runs one of its targets
(--target):
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

Inkling2 (inkling-2-mini, inkling-2-mini-varloop, or a checkpoint of model_type inkling2_loop) runs
without a vLLM engine, on its own tree runner. By default that is the static
runner on tensor cores with the drafter's CUDA graphs, and packed experts when
humming-kernels >= 0.1.20 is importable (--humming DIR), dense ones otherwise
(slower: the engine warns);
--runner, --experts, --no-tensor-cores and --no-drafter-cuda-graph select the
other paths (--runner reference: one token at a time through the reference
forward, the runner the others are checked against). Its target is original
(per-loop KV): it has no KV sharing. The registry has the DSpark drafter of
inkling-2-mini-varloop and none for inkling-2-mini yet (pass --drafter or
--block-size 1). Its EOS ids are fixed to [200006].
--method ar is the same runner decoding one token at a time; it is not the
vendor's engine (scripts/inkling2/native_baseline.py measures that one).
--loops R runs a variable-loop checkpoint with R passes. --shard I/N runs every
N-th prompt of the order, for one process per GPU with its own --out.

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
# Families without an engine model in this vLLM build: --method ar runs their own runner.
NO_AR_ENGINE = ("inkling2",)


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
    # Inkling2's runtime (the engine's defaults and the flags): its runners and arithmetics can differ at
    # near-ties and its drafter graphs change the drafts, so a run resumes only under the runtime that
    # started it. Recorded where it departs from the batched runner on the reference arithmetic.
    runtime = getattr(args, "runtime", None) or getattr(args, "engine_kwargs", {})
    if runtime.get("runner") not in (None, "batched"):
        identity["runner"] = runtime["runner"]
    for key in ("tensor_cores", "experts", "drafter_cuda_graph"):
        if runtime.get(key) not in (None, False):
            identity[key] = runtime[key]
    # Whether Humming's expert GEMM runs: from --humming DIR or the environment's own installation.
    if getattr(args, "expert_backend", None) == "humming":
        identity["humming"] = True
    if getattr(args, "engine_kwargs", {}).get("drafter_of_other_target"):  # a deliberate transfer
        identity["drafter_of_other_target"] = True
    # Another number of passes is another model output; the checkpoint's own count is not recorded.
    for key in ("loops", "drafter_of_other_loops"):
        if getattr(args, "engine_kwargs", {}).get(key):
            identity[key] = args.engine_kwargs[key]
    if getattr(args, "proposal_top_p", None) is not None:  # other proposals: other draws under a seed
        identity["proposal_top_p"] = args.proposal_top_p
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
        **args.engine_kwargs,
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
        eos_token_ids=args.eos_token_ids,  # None: the engine reads the checkpoint's EOS ids
        warmup_requests=args.warmup,
        journal_path=str(out.resolve()),
        **({} if args.proposal_top_p is None else {"proposal_top_p": args.proposal_top_p}),
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


TABLE_BENCHMARK = {"aime24": "aime24/25", "aime25": "aime24/25"}  # one column in the paper's tables


def summarize(paths: list[Path]) -> None:
    print(f"{'run':40s} {'benchmark':12s} {'n':>5s} {'tok/s':>9s} {'tok/s (rep.-filtered)':>22s}")
    for path in paths:
        rows = read_jsonl(path)
        groups = collections.defaultdict(list)
        for row in rows:
            groups[TABLE_BENCHMARK.get(row["benchmark"], row["benchmark"])].append(row)
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
                        help="registry names only: w64 shared-KV target (default) or original base model "
                        "(inkling-2-mini: original only)")
    parser.add_argument("--revision", help="revision of the target repository (replaces a base model's pin)")
    parser.add_argument("--drafter", help="drafter export (local path, Hugging Face id, or id@revision); "
                        "overrides a registry name's drafter")
    parser.add_argument("--drafter-kind", choices=registry.DRAFTER_KINDS,
                        help="registry names only: the drafter of the target to run, dflash (the paper's; the default "
                        "where the model has one) or dspark; an export given by --drafter tells its own kind")
    parser.add_argument("--prompts", type=Path, help="SUITE.prompts.jsonl from build_prompts.py")
    parser.add_argument("--out", type=Path, help="results JSONL (appended; finished ids of the same run are skipped)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--proposal-top-p", type=float, metavar="P",
                        help="2DSD sampling: the top-p of the early-loop readouts, which only propose (default 1: "
                             "not cut). Any value keeps the distribution of the outputs; the value of --top-p "
                             "cuts the proposals like the target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="first N prompts of the shuffled order")
    parser.add_argument("--warmup", type=int, default=2, help="64-token warmup requests")
    parser.add_argument("--stage", choices=("stage1", "stage2a", "stage2b"),
                        help="2DSD variant: stage1 drafts at r1 only, stage2a adds deeper promotion at r2, stage2b "
                        "also forks at r2 (default: the paper's setting for the model)")
    parser.add_argument("--block-size", type=int,
                        help="drafted block size k (default 16, Inkling2 8; 1 runs no drafter)")
    parser.add_argument("--d1", type=int, help="early readout loop r1 (default 1)")
    parser.add_argument("--d2", type=int, help="deeper readout loop r2 (default 2 when R = 3 or 4)")
    parser.add_argument("--kv-policy", choices=("w64", "native"),
                        help="Dory: w64 last-loop sharing or native per-loop KV; set by --target for a registry "
                        "name, required for a Dory checkpoint given by path or id. Inkling2 runs native only")
    parser.add_argument("--experts", choices=("dense", "packed"),
                        help="Inkling2 only: keep the experts packed (default on the static runner with "
                        "humming-kernels >= 0.1.20, see --humming) or dequantized on the GPU (49 GiB of weights "
                        "instead of 18 GiB)")
    parser.add_argument("--runner", choices=("batched", "static", "reference"),
                        help="Inkling2 only: the static tree runner (default: the measured one), the eager "
                        "batched one or the reference path, which the static one is checked against")
    parser.add_argument("--tensor-cores", action=argparse.BooleanOptionalAction,
                        help="Inkling2 only: FastInkling2, the served arithmetic with matrix products on tensor "
                        "cores; the static runner then also reads keys and values with its attention kernel "
                        "(default: on)")
    parser.add_argument("--drafter-cuda-graph", action=argparse.BooleanOptionalAction,
                        help="Inkling2 only: the drafter adapter replays its drafting from CUDA graphs "
                        "(default: on)")
    parser.add_argument("--drafter-of-other-target", action="store_true",
                        help="Inkling2 only: run a drafter that was trained against another checkpoint (refused "
                             "by default: checkpoints of one architecture take each other's drafters silently)")
    parser.add_argument("--loops", type=int, metavar="R",
                        help="Inkling2 variable-loop checkpoints only: run R passes instead of the checkpoint's "
                             "own count; the output is then the readout of pass R, another model output")
    parser.add_argument("--drafter-of-other-loops", action="store_true",
                        help="Inkling2 only, with --loops: draft with a drafter that was trained at another "
                             "number of passes of the same checkpoint (refused by default)")
    parser.add_argument("--humming", type=Path, metavar="DIR",
                        help="Inkling2 only: directory of humming-kernels >= 0.1.20 (uv pip install --target DIR "
                        "--no-deps humming-kernels==0.1.20), put first on the import path. On the default "
                        "runtime (static runner, tensor cores, packed experts) the routed experts then run on "
                        "Humming's packed expert GEMM; an error with a runtime that would not use it")
    parser.add_argument("--shard", help="I/N: only prompts I, I + N, ... of the shuffled (and limited) order; "
                        "give every shard its own --out")
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
                                    kv_policy=args.kv_policy, drafter_kind=args.drafter_kind,
                                    with_drafter=args.method == "2dsd" and args.block_size != 1)
    except ValueError as error:
        parser.error(str(error))
    family = resolved.pop("family")
    vars(args).update(resolved)
    if args.method == "2dsd" and args.drafter is None and args.block_size != 1:
        parser.error("2DSD needs --drafter (or --block-size 1)")
    if family is None:  # outside the registry
        from vllm.v1.spec_decode.twodsd import model_type

        try:  # an unknown model_type is an error: no model runs as another family
            family = registry.model_type_family(model_type(args.model, args.revision))
        except ValueError as error:
            parser.error(str(error))
    if family not in registry.UNADAPTED and args.kv_policy is not None:
        parser.error("--kv-policy applies to Dory only")
    if family == "dory" and args.kv_policy is None:
        parser.error("--kv-policy is required for Dory (w64 or native)")
    own_runner = family in NO_AR_ENGINE
    if own_runner:
        if args.kv_policy not in (None, "native"):
            parser.error("Inkling2 runs with per-loop KV only (--kv-policy native)")
        args.kv_policy = "native"
    elif args.experts is not None or args.runner is not None:
        parser.error("--experts and --runner apply to Inkling2 only")
    elif args.tensor_cores is not None or args.drafter_cuda_graph is not None or args.humming:
        parser.error("--tensor-cores, --drafter-cuda-graph and --humming apply to Inkling2 only")
    elif args.drafter_of_other_target:
        parser.error("--drafter-of-other-target applies to Inkling2 only")
    elif args.loops is not None or args.drafter_of_other_loops:
        parser.error("--loops and --drafter-of-other-loops apply to Inkling2 only")
    if args.proposal_top_p is not None and (args.method != "2dsd" or args.temperature <= 0):
        parser.error("--proposal-top-p applies to 2DSD sampling")
    if args.humming:
        sys.path.insert(0, str(args.humming))
    args.engine_kwargs = {}
    if own_runner:
        from vllm.v1.spec_decode.twodsd import inkling2_expert_backend, inkling2_runtime

        chosen = {"experts": args.experts, "runner": args.runner, "tensor_cores": args.tensor_cores,
                  "drafter_cuda_graph": None if args.method == "ar" else args.drafter_cuda_graph}  # ar: no drafter
        args.engine_kwargs.update({name: value for name, value in chosen.items() if value is not None})
        # The runtime the engine takes by its defaults and these flags: it is part of the run identity.
        args.runtime = inkling2_runtime(args.engine_kwargs)
        args.expert_backend = inkling2_expert_backend(args.runtime)
        if args.method == "ar":
            args.runtime.pop("drafter_cuda_graph", None)
        if args.humming and args.expert_backend != "humming":
            parser.error(f"--humming {args.humming}: Humming's expert GEMM would not run (it needs humming-kernels "
                         ">= 0.1.20 in that directory, the static runner, tensor cores and packed experts)")
    if args.drafter_of_other_target and args.method != "ar":
        args.engine_kwargs["drafter_of_other_target"] = True
    if args.loops is not None:
        args.engine_kwargs["loops"] = args.loops
    if args.drafter_of_other_loops and args.method != "ar":
        args.engine_kwargs["drafter_of_other_loops"] = True
    if own_runner and args.method == "ar":  # the runner decodes one token at a time, without a drafter
        args.engine_kwargs["plain"] = True
        args.drafter = None
    args.eos_token_ids = registry.eos_token_ids(family)
    print(f"target {args.model}" + (f"@{args.revision}" if args.revision else "") + f", drafter {args.drafter}, "
          f"kv_policy {args.kv_policy or 'from config.json'}"
          + (f", expert backend {args.expert_backend}" if own_runner else ""), file=sys.stderr)
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
        (run_2dsd if args.method == "2dsd" or own_runner else run_ar)(args, pending, args.out, identity)
    summarize([args.out])


if __name__ == "__main__":
    main()
