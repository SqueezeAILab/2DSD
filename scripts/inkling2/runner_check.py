#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Greedy agreement of the Inkling 2 tree runners with each other and with the official server's rollouts.

For each greedy rollout row (prompt_ids and the official server's output_ids), every runner of --runners decodes
--tokens tokens plainly from the prompt (the 2DSD executor with k = 1, stage1, d1 = R). The script reports where
each runner's output first leaves the official output and where the runners leave each other. At every such
position it also reports the top-1 / top-2 logit margin of the reference forward, teacher-forced on the
official trajectory. A divergence at a near-tie (margin below --tie) is rounding. A divergence at a clear margin
is a defect to look into.

  python scripts/inkling2/runner_check.py --model SNAPSHOT --trajectories greedy-dev.jsonl --limit 8 \\
      --tokens 256 --runners static batched --out runner-check.json

Run with the release environment built from this tree, or with PYTHONPATH=vllm-fork.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

RUNNERS = ("static", "batched", "reference")


def runner_class(name):
    from vllm.model_executor.models.inkling2_2d import Inkling2TreeRunner
    from vllm.model_executor.models.inkling2_batch import Inkling2BatchRunner
    from vllm.model_executor.models.inkling2_graph import Inkling2GraphRunner

    return {"static": Inkling2GraphRunner, "batched": Inkling2BatchRunner, "reference": Inkling2TreeRunner}[name]


def first_divergence(a, b):
    """The first position where two token lists differ, or None when one is a prefix of the other."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def reference_margins(model, prompt, official, torch):
    """Top-1 minus top-2 logit of the reference forward at every output position, teacher-forced on the
    official trajectory (position i predicts official[i])."""
    from vllm.model_executor.models.inkling2_reference import Inkling2Cache

    loops = model.config.num_passes
    tokens = torch.tensor(prompt + official[:-1], dtype=torch.long, device=model.device)
    out = model.forward(tokens, num_passes=loops, depths=(loops,), cache=Inkling2Cache(), pass_end=False)
    hidden = out.readouts[loops].hidden[len(prompt) - 1:]
    top = model.readout_logits(hidden).float().topk(2, dim=-1)
    return {"margin": (top.values[:, 0] - top.values[:, 1]).tolist(), "argmax": top.indices[:, 0].tolist()}


def decode(model, cls, prompt, tokens, eos, torch, options):
    from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor

    loops = model.config.num_passes
    runner = cls(model, {loops}, max_len=len(prompt) + tokens + 16, **options)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = FinalizedTreeExecutor(runner, None, 1, "stage1", loops, 2).generate(prompt, tokens, eos)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return result["output_ids"], time.perf_counter() - started


def check(args):
    import torch
    from vllm.model_executor.models.inkling2_fast import FastInkling2
    from vllm.model_executor.models.inkling2_reference import NativeInkling2

    cls = FastInkling2 if args.tensor_cores else NativeInkling2
    model = cls.from_pretrained(args.model, device=args.device, precision=args.precision, dtype=torch.bfloat16,
                                experts=args.experts)
    rows = [json.loads(line) for line in Path(args.trajectories).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("sampling") is None and row.get("output_ids")][:args.limit]
    eos = [args.eos] if args.eos is not None else []
    records = []
    for row in rows:
        prompt, official = list(row["prompt_ids"]), list(row["output_ids"])[:args.tokens]
        with torch.inference_mode():
            ref = reference_margins(model, prompt, official, torch)
        outputs, seconds = {}, {}
        for name in args.runners:
            options = {"use_cuda_graph": not args.eager} if name == "static" else {}
            outputs[name], seconds[name] = decode(model, runner_class(name), prompt, len(official), eos, torch,
                                                  options)
        pairs = [("official", name) for name in args.runners] + list(itertools.combinations(args.runners, 2))
        found = {}
        left = {name: first_divergence(official, outputs[name]) for name in args.runners}
        for a, b in pairs:
            x = official if a == "official" else outputs[a]
            at = first_divergence(x, outputs[b])
            # The reference margin is read on the official trajectory: it is the pair's own context only up to the
            # point where the pair's runners leave the official output.
            valid = at is not None and all(left[n] is None or at <= left[n] for n in (a, b) if n != "official")
            margin = ref["margin"][at] if valid and at < len(ref["margin"]) else None
            found[f"{a}~{b}"] = {"first_divergence": at, "reference_margin": margin,
                                 "near_tie": None if margin is None else margin < args.tie,
                                 "agree": sum(p == q for p, q in zip(x, outputs[b])), "of": min(len(x), len(outputs[b]))}
        teacher = sum(p == q for p, q in zip(ref["argmax"], official))
        records.append({"id": row.get("id"), "source": row.get("source"), "tokens": len(official),
                        "reference_teacher_forced_agree": teacher, "pairs": found,
                        "seconds": seconds, "tokens_per_second": {k: len(official) / v for k, v in seconds.items()}})
        print(json.dumps(records[-1]), flush=True)
    return {"records": records, "settings": {"runners": args.runners, "tokens": args.tokens, "tie": args.tie,
                                             "experts": args.experts, "precision": args.precision,
                                             "tensor_cores": args.tensor_cores,
                                             "graphs": not args.eager, "torch": torch.__version__}}


def summarize(result):
    """Per pair: sequences identical, divergences at a near-tie and at a clear margin."""
    lines, pairs = [], {}
    for record in result["records"]:
        for pair, value in record["pairs"].items():
            pairs.setdefault(pair, []).append(value)
    for pair, values in pairs.items():
        same = sum(v["first_divergence"] is None for v in values)
        tie = sum(v["near_tie"] is True for v in values)
        clear = [v for v in values if v["near_tie"] is False]
        later = sum(v["first_divergence"] is not None and v["near_tie"] is None for v in values)
        lines.append(f"{pair:22s} identical {same}/{len(values)}, near-tie divergences {tie}, clear-margin "
                     f"divergences {len(clear)}" + (f" (margins {[round(v['reference_margin'], 3) for v in clear]})"
                                                    if clear else "")
                     + (f", {later} after leaving the official output (margin not read)" if later else ""))
    speed = {}
    for record in result["records"]:
        for name, value in record["tokens_per_second"].items():
            speed.setdefault(name, []).append(value)
    lines.append("plain decoding tok/s (whole run, first call includes graph capture): "
                 + ", ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in speed.items()))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="local snapshot directory of the checkpoint")
    parser.add_argument("--trajectories", type=Path, required=True, help="greedy rollout rows (sampling null)")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256, help="output tokens compared per row")
    parser.add_argument("--runners", nargs="+", choices=RUNNERS, default=["static", "batched"])
    parser.add_argument("--tie", type=float, default=0.25, help="reference margin below which a divergence is a tie")
    parser.add_argument("--eos", type=int, default=200006)
    parser.add_argument("--experts", choices=("dense", "packed"), default="dense")
    parser.add_argument("--precision", default="served")
    parser.add_argument("--tensor-cores", action="store_true",
                        help="FastInkling2: the served arithmetic with its matrix products on tensor cores")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eager", action="store_true", help="static runner without CUDA graphs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = check(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(summarize(result))


if __name__ == "__main__":
    main()
