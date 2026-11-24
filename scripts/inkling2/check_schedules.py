#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Does every 2DSD schedule name the tokens of plain decoding on a real Inkling 2 checkpoint?

The static runner decodes a position with the same arithmetic whether the positions before it are
committed (plain decoding) or still in the tree of a drafted block, so a greedy schedule must give the
output of the same runner at k = 1, token for token. This script checks that on a GPU with the real
weights, at prompt lengths you choose. The defaults sit on both sides of 8,192 tokens, a boundary of the
runner's prefix buckets (the place where arithmetic that depends on the committed length would show);
the generation then crosses that bucket in every schedule.

  python scripts/inkling2/check_schedules.py --model inkling-2-mini-varloop --drafter EXPORT \
      [--lengths 8191 8192 8193] [--new 18] [--humming DIR] [--out DIR] \
      [--loops R] [--schedules r1-2 stage2b-1-3 ...]

The prompt is generated text cut to the exact number of tokens, the same for every checkpoint with the
released tokenizer. Each (length, schedule) runs in its own process: one engine per process. Exit status
1 if a schedule differs from plain decoding. Run it on the code and drafter of an evaluation before
the evaluation, and at lengths near its budget (Hard: 32,768).

A schedule is r1-D (Stage 1 that drafts at depth D; r1-R is block drafting at the final pass) or
stage2a-D1-D2 / stage2b-D1-D2 (stage2a and stage2b alone: depths 1 and 2). --schedules all checks every
schedule of the pass count: Stage 1 at every depth and Stages 2a and 2b at every pair of depths. --loops R
runs a variable-loop checkpoint with R passes; plain decoding is then the same runner at R passes, the
default schedules are Stage 1 at every depth and Stage 2b at every pair of depths, and a drafter trained
at another R drafts (a deliberate transfer: the check is that the schedule still names plain decoding's
tokens).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ("r1-3", "r1-1", "stage2b")  # at the checkpoint's own three passes


def schedule(name, loops):
    """Engine options of a schedule name at R = loops passes; None for plain decoding ("ar")."""
    if name == "ar":
        return None
    kind, *depths = name.split("-")
    try:
        depths = [int(depth) for depth in depths]
    except ValueError:
        depths = None
    if kind == "r1" and depths and len(depths) == 1 and 1 <= depths[0] <= loops:
        return dict(stage="stage1", d1=depths[0])
    if kind in ("stage2a", "stage2b") and depths is not None and len(depths) in (0, 2):
        d1, d2 = depths or (1, 2)
        if 1 <= d1 < d2 < loops:
            return dict(stage=kind, d1=d1, d2=d2)
    raise ValueError(f"{name!r} is not a schedule of R = {loops}: r1-D with D in 1..R, or stage2a-D1-D2 / "
                     "stage2b-D1-D2 with D1 < D2 < R")


def every_schedule(loops, stages=("stage2b",)):
    """Stage 1 at every depth (the final one first) and each of `stages` at every pair of depths."""
    return ([f"r1-{depth}" for depth in range(loops, 0, -1)]
            + [f"{stage}-{d1}-{d2}" for stage in stages for d1 in range(1, loops) for d2 in range(d1 + 1, loops)])


def prompt_ids(tokenizer_json, total):
    """Generated text of exactly `total` tokens: log lines, a question and an instruction."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    lead = encode("The following notes describe a new lighthouse archive.\n")
    end = encode("\nMira asks which crate has the missing blue lens. Answer in one sentence.\n")
    audit = encode("\nAudit 16499. Predict which numbered bay the keeper will choose and why.\n")
    index = 0
    while len(lead) + len(end) + len(audit) < total:
        lead += encode(f"Log {16493 + index}. In bay {(7 * index) % 29}, box {(23 * index) % 41} "
                       f"holds the green chart and lens {(19 * index) % 61}.\n")
        index += 1
    ids = lead[:total - len(end) - len(audit)] + end + audit
    if len(ids) != total:
        raise ValueError(f"cannot build a prompt of {total} tokens")
    return ids


def one(args, length, name):
    """One schedule at one prompt length in this process; writes its tokens."""
    if args.humming:
        sys.path.insert(0, str(args.humming))
    from vllm.v1.spec_decode.twodsd import TwoDSD

    entry = json.loads((ROOT / "configs" / "models.json").read_text())["models"][args.model]
    chosen = schedule(name, args.loops or trained_loops(args.drafter))
    options = dict(plain=True) if chosen is None else dict(chosen, block_size=args.block_size,
                                                           drafter_cuda_graph=True)
    if args.loops:
        options["loops"] = args.loops
        if chosen is not None and trained_loops(args.drafter) != args.loops:
            options["drafter_of_other_loops"] = True
    engine = TwoDSD(entry["repo_id"], None if chosen is None else str(args.drafter),
                    max_model_len=length + args.new + 1, revision=entry["revision"], seed=0,
                    runner="static", experts="packed", tensor_cores=True, **options)
    ids = prompt_ids(Path(engine.model) / "tokenizer.json", length)
    [result] = engine.generate([ids], max_new_tokens=[args.new], temperature=0.0, top_p=1.0, eos_token_ids=[],
                               warmup_requests=0)
    args.out.mkdir(parents=True, exist_ok=True)
    output_file(args, length, name).write_text(json.dumps({"output_ids": list(result["output_ids"])}))


def trained_loops(drafter):
    """The number of passes the drafter export was trained at."""
    return json.loads((Path(drafter) / "config.json").read_text())["loopedtf_drafter"]["target_loops"]


def output_file(args, length, name):
    return args.out / (f"{length}-{name}.json" if not args.loops else f"{length}-R{args.loops}-{name}.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="an Inkling 2 name of configs/models.json")
    parser.add_argument("--drafter", required=True, type=Path, help="the drafter export of that model")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8191, 8192, 8193], help="prompt tokens")
    parser.add_argument("--new", type=int, default=18, help="tokens to generate")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--schedules", nargs="+", help=f"default: {' '.join(DEFAULT)}; with --loops, Stage 1 at "
                        "every depth and Stage 2b at every pair of depths; all: Stage 1 at every depth and "
                        "Stages 2a and 2b at every pair")
    parser.add_argument("--loops", type=int, metavar="R", help="variable-loop checkpoints: run R passes "
                        "(default: the checkpoint's own count)")
    parser.add_argument("--humming", type=Path, help="directory of the humming-kernels release to import")
    parser.add_argument("--out", type=Path, default=Path("runs/check-schedules"))
    parser.add_argument("--one", nargs=2, metavar=("LENGTH", "SCHEDULE"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one:
        return one(args, int(args.one[0]), args.one[1])
    loops = args.loops or trained_loops(args.drafter)
    if args.schedules == ["all"]:
        names = every_schedule(loops, ("stage2a", "stage2b"))
    else:
        names = args.schedules or (every_schedule(loops) if args.loops else list(DEFAULT))
    for name in names:
        if name == "ar" or schedule(name, loops) is None:
            parser.error("--schedules lists drafted schedules; plain decoding always runs")
    failed = False
    for length in args.lengths:
        outputs = {}
        for name in ("ar", *names):
            subprocess.run([sys.executable, __file__, *(argv or sys.argv[1:]), "--one", str(length), name], check=True)
            outputs[name] = json.loads(output_file(args, length, name).read_text())["output_ids"]
        for name in names:
            reference, tokens = outputs["ar"], outputs[name]
            at = next((i for i, (a, b) in enumerate(zip(reference, tokens)) if a != b), None)
            same = at is None and len(reference) == len(tokens)
            failed |= not same
            print(f"prompt {length:6d}  {name:12s} " + ("= plain decoding" if same else
                  f"DIFFERS at output {at}: plain decoding {reference[at]}, {name} {tokens[at]}"), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
