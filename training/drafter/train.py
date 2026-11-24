"""Colocated depth-conditioned drafter training on a frozen looped target (torchrun, DDP).

Every rank runs the target on its own packs of whole traces and computes loop-1..R features online:
an original target with one teacher-forced forward and layer hooks; a shared-KV (W64) target by
exact chunk replay (chunk = W). Dory shares without adaptation, so its drafter always trains on the
original per-loop Dory (--target original) and serves both of its targets. Several drafters can train
from the same features, so the dominant target cost is paid once and all drafters see identical data
in identical order.

  torchrun --standalone --nproc-per-node 8 -m training.drafter.train \\
      --config configs/drafter/ouro-1.4b-thinking-w64.json --target-path W64_CHECKPOINT \\
      --traces TRACES --out RUN [--resume]

  --drafter NAME:LAYERS:D_WEIGHTS[:RECIPE]   e.g. drafter:5:1=0.5,2=0.125,3=0.125,4=0.25
  RECIPE is ce-dpace (default: DFlash, the paper's) or dspark (DSpark with speculators' heads, loss and
  optimizers; training.drafter.model). Both read the same features, blocks and anchor depths.
Flags override the --config file. --resume continues after the last completed epoch saved in --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ..common import registry
from ..common.cli import parse_with_config
from ..common.io import file_sha
from .blocks import INPUT_FORMAT, Blocks, block_labels, sample_blocks
from .data import load_traces, pack, packed_stream
from .model import (DORY_MASK_TOKEN, DSPARK, LOSS, MARKOV_HEADS, MASK_TOKEN, RECIPES, DepthDFlash,
                    acceptance_counts, drafter_config, drafter_forward)
from .targets import ReplayTarget, native_target, target_feature_layers


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="JSON defaults (configs/drafter/*.json); flags override it")
    ap.add_argument("--model", choices=registry.names(), help="registry name of the looped model")
    ap.add_argument("--target", choices=registry.VARIANTS, default="original",
                    help="original: per-loop-KV target, features from a teacher-forced forward; "
                         "w64: shared-KV target, features and labels from exact chunk replay")
    ap.add_argument("--target-path", help="local directory or HF repo id of the target checkpoint "
                                          "(default: the repository in configs/models.json)")
    ap.add_argument("--window", type=int, default=64, help="shared read window of a w64 target")
    ap.add_argument("--replay-batch", type=int, default=16, help="documents per exact replay batch")
    ap.add_argument("--replay-batch-tokens", type=int, default=8 * 16384,
                    help="padded tokens per exact replay batch (its history KV buffers scale with this)")
    ap.add_argument("--replay-prompt-tokens", type=int, default=16384,
                    help="padded prompt tokens per replay batch (the prompt pass keeps every loop's KV)")
    ap.add_argument("--replay-attention", choices=("flash", "sdpa"), default="flash",
                    help="exact replay attention: FlashAttention + CUDA graphs, or the masked SDPA reference "
                         "(Dory: the attention of its teacher-forced forward)")
    ap.add_argument("--replay-group-tokens", type=int, default=131072,
                    help="replay consecutive packs together until their documents hold this many tokens")
    ap.add_argument("--target-experts", choices=("dense", "packed"), default="dense",
                    help="Inkling 2: hold every expert dequantized on the GPU (49 GiB of weights) or packed")
    ap.add_argument("--target-tensor-cores", action="store_true",
                    help="Inkling 2: run the target as FastInkling2 (tensor-core products, grouped expert GEMM, "
                         "attention kernel); needs a GPU and dense experts for the full speed-up")
    ap.add_argument("--mask-token", type=int, help="embedding row that initialises the mask (default: Dory "
                                                   f"{DORY_MASK_TOKEN}, Inkling 2 its first unused row, otherwise "
                                                   f"{MASK_TOKEN})")
    ap.add_argument("--traces", type=Path, help="directory with train.jsonl and dev.jsonl")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--drafter", nargs="+", help="NAME:LAYERS:D_WEIGHTS[:ce-dpace|dspark] per drafter")
    ap.add_argument("--markov-head", choices=MARKOV_HEADS, default=DSPARK["markov_head"],
                    help="dspark: speculators' sequential head (vanilla: first-order Markov bias)")
    ap.add_argument("--markov-rank", type=int, default=DSPARK["markov_rank"], help="dspark: rank of the head")
    ap.add_argument("--sample-from-anchor", action="store_true",
                    help="dspark: the given token's slot drafts the first token (the DSpark paper's block: "
                         "k_max - 1 slots). Default: the given token and k_max - 1 masks (k_max slots)")
    ap.add_argument("--feat-layers", help="comma-separated 0-based layers inside one loop (default: DFlash rule; "
                                          "Dory: its physical sublayer taps 3,19,35,51,67)")
    ap.add_argument("--k-max", type=int, default=16,
                    help="block length: the given token and the k_max - 1 tokens drafted after it")
    ap.add_argument("--blocks", type=int, default=1024, help="blocks per packed sequence and optimizer step")
    ap.add_argument("--micro-blocks", type=int, default=256, help="blocks per forward (gradient accumulation)")
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--reuse", type=int, default=2, help="optimizer steps per target forward (fresh blocks each)")
    ap.add_argument("--lr", type=float, default=6e-4, help="AdamW learning rate")
    ap.add_argument("--muon-lr", type=float, default=DSPARK["muon_lr"],
                    help="dspark: learning rate of the 2-D weights, which Muon updates")
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--wd", type=float, default=0.01,
                    help="ce-dpace: AdamW weight decay (dspark uses speculators' recipe: AdamW "
                         f"{DSPARK['weight_decay']}, Muon {DSPARK['muon_weight_decay']})")
    ap.add_argument("--train-limit", type=int, default=0, help="smoke runs: domain-stratified train subset size")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke runs: stop early")
    ap.add_argument("--save-every", type=int, default=0, metavar="STEPS",
                    help="also write last.pt every this many steps, so that a run that dies inside an epoch "
                         "leaves weights to export (the run still resumes from the last finished epoch)")
    ap.add_argument("--eval-docs", type=int, default=240)
    ap.add_argument("--eval-stride", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue after the last completed epoch saved in --out")
    ap.add_argument("--allow-other-target", action="store_true",
                    help="a deliberate transfer: train on traces that another model generated, or (Dory, Inkling 2) "
                         "against a --target-path that is not the --model's pinned checkpoint. Refused by default")
    ap.add_argument("--init", type=Path, metavar="RESUME_PT",
                    help="start from the drafter weights of another run's resume.pt, with a fresh optimizer and "
                         "schedule (fine-tuning on another target's traces); ignored once --out has a resume.pt "
                         "and --resume is given")
    ap.add_argument("--eval-only", type=Path, metavar="RESUME_PT",
                    help="load the drafters of a resume.pt, run the dev evaluation (eval_only.json in --out) and "
                         "exit: the same drafter under another target arithmetic or feature source")
    return ap


def parse_drafter(spec):
    """(name, layers, anchor depth weights, recipe) of a --drafter value."""
    name, layers, weights, *rest = spec.split(":")
    recipe = rest[0] if rest else LOSS
    if recipe not in RECIPES or len(rest) > 1:
        raise ValueError(f"{name}: the recipe is one of {', '.join(RECIPES)}")
    weights = {int(k): float(v) for k, v in (kv.split("=") for kv in weights.split(","))}
    return name, int(layers), weights, recipe


def build_optimizers(module, recipe, args):
    """ce-dpace: AdamW over every parameter. dspark: speculators' split, Muon over the 2-D weights and AdamW
    over the rest (norms, mask, Markov factors, confidence head)."""
    if recipe != "dspark":
        return [torch.optim.AdamW(module.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd,
                                  fused=True)]
    from speculators.train.optimizers import split_named_params_for_muon

    muon, adamw = split_named_params_for_muon(module)
    return [torch.optim.Muon(muon, lr=args.muon_lr, momentum=DSPARK["muon_momentum"],
                             weight_decay=DSPARK["muon_weight_decay"], ns_steps=5,
                             adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(adamw, lr=args.lr, weight_decay=DSPARK["weight_decay"], fused=True)]


def trace_expectation(model, family, target_dir, target_path=None):
    """What a trace row must say to train the drafter of registry model `model` (data.load_traces).

    Every family: the row's `model`. Dory and Inkling 2 have one pinned checkpoint per name (no KV-sharing
    adaptation), so their traces also carry that checkpoint's config hash, and a `target_path` must be a
    copy of it: another checkpoint of the same architecture gets its own registry name. Ouro and Nanbeige
    train both of their drafters on the traces of the shared-KV checkpoint, so their hash is not compared."""
    expect = {"model": model}
    if family in registry.UNADAPTED:
        sha = file_sha(Path(target_dir) / "config.json")
        if target_path:
            pinned = registry.resolve(model, allow_patterns=["config.json"])
            if file_sha(Path(pinned) / "config.json") != sha:
                raise ValueError(f"--target-path {target_path} is not the checkpoint of {model} "
                                 f"({registry.label(model)}). Register it under its own name in configs/models.json, "
                                 "or pass --allow-other-target")
        expect["target_config_sha256"] = sha
    return expect


def resume_entry(d):
    """One drafter's entry of resume.pt."""
    return {"model": d["module"].state_dict(), "opts": [o.state_dict() for o in d["opts"]],
            "scheds": [x.state_dict() for x in d["scheds"]],
            "optimizers": [type(o).__name__ for o in d["opts"]], "best": d["best"]}


def load_resume_entry(d, saved):
    """Restore one drafter (module, optimizers, schedules, best score) from its resume.pt entry.

    An entry of the release format (one AdamW, saved as ``opt`` and ``sched``) reads as a list of one. An
    entry of another recipe, with another number or kind of optimizers, is refused before anything is loaded.
    """
    release = "opts" not in saved
    opts, scheds = ([saved["opt"]], [saved["sched"]]) if release else (saved["opts"], saved["scheds"])
    kinds = ["AdamW"] if release else saved.get("optimizers")  # absent in files written before it was saved
    current = [type(o).__name__ for o in d["opts"]]
    if len(opts) != len(current) or len(scheds) != len(current) or (kinds is not None and list(kinds) != current):
        raise ValueError(f"resume.pt holds the optimizers {kinds or len(opts)} and this recipe has {current}: "
                         "it was written by another recipe (start from its weights with --init)")
    d["module"].load_state_dict(saved["model"])
    for target, state in zip((*d["opts"], *d["scheds"]), (*opts, *scheds)):
        target.load_state_dict(state)
    d["best"] = saved["best"]


def split(b, size):
    for s in range(0, b.a.numel(), size):
        yield Blocks(*(getattr(b, f)[s:s + size] for f in ("a", "d", "lif", "k", "doc", "valid")))


def main(argv=None):
    parser = build_parser()
    args = parse_with_config(parser, argv)
    for flag in ("model", "traces", "out", "drafter"):
        if not getattr(args, flag):
            parser.error(f"--{flag} is required (on the command line or in --config)")
    family = registry.family(args.model)
    dory = family == "dory"
    if family in registry.UNADAPTED and args.target != "original":
        parser.error("Dory and Inkling 2 have no adapted checkpoint: their drafter trains on the original "
                     "per-loop model (--target original); Dory's serves both of its targets")
    if args.mask_token is None and family != "inkling2":  # Inkling 2: set once its target is loaded
        args.mask_token = DORY_MASK_TOKEN if dory else MASK_TOKEN

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    traces, out = args.traces, args.out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                   for k, v in vars(args).items()}, indent=2))

    def log(msg):
        if rank == 0:
            print(msg, flush=True)
            with open(out / "log.txt", "a") as f:
                f.write(msg + "\n")

    # Rank 0 downloads a registry checkpoint once; the other ranks then find it in the cache.
    if rank == 0:
        target_dir = registry.resolve_target(args.model, args.target, args.target_path)
    if distributed:
        dist.barrier()
    if rank != 0:
        target_dir = registry.resolve_target(args.model, args.target, args.target_path)
    target_cfg = json.loads((target_dir / "config.json").read_text())
    expect = None if args.allow_other_target else trace_expectation(args.model, family, target_dir, args.target_path)
    feat_layers = ([int(x) for x in args.feat_layers.split(",")] if args.feat_layers
                   else target_feature_layers(target_cfg))
    target = (ReplayTarget(args.model, target_dir, feat_layers, dev, args.window, args.replay_batch,
                           args.replay_attention, args.replay_batch_tokens, args.replay_prompt_tokens)
              if args.target == "w64" else native_target(args.model, feat_layers, dev, path=target_dir,
                                                         attention=args.replay_attention,
                                                         experts=args.target_experts,
                                                         tensor_cores=args.target_tensor_cores))
    R = target.R
    if args.mask_token is None:  # the first embedding row past the served vocabulary
        args.mask_token = target.lm_head.weight.shape[0]
    drafters = []
    for spec in args.drafter:
        name, layers, d_weights, recipe = parse_drafter(spec)
        if set(d_weights) - set(range(1, R + 1)):
            raise ValueError(f"{name}: anchor depths must lie in 1..{R}")
        cfg = drafter_config(layers, target.model.config)
        heads = ({"vocab": target.lm_head.weight.shape[0], "rank": args.markov_rank, "type": args.markov_head,
                  "sample_from_anchor": args.sample_from_anchor} if recipe == "dspark" else None)
        module = DepthDFlash(cfg, len(feat_layers), heads=heads).to(dev)
        with torch.no_grad():
            module.mask_emb.copy_(target.embed.weight[args.mask_token].float())
        wrapped = (torch.nn.parallel.DistributedDataParallel(module, device_ids=[local], broadcast_buffers=False)
                   if distributed else module)
        drafters.append({"name": name, "layers": layers, "d_weights": d_weights, "recipe": recipe, "cfg": cfg,
                         "heads": heads, "module": module, "model": wrapped,
                         "rng": random.Random(args.seed * 1000 + rank * 17 + len(drafters)),
                         "params": sum(p.numel() for p in module.parameters()), "best": -1.0})

    train_docs, tstats = load_traces(traces / "train.jsonl", args.seq_len, args.train_limit, args.seed, expect)
    dev_docs, dstats = load_traces(traces / "dev.jsonl", args.seq_len, expect=expect)
    dev_docs = sorted(dev_docs, key=lambda d: d["id"])[:: max(1, len(dev_docs) // args.eval_docs)][:args.eval_docs]
    packs_per_epoch = len(pack(train_docs, args.seq_len, random.Random(0))) // world
    total_steps = packs_per_epoch * args.epochs * args.reuse
    epochs = math.ceil(total_steps / (packs_per_epoch * args.reuse))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    for d in drafters:
        d["opts"] = build_optimizers(d["module"], d["recipe"], args)
        warm = max(1, int(args.warmup * total_steps))
        d["scheds"] = [torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s, warm=warm: (s + 1) / warm if s < warm else
            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))) for opt in d["opts"]]
    log(json.dumps({"model": args.model, "target": args.target, "feat_layers": feat_layers, "world": world,
                    "train": tstats, "dev_eval_docs": len(dev_docs),
                    "packs_per_rank_epoch": packs_per_epoch, "total_steps": total_steps, "epochs": epochs,
                    "drafters": {d["name"]: {"layers": d["layers"], "d_weights": d["d_weights"],
                                             "recipe": d["recipe"], "heads": d["heads"],
                                             "optimizers": [type(o).__name__ for o in d["opts"]],
                                             "params_M": round(d["params"] / 1e6, 1)}
                                 for d in drafters}}))

    @torch.no_grad()
    def quick_eval(tag):
        """Deterministic anchors on a dev subset: d = 1..R with L_if = (R - d) * k_max / 2, block k_max."""
        K = args.k_max
        settings = [(d, (R - d) * (K // 2)) for d in range(1, R + 1)]
        # attempted and accepted per slot; [native head matches, blocks]; dspark: [sum of 1 - TV, sum of the
        # confidence head's absolute error, slots]; [sum of the training loss over its slots, slots]
        acc = {(d["name"], s): [torch.zeros(K - 1, device=dev), torch.zeros(K - 1, device=dev),
                                torch.zeros(2, device=dev), torch.zeros(3, device=dev), torch.zeros(2, device=dev)]
               for d in drafters for s in settings}
        for d in drafters:
            d["module"].eval()
        for members, p in packed_stream(target, dev_docs, pack(dev_docs, args.seq_len, None)[rank::world],
                                        args.seq_len, dev, args.replay_group_tokens):
            for depth, lif_nom in settings:
                rows = []
                for di, (_start, rs, end, _dom) in enumerate(p.docs):
                    for a in range(rs - 1 if depth == R else rs, end - 2, args.eval_stride):
                        rows.append((a, depth, max(0, min(lif_nom, a - rs)) if depth < R else 0, K, di))
                for c in range(0, len(rows), args.micro_blocks):
                    chunk = rows[c:c + args.micro_blocks]
                    pad = args.micro_blocks - len(chunk)
                    t = lambda j, fill: torch.tensor([r[j] for r in chunk] + [fill] * pad, device=dev)
                    b = Blocks(t(0, 0), t(1, R), t(2, 0), t(3, 0), t(4, -2),
                               torch.tensor([True] * len(chunk) + [False] * pad, device=dev))
                    for d in drafters:
                        loss, m = drafter_forward(
                            d["module"], target, p, b, R, K, recipe=d["recipe"],
                            anchor_predicts=bool(d["heads"] and d["heads"]["sample_from_anchor"]))
                        att, ok = acceptance_counts(m["correct"], m["valid"])
                        e = acc[(d["name"], (depth, lif_nom))]
                        e[0] += att
                        e[1] += ok
                        e[2] += torch.stack([m["native_ok"][b.valid].sum(), b.valid.sum()]).float()
                        if "overlap" in m:
                            e[3] += torch.stack([m["overlap"], m["confidence_error"], m["n_valid"]]).float()
                        e[4] += torch.stack([loss.float() * m["n_valid"], m["n_valid"]])
        if distributed:
            for v in acc.values():
                for x in v:
                    dist.all_reduce(x)
        results = {}
        for d in drafters:
            rows = {}
            for depth, lif in settings:
                att, ok, nat, dist_stats, loss_stats = acc[(d["name"], (depth, lif))]
                a_i = (ok / att.clamp(min=1)).tolist()
                A, tau = 1.0, 1.0
                for x in a_i:
                    A *= x
                    tau += A
                rows[f"d{depth}"] = {"L_if": lif, "a_i": a_i, "ahat": (ok.sum() / att.sum().clamp(min=1)).item(),
                                     f"tau_k{K}": tau, "alpha_native_head": (nat[0] / nat[1].clamp(min=1)).item(),
                                     "val_loss": (loss_stats[0] / loss_stats[1].clamp(min=1)).item()}
                if d["recipe"] == "dspark":  # over every slot, teacher-forced: 1 - TV and the confidence error
                    slots = dist_stats[2].clamp(min=1)
                    rows[f"d{depth}"].update(overlap=(dist_stats[0] / slots).item(),
                                             confidence_abs_error=(dist_stats[1] / slots).item())
            results[d["name"]] = rows
            d["module"].train()
            key = [f"d{x}" for x in sorted(d["d_weights"])]
            d["score"] = (sum(rows[k]["ahat"] for k in key if k in ("d1", f"d{R}"))
                          / max(1, sum(k in ("d1", f"d{R}") for k in key)))
            if rank == 0:
                log(f"[eval {tag}] {d['name']}: " + " | ".join(
                    f"{k} a1..{min(4, K - 1)}=" + "/".join(f"{x:.3f}" for x in v["a_i"][:4]) + " "
                    f"ahat={v['ahat']:.3f} tau{K}={v[f'tau_k{K}']:.2f} head={v['alpha_native_head']:.3f} "
                    f"val_loss={v['val_loss']:.4f}"
                    + (f" overlap={v['overlap']:.3f} conf_err={v['confidence_abs_error']:.3f}"
                       if "overlap" in v else "")
                    for k, v in rows.items()))
        if rank == 0:
            (out / f"eval_{tag}.json").write_text(json.dumps(results, indent=2))
        return results

    def save(d, tag, step, epoch):
        if rank != 0:
            return
        path = out / d["name"]
        path.mkdir(exist_ok=True)
        torch.save({"model": d["module"].state_dict(), "cfg": d["cfg"].to_dict(), "feat_layers": feat_layers,
                    "d_weights": d["d_weights"], "recipe": d["recipe"],
                    "heads": d["heads"] and {key: d["heads"][key] for key in ("rank", "type", "sample_from_anchor")},
                    "k_max": args.k_max, "learn_mask": True, "mask_token": args.mask_token,
                    "input_format": INPUT_FORMAT, "target": args.model, "target_variant": args.target,
                    "target_spec": args.target_path, "window": args.window if args.target == "w64" else None,
                    "target_config_sha256": file_sha(Path(target_dir) / "config.json"),  # export compares it
                    "target_loops": R,
                    "step": step, "epoch": epoch,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                   path / f"{tag}.pt")

    def save_resume(epoch, step):
        """Everything needed to continue at the next epoch; packing and block sampling are seeded per epoch."""
        if rank != 0:
            return
        torch.save({"epoch": epoch, "step": step, "total_steps": total_steps,
                    "drafters": {d["name"]: resume_entry(d) for d in drafters}},
                   out / "resume.tmp")
        (out / "resume.tmp").replace(out / "resume.pt")

    if args.eval_only:
        state = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        for d in drafters:
            d["module"].load_state_dict(state["drafters"][d["name"]]["model"])
        quick_eval("only")
        if distributed:
            dist.destroy_process_group()
        return

    step, done, start_epoch = 0, False, 0
    if args.init and not (args.resume and (out / "resume.pt").exists()):
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        for d in drafters:
            d["module"].load_state_dict(state["drafters"][d["name"]]["model"])
        log(f"initialised the drafters from {args.init} (after its epoch {state['epoch']})")
        del state
    if args.resume and (out / "resume.pt").exists():
        state = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if state["total_steps"] != total_steps:
            raise ValueError("resume.pt was written for a different update budget")
        for d in drafters:
            load_resume_entry(d, state["drafters"][d["name"]])
        step, start_epoch = state["step"], state["epoch"] + 1
        log(f"resumed after epoch {state['epoch']} at step {step}")
    stats = {d["name"]: {"loss": 0.0, "n": 0, "ms": 0.0, "slot1": {k: [0, 0] for k in range(1, R + 1)}}
             for d in drafters}
    t_target, n_packs = 0.0, 0
    for epoch in range(start_epoch, epochs):
        for i, d in enumerate(drafters):
            d["rng"] = random.Random(f"{args.seed}:{rank}:{i}:{epoch}")
        packs = pack(train_docs, args.seq_len, random.Random(epoch))
        packs = packs[: (len(packs) // world) * world][rank::world]
        stream = packed_stream(target, train_docs, packs, args.seq_len, dev, args.replay_group_tokens)
        while True:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            members, p = next(stream, (None, None))
            if p is None:
                break
            torch.cuda.synchronize()
            t_target += time.perf_counter() - t0
            n_packs += 1
            for _ in range(args.reuse):
                for d in drafters:
                    t1 = time.perf_counter()
                    b = sample_blocks(p, args.blocks, R, args.k_max, d["d_weights"], d["rng"])
                    n_total = int(block_labels(p, b, args.k_max, chain=d["recipe"] != "dspark")[1].sum())
                    for opt in d["opts"]:
                        opt.zero_grad(set_to_none=True)
                    chunks = list(split(b, args.micro_blocks))
                    loss_sum = 0.0
                    for i, chunk in enumerate(chunks):
                        sync = (not distributed) or i == len(chunks) - 1
                        ctx = d["model"].no_sync() if not sync else torch.enable_grad()
                        with ctx:
                            loss, m = drafter_forward(
                                d["model"], target, p, chunk, R, args.k_max, n_total=n_total, recipe=d["recipe"],
                                anchor_predicts=bool(d["heads"] and d["heads"]["sample_from_anchor"]))
                            loss.backward()
                        loss_sum += loss.item()
                        s = stats[d["name"]]
                        for depth in range(1, R + 1):
                            sel = (chunk.d == depth) & m["valid"][:, 0]
                            s["slot1"][depth][0] += int(m["correct"][sel, 0].sum())
                            s["slot1"][depth][1] += int(sel.sum())
                    torch.nn.utils.clip_grad_norm_(d["module"].parameters(), 1.0)
                    for opt in d["opts"]:
                        opt.step()
                    for sched in d["scheds"]:
                        sched.step()
                    torch.cuda.synchronize()
                    s = stats[d["name"]]
                    s["loss"] += loss_sum
                    s["n"] += 1
                    s["ms"] += (time.perf_counter() - t1) * 1e3
                step += 1
                if args.save_every and step % args.save_every == 0:
                    for d in drafters:
                        save(d, "last", step, epoch)
                if step % args.log_every == 0:
                    parts = []
                    for d in drafters:
                        s = stats[d["name"]]
                        acc1 = " ".join(f"d{k}={c / max(1, t):.3f}" for k, (c, t) in s["slot1"].items() if t)
                        parts.append(f"{d['name']} loss {s['loss'] / max(1, s['n']):.4f} slot1 {acc1} "
                                     f"{s['ms'] / max(1, s['n']):.0f}ms")
                        stats[d["name"]] = {"loss": 0.0, "n": 0, "ms": 0.0,
                                            "slot1": {k: [0, 0] for k in range(1, R + 1)}}
                    log(f"ep {epoch} step {step}/{total_steps} lr {drafters[0]['scheds'][-1].get_last_lr()[0]:.2e} | "
                        f"target {t_target / max(1, n_packs) * 1e3:.0f}ms/pack | "
                        f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G | " + " || ".join(parts))
                if step >= total_steps:
                    done = True
                    break
            if done:
                break
        # The evaluation runs the target on its own packs: release the last training pack and the gradients.
        p = members = stream = None
        for d in drafters:
            for opt in d["opts"]:
                opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        quick_eval(f"ep{epoch}")
        for d in drafters:
            save(d, "last", step, epoch)
            if d["score"] > d["best"]:
                d["best"] = d["score"]
                save(d, "best", step, epoch)
        save_resume(epoch, step)
        if done:
            break
    log("done")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
