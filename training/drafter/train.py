"""Colocated depth-conditioned DFlash drafter training on a frozen looped target (torchrun, DDP).

Every rank runs the target on its own packs of whole traces and computes loop-1..R features online:
an original target with one teacher-forced forward and layer hooks; a shared-KV (W64) target by
exact chunk replay (chunk = W). Dory shares without adaptation, so its drafter always trains on the
original per-loop Dory (--target original) and serves both of its targets. Several drafters can train
from the same features, so the dominant target cost is paid once and all drafters see identical data
in identical order.

  torchrun --standalone --nproc-per-node 8 -m training.drafter.train \\
      --config configs/drafter/ouro-1.4b-thinking-w64.json --target-path W64_CHECKPOINT \\
      --traces TRACES --out RUN [--resume]

  --drafter NAME:LAYERS:D_WEIGHTS[:ce-dpace]   e.g. drafter:5:1=0.5,2=0.125,3=0.125,4=0.25
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
from .blocks import INPUT_FORMAT, Blocks, block_labels, sample_blocks
from .data import load_traces, pack, packed_stream
from .model import DORY_MASK_TOKEN, LOSS, MASK_TOKEN, DepthDFlash, acceptance_counts, drafter_config, drafter_forward
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
    ap.add_argument("--mask-token", type=int, help="embedding row that initialises the mask "
                                                   f"(default: Dory {DORY_MASK_TOKEN}, otherwise {MASK_TOKEN})")
    ap.add_argument("--traces", type=Path, help="directory with train.jsonl and dev.jsonl")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--drafter", nargs="+", help="NAME:LAYERS:D_WEIGHTS[:ce-dpace] per drafter")
    ap.add_argument("--feat-layers", help="comma-separated 0-based layers inside one loop (default: DFlash rule; "
                                          "Dory: its physical sublayer taps 3,19,35,51,67)")
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=1024, help="blocks per packed sequence and optimizer step")
    ap.add_argument("--micro-blocks", type=int, default=256, help="blocks per forward (gradient accumulation)")
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--reuse", type=int, default=2, help="optimizer steps per target forward (fresh blocks each)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--train-limit", type=int, default=0, help="smoke runs: domain-stratified train subset size")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke runs: stop early")
    ap.add_argument("--eval-docs", type=int, default=240)
    ap.add_argument("--eval-stride", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue after the last completed epoch saved in --out")
    return ap


def parse_drafter(spec):
    name, layers, weights, *rest = spec.split(":")
    loss = rest[0] if rest else LOSS
    if loss != LOSS:
        raise ValueError(f"{name}: only the {LOSS} loss is supported")
    return name, int(layers), {int(k): float(v) for k, v in (kv.split("=") for kv in weights.split(","))}, loss


def split(b, size):
    for s in range(0, b.a.numel(), size):
        yield Blocks(*(getattr(b, f)[s:s + size] for f in ("a", "d", "lif", "k", "doc", "valid")))


def main(argv=None):
    parser = build_parser()
    args = parse_with_config(parser, argv)
    for flag in ("model", "traces", "out", "drafter"):
        if not getattr(args, flag):
            parser.error(f"--{flag} is required (on the command line or in --config)")
    dory = registry.family(args.model) == "dory"
    if dory and args.target != "original":
        parser.error("Dory shares without adaptation: its drafter trains on the original per-loop Dory "
                     "(--target original) and serves both of its targets")
    if args.mask_token is None:
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
    feat_layers = ([int(x) for x in args.feat_layers.split(",")] if args.feat_layers
                   else target_feature_layers(target_cfg))
    target = (ReplayTarget(args.model, target_dir, feat_layers, dev, args.window, args.replay_batch,
                           args.replay_attention, args.replay_batch_tokens, args.replay_prompt_tokens)
              if args.target == "w64" else native_target(args.model, feat_layers, dev, path=target_dir,
                                                         attention=args.replay_attention))
    R = target.R
    drafters = []
    for spec in args.drafter:
        name, layers, d_weights, loss = parse_drafter(spec)
        if set(d_weights) - set(range(1, R + 1)):
            raise ValueError(f"{name}: anchor depths must lie in 1..{R}")
        cfg = drafter_config(layers, target.model.config)
        module = DepthDFlash(cfg, len(feat_layers)).to(dev)
        with torch.no_grad():
            module.mask_emb.copy_(target.embed.weight[args.mask_token].float())
        wrapped = (torch.nn.parallel.DistributedDataParallel(module, device_ids=[local], broadcast_buffers=False)
                   if distributed else module)
        drafters.append({"name": name, "layers": layers, "d_weights": d_weights, "loss": loss, "cfg": cfg,
                         "module": module, "model": wrapped,
                         "rng": random.Random(args.seed * 1000 + rank * 17 + len(drafters)),
                         "params": sum(p.numel() for p in module.parameters()), "best": -1.0})

    train_docs, tstats = load_traces(traces / "train.jsonl", args.seq_len, args.train_limit, args.seed)
    dev_docs, dstats = load_traces(traces / "dev.jsonl", args.seq_len)
    dev_docs = sorted(dev_docs, key=lambda d: d["id"])[:: max(1, len(dev_docs) // args.eval_docs)][:args.eval_docs]
    packs_per_epoch = len(pack(train_docs, args.seq_len, random.Random(0))) // world
    total_steps = packs_per_epoch * args.epochs * args.reuse
    epochs = math.ceil(total_steps / (packs_per_epoch * args.reuse))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    for d in drafters:
        d["opt"] = torch.optim.AdamW(d["module"].parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd,
                                     fused=True)
        warm = max(1, int(args.warmup * total_steps))
        d["sched"] = torch.optim.lr_scheduler.LambdaLR(
            d["opt"], lambda s, warm=warm: (s + 1) / warm if s < warm else
            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    log(json.dumps({"model": args.model, "target": args.target, "feat_layers": feat_layers, "world": world,
                    "train": tstats, "dev_eval_docs": len(dev_docs),
                    "packs_per_rank_epoch": packs_per_epoch, "total_steps": total_steps, "epochs": epochs,
                    "drafters": {d["name"]: {"layers": d["layers"], "d_weights": d["d_weights"], "loss": d["loss"],
                                             "params_M": round(d["params"] / 1e6, 1)}
                                 for d in drafters}}))

    @torch.no_grad()
    def quick_eval(tag):
        """Deterministic anchors on a dev subset: d = 1..R with L_if = (R - d) * 8, block 16."""
        K = args.k_max
        settings = [(d, (R - d) * 8) for d in range(1, R + 1)]
        acc = {(d["name"], s): [torch.zeros(K - 1, device=dev), torch.zeros(K - 1, device=dev),
                                torch.zeros(2, device=dev)]
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
                        _, m = drafter_forward(d["module"], target, p, b, R, K)
                        att, ok = acceptance_counts(m["correct"], m["valid"])
                        e = acc[(d["name"], (depth, lif_nom))]
                        e[0] += att
                        e[1] += ok
                        e[2] += torch.stack([m["native_ok"][b.valid].sum(), b.valid.sum()]).float()
        if distributed:
            for v in acc.values():
                for x in v:
                    dist.all_reduce(x)
        results = {}
        for d in drafters:
            rows = {}
            for depth, lif in settings:
                att, ok, nat = acc[(d["name"], (depth, lif))]
                a_i = (ok / att.clamp(min=1)).tolist()
                A, tau = 1.0, 1.0
                for x in a_i:
                    A *= x
                    tau += A
                rows[f"d{depth}"] = {"L_if": lif, "a_i": a_i, "ahat": (ok.sum() / att.sum().clamp(min=1)).item(),
                                     "tau_k16": tau, "alpha_native_head": (nat[0] / nat[1].clamp(min=1)).item()}
            results[d["name"]] = rows
            d["module"].train()
            key = [f"d{x}" for x in sorted(d["d_weights"])]
            d["score"] = (sum(rows[k]["ahat"] for k in key if k in ("d1", f"d{R}"))
                          / max(1, sum(k in ("d1", f"d{R}") for k in key)))
            if rank == 0:
                log(f"[eval {tag}] {d['name']}: " + " | ".join(
                    f"{k} a1..4={v['a_i'][0]:.3f}/{v['a_i'][1]:.3f}/{v['a_i'][2]:.3f}/{v['a_i'][3]:.3f} "
                    f"ahat={v['ahat']:.3f} tau16={v['tau_k16']:.2f}"
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
                    "d_weights": d["d_weights"], "loss": d["loss"], "learn_mask": True, "mask_token": args.mask_token,
                    "input_format": INPUT_FORMAT, "target": args.model, "target_variant": args.target,
                    "target_spec": args.target_path, "window": args.window if args.target == "w64" else None,
                    "target_loops": R,
                    "step": step, "epoch": epoch,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                   path / f"{tag}.pt")

    def save_resume(epoch, step):
        """Everything needed to continue at the next epoch; packing and block sampling are seeded per epoch."""
        if rank != 0:
            return
        torch.save({"epoch": epoch, "step": step, "total_steps": total_steps,
                    "drafters": {d["name"]: {"model": d["module"].state_dict(), "opt": d["opt"].state_dict(),
                                             "sched": d["sched"].state_dict(), "best": d["best"]}
                                 for d in drafters}},
                   out / "resume.tmp")
        (out / "resume.tmp").replace(out / "resume.pt")

    step, done, start_epoch = 0, False, 0
    if args.resume and (out / "resume.pt").exists():
        state = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if state["total_steps"] != total_steps:
            raise ValueError("resume.pt was written for a different update budget")
        for d in drafters:
            saved = state["drafters"][d["name"]]
            d["module"].load_state_dict(saved["model"])
            d["opt"].load_state_dict(saved["opt"])
            d["sched"].load_state_dict(saved["sched"])
            d["best"] = saved["best"]
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
                    n_total = int(block_labels(p, b, args.k_max)[1].sum())
                    d["opt"].zero_grad(set_to_none=True)
                    chunks = list(split(b, args.micro_blocks))
                    loss_sum = 0.0
                    for i, chunk in enumerate(chunks):
                        sync = (not distributed) or i == len(chunks) - 1
                        ctx = d["model"].no_sync() if not sync else torch.enable_grad()
                        with ctx:
                            loss, m = drafter_forward(d["model"], target, p, chunk, R, args.k_max, n_total=n_total)
                            loss.backward()
                        loss_sum += loss.item()
                        s = stats[d["name"]]
                        for depth in range(1, R + 1):
                            sel = (chunk.d == depth) & m["valid"][:, 0]
                            s["slot1"][depth][0] += int(m["correct"][sel, 0].sum())
                            s["slot1"][depth][1] += int(sel.sum())
                    torch.nn.utils.clip_grad_norm_(d["module"].parameters(), 1.0)
                    d["opt"].step()
                    d["sched"].step()
                    torch.cuda.synchronize()
                    s = stats[d["name"]]
                    s["loss"] += loss_sum
                    s["n"] += 1
                    s["ms"] += (time.perf_counter() - t1) * 1e3
                step += 1
                if step % args.log_every == 0:
                    parts = []
                    for d in drafters:
                        s = stats[d["name"]]
                        acc1 = " ".join(f"d{k}={c / max(1, t):.3f}" for k, (c, t) in s["slot1"].items() if t)
                        parts.append(f"{d['name']} loss {s['loss'] / max(1, s['n']):.4f} slot1 {acc1} "
                                     f"{s['ms'] / max(1, s['n']):.0f}ms")
                        stats[d["name"]] = {"loss": 0.0, "n": 0, "ms": 0.0,
                                            "slot1": {k: [0, 0] for k in range(1, R + 1)}}
                    log(f"ep {epoch} step {step}/{total_steps} lr {drafters[0]['sched'].get_last_lr()[0]:.2e} | "
                        f"target {t_target / max(1, n_packs) * 1e3:.0f}ms/pack | "
                        f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G | " + " || ".join(parts))
                if step >= total_steps:
                    done = True
                    break
            if done:
                break
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
