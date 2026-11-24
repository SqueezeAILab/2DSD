#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Does an exported drafter draft at run time what its training module drafts?

The target verifies every drafted token, so a 2DSD run names the right tokens whatever the drafter
proposes: a drafter that drafts something else than it was trained to costs speed and shows nowhere in the
outputs. This script compares the two sides directly, on the real target and held-out traces:

  training  the training module (training.drafter.model.DepthDFlash) with the export's weights, in the
            arithmetic of training (BF16 autocast), its tokens decided left to right;
  run time  the adapter that the 2DSD workers load (ContextDFlashAdapter.from_export), given the same
            target features, in-flight span and given token.

  python scripts/check_drafter_export.py --model dory1.0-4b --drafter EXPORT --traces dev.jsonl \\
      [--target original] [--target-path CHECKPOINT] [--docs 8] [--stride 96] [--experts packed]

It prints, per anchor depth, the share of drafted tokens that are the same on both sides (all slots and the
first slot) and, for a DSpark export, the mean absolute difference of the confidences. A DSpark token
depends on the token drafted before it, so one differing token changes the rest of its block: the first
slot is the clean measure, and for its disagreements the logit gap tells a near-tie from another model.
Exit status 1 when the first slot agrees less often than --min-agreement (default 0.98: BF16 sums in
another order flip a near-tie now and then; a wrong head, scale or slot layout is far below). The target
is the per-loop model of the registry name; --target w64 replays its shared-KV checkpoint, as training on
that checkpoint's traces does, and --target-path names the checkpoint the drafter was trained against
when it is not the registry's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def training_module(export, target, device):
    """The export's weights in the training module, and its metadata."""
    import torch
    from safetensors.torch import load_file

    from training.drafter.model import DepthDFlash, drafter_config

    config = json.loads((export / "config.json").read_text())
    info = config["loopedtf_drafter"]
    if info.get("input_format") != "loop-context":
        raise ValueError("check_drafter_export.py checks loop-context exports")
    dspark = info.get("dspark")
    if dspark and dspark["markov_head_type"] != "vanilla":
        raise ValueError("the left-to-right reference covers the first-order (vanilla) Markov head")
    heads = dspark and {"vocab": target.lm_head.weight.shape[0], "rank": dspark["markov_rank"],
                        "type": dspark["markov_head_type"]}
    module = DepthDFlash(drafter_config(config["num_hidden_layers"], target.model.config),
                         len(info["feature_layers_per_loop"]), heads=heads or None)
    state = {name: value.float() for name, value in load_file(export / "model.safetensors").items()
             if name != "lm_head.weight"}
    mask = export / "mask_embedding.pt"
    if mask.exists():
        state["mask_emb"] = torch.load(mask, map_location="cpu")["embedding"].float()
    else:  # a frozen mask: the embedding row of the export's mask token
        state["mask_emb"] = target.embed.weight[config["dflash_config"]["mask_token_id"]].float().cpu()
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("rotary.") for name in missing):
        raise ValueError("the export does not fill the training module: "
                         f"missing {missing}, unexpected {unexpected}")
    return module.to(device).eval(), info, config["block_size"]


def training_drafts(module, target, p, b, R, K, anchor_predicts):
    """Tokens [NB, K - 1] and confidences (DSpark) the training module drafts for blocks b, decided left to
    right from the given token, in the arithmetic of the training forward; and its logits of the first
    drafted token [NB, V]."""
    import torch

    from training.drafter.blocks import build_context_inputs, build_context_mask
    from training.drafter.model import softcap

    inp = build_context_inputs(p, b, R, K, chain=False, anchor_predicts=anchor_predicts)
    mask, _ = build_context_mask(p, b, R, K, anchor_predicts=anchor_predicts)
    position_ids = torch.cat([p.pos.repeat(R), inp["q_pos"]])[None]
    cap = getattr(target.lm_head, "softcap", None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = module(p.feats.reshape(-1, p.feats.shape[-1])[None], target.embed(inp["q_tok"])[None],
                        inp["q_is_mask"], position_ids, mask)[0][inp["q_index"]]  # [NB, K - 1, H]
        if module.markov_head is None:
            logits = target.lm_head(hidden)
            logits = softcap(logits, cap) if cap else logits.float()
            return logits.argmax(-1), None, logits[:, 0]
        previous = p.input_ids[(b.a + 1).clamp(max=p.input_ids.numel() - 1)]  # the given token
        tokens, confidences, first = [], [], None
        for slot in range(hidden.shape[1]):
            logits, confidence = module.head_outputs(hidden[:, slot:slot + 1], previous[:, None], target.lm_head)
            logits = softcap(logits, cap) if cap else logits.float()
            first = logits[:, 0] if first is None else first
            previous = logits[:, 0].argmax(-1)
            tokens.append(previous)
            confidences.append(torch.sigmoid(confidence[:, 0].float()))
    return torch.stack(tokens, 1), torch.stack(confidences, 1), first


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a name of configs/models.json")
    parser.add_argument("--drafter", required=True, type=Path, help="the drafter export to check")
    parser.add_argument("--traces", required=True, type=Path, help="held-out trace file (dev.jsonl)")
    parser.add_argument("--target", choices=("original", "w64"), default="original",
                        help="the target whose features both sides read (default: the per-loop model)")
    parser.add_argument("--target-path", help="local directory or Hugging Face id of the target checkpoint "
                        "(default: the registry's checkpoint of --target)")
    parser.add_argument("--docs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--stride", type=int, default=96, help="anchors: every this many response positions")
    parser.add_argument("--micro-blocks", type=int, default=64)
    parser.add_argument("--experts", choices=("dense", "packed"), default="dense", help="Inkling2 targets only")
    parser.add_argument("--min-agreement", type=float, default=0.98)
    parser.add_argument("--out", type=Path, help="also write the result as JSON")
    args = parser.parse_args(argv)
    import torch

    from training.common import registry
    from training.drafter.blocks import Blocks
    from training.drafter.data import build_packed, load_traces, pack
    from training.drafter.targets import ReplayTarget, native_target

    device = torch.device("cuda")
    export = json.loads((args.drafter / "config.json").read_text())
    info = export["loopedtf_drafter"]
    layers = info["feature_layers_per_loop"]
    target_dir = registry.resolve_target(args.model, args.target, args.target_path)
    if args.target == "w64":
        target = ReplayTarget(args.model, target_dir, layers, device, batch_tokens=65536)
    else:
        target = native_target(args.model, layers, device, path=target_dir, experts=args.experts,
                               tensor_cores=registry.family(args.model) == "inkling2")
    R = target.R
    if info.get("target_loops", R) != R:
        raise ValueError(f"the drafter was trained at R = {info['target_loops']} and the target runs R = {R}")
    module, info, K = training_module(args.drafter, target, device)
    anchor_predicts = bool(info.get("dspark") and info["dspark"]["sample_from_anchor"])
    block = K + anchor_predicts  # the given token and the drafted ones
    head = target.lm_head
    cap = getattr(head, "softcap", None)
    if cap and not info.get("dspark"):  # the workers hand a DFlash drafter the capped head
        from training.drafter.model import softcap

        head = lambda hidden: softcap(target.lm_head(hidden), cap)
    docs, _ = load_traces(args.traces, args.seq_len, args.docs, 0)
    counts = {d: {"same": 0, "slots": 0, "first": 0, "blocks": 0, "confidence": 0.0} for d in range(1, R + 1)}
    margins = []  # first-slot disagreements: the training module's logit gap between the two tokens
    with torch.no_grad():
        # The target's features first: importing vLLM turns PyTorch's cuDNN attention off for the process,
        # and an SDPA forward over a long document (the fallback without vLLM's FlashAttention) then
        # materializes its attention matrix.
        packs = [build_packed(target, docs, members, args.seq_len, device)
                 for members in pack(docs, args.seq_len, None)]
        from vllm.v1.spec_decode.looped_dflash_context import ContextDFlashAdapter

        # The head of the run-time path: as the workers pass it (before the softcap for a DSpark export).
        adapter = ContextDFlashAdapter.from_export(args.drafter, embed=target.embed, lm_head=head,
                                                   max_len=args.seq_len, device=device, dtype=torch.bfloat16,
                                                   use_cuda_graph=False)
        for p in packs:
            for index, (start, begin, end, _domain) in enumerate(p.docs):
                rows = []
                for depth in range(1, R + 1):
                    nominal = (R - depth) * (block // 2)
                    for a in range(begin - 1 if depth == R else begin, end - block - 1, args.stride):
                        rows.append((a, depth, max(0, min(nominal, a - begin)) if depth < R else 0))
                for at in range(0, len(rows), args.micro_blocks):
                    chunk = rows[at:at + args.micro_blocks]
                    column = lambda j: torch.tensor([r[j] for r in chunk], device=device)
                    b = Blocks(column(0), column(1), column(2), torch.full((len(chunk),), block, device=device),
                               torch.full((len(chunk),), index, device=device),
                               torch.ones(len(chunk), dtype=torch.bool, device=device))
                    want, want_confidence, first = training_drafts(module, target, p, b, R, block, anchor_predicts)
                    for row, (a, depth, lif) in enumerate(chunk):
                        # One document's positions: the adapter's prefix starts at the document.
                        verified = a + 1 if depth == R else a - lif
                        request = {"prefix_features": p.feats[R - 1, start:], "prefix_len": verified - start,
                                   "span_features": None if depth == R else p.feats[depth - 1, verified:a + 1],
                                   "head": int(p.input_ids[a + 1]), "k": block}
                        adapter.reset()
                        [draft] = adapter.draft_context_many([request])
                        got = draft.token_ids.to(device)
                        same = got == want[row, :got.numel()]
                        entry = counts[depth]
                        entry["same"] += int(same.sum())
                        entry["slots"] += got.numel()
                        entry["first"] += int(same[0])
                        entry["blocks"] += 1
                        if not bool(same[0]):  # the training module's gap from its own choice to that token
                            margins.append(float(first[row].max() - first[row, got[0]]))
                        if want_confidence is not None and draft.confidence is not None:
                            entry["confidence"] += float((draft.confidence.to(device).float()
                                                          - want_confidence[row, :got.numel()]).abs().mean())
    result = {"model": args.model, "target": args.target, "drafter": str(args.drafter), "block_size": K,
              "dspark": bool(info.get("dspark")), "docs": len(docs),
              "depths": {str(d): {"blocks": e["blocks"], "agreement": e["same"] / max(e["slots"], 1),
                                  "first_slot_agreement": e["first"] / max(e["blocks"], 1),
                                  "confidence_abs_difference": e["confidence"] / max(e["blocks"], 1)}
                         for d, e in counts.items()}}
    slots = sum(e["slots"] for e in counts.values())
    result["agreement"] = sum(e["same"] for e in counts.values()) / max(slots, 1)
    for d, e in result["depths"].items():
        print(f"depth {d}: {e['blocks']:5d} blocks, drafted tokens equal {e['agreement']:.4f} "
              f"(first slot {e['first_slot_agreement']:.4f})"
              + (f", confidence differs by {e['confidence_abs_difference']:.4f}" if result["dspark"] else ""))
    print(f"all depths: {result['agreement']:.4f} of {slots} drafted tokens are the training module's")
    if margins:
        margins.sort()
        result["first_slot_margins"] = {"count": len(margins), "median": margins[len(margins) // 2],
                                        "max": margins[-1]}
        print(f"first-slot disagreements: {len(margins)}; the training module's logit gap between the two tokens: "
              f"median {margins[len(margins) // 2]:.3f}, largest {margins[-1]:.3f} (a near-tie is well below 0.1)")
    firsts = sum(e["first"] for e in counts.values()) / max(sum(e["blocks"] for e in counts.values()), 1)
    result["first_slot_agreement"] = firsts
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    sys.exit(0 if firsts >= args.min_agreement else 1)


if __name__ == "__main__":
    main()
