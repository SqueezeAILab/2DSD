"""Trace loading, fixed-length packing, and per-pack target features.

A trace file holds one JSON object per line with at least id, domain, source, prompt_ids,
output_ids, finish_reason, degenerate and sampling (null for greedy), as written by
training.drafter.traces collect.
"""
import json
import random
from pathlib import Path

import torch

from .blocks import Packed


def load_traces(path, max_len, limit=0, seed=0, expect=None):
    """Non-degenerate traces of one pool file; `limit` keeps a seeded, domain-stratified subset.

    `expect` maps fields of a trace row (its `model`, its `target_config_sha256`) to the values of the model
    the drafter trains for: a row that names another one is refused, so that the traces of one model do not
    train the drafter of another by accident."""
    docs, stats = [], {"rows": 0, "degenerate": 0, "too_short": 0, "length_capped": 0, "loop_stopped": 0, "sampled": 0}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        stats["rows"] += 1
        for key, want in (expect or {}).items():
            if r.get(key) is not None and r[key] != want:
                raise ValueError(f"{path}: trace {r['id']} has {key} {r[key]!r} and the drafter trains for {want!r}. "
                                 "These are the traces of another model (--allow-other-target trains on them anyway)")
        if r["degenerate"]:
            stats["degenerate"] += 1
            continue
        if len(r["output_ids"]) < 8:
            stats["too_short"] += 1
            continue
        ids = (r["prompt_ids"] + r["output_ids"])[:max_len]
        stats["length_capped"] += r["finish_reason"] == "length"
        # A loop-stopped response was already cut at the loop onset during generation.
        stats["loop_stopped"] += r["finish_reason"] == "loop"
        stats["sampled"] += bool(r.get("sampling"))
        docs.append({"id": r["id"], "domain": r["domain"], "source": r["source"], "ids": ids,
                     "resp_start": len(r["prompt_ids"]), "sampled": bool(r.get("sampling"))})
    if limit and limit < len(docs):
        rng = random.Random(seed)
        by_domain = {}
        for d in docs:
            by_domain.setdefault(d["domain"], []).append(d)
        keep = []
        for rows in by_domain.values():
            rng.shuffle(rows)
            keep += rows[:round(limit * len(rows) / len(docs))]
        docs = sorted(keep, key=lambda d: d["id"])
    stats["docs"] = len(docs)
    return docs, stats


def pack(docs, T, rng):
    """First-fit packing of whole documents into sequences of at most T tokens."""
    order = list(range(len(docs)))
    if rng is not None:
        rng.shuffle(order)
    packs, fill = [], []
    for i in order:
        n = len(docs[i]["ids"])
        for j, f in enumerate(fill):
            if f + n <= T:
                packs[j].append(i)
                fill[j] += n
                break
        else:
            packs.append([i])
            fill.append(n)
    return packs


def build_packed(target, docs, members, T, device, computed=None):
    """Run the frozen target on each document of a pack and lay everything out in packed coordinates.

    computed optionally holds (features, loop states) per member, already replayed by packed_stream.
    """
    R, F = target.R, len(target.feat_layers) * target.H
    input_ids = torch.zeros(T, dtype=torch.long, device=device)
    doc_ids = torch.full((T,), -1, dtype=torch.long, device=device)
    pos = torch.zeros(T, dtype=torch.long, device=device)
    feats = torch.zeros(R, T, F, dtype=torch.bfloat16, device=device)
    loop_out = torch.zeros(R, T, target.H, dtype=torch.bfloat16, device=device)
    if computed is None and hasattr(target, "features_many"):
        computed = target.features_many([docs[m] for m in members])
    spans, off = [], 0
    for di, m in enumerate(members):
        ids = torch.tensor(docs[m]["ids"], dtype=torch.long, device=device)
        n = ids.numel()
        input_ids[off:off + n] = ids
        doc_ids[off:off + n] = di
        pos[off:off + n] = torch.arange(n, device=device)
        feats[:, off:off + n], loop_out[:, off:off + n] = computed[di] if computed else target.features(ids)
        spans.append((off, off + docs[m]["resp_start"], off + n, docs[m]["domain"]))
        off += n
    final_argmax = target.readout_argmax(loop_out[R - 1])
    sampled = torch.tensor([docs[m].get("sampled", False) for m in members], dtype=torch.bool, device=device)
    return Packed(input_ids, doc_ids, pos, spans, feats, final_argmax, loop_out, sampled)


def packed_stream(target, docs, packs, T, device, group_tokens=0):
    """Yield (pack members, Packed) in pack order.

    Exact replay of a shared-KV target is latency bound for a single document. Consecutive packs are
    replayed together (target.features_many batches them by length) in groups of group_tokens // T
    packs, so a group holds at most group_tokens tokens; 0 replays each pack on its own. Groups are
    counted in packs, not document tokens: every DDP rank holds the same number of packs, so all ranks
    replay at the same steps instead of one rank's replay stalling the others at the gradient all-reduce.
    """
    if not group_tokens or not hasattr(target, "features_many"):
        for members in packs:
            yield members, build_packed(target, docs, members, T, device)
        return
    size = max(1, group_tokens // T)
    for start in range(0, len(packs), size):
        group = packs[start:start + size]
        computed = iter(target.features_many([docs[m] for g in group for m in g]))
        for g in group:
            yield g, build_packed(target, docs, g, T, device, [next(computed) for _ in g])
