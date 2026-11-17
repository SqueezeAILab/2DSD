"""Depth-conditioned DFlash blocks: sampling, labels and chain mask, loop-context queries and mask.

Notation for one block (code name -> paper name):
  a      ("anchor" in code) the last position whose target feature is injected, just before the paper's anchor
  d      its depth in 1..R; d = R means fully verified, the original DFlash setting
  L_if   in-flight positions a-L_if .. a-1 before it: drafted but not yet verified
  v      last verified position (a for d = R, else a - L_if - 1); positions <= v use loop-R features
  slot 0 ("head" in code; the paper's anchor) the given token at a+1; slots 1..k-1 predict a+2 .. a+k

Loop-context input: positions <= v inject loop-R features and v+1 .. a inject loop-d features, so the
in-flight span is target context like the verified prefix. Queries are the k block slots (slot 0 is
the head token, slots >= 1 the mask), attending bidirectionally within the block.
KV layout: [loop-1 .. loop-R features of every packed position (R * T) | queries].
d = R gives v = a and an empty loop-d span, i.e. standard DFlash.
"""
import random
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import create_block_mask

_compiled_create_block_mask = torch.compile(create_block_mask)
INPUT_FORMAT = "loop-context"


@dataclass
class Packed:
    """One packed training sequence of fixed length T (padded)."""
    input_ids: torch.Tensor    # [T] long
    doc_ids: torch.Tensor      # [T] long, -1 = padding
    pos: torch.Tensor          # [T] long, position inside its document
    docs: list                 # [(start, resp_start, end, domain)] in packed coordinates, end exclusive
    feats: torch.Tensor        # [R, T, F] bf16 target features
    final_argmax: torch.Tensor  # [T] long: argmax p_T(x_{t+1} | x_<=t)
    loop_out: torch.Tensor | None = None  # [R, T, H] readout inputs (target.readout_*), not drafter inputs
    doc_sampled: torch.Tensor | None = None  # [docs] bool: the trace was sampled at T > 0, not greedy


@dataclass
class Blocks:
    a: torch.Tensor     # [NB] anchor packed index
    d: torch.Tensor     # [NB] anchor depth 1..R
    lif: torch.Tensor   # [NB] in-flight length
    k: torch.Tensor     # [NB] block length (slot 0 + k-1 drafted slots)
    doc: torch.Tensor   # [NB] document id
    valid: torch.Tensor  # [NB] bool (padding blocks are invalid)


def sample_blocks(p: Packed, nb: int, R: int, k_max: int, d_weights: dict, rng: random.Random) -> Blocks:
    """Randomize the schedule-dependent inputs: d, L_if and the anchor position (blocks of k_max)."""
    l_max = (R - 1) * k_max
    ds, dws = zip(*sorted(d_weights.items()))
    spans = []
    for di, (start, rs, end, _dom) in enumerate(p.docs):
        lo_R, hi = rs - 1, end - 3  # need a+2 <= end-1 so that slot 1 exists
        if hi >= rs:
            spans.append((di, start, rs, hi, hi - rs + 1))
    out = {k: [] for k in ("a", "d", "lif", "k", "doc", "valid")}
    if spans:
        weights = [s[-1] for s in spans]
        for _ in range(nb):
            di, start, rs, hi, _n = rng.choices(spans, weights)[0]
            d = rng.choices(ds, dws)[0]
            lo = rs - 1 if d == R else rs  # an unverified anchor is itself a drafted (response) position
            a = rng.randint(lo, hi)
            if d == R:
                lif = 0
            elif rng.random() < 0.7:  # near the steady-state value (R - d) * k_prev
                lif = (R - d) * rng.randint(1, k_max)
            else:
                lif = rng.randint(0, l_max)
            lif = max(0, min(lif, l_max, a - rs))  # in-flight positions are drafted response tokens
            k = k_max
            for key, val in (("a", a), ("d", d), ("lif", lif), ("k", k), ("doc", di), ("valid", True)):
                out[key].append(val)
    while len(out["a"]) < nb:
        for key, val in (("a", 0), ("d", R), ("lif", 0), ("k", 0), ("doc", -2), ("valid", False)):
            out[key].append(val)
    dev = p.input_ids.device
    order = sorted(range(nb), key=lambda i: out["a"][i])  # contiguous anchors -> denser flex blocks
    t = lambda key, dt=torch.long: torch.tensor([out[key][i] for i in order], dtype=dt, device=dev)
    return Blocks(t("a"), t("d"), t("lif"), t("k"), t("doc"), t("valid", torch.bool))


def block_labels(p: Packed, b: Blocks, k_max: int):
    """Labels [NB, k_max-1] and their validity for drafted slots i >= 1.

    Slot i predicts the token at a+1+i; its label is argmax p_T at a+i. Chain mask: slot i is a valid
    greedy target only if the trace followed argmax p_T for every token from slot 0 through slot i-1
    (trace/label numerics can disagree, ~0.5%/token). In a sampled trace the head is a sample, as a
    sampled anchor or bonus is at T > 0, so the chain starts at slot 1 there: later slots still follow
    the argmax chain after the head.
    """
    dev = p.input_ids.device
    T = p.input_ids.numel()
    i = torch.arange(1, k_max, device=dev)[None, :]
    lab_pos = (b.a[:, None] + i).clamp(0, T - 1)
    labels = p.final_argmax[lab_pos]
    doc_end = torch.tensor([e for (_s, _r, e, _d) in p.docs] + [0], device=dev)[b.doc.clamp(min=-1)]
    in_doc = (b.a[:, None] + 1 + i) < doc_end[:, None]
    m = torch.arange(0, k_max - 1, device=dev)[None, :]
    nxt = (b.a[:, None] + 1 + m).clamp(0, T - 1)
    agree = p.input_ids[nxt] == p.final_argmax[(b.a[:, None] + m).clamp(0, T - 1)]
    if p.doc_sampled is not None:
        agree[:, 0] |= p.doc_sampled[b.doc.clamp(min=0)]
    agree = agree.long()
    chain_ok = torch.cumprod(agree, dim=1).bool()  # chain_ok[:, i-1] covers slots 0..i-1
    lab_valid = in_doc & chain_ok & (i < b.k[:, None]) & b.valid[:, None]
    return labels, lab_valid


def verified_end(b: Blocks, R: int):
    """Last position injected with loop-R features: a for d = R, else a - L_if - 1."""
    return torch.where(b.d == R, b.a, b.a - b.lif - 1)


def build_context_inputs(p: Packed, b: Blocks, R: int, k_max: int):
    """loop-context queries: slot 0 holds the head token at a+1, slots 1..k-1 the mask."""
    dev = p.input_ids.device
    T = p.input_ids.numel()
    nb = b.a.numel()
    slot = torch.arange(k_max, device=dev)[None, :]
    q_valid = (slot < b.k[:, None]) & b.valid[:, None]
    q_tok = p.input_ids[(b.a[:, None] + 1 + slot).clamp(0, T - 1)]
    q_is_mask = (slot >= 1).expand(nb, k_max)
    q_pos = p.pos[b.a][:, None] + 1 + slot
    labels, lab_valid = block_labels(p, b, k_max)
    q_index = torch.arange(nb, device=dev)[:, None] * k_max + torch.arange(1, k_max, device=dev)[None, :]
    return dict(q_tok=q_tok.reshape(-1), q_is_mask=q_is_mask.reshape(-1), q_pos=q_pos.reshape(-1),
                q_valid=q_valid.reshape(-1), labels=labels, lab_valid=lab_valid, q_index=q_index, S=k_max)


def build_context_mask(p: Packed, b: Blocks, R: int, k_max: int):
    mask_mod, q_len, kv_len = context_mask_mod(p, b, R, k_max)
    return _compiled_create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len,
                                       device=p.input_ids.device), mask_mod


def context_mask_mod(p: Packed, b: Blocks, R: int, k_max: int):
    """loop-context mask_mod plus (Q_LEN, KV_LEN). KV: [loop r features at kv = r * T + t | queries]."""
    T = p.input_ids.numel()
    nb = b.a.numel()
    Q = nb * k_max
    C = R * T
    doc = p.doc_ids
    blk_doc, blk_a, blk_d, blk_k, blk_ok = b.doc, b.a, b.d, b.k, b.valid
    blk_v = verified_end(b, R)

    def mask_mod(_b, _h, q, kv):
        jb = q // k_max
        sq = q - jb * k_max
        q_ok = blk_ok[jb] & (sq < blk_k[jb])
        is_ctx = kv < C
        kvc = torch.clamp(kv, max=C - 1)
        loop = kvc // T
        t = kvc - loop * T
        same_doc = is_ctx & (doc[t] == blk_doc[jb])
        prefix = same_doc & (loop == R - 1) & (t <= blk_v[jb])
        span = same_doc & (loop == blk_d[jb] - 1) & (t > blk_v[jb]) & (t <= blk_a[jb])
        kq = torch.clamp(kv - C, min=0)
        kb = kq // k_max
        ks = kq - kb * k_max
        is_q = kv >= C
        block = is_q & (kb == jb) & (ks < blk_k[jb])
        selfloop = is_q & (kq == q)
        return (q_ok & (prefix | span | block)) | ((~q_ok) & selfloop)

    return mask_mod, Q, C + Q
