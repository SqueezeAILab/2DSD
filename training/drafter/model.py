"""Depth-conditioned drafter on speculators' Qwen3 DFlash layers (target features injected as K/V).

The drafter has the target's width; fc(F -> H) + RMSNorm projects the concatenated feature layers to
the context that every drafter layer reads as K/V. The target's embedding and LM head are shared and
frozen; the mask vector is learned. For Dory the head is its full-vocabulary nGPT head (FP32 scale after
the GEMM), which expects unit-norm inputs, so the drafter's final RMSNorm weight starts at 1/sqrt(H)
instead of 1 (still trained).

A block of length k drafts the k - 1 tokens after a given token (DFlash's anchor). Two recipes:

"ce-dpace"  DFlash, the paper's recipe. The block's slots are the given token and k - 1 masks, each mask
            predicting the token at its own position. CE on the target's last-loop argmax with D-PACE slot
            weights, over the slots the chain mask keeps (training.drafter.blocks).
"dspark"    DSpark (arXiv 2607.05147) as speculators implements and trains it, on the same backbone.
            speculators' MarkovHead adds a bias to the logits of every drafted token from the token before
            it (the given token, then the tokens drafted before), and its ConfidenceHead predicts every
            drafted token's acceptance from its slot's backbone state and that token's Markov embedding.
            Training is teacher-forced on the trace and covers every drafted token of a block; the loss is
            speculators' DSpark loss (DSPARK below) against the target's last-loop distribution at each
            position. At run time the tokens are decided left to right after the one backbone pass. A
            target whose head has a logit softcap (Inkling 2) gets it after the bias.
            The slots are DFlash's by default (speculators' sample_from_anchor = False): the given token and
            k - 1 masks, so a block of 8 is the given token and 7 masks, as in the DSpark drafter bundled
            with Inkling 2. With anchor_predicts (sample_from_anchor = True, the DSpark paper's layout) the
            given token's slot is the first prediction: k - 1 slots, the given token and k - 2 masks, every
            slot predicting the token after its position.
"""
import json

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from speculators.losses import resolve_loss_config
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dspark.metrics import compute_metrics as dspark_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.utils import flatten_rope_parameters
from torch.nn.attention.flex_attention import flex_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .blocks import block_labels, build_context_inputs, build_context_mask

MASK_TOKEN = 16  # embedding row that initialises the learned mask (Ouro <empty_output>, never generated)
DORY_MASK_TOKEN = 999  # Dory's reserved <SPECIAL_999>, never emitted; also its exported mask_token_id
LOSS = "ce-dpace"
RECIPES = ("ce-dpace", "dspark")
# DSpark as speculators trains it (its published Qwen3 and Kimi K3 recipes): 0.1 CE on the target's argmax
# + 0.9 TV to the target's distribution + 1.0 BCE of the confidence head against 1 - TV, each slot weighted
# exp(-slot / gamma) (gamma 4 for a block of 8) and the sum divided by the number of slots; Muon on the 2-D
# weights and AdamW on the rest (norms, Markov factors, confidence head, mask), cosine after 4% warmup.
DSPARK = {"loss": {"ce": 0.1, "tv": 0.9}, "gamma": 4.0, "confidence_alpha": 1.0,
          "markov_rank": 256, "markov_head": "vanilla",
          "muon_lr": 3e-3, "muon_momentum": 0.95, "muon_weight_decay": 0.1, "lr": 6e-4, "weight_decay": 0.0}
MARKOV_HEADS = ("vanilla", "gated", "rnn")


# speculators 0.9.0 calls flex_attention eagerly, which materializes dense scores (24 GiB for one
# 256-block micro-batch). The compiled kernel keeps the block-sparse path.
_compiled_flex_attention = torch.compile(flex_attention)


def compiled_flex_attention_forward(module, query, key, value, attention_mask, scaling=None, **_kwargs):
    out = _compiled_flex_attention(query.contiguous(), key.contiguous(), value.contiguous(), block_mask=attention_mask,
                                   enable_gqa=query.shape[1] != key.shape[1], scale=scaling)
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("loopedtf_flex_attention", compiled_flex_attention_forward)


class _Softcap(torch.autograd.Function):
    """cap * tanh(x / cap) in FP32 that holds one tensor of x's shape, where the three operations hold three:
    the result is computed in place on the FP32 copy of x, and it is all the gradient needs
    (dy/dx = 1 - (y / cap)^2). Logits over a 200K vocabulary make that difference gigabytes."""

    @staticmethod
    def forward(ctx, x, cap):
        y = x.float() if x.dtype != torch.float32 else x.clone()
        y.div_(cap).tanh_().mul_(cap)
        ctx.cap, ctx.dtype = cap, x.dtype
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        scale = (y / ctx.cap).square_().neg_().add_(1.0)
        return scale.mul_(grad).to(ctx.dtype), None


def softcap(logits, cap):
    """The target head's logit softcap, cap * tanh(logits / cap), as FP32."""
    return _Softcap.apply(logits, float(cap))


def drafter_config(num_layers, target_config):
    t = target_config
    dory = getattr(t, "model_type", None) == "dory"
    # A mixture-of-experts target's intermediate_size is one expert's width; its drafter gets a 4H MLP.
    intermediate = 4 * t.hidden_size if hasattr(t, "n_routed_experts") else t.intermediate_size
    cfg = Qwen3Config(
        hidden_size=t.hidden_size, intermediate_size=intermediate, num_hidden_layers=num_layers,
        num_attention_heads=t.num_attention_heads, num_key_value_heads=t.num_key_value_heads,
        head_dim=getattr(t, "head_dim", None) or t.hidden_size // t.num_attention_heads,
        rms_norm_eps=getattr(t, "rms_norm_eps", 1e-6), max_position_embeddings=65536, attention_bias=False,
        layer_types=["full_attention"] * num_layers, use_sliding_window=False,
        rope_parameters={"rope_type": "default", "rope_theta": (t.rope_parameters["full_attention"]["rope_theta"]
                                                                if dory else getattr(t, "rope_theta", 1e6))},
    )
    cfg._attn_implementation = "loopedtf_flex_attention"  # noqa: SLF001
    if dory:  # the nGPT head reads unit-norm states
        cfg.loopedtf_output_norm_scale = t.hidden_size ** -0.5
    return cfg


class DepthDFlash(nn.Module):
    """heads = {"vocab", "rank", "type"} adds DSpark's Markov and confidence heads over the head's vocabulary."""

    def __init__(self, cfg, n_feat_layers, init_std=0.02, heads=None):
        super().__init__()
        H = cfg.hidden_size
        self.cfg = cfg
        self.layers = nn.ModuleList(Qwen3DFlashDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.fc = nn.Linear(n_feat_layers * H, H, bias=False)
        self.hidden_norm = Qwen3RMSNorm(H, eps=cfg.rms_norm_eps)
        self.norm = Qwen3RMSNorm(H, eps=cfg.rms_norm_eps)
        with torch.no_grad():
            self.norm.weight.fill_(getattr(cfg, "loopedtf_output_norm_scale", 1.0))
        self.rotary = Qwen3RotaryEmbedding(flatten_rope_parameters(cfg))
        # DFlash borrows an unused target embedding row as a frozen mask. Ouro has no padding rows and the rows
        # of never-seen tokens decayed to norm ~0.007 (vs ~1.1), which RMSNorm's eps then shrinks to ~15% of a
        # token's scale, so the mask is learned. The caller initialises it from MASK_TOKEN's row.
        self.mask_emb = nn.Parameter(torch.zeros(H))
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=init_std)
        # speculators builds its DSpark heads after the backbone's initialisation; they keep their own.
        self.markov_head = self.confidence_head = None
        if heads:
            self.markov_head = MarkovHead(verifier_vocab_size=heads["vocab"], draft_vocab_size=heads["vocab"],
                                          markov_rank=heads["rank"], hidden_size=H, head_type=heads["type"])
            self.confidence_head = ConfidenceHead(H + heads["rank"])

    def forward(self, ctx_raw, q_emb, q_is_mask, position_ids, block_mask):
        """ctx_raw [1, C, F] target features, q_emb [1, Q, H], q_is_mask [Q], position_ids [1, C + Q]."""
        ctx = self.hidden_norm(self.fc(ctx_raw))
        h = torch.where(q_is_mask[None, :, None], self.mask_emb.to(q_emb.dtype), q_emb)
        pe = self.rotary(h, position_ids)
        for layer in self.layers:
            h = layer(hidden_states=h, target_hidden=ctx, attention_mask=block_mask,
                      position_ids=position_ids, position_embeddings=pe)
        return self.norm(h)

    def head_outputs(self, hidden, previous, lm_head):
        """DSpark's outputs for the slots of blocks: hidden [NB, S, H] and the tokens `previous` [NB, S]
        before the tokens they predict -> logits [NB, S, V] with the Markov bias (before any softcap) and
        confidence logits [NB, S]."""
        embedding = self.markov_head.prev_embeddings(previous)
        logits = lm_head(hidden) + self.markov_head.block_bias(prev_token_ids=previous, hidden_states=hidden,
                                                               prev_emb=embedding)
        confidence = self.confidence_head(torch.cat([hidden, embedding.to(hidden.dtype)], dim=-1))
        return logits, confidence


def dspark_loss(logits, target_logits, confidence, valid, recipe=DSPARK):
    """speculators' DSpark loss and metrics over drafted slots.

    logits and target_logits [NB, S, V] are the drafter's (Markov-corrected) and the target's for the S
    drafted tokens of each block, confidence [NB, S] the confidence logits and valid [NB, S] the tokens that
    count. Drafted token s has the weight exp(-s / gamma), which is speculators' weight in either slot
    layout (there the given token's own slot, when it does not predict, has weight 0), and the loss is
    divided by the number of valid tokens.
    """
    blocks, slots, vocab = logits.shape
    kind = "fused" if logits.is_cuda else "eager"  # the fused kernels never materialize [slots, V] probabilities
    flat = lambda x: x.reshape(1, blocks * slots, *x.shape[2:])
    return dspark_metrics(flat(logits), flat(target_logits), flat(confidence), flat(valid), slots,
                          loss_config=resolve_loss_config(json.dumps(recipe["loss"]), kind),
                          tv_loss_fn=resolve_loss_config("tv", kind)["tv"][0], gamma=recipe["gamma"],
                          confidence_head_alpha=recipe["confidence_alpha"],
                          per_position_loss_weight="fixed-exp-decay", sample_from_anchor=True)


def drafter_forward(drafter, target, p, b, R, k_max, n_total=None, recipe=LOSS, anchor_predicts=False):
    """The drafter's loss over valid block slots, and per-slot correctness against the target's argmax.

    "ce-dpace": CE on the target's last-loop argmax with D-PACE slot weights (speculators' DFlash default)
    over the chain-masked slots. "dspark": dspark_loss over every drafted token of the block, in the slot
    layout anchor_predicts selects (training.drafter.blocks). n_total normalizes the loss when blocks are
    split into micro-batches. The returned correctness and its `valid` cover the
    chain-masked slots under either recipe: there the trace tokens that teacher forcing feeds the Markov head
    are the tokens a correct greedy draft has produced, so the counts are greedy acceptance.
    """
    if recipe not in RECIPES:
        raise ValueError(f"unknown drafter recipe {recipe!r}; supported: {', '.join(RECIPES)}")
    dspark = recipe == "dspark"
    if anchor_predicts and not dspark:
        raise ValueError("only the dspark recipe drafts from the given token's slot")
    inp = build_context_inputs(p, b, R, k_max, chain=not dspark, anchor_predicts=anchor_predicts)
    block_mask, _ = build_context_mask(p, b, R, k_max, anchor_predicts=anchor_predicts)
    ctx_raw = p.feats.reshape(-1, p.feats.shape[-1])[None]
    position_ids = torch.cat([p.pos.repeat(R), inp["q_pos"]])[None]
    q_emb = target.embed(inp["q_tok"])[None]
    module = getattr(drafter, "module", drafter)
    cap = getattr(target.lm_head, "softcap", None)
    sel = inp["lab_valid"]
    n_valid = sel.sum()
    denom = n_valid.clamp(min=1) if n_total is None else max(int(n_total), 1)
    extra = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = drafter(ctx_raw, q_emb, inp["q_is_mask"], position_ids, block_mask)[0]
        if dspark:  # drafted token i follows the trace token at a + i: the given token, then the trace
            previous = inp["q_tok"].view(-1, inp["S"])[:, :k_max - 1]
            logits, confidence = module.head_outputs(hidden[inp["q_index"]], previous, target.lm_head)
        else:
            logits = target.lm_head(hidden[inp["q_index"][sel]])
        if cap:
            logits = softcap(logits, cap)
    if dspark:
        slots = torch.arange(k_max - 1, device=sel.device)[None, :]
        positions = (b.a[:, None] + 1 + slots).clamp(0, p.input_ids.numel() - 1)  # the labels' positions
        with torch.no_grad():  # in parts into one tensor: a readout holds several [rows, V] tensors at once
            states = p.loop_out[R - 1, positions.reshape(-1)]
            target_logits = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
            flat = target_logits.view(-1, logits.shape[-1])
            for start in range(0, states.shape[0], 256):
                flat[start:start + 256] = target.readout_logits(states[start:start + 256])
        mean, metrics = dspark_loss(logits.float(), target_logits, confidence.float(), sel)
        total = mean * n_valid  # speculators divides by the valid slots of this forward
        accept = block_labels(p, b, k_max)[1]
        correct = (logits.argmax(-1) == target_logits.argmax(-1)) & accept
        extra = {"overlap": metrics["accept_rate_sum"], "confidence_error": metrics["confidence_abs_error_sum"]}
    else:
        labels = inp["labels"][sel]
        ce = F.cross_entropy(logits.float(), labels, reduction="none")
        total = (ce * dpace_weights(ce, sel)).sum()
        accept, correct = sel, torch.zeros_like(sel)
        correct[sel] = logits.argmax(-1) == labels
    # 0 * sum keeps every parameter in the graph when no slot is valid, so DDP ranks stay in lockstep.
    loss = total / denom + 0.0 * hidden.float().sum()
    native = target.readout_argmax(p.loop_out[(b.d - 1).clamp(0, R - 1), b.a])
    return loss, dict(correct=correct, valid=accept, native_ok=native == p.final_argmax[b.a], n_valid=n_valid,
                      **extra)


def dpace_weights(ce, valid, alpha=0.5):
    """D-PACE slot weights as in speculators: w_k = sum_{j>=k} prod_{i<=j} s_i over the valid slots of a block,
    s_i = (1 - alpha) q_i + alpha, q_i = exp(-CE_i) the drafter's probability of the label (no gradient).

    A smoothed derivative of the expected accepted length sum_j prod_{i<=j} q_i with respect to log q_k.
    ce holds the valid slots of valid [blocks, slots]; returns weights in the same order.
    """
    s = torch.ones(valid.shape, dtype=torch.float32, device=valid.device)
    s[valid] = (1 - alpha) * torch.exp(-ce.detach().float()) + alpha
    prefix = torch.cumprod(s, dim=1) * valid
    return prefix.flip(1).cumsum(1).flip(1)[valid]


def acceptance_counts(correct, valid):
    """Per-slot conditional acceptance counts; valid slots of a block form a prefix.

    attempted_i: slot i valid and slots 1..i-1 all accepted; accepted_i: also correct.
    """
    ok = correct & valid
    run = torch.cumprod(ok.long(), dim=1).bool()
    prev = torch.cat([torch.ones_like(run[:, :1]), run[:, :-1]], dim=1)
    attempted = valid & prev
    return attempted.sum(0), (attempted & correct).sum(0)
