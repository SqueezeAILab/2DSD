"""Depth-conditioned DFlash drafter on speculators' Qwen3 DFlash layers (target features injected as K/V).

The drafter has the target's width; fc(F -> H) + RMSNorm projects the concatenated feature layers to
the context that every drafter layer reads as K/V. The target's embedding and LM head are shared and
frozen; the mask vector is learned. Loss: CE on the target's last-loop argmax with D-PACE slot weights.
For Dory the head is its full-vocabulary nGPT head (FP32 scale after the GEMM), which expects unit-norm
inputs, so the drafter's final RMSNorm weight starts at 1/sqrt(H) instead of 1 (still trained).
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.utils import flatten_rope_parameters
from torch.nn.attention.flex_attention import flex_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .blocks import build_context_inputs, build_context_mask

MASK_TOKEN = 16  # embedding row that initialises the learned mask (Ouro <empty_output>, never generated)
DORY_MASK_TOKEN = 999  # Dory's reserved <SPECIAL_999>, never emitted; also its exported mask_token_id
LOSS = "ce-dpace"


# speculators 0.9.0 calls flex_attention eagerly, which materializes dense scores (24 GiB for one
# 256-block micro-batch). The compiled kernel keeps the block-sparse path.
_compiled_flex_attention = torch.compile(flex_attention)


def compiled_flex_attention_forward(module, query, key, value, attention_mask, scaling=None, **_kwargs):
    out = _compiled_flex_attention(query.contiguous(), key.contiguous(), value.contiguous(), block_mask=attention_mask,
                                   enable_gqa=query.shape[1] != key.shape[1], scale=scaling)
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("loopedtf_flex_attention", compiled_flex_attention_forward)


def drafter_config(num_layers, target_config):
    t = target_config
    dory = getattr(t, "model_type", None) == "dory"
    cfg = Qwen3Config(
        hidden_size=t.hidden_size, intermediate_size=t.intermediate_size, num_hidden_layers=num_layers,
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
    def __init__(self, cfg, n_feat_layers, init_std=0.02):
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

    def forward(self, ctx_raw, q_emb, q_is_mask, position_ids, block_mask):
        """ctx_raw [1, C, F] target features, q_emb [1, Q, H], q_is_mask [Q], position_ids [1, C + Q]."""
        ctx = self.hidden_norm(self.fc(ctx_raw))
        h = torch.where(q_is_mask[None, :, None], self.mask_emb.to(q_emb.dtype), q_emb)
        pe = self.rotary(h, position_ids)
        for layer in self.layers:
            h = layer(hidden_states=h, target_hidden=ctx, attention_mask=block_mask,
                      position_ids=position_ids, position_embeddings=pe)
        return self.norm(h)


def drafter_forward(drafter, target, p, b, R, k_max, n_total=None):
    """D-PACE-weighted CE over valid block slots, and per-slot correctness.

    The label is the target's last-loop argmax; D-PACE slot weights follow speculators' DFlash
    default. n_total normalizes the loss when blocks are split into micro-batches.
    """
    inp = build_context_inputs(p, b, R, k_max)
    block_mask, _ = build_context_mask(p, b, R, k_max)
    ctx_raw = p.feats.reshape(-1, p.feats.shape[-1])[None]
    position_ids = torch.cat([p.pos.repeat(R), inp["q_pos"]])[None]
    q_emb = target.embed(inp["q_tok"])[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = drafter(ctx_raw, q_emb, inp["q_is_mask"], position_ids, block_mask)[0]
        sel = inp["lab_valid"]
        logits = target.lm_head(hidden[inp["q_index"][sel]])
    labels = inp["labels"][sel]
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    weight = dpace_weights(ce, sel)
    n_valid = sel.sum()
    denom = n_valid.clamp(min=1) if n_total is None else max(int(n_total), 1)
    # 0 * sum keeps every parameter in the graph when no slot is valid, so DDP ranks stay in lockstep.
    loss = (ce * weight).sum() / denom + 0.0 * hidden.float().sum()
    correct = torch.zeros_like(sel)
    correct[sel] = logits.argmax(-1) == labels
    native = target.readout_argmax(p.loop_out[(b.d - 1).clamp(0, R - 1), b.a])
    return loss, dict(correct=correct, valid=sel, native_ok=native == p.final_argmax[b.a], n_valid=n_valid)


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
