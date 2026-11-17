"""The last-loop KV sharing rule and the differentiable two-pass (K2) student forward.

Shared rule with window W (W counts the query itself): in every loop except the last, a response
query at position t reads the last-loop post-RoPE KV of the same physical layer for positions
j <= t - W and its own loop's KV for t - W < j <= t. The prompt is computed natively in one pass.
The last loop and the first physical layer of loop 1 always read same-pass causal KV.

K2 (training): P1 is the current student's native no-grad pass, of which only the detached
last-loop KV is kept. P2 computes prompt queries natively and response queries with the
query-local native band plus P1 last-loop KV outside it. Reading P1's last-loop KV for old
positions is the only approximation; the exact model (training.sharekv.replay) reads its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint


_FLEX = None


def compiled_flex():
    global _FLEX
    if _FLEX is None:
        # Train/eval use two fixed sequence sizes, four diagnostic windows, and
        # grad/no-grad variants. The default eight graphs is insufficient.
        # Fullgraph prevents a silent dense eager fallback for a 32K attention.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
        _FLEX = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    return _FLEX


@dataclass
class Layout:
    documents: torch.Tensor
    positions: torch.Tensor
    is_prompt: torch.Tensor

    def validate(self):
        if self.documents.ndim != 2 or self.positions.shape != self.documents.shape:
            raise ValueError("layout tensors must have matching [batch, sequence] shapes")
        if self.is_prompt.shape != self.documents.shape:
            raise ValueError("is_prompt must match document IDs")


class AttentionPlan:
    """Two reusable BlockMasks: native causal and shared-with-native-prompt."""

    def __init__(self, layout: Layout, window: int, backend: str = "flex"):
        layout.validate()
        if window < 1:
            raise ValueError("W includes self and must be >= 1")
        if backend not in ("flex", "dense"):
            raise ValueError("attention backend must be flex or dense")
        self.layout, self.window, self.backend = layout, window, backend
        self.batch, self.length = layout.documents.shape
        docs, pos, prompt = layout.documents, layout.positions, layout.is_prompt
        length = self.length

        def native(b, h, q, k):
            # FlexAttention may visit padded block indices past the sequence.
            qq, kk = q.clamp(max=length - 1), k.clamp(max=length - 1)
            valid = (q < length) & (k < length)
            same = (docs[b, qq] == docs[b, kk]) & (docs[b, qq] >= 0)
            # Pad queries attend only themselves, preventing empty softmax rows.
            pad = (docs[b, qq] < 0) & (q == k)
            return valid & ((same & (pos[b, kk] <= pos[b, qq])) | pad)

        def shared(b, h, q, k):
            qq, kk = q.clamp(max=length - 1), (k % length).clamp(max=length - 1)
            valid = (q < length) & (k < 2 * length)
            same = (docs[b, qq] == docs[b, kk]) & (docs[b, qq] >= 0)
            causal = pos[b, kk] <= pos[b, qq]
            old = pos[b, kk] <= pos[b, qq] - window
            use_memory = (k < length) & ~prompt[b, qq] & old
            use_current = (k >= length) & (prompt[b, qq] | ~old)
            pad = (docs[b, qq] < 0) & (k == q + length)
            return valid & ((same & causal & (use_memory | use_current)) | pad)

        self.native_mod, self.shared_mod = native, shared
        if backend == "flex":
            kw = dict(B=self.batch, H=None, Q_LEN=length, device=docs.device,
                      _compile=True)
            self.native_mask = create_block_mask(native, KV_LEN=length, **kw)
            self.shared_mask = create_block_mask(shared, KV_LEN=2 * length, **kw)
        else:
            if length > 4096:
                raise ValueError("dense attention is a small semantic oracle, not a 32K backend")
            b = torch.arange(self.batch, device=docs.device)[:, None, None]
            q = torch.arange(length, device=docs.device)[None, :, None]
            k = torch.arange(length, device=docs.device)[None, None, :]
            self.native_mask = native(b, 0, q, k)[:, None]
            k = torch.arange(2 * length, device=docs.device)[None, None, :]
            self.shared_mask = shared(b, 0, q, k)[:, None]

    def attend(self, q, k, v, *, shared=False, scale=None):
        mask = self.shared_mask if shared else self.native_mask
        if self.backend == "flex":
            return compiled_flex()(q, k, v, block_mask=mask, scale=scale,
                                   enable_gqa=q.shape[1] != k.shape[1])
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale,
                                              enable_gqa=q.shape[1] != k.shape[1])


def qkv(layer, hidden, rotary):
    a = layer.self_attn
    shape = (*hidden.shape[:-1], -1, a.head_dim)
    x = layer.input_layernorm(hidden)
    q, k, v = (proj(x).view(shape).transpose(1, 2)
               for proj in (a.q_proj, a.k_proj, a.v_proj))
    if getattr(a, "q_layernorm", None) is not None:
        q = a.q_layernorm(q)
    if getattr(a, "k_layernorm", None) is not None:
        k = a.k_layernorm(k)
    cos, sin = (x.unsqueeze(1) for x in rotary)

    def rotate(x):
        left, right = x.chunk(2, -1)
        return torch.cat([-right, left], -1)

    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin, v


def finish_layer(layer, hidden, attention):
    a = layer.self_attn
    attention = attention.transpose(1, 2).reshape(*hidden.shape[:-1], -1).contiguous()
    if hasattr(layer, "input_layernorm_2"):
        hidden = hidden + layer.input_layernorm_2(a.o_proj(attention))
        return hidden + layer.post_attention_layernorm_2(
            layer.mlp(layer.post_attention_layernorm(hidden)))
    # Nanbeige uses ordinary pre-norm residuals, without Ouro's branch norms.
    hidden = hidden + a.o_proj(attention)
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


def loop_count(model):
    body = model.model
    return body.total_ut_steps if hasattr(body, "total_ut_steps") else body._get_num_loops()


def rotary_embeddings(body, hidden, positions):
    if hasattr(body, "rotary_emb"):
        return body.rotary_emb(hidden, positions)
    # All layers of the supported Nanbeige checkpoint use the same RoPE.
    return body.layers[0].self_attn.rotary_emb(hidden, positions)


def pad_token(model):
    value = getattr(model.config, "pad_token_id", None)
    if value is None:
        value = getattr(model.config, "eos_token_id", None)
    return int(value) if value is not None else 0


def layer_forward(layer, hidden, rotary, plan, memory=None):
    q, k, v = qkv(layer, hidden, rotary)
    current = k, v
    if memory is not None:
        k, v = (torch.cat([old, now], dim=2) for old, now in zip(memory, current))
    attention = plan.attend(q, k, v, shared=memory is not None,
                            scale=layer.self_attn.scaling)
    return finish_layer(layer, hidden, attention), current


def native_forward(model, ids, layout, *, backend="flex", capture="final",
                   return_loops=False, layer_hook=None):
    """Caller controls grad mode; teacher/P1 callers explicitly use no_grad.

    layer_hook(loop, index, hidden) sees every layer output, the residual
    stream before the loop-end norm.
    """
    body = model.model
    hidden = body.embed_tokens(ids)
    rotary = rotary_embeddings(body, hidden, layout.positions)
    plan = AttentionPlan(layout, 1, backend)
    loops, banks, states = loop_count(model), {}, []
    for loop in range(loops):
        for index, layer in enumerate(body.layers):
            hidden, pair = layer_forward(layer, hidden, rotary, plan)
            if layer_hook is not None:
                layer_hook(loop, index, hidden)
            if capture == "all" or (capture == "final" and loop == loops - 1):
                banks[(loop, index)] = tuple(x.detach() for x in pair)
        hidden = body.norm(hidden)
        if return_loops:
            states.append(hidden)
    return hidden, banks, states


def k2_forward(model, ids, layout, *, window=1, backend="flex",
               checkpoint_layers=2, return_loops=False, memory=None,
               return_memory=False):
    """P1 native no-grad -> P2 shared grad. Prompt remains native in P2.

    Optional memory exists for semantic tests only; production computes it anew
    from the current student on every microbatch. It must always be detached.
    """
    body = model.model
    if return_memory and torch.is_grad_enabled():
        raise ValueError("Jacobi reference KV capture is inference-only")
    if memory is None:
        with torch.no_grad():
            _, memory, _ = native_forward(model, ids, layout, backend=backend)
    if any(x.requires_grad for pair in memory.values() for x in pair):
        raise ValueError("P1 memory must be detached")
    hidden = body.embed_tokens(ids)
    rotary = rotary_embeddings(body, hidden, layout.positions)
    plan = AttentionPlan(layout, window, backend)
    loops, states, written = loop_count(model), [], {}
    group_size = max(1, checkpoint_layers)
    for loop in range(loops):
        for start in range(0, len(body.layers), group_size):
            stop = min(start + group_size, len(body.layers))

            def group(h, begin=start, end=stop, depth=loop):
                for index in range(begin, end):
                    native = depth == loops - 1 or (depth, index) == (0, 0)
                    old = None if native else memory[(loops - 1, index)]
                    h, pair = layer_forward(body.layers[index], h, rotary, plan, old)
                    if return_memory and depth == loops - 1:
                        written[(depth, index)] = tuple(x.detach() for x in pair)
                return h

            if checkpoint_layers and torch.is_grad_enabled():
                hidden = checkpoint(group, hidden, use_reentrant=False)
            else:
                hidden = group(hidden)
        hidden = body.norm(hidden)
        if return_loops:
            states.append(hidden)
    return (hidden, states, written) if return_memory else (hidden, states)
