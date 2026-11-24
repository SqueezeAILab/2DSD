# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Looped Inkling2 (Inkling-2-Mini) for teacher-forced features, per-depth readouts and the lossless reference.

Runs the tensors of vllm.model_executor.models.inkling2_weights (the fork has no Inkling2 model class and
Transformers 5.17 no Inkling2 model). Written from vLLM PR #69028 at b426b46e (V/ = vllm/models/inkling2/nvidia/,
the function served on Hopper) and transformers PR #51050 at 6173a2f (T = src/transformers/models/inkling2_loop/
modeling_inkling2_loop.py); nothing is imported from either. Every formula was read in both; where they differ
numerically this module follows V/ (list at the end).

Schedule (V/loop.py:339-342, T:1363-1440). Layers [0, core_start) run once (prelude), the core [core_start,
core_end) num_passes times, layers [core_end, L) once (coda): 6 + 3 * 12 + 6 = 48 layer visits on Mini, visit
ids 0-5, 6-17 / 18-29 / 30-41 and 42-47, with 48 standing for the final-norm output. A pass is 1-based here and
equals the readout depth it completes.

State (V/model.py:5-8). There is no residual stream. A token holds completed blocks (block 0 is the normed
embedding) and a running partial block, the sum of the layer outputs of the current block of
n_layers_per_attn_res_block layers. Every sublayer input is attn_res: with x_n the blocks and the partial,
logit_n = (q . x_n) rsqrt(mean(x_n^2) + 1e-6) for a query q scaled by H ** -0.5, mix = sum_n softmax_n x_n
rsqrt(mean(x_n^2) + 1e-6) (V/ops/attn_res.py:85-116, T:838-853). Same token only.

Layer i (V/loop.py:199-316, T:876-910):
    x      = norm(attn_res(blocks, partial; attn_queries[i - 1]))   one mix on a sliding-window layer, separate
                                                                    q / k / v mixes and norms on a full one;
                                                                    layer 0 is norm(embedding) with no mix
    a      = conv(attention(x))
    y      = norm_mlp(attn_res(blocks, partial + a; mlp_queries[i]))
    partial = (partial + a) + conv(moe(y))
The first layer of a block closes the previous one instead: the partial joins the blocks before the MLP mix,
whose last source is a alone, and the new partial is a + conv(moe(y)).

Attention (V/attention.py:362-400, V/ops/sconv.py:706-813, V/ops/norm.py:73-82 and 116-185,
V/ops/attention_sm90.py:153-231, V/ops/attention.py:272-299, T:111-142 and 176-270). q, the relative vector r,
the output gate and the sink logit come from the q input, k from the k input and v from the v input. q and r
are multiplied by a learned per-head scale and L2-clipped (qk_l2_clip_threshold, rel_l2_clip_threshold); on
full-attention layers both are then multiplied by tau = 1 + alpha log(max((pos + 1) / n_floor, 1)). k and v go
through a short conv before caching; then k is L2-clipped (no scale) and v RMS-normalized per head without a
weight. For backward distance d = pos_q - pos_k the score is q . k head_dim ** -0.5 + (r @ rel_proj)[d]; keys
with d < 0, and on a sliding layer d >= window, are masked (a sliding layer sees itself and window - 1 earlier
keys); a full layer keeps a key with d >= rel_extent and gives it zero bias. The output is
softmax-weighted v times sigmoid(gate) times sigmoid(logsumexp(scores) - sink), i.e. the sink is one more
logit with a zero value, followed by the output projection.

Short conv (V/ops/sconv.py:12-22 and 96-183, T:306-325 and 396-455), kernel W = 4, causal, zero before position
0, on four streams per layer: the k and v projections, the attention output and the MoE output.
    static   y[t] = (sum_k w[c, k] x[t - (W - 1) + k] + x[t]) (W + 1) ** -0.5
    dynamic  y[t] = (sum_l (w[l, c] + dyn[t, head(c), l]) x[t - l] + x[t]) (2 W + 1) ** -0.5
The dynamic form is the k and v conv of full-attention layers, dyn projected from the token's own k or v input.
The state of a stream is its last W - 1 inputs.

MoE (V/moe.py:261-312 and 385-395, V/ops/moe.py:20-64 and 109-145, V/ops/moe_sm90.py:111-157 and 286-316,
V/ops/activation.py:47-78, T:787-811 and 1163-1172). Router logits are x @ gate.weight.T; the top-k experts by
sigmoid(logit) + bias are selected, ties to the lower index; their weights are the sigmoids alone renormalized
over the selection, times route_scale. An expert is down(softcap_4(g) sigmoid(g) softcap_25(u)) with (g, u) the
interleaved rows of its gate/up projection, softcap_c(z) = c tanh(z / c) and the sigmoid on the raw gate.

Loop (V/loop.py:347-393, V/ops/loop.py:5-8, T:89-96 and 1398-1424). Every pass restarts from the prelude's
blocks and partial. Its only input from the previous pass is the bank, that pass's core blocks of the same
token ([B2, B3] on Mini). In passes after the first, each core layer reads four messages (streams attn, k, v,
mlp) as attn_res over the bank, and each of its inputs is the controller's blend of the mix before its norm
with a message: with h = down([rms(dest), rms(message)]) of rank gate_rank,
(1 - tanh(forget(h))) dest + tanh(gain(h)) message. A sliding layer then has three different q / k / v inputs
from its one mix. The first pass has no message and bypasses the controller (a zero message is not a bypass).
No parameter depends on the pass; a pass selects only its own keys, values and conv states.

Readout (V/loop.py:374-397, V/model.py:583-587, T:1442-1443 and 1533-1535). The coda continues from the last
pass's state; the output is norm(attn_res(blocks, partial; output_query)), the unembedding over the served
vocabulary (the padding rows of the table are never predicted) and softcap 30 tanh(x / 30). A depth-p readout
is the coda on the state after pass p with keys, values and conv states of its own, which is the model
configured with p passes; upstream only offers the latter. forward() returns every requested depth from one
run of the prelude and the core.

Precision. "served" keeps FP32 arithmetic and rounds to BF16 where the Hopper path stores an activation; these
are all of its rounding points: the embedding norm, every sublayer input after its norm, the controller's
normalized pair and rank-r hidden, all attention projections and dynamic taps, q and r after the clip and again
after tau, the relative logits, k and v after the conv and again after the clip / norm, the attention
probabilities entering the product with v, the gated output, the output projection, both hidden-stream conv
outputs, the partial block, an expert's gate/up row, its activation (which carries the routing weight) and its
down row, and the sum of the selected experts. FP32 and never rounded: mixes, messages, blends, scores, router
logits and weights, the FP32 accumulator of an expert product times its FP32 scale, the logits. "exact" is
FP64 with no rounding; "served-fp64" keeps the rounding points with FP64 between them and "exact-fp32" drops
them in FP32. The weights are exact in every case (inkling2_weights). An FP32 product depends in its last bit
on the shape of the batch, which now and then moves a BF16 value by one step, so "served" is the same function
for every chunking only up to BF16 noise (as upstream is between prefill and decode); "served-fp64" rounds the
same quantities from FP64 and gives the same BF16 states however the tokens are chunked.

Where the sources differ, and what is done here:
  - Routing weight: V/ applies it inside the activation, before the BF16 rounding and the down projection
    (V/moe.py:272-273); T after the down projection (T:804). Here: as V/.
  - Weight normalization: V/ exp(logsigmoid - logsumexp) with the pairwise order of V/ops/moe.py:21-28 for
    eight experts; T sigmoid / (sum + 1e-20) (T:1170-1171). Here: as V/.
  - Top-k ties: V/ lower index; T torch.topk. Here: as V/.
  - Expert weights: V/ keeps the per-expert FP32 scale apart and applies it to the FP32 accumulator; T folds it
    into the dense weight and casts (core_model_loading.py:814-815). Here: as V/.
  - Scores and logits: V/ accumulates scores in FP32 and returns FP32 logits; T computes scores and the head
    in the model dtype and casts (T:126, T:1534). Here: as V/.
  - Sink: V/ sigmoid(logsumexp - sink) on the output; T an extra softmax column (T:132-135). Same function.
  - Bank: V/ reads exactly two bank blocks (V/loop.py:228-230); T all core blocks of the previous pass
    (T:1400-1401). Here: all core blocks, the same on Mini.
  - Controller RMS epsilon: V/ the constant 1e-6; T rms_norm_eps. Equal on the released checkpoints; here the
    loader's message_norm_eps (V/).
  - Pass count: V/ refuses a non-default count on a fixed-loop checkpoint; T accepts any. Here: any.
  - Layer kinds: V/ reads local_layer_ids; T reads layer_types and defaults to full attention on every sixth
    layer. The same on the released checkpoints; here local_layer_ids (the loader).
Not reproduced, all below BF16 rounding: the order of the FP32 accumulation inside cuBLAS, Triton and Humming
GEMMs, fused multiply-adds, approximate rsqrt / exp / tanh, and the key-block order of the attention kernel
(its probabilities are rounded relative to the running maximum of the blocks seen so far; here relative to the
row maximum). Upstream itself is not bitwise equal between prefill and decode. Not implemented: the activation
quantization of the Blackwell path (Precision.activation_quantization is a placeholder and raises) and
loop_share_cache, the serving option of variable-loop checkpoints that gives all passes of a core layer one KV
cache (V/attention.py:177-179).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from vllm.model_executor.models.inkling2_weights import (
    GATE_ROWS,
    LLM_PREFIX,
    LOOP_STREAMS,
    UP_ROWS,
    Inkling2PackedExperts,
    layer_prefix,
    load_inkling2_checkpoint,
)

ROUTER_GROUP = 1024  # router logits accumulate in ordered groups of this many inputs (V/ops/linear.py:275-282)


@dataclass(frozen=True)
class Precision:
    """Arithmetic of the forward: `dtype` between rounding points, `storage` at them (None: no rounding)."""

    name: str
    dtype: torch.dtype
    storage: torch.dtype | None
    # Blackwell quantizes the expert inputs to NVFP4 / MXFP8; Hopper does not. Not implemented.
    activation_quantization: bool = False

    def store(self, x):
        """x at a rounding point: rounded once to the storage dtype, still held in the arithmetic dtype."""
        return x if self.storage is None else x.to(self.storage).to(self.dtype)


PRECISIONS = {
    "served": Precision("served", torch.float32, torch.bfloat16),
    "served-fp64": Precision("served-fp64", torch.float64, torch.bfloat16),
    "exact": Precision("exact", torch.float64, None),
    "exact-fp32": Precision("exact-fp32", torch.float32, None),
}


def drafter_feature_layers(num_layers):
    """Layers whose inputs are a depth's drafter features, by DFlash's rule (also DSpark's) along that
    depth's path of num_layers layers (prelude, one core pass, coda): the states after layers
    round(1 + j (L - 4) / 4), j = 0..4. Inkling 2 has no residual stream, so the state after layer l is read
    as the input of layer l + 1 (the running partial block). Mini: the inputs of layers 2 (prelude), 7, 12
    and 17 (the depth's core pass) and 22 (the coda on that pass's state), as Dory's taps lie in its input
    block, its recurrent block and its output block."""
    return sorted({round(1 + j * (num_layers - 4) / 4) + 1 for j in range(5)})


def visit_layers(config, num_passes=None):
    """(physical layer, pass) of every layer visit in execution order; the prelude and the coda are pass 0."""
    passes = config.num_passes if num_passes is None else num_passes
    core = config.core_layers
    return ([(i, 0) for i in range(core.start)] + [(i, p) for p in range(1, passes + 1) for i in core]
            + [(i, 0) for i in range(core.stop, config.num_hidden_layers)])


def num_layer_visits(config, num_passes=None):
    """Layer visits of the model run with num_passes passes; this id is its final-norm output."""
    return len(visit_layers(config, num_passes))


def rms(x, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def l2_clip(x, threshold, eps):
    """x scaled down to L2 norm `threshold` over the last axis when longer."""
    norm = x.square().sum(-1, keepdim=True).sqrt()
    return x * (threshold / (norm + eps)).clamp(max=1.0)


def softcap(x, cap):
    return x if cap is None or cap <= 0 else cap * torch.tanh(x / cap)


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def attn_res(blocks, last, queries, eps):
    """Block attention residual: mixes [Q, T, H] of the sources [*blocks, last] ([T, H] each) for the scaled
    queries [Q, H], before any norm. Sources are scored and mixed RMS-normalized."""
    x = torch.stack((*blocks, last))
    inv_rms = torch.rsqrt(x.square().mean(-1) + eps).T  # [T, N]
    logits = torch.einsum("ntd,qd->qtn", x, queries) * inv_rms
    probs = torch.exp(logits - logits.amax(-1, keepdim=True))
    return torch.einsum("qtn,ntd->qtd", probs * inv_rms, x) * (1.0 / probs.sum(-1, keepdim=True))


def short_conv(x, weight, history=None, dynamic=None):
    """One stream's causal short conv with skip, before rounding.

    x [T, C] are the stream's inputs at consecutive positions and history [W - 1, C] those of the positions
    just before (None at the start of a sequence: inputs before position 0 are zero). Static: weight [C, W],
    oldest tap first. Dynamic: weight [W, C] and dynamic [T, heads, W], newest tap first, the per-token deltas
    shared within a head. Returns y [T, C] and the history of the next call.
    """
    kernel = weight.shape[1] if dynamic is None else weight.shape[0]
    tokens, channels = x.shape
    if history is None:
        history = x.new_zeros(kernel - 1, channels)
    context = torch.cat((history, x))  # context[kernel - 1 + t] is x[t]
    acc = torch.zeros_like(x)
    if dynamic is None:
        for k in range(kernel):
            acc = acc + context[k:k + tokens] * weight[:, k]
        scale = (kernel + 1) ** -0.5
    else:
        head_dim = channels // dynamic.shape[1]
        for lag in range(kernel):
            taps = weight[lag] + dynamic[:, :, lag].repeat_interleave(head_dim, dim=1)
            acc = acc + context[kernel - 1 - lag:kernel - 1 - lag + tokens] * taps
        scale = (2 * kernel + 1) ** -0.5
    # A copy: a view would keep the whole chunk alive for as long as the history is cached.
    return (acc + x) * scale, context[tokens:].clone()


def relative_attention(query, keys, values, rel_states, rel_proj, gate, sink, *, first_query=0, first_key=0,
                       window=None, precision=PRECISIONS["exact"], max_elements=1 << 26):
    """Gated relative-bias attention of consecutive queries over consecutive cached keys, before rounding.

    query and gate [T, heads, D], rel_states [T, heads, d_rel] (clipped), rel_proj [d_rel, extent], sink
    [T, heads]; keys and values [N, kv_heads, D] as cached (clipped keys, normalized values). first_query and
    first_key are the absolute positions of query[0] and keys[0]; head h reads key head h // (heads / kv_heads).
    window is None for full attention. Queries are processed in chunks of at most max_elements scores, and a
    sliding layer loads only the keys a chunk can see. Returns [T, heads, D].
    """
    tokens, heads, dim = query.shape
    kv_heads, extent = keys.shape[1], rel_proj.shape[1]
    group = heads // kv_heads
    if window is None:
        chunk = max(1, max_elements // (heads * max(1, keys.shape[0])))
    else:
        chunk = max(1, int((math.sqrt(window * window + 4 * max_elements / heads) - window) / 2))
    out = query.new_empty(tokens, heads, dim)
    for a in range(0, tokens, chunk):
        b = min(tokens, a + chunk)
        low = first_key if window is None else max(first_key, first_query + a - (window - 1))
        high = first_query + b
        k, v = keys[low - first_key:high - first_key], values[low - first_key:high - first_key]
        dist = (torch.arange(first_query + a, high, device=query.device)[:, None]
                - torch.arange(low, high, device=query.device)[None, :])  # [queries, keys]
        visible = dist >= 0
        if window is not None:
            visible &= dist < window
        scores = torch.einsum("qhgd,nhd->qhgn", query[a:b].view(b - a, kv_heads, group, dim), k) * dim ** -0.5
        rel = precision.store(rel_states[a:b] @ rel_proj)  # [queries, heads, extent]
        bias = rel.gather(2, dist.clamp(0, extent - 1)[:, None, :].expand(-1, heads, -1))
        bias = bias.masked_fill(((dist < 0) | (dist >= extent))[:, None, :], 0.0)
        scores = scores + bias.view(b - a, kv_heads, group, -1)
        scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
        top = scores.amax(-1, keepdim=True)
        probs = torch.exp(scores - top)
        total = probs.sum(-1, keepdim=True)
        mixed = torch.einsum("qhgn,nhd->qhgd", precision.store(probs), v) / total
        lse = (top + torch.log(total)).reshape(b - a, heads)
        out[a:b] = mixed.reshape(b - a, heads, dim) * sigmoid(gate[a:b]) * sigmoid(lse - sink[a:b])[..., None]
    return out


def route_experts(logits, bias, top_k, route_scale):
    """Router: ids [T, top_k] of the experts with the largest sigmoid(logit) + bias, best first and ties to the
    lower index, and their weights [T, top_k]: the sigmoids renormalized over the selection, times route_scale."""
    ids = torch.sort(sigmoid(logits) + bias, dim=-1, descending=True, stable=True).indices[:, :top_k]
    selected = logits.gather(1, ids)
    log_probs = selected.clamp(max=0.0) - torch.log1p(torch.exp(-selected.abs()))
    top = log_probs.amax(-1, keepdim=True)
    shifted = torch.exp(log_probs - top)
    if top_k == 8:
        s = shifted.unbind(-1)
        total = ((s[0] + s[2]) + (s[1] + s[3])) + ((s[4] + s[6]) + (s[5] + s[7]))
    else:
        total = shifted.sum(-1)
    return ids, torch.exp(log_probs - (top + torch.log(total)[:, None])) * route_scale


def situ(gate, up, weight, gate_cap, up_cap):
    """Softcapped SwiGLU of an expert times its routing weight: the sigmoid sees the raw gate."""
    return ((softcap(gate, gate_cap) * sigmoid(gate)) * softcap(up, up_cap)) * weight


@dataclass
class Inkling2Plane:
    """Keys, values and short-conv state of one layer in one pass (or, for a coda layer, one readout depth).

    keys and values [N, kv_heads, D] are those cached for positions first .. first + N - 1 (all positions on a
    full-attention layer, the last window - 1 on a sliding one); conv maps a stream ("k", "v", "attn", "mlp")
    to its last kernel - 1 inputs.
    """

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    first: int = 0
    conv: dict = field(default_factory=dict)


class Inkling2Cache:
    """State of incremental decoding: planes[(layer, index)] with index 0 for a prelude layer, the pass for a
    core layer and the readout depth for a coda layer, and the number of tokens already processed. A cache
    belongs to one model, one precision and one (num_passes, depths)."""

    def __init__(self):
        self.length = 0
        self.planes = {}
        self.signature = None

    def bind(self, signature):
        if self.signature not in (None, signature):
            raise ValueError(f"the cache was written with (num_passes, depths) = {self.signature}, not {signature}")
        self.signature = signature

    def plane(self, layer, index):
        return self.planes.setdefault((layer, index), Inkling2Plane())


@dataclass
class Inkling2Readout:
    """Readout at one depth: the coda on the state after that pass, with keys, values and conv states of its own.

    hidden [T, H] is the final-norm output (the unembedding's input), argmax [T] the served-vocabulary argmax,
    logits [T, V] the softcapped logits if requested, experts [coda layers, T, top_k] the routed expert ids,
    taps the inputs [T, H] of the requested coda layers at this depth.
    """

    hidden: torch.Tensor
    argmax: torch.Tensor
    logits: torch.Tensor | None
    experts: torch.Tensor
    taps: dict = field(default_factory=dict)


@dataclass
class Inkling2Output:
    """Result of NativeInkling2.forward.

    readouts[d] for every requested depth; depth num_passes is the model's own output. features[v] [T, H] is
    the running partial block at the input of visit v of the num_passes model, as upstream captures it, and
    features[num_layer_visits] its final-norm output. At a pass restart that is the prelude's block again, so
    the state a pass ends in is pass_end[p] [core blocks, T, H] ([B2_p, B3_p] on Mini). experts
    [prelude + core visits, T, top_k] are the routed expert ids of the visits before the coda.
    """

    num_passes: int
    core: range
    num_hidden_layers: int
    readouts: dict
    features: dict
    pass_end: dict
    experts: torch.Tensor

    def routed_experts(self, depth):
        """Expert ids [visits, T, top_k] of the model with `depth` passes in visit order, and the physical
        layer of every visit."""
        core = list(self.core)
        layers = [*range(self.core.start), *core * depth, *range(self.core.stop, self.num_hidden_layers)]
        return torch.cat((self.experts[:self.core.start + depth * len(core)], self.readouts[depth].experts)), layers


@dataclass
class _Run:
    positions: torch.Tensor
    start: int
    cache: Inkling2Cache | None

    def plane(self, layer, index):
        return None if self.cache is None else self.cache.plane(layer, index)


class NativeInkling2:
    """Reference forward over an Inkling2Checkpoint.

    forward(tokens) is full-sequence teacher forcing; forward(chunk, cache=cache) continues a sequence from an
    Inkling2Cache, one token or several at a time, and gives the same function. Activations are held in the
    arithmetic dtype of `precision` (a name of PRECISIONS or a Precision); the checkpoint may be loaded in any
    dtype and with dense or packed experts (packed dequantizes only the routed experts, on every use).
    """

    def __init__(self, checkpoint, precision="served", attention_elements=1 << 26, logit_elements=1 << 26):
        self.checkpoint, self.config = checkpoint, checkpoint.config
        self.precision = PRECISIONS[precision] if isinstance(precision, str) else precision
        if self.precision.activation_quantization:
            raise NotImplementedError("Blackwell activation quantization (NVFP4 / MXFP8 expert inputs) is not "
                                      "implemented; the reference follows the Hopper path (BF16 expert inputs)")
        c = self.config
        if checkpoint.expert_mode == "none" or len(checkpoint.layers) != c.num_hidden_layers:
            raise ValueError("the forward needs every layer of the checkpoint and its experts")
        if any(heads % kv_heads for heads, kv_heads, _ in map(c.attention_dims, range(c.num_hidden_layers))):
            raise ValueError("attention heads must be a multiple of key/value heads")
        self.device = checkpoint.device
        self.attention_elements, self.logit_elements = attention_elements, logit_elements
        self._unembedding = None

    @classmethod
    def from_pretrained(cls, directory, device="cpu", precision="served", *, dtype=torch.bfloat16, experts="dense",
                        **options):
        """Model of a snapshot directory; dtype and experts are those of load_inkling2_checkpoint and options
        the constructor's."""
        return cls(load_inkling2_checkpoint(directory, device=device, dtype=dtype, experts=experts), precision,
                   **options)

    # Weights, in the arithmetic dtype. BF16 and FP32 values are exact in FP32 and FP64.

    def _w(self, name):
        return self.checkpoint.tensors[LLM_PREFIX + name].to(self.precision.dtype)

    def _lw(self, layer, name):
        return self.checkpoint.tensors[layer_prefix(layer) + name].to(self.precision.dtype)

    def _linear(self, x, layer, name):
        return x @ self._lw(layer, name).T

    def _query(self, name):
        """AttnRes pseudo-queries [Q, H], scaled by H ** -0.5 as upstream scales them at load."""
        hidden = self.config.hidden_size
        return self._w(name).reshape(-1, hidden) * hidden ** -0.5

    def _message_queries(self, layer):
        """The four message queries of a core layer, in LOOP_STREAMS order (attn, k, v, mlp)."""
        row = 2 * (layer - self.config.core_start)
        queries, kv = self._w("middle_cycle_cross_pass_queries"), self._w("middle_cycle_cross_pass_kv_queries")
        return torch.stack((queries[row], kv[row], kv[row + 1], queries[row + 1])) * self.config.hidden_size ** -0.5

    def _norm(self, x, weight):
        return self.precision.store(rms(x, self.config.rms_norm_eps) * weight)

    # Sublayers.

    def _blend(self, layer, stream, dest, message):
        """Controller of one stream: the mix `dest` before its norm blended with `message`, both [T, H]."""
        p, eps = self.precision, self.config.message_norm_eps
        prefix = f"controllers.{stream}."
        pair = p.store(torch.cat((rms(dest, eps), rms(message, eps)), -1))
        hidden = p.store(self._linear(pair, layer, prefix + "down_weight"))
        forget = torch.tanh(self._linear(hidden, layer, prefix + "forget_weight"))
        return (1.0 - forget) * dest + torch.tanh(self._linear(hidden, layer, prefix + "gain_weight")) * message

    def _conv(self, plane, stream, x, weight, dynamic=None):
        y, history = short_conv(x, weight, None if plane is None else plane.conv.get(stream), dynamic)
        if plane is not None:
            plane.conv[stream] = history
        return self.precision.store(y)

    def _key_value(self, layer, plane, stream, x):
        """The k or v projection of x after its short conv: [T, kv_heads, D], before the clip or norm."""
        c, p = self.config, self.precision
        _, kv_heads, dim = c.attention_dims(layer)
        projected = p.store(self._linear(x, layer, f"attn.w{stream}_dv.weight"))
        name = f"attn.{stream}_sconv."
        if c.is_local(layer):
            y = self._conv(plane, stream, projected, self._lw(layer, name + "weight").squeeze(1))
        else:
            taps = p.store(self._linear(x, layer, name + "weight_proj")).view(-1, kv_heads, c.sconv_kernel_size)
            y = self._conv(plane, stream, projected, self._lw(layer, name + "static_w"), taps)
        return y.view(-1, kv_heads, dim)

    def _attention(self, run, layer, plane, inputs):
        """Attention of `layer` on its normed inputs ([x] or [x_q, x_k, x_v]): [T, H] after the output
        projection. Appends the tokens' keys and values to `plane`."""
        c, p = self.config, self.precision
        x_q, x_k, x_v = inputs if len(inputs) == 3 else inputs * 3
        heads, _, dim = c.attention_dims(layer)
        local, tokens = c.is_local(layer), x_q.shape[0]
        tau = None
        if not local and c.log_scaling_n_floor is not None:
            ratio = (run.positions + 1).to(p.dtype) * (1.0 / c.log_scaling_n_floor)
            tau = (1.0 + c.log_scaling_alpha * torch.log(ratio.clamp(min=1.0)))[:, None, None]

        def clipped(name, width, scale, threshold):
            x = p.store(self._linear(x_q, layer, f"attn.{name}.weight")).view(tokens, heads, width)
            x = p.store(l2_clip(x * self._lw(layer, f"attn.{scale}.scale"), threshold, c.l2_clip_eps))
            return x if tau is None else p.store(x * tau)

        query = clipped("wq_du", dim, "q_l2_clip", c.qk_l2_clip_threshold)
        rel_states = clipped("wr_du", c.d_rel, "rel_l2_clip", c.rel_l2_clip_threshold)
        gate = p.store(self._linear(x_q, layer, "attn.gate_attn.weight")).view(tokens, heads, dim)
        sink = p.store(self._linear(x_q, layer, "attn.attn_sink_proj.weight"))
        if plane is not None:  # refused before the plane's conv state advances
            held = 0 if plane.keys is None else plane.first + plane.keys.shape[0]
            if held != run.start:
                raise ValueError(f"the cache of layer {layer} ends before position {held}, not {run.start}")
        keys = p.store(l2_clip(self._key_value(layer, plane, "k", x_k), c.qk_l2_clip_threshold, c.l2_clip_eps))
        values = p.store(rms(self._key_value(layer, plane, "v", x_v), c.rms_norm_eps))
        first_key = run.start
        if plane is not None and plane.keys is not None:
            keys, values = torch.cat((plane.keys, keys)), torch.cat((plane.values, values))
            first_key = plane.first
        window = c.sliding_window_size if local else None
        out = self._attend(query, keys, values, rel_states, self._lw(layer, "attn.rel_logits_proj.proj"),
                           gate, sink, first_query=run.start, first_key=first_key, window=window)
        if plane is not None:
            # The next query is at run.start + tokens and a sliding layer shows it window - 1 earlier keys.
            drop = max(0, run.start + tokens - (window - 1) - first_key) if local else 0
            if drop:  # copies, so the plane does not keep the dropped keys' memory
                keys, values = keys[drop:].clone(), values[drop:].clone()
            plane.keys, plane.values, plane.first = keys, values, first_key + drop
        return p.store(self._linear(out.view(tokens, -1), layer, "attn.wo_ud.weight"))

    def _attend(self, query, keys, values, rel_states, rel_proj, gate, sink, *, first_query, first_key, window):
        """The gated attention output [T, heads, D] at its rounding point (relative_attention)."""
        return self.precision.store(relative_attention(
            query, keys, values, rel_states, rel_proj, gate, sink, first_query=first_query, first_key=first_key,
            window=window, precision=self.precision, max_elements=self.attention_elements))

    def _expert(self, experts, index):
        """Dense factors and FP32 scales (w13, scale, w2, scale) of one expert, in the arithmetic dtype."""
        dtype = self.precision.dtype
        if isinstance(experts, Inkling2PackedExperts):
            (w13, s13), (w2, s2) = (experts.dequantize(name, index, dtype) for name in ("w13", "w2"))
            return w13[0], s13[0].to(dtype), w2[0], s2[0].to(dtype)
        return (experts.w13[index].to(dtype), experts.w13_scale[index].to(dtype),
                experts.w2[index].to(dtype), experts.w2_scale[index].to(dtype))

    def _moe(self, layer, x):
        """Routed experts on the normed MLP input x [T, H]: output [T, H] and expert ids [T, top_k]."""
        c, p = self.config, self.precision
        gate = self._lw(layer, "mlp.gate.weight")
        logits = sum(x[:, k:k + ROUTER_GROUP] @ gate[:, k:k + ROUTER_GROUP].T
                     for k in range(0, c.hidden_size, ROUTER_GROUP))
        ids, weights = route_experts(logits, self._lw(layer, "mlp.gate.bias"), c.num_experts_per_tok, c.route_scale)
        experts = self.checkpoint.experts(layer)
        # One row per (token, slot), held as it is stored.
        rows = x.new_zeros((x.shape[0], c.num_experts_per_tok, c.hidden_size), dtype=p.storage or p.dtype)
        for index in torch.unique(ids).tolist():
            token, slot = torch.nonzero(ids == index, as_tuple=True)
            w13, s13, w2, s2 = self._expert(experts, index)
            gate_up = p.store((x[token] @ w13.T) * s13)
            activated = p.store(situ(gate_up[:, GATE_ROWS], gate_up[:, UP_ROWS], weights[token, slot, None],
                                     c.mlp_gate_softcap, c.mlp_up_softcap))
            rows[token, slot] = p.store((activated @ w2.T) * s2).to(rows.dtype)
        out = rows[:, 0].to(p.dtype)
        for slot in range(1, c.num_experts_per_tok):
            out = out + rows[:, slot].to(p.dtype)
        return p.store(out), ids

    def _layer(self, run, layer, index, blocks, partial, bank):
        """One visit of `layer` on a token state (completed blocks, running partial), with its plane `index`
        and, in a pass after the first, the bank of the previous pass. Returns the new state and the routed
        expert ids. The state tensors are never modified in place."""
        c, p = self.config, self.precision
        plane = run.plane(layer, index)
        local = c.is_local(layer)
        names = ("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm")
        norms = [self._lw(layer, name + ".weight") for name in names]
        messages = None
        if layer == 0:
            inputs = [self._norm(partial, weight) for weight in norms]
        else:
            mixes = attn_res(blocks, partial, self._query(f"attn_res.attn_queries.{layer - 1}"), c.attn_res_eps)
            if bank is None:
                inputs = [self._norm(mix, weight) for mix, weight in zip(mixes, norms)]
            else:
                messages = attn_res(bank[:-1], bank[-1], self._message_queries(layer), c.attn_res_eps)
                # A sliding layer blends its one mix with the q, k and v messages and shares one norm.
                inputs = [self._norm(self._blend(layer, stream, mixes[0 if local else s], messages[s]),
                                     norms[0 if local else s]) for s, stream in enumerate(LOOP_STREAMS[:3])]
        attn = self._attention(run, layer, plane, inputs)
        attn = self._conv(plane, "attn", attn, self._lw(layer, "attn_sconv.weight").squeeze(1))
        if layer % c.n_layers_per_attn_res_block == 0:
            blocks, last = [*blocks, partial], attn
        else:
            last = partial + attn
        mix = attn_res(blocks, last, self._query(f"attn_res.mlp_queries.{layer}"), c.attn_res_eps)[0]
        if messages is not None:
            mix = self._blend(layer, LOOP_STREAMS[3], mix, messages[3])
        mlp, ids = self._moe(layer, self._norm(mix, self._lw(layer, "mlp_norm.weight")))
        mlp = self._conv(plane, "mlp", mlp, self._lw(layer, "mlp_sconv.weight").squeeze(1))
        return blocks, p.store(last + mlp), ids

    # Embedding and readout head.

    def embed_tokens(self, tokens):
        """Token ids [T] -> the normed embeddings [T, H] that start the prelude."""
        return self._norm(self.checkpoint.embedding[tokens].to(self.precision.dtype), self._w("embed_norm.weight"))

    def readout_hidden(self, blocks, partial):
        """A token state after the coda -> final-norm states [T, H], the unembedding's input."""
        mix = attn_res(blocks, partial, self._query("attn_res.output_query"), self.config.attn_res_eps)[0]
        return self._norm(mix, self._w("norm.weight"))

    def readout_logits(self, hidden):
        """Final-norm states [N, H] -> softcapped logits [N, V] over the served vocabulary."""
        if self._unembedding is None:
            self._unembedding = self.checkpoint.unembedding.to(self.precision.dtype)
        logits = hidden.to(self.precision.dtype) @ self._unembedding.T
        return softcap(logits, self.config.final_logit_softcapping)

    def readout_top(self, hidden, k=1):
        """Final-norm states [N, H] -> the k largest logits [N, k] and their token ids, best first."""
        chunk = max(1, self.logit_elements // self.config.served_vocab_size)
        top = [self.readout_logits(rows).topk(k, dim=-1) for rows in hidden.split(chunk)]
        return torch.cat([t.values for t in top]), torch.cat([t.indices for t in top])

    def readout_argmax(self, hidden):
        """Final-norm states [N, H] -> served-vocabulary argmax [N], without holding all logits."""
        chunk = max(1, self.logit_elements // self.config.served_vocab_size)
        return torch.cat([self.readout_logits(rows).argmax(-1) for rows in hidden.split(chunk)])

    @torch.no_grad()
    def forward(self, tokens, *, num_passes=None, depths=None, taps=(), cache=None, return_logits=False,
                pass_end=True, release_experts=False, coda_taps=()):
        """Teacher-forced forward of a token vector [T] (or of the next T tokens of `cache`).

        num_passes is the number of core passes (default: the checkpoint's; any positive count runs) and
        depths the readout depths within 1..num_passes (default: num_passes only, the model's own output). taps
        are visit ids of the num_passes model whose input block is returned; ids in the coda and the final id
        need depth num_passes. coda_taps are coda layers whose input block every readout returns (its taps):
        the coda at each depth, where taps reach only the model's own. return_logits adds the full logits of
        every depth, pass_end the end-of-pass blocks, and release_experts drops each layer's experts from the
        checkpoint after its last use.
        """
        c = self.config
        core, layers = c.core_layers, c.num_hidden_layers
        passes = c.num_passes if num_passes is None else num_passes
        depths = (passes,) if depths is None else tuple(sorted(set(depths)))
        if passes < 1 or not depths or depths[0] < 1 or depths[-1] > passes:
            raise ValueError(f"depths {depths} must lie in 1..num_passes = {passes}")
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        if tokens.ndim != 1 or not tokens.numel():
            raise ValueError("the forward takes a nonempty token vector")
        if int(tokens.min()) < 0 or int(tokens.max()) >= c.served_vocab_size:
            raise ValueError(f"token ids must lie in [0, {c.served_vocab_size})")
        final, coda = layers + (passes - 1) * len(core), core.start + passes * len(core)
        taps = set(taps)
        if any(not 0 <= visit <= final for visit in taps):
            raise ValueError(f"taps must be visit ids in [0, {final}]")
        if passes not in depths and any(visit >= coda for visit in taps):
            raise ValueError(f"taps from visit {coda} on need a readout at depth {passes}")
        coda_taps = set(coda_taps)
        if any(not core.stop <= layer < layers for layer in coda_taps):
            raise ValueError(f"coda_taps are layers in [{core.stop}, {layers})")
        if cache is not None:
            cache.bind((passes, depths))
        start = 0 if cache is None else cache.length
        run = _Run(start + torch.arange(tokens.numel(), device=self.device), start, cache)
        release = self.checkpoint.drop_experts if release_experts else (lambda layer: None)
        features, routed, ends, states = {}, [], {}, {}

        def tap(visit, partial):
            if visit in taps:
                features[visit] = partial

        blocks = []
        partial = self.embed_tokens(tokens)
        for layer in range(core.start):
            tap(layer, partial)
            blocks, partial, ids = self._layer(run, layer, 0, blocks, partial, None)
            routed.append(ids)
            release(layer)
        prelude, bank = (blocks, partial), None
        for depth in range(1, passes + 1):
            blocks, partial = prelude
            for layer in core:
                tap(layer + (depth - 1) * len(core), partial)
                blocks, partial, ids = self._layer(run, layer, depth, blocks, partial, bank)
                routed.append(ids)
            bank = [*blocks[len(prelude[0]) + 1:], partial]
            if pass_end:
                ends[depth] = torch.stack(bank)
            if depth in depths:
                states[depth] = (blocks, partial)
        for layer in core:
            release(layer)
        coda_routed = {depth: [] for depth in depths}
        coda_features = {depth: {} for depth in depths}
        for layer in range(core.stop, layers):
            for depth in depths:
                blocks, partial = states[depth]
                if layer in coda_taps:
                    coda_features[depth][layer] = partial
                if depth == passes:
                    tap(layer + (passes - 1) * len(core), partial)
                blocks, partial, ids = self._layer(run, layer, depth, blocks, partial, None)
                states[depth] = (blocks, partial)
                coda_routed[depth].append(ids)
            release(layer)
        readouts = {}
        for depth in depths:
            blocks, partial = states[depth]
            hidden = self.readout_hidden(blocks, partial)
            experts = torch.stack(coda_routed[depth]) if coda_routed[depth] else routed[0].new_empty(
                (0, *routed[0].shape))
            logits = self.readout_logits(hidden) if return_logits else None
            argmax = self.readout_argmax(hidden) if logits is None else logits.argmax(-1)
            readouts[depth] = Inkling2Readout(hidden, argmax, logits, experts, coda_features[depth])
        if passes in depths:
            tap(final, readouts[passes].hidden)
        if cache is not None:
            cache.length += tokens.numel()
        return Inkling2Output(passes, core, layers, readouts, features, ends, torch.stack(routed))
