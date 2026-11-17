# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Last-loop KV sharing for vLLM AR / speculative decoding of looped models.

Native prompt and last-loop shared history with a same-loop recent band.

Final-loop and lexical-reader caches keep the layer's own span (full attention, or
its sliding window). Other loop caches use SlidingWindowSpec(W): paged storage of
W-1 recent tokens plus current scratch. In a sliding-window layer of span S the
final-loop history is j in [t - S + 1, t - W].
The two disjoint history sources share one softmax through an FP32 LSE merge.
Decode reads the band and merges in one Triton kernel (_band_merge); prefill uses
FlashAttention for both sources and merge_attn_states.
"""
from __future__ import annotations

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def _write(layer, cache, slots, key, value):
    """Scatter [N, H, D] K/V (token-strided views) into a paged cache."""
    # reshape_and_cache_flash supports a strided token dimension but assumes
    # contiguous H*D within each token. Splitting packed [N,H,K+V] gives an
    # interleaved-head view; normalize it before copying the first writer.
    if key.stride(-1) != 1 or key.stride(-2) != key.shape[-1]:
        key = key.contiguous()
    if value.stride(-1) != 1 or value.stride(-2) != value.shape[-1]:
        value = value.contiguous()
    layer.impl.do_kv_cache_update(layer, key, value, cache, slots)


def self_state(query, key, value, scale):
    """Per-query self partial: output [N, Hq, D] and LSE [Hq, N] (fp32)."""
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1) if groups > 1 else key
    value = value.repeat_interleave(groups, dim=1) if groups > 1 else value
    acc = torch.promote_types(query.dtype, torch.float32)
    lse = (query.to(acc) * key.to(acc)).sum(-1).mul_(scale).transpose(0, 1).contiguous()
    return value.contiguous(), lse


@triton.jit
def _band_merge(
    query, cache, block_table, seq_lens, query_start_loc, memory_out, memory_lse, output,
    new_key, new_value, slot_mapping,
    num_reqs,
    q_token_stride, q_head_stride,
    cache_block_stride, cache_head_stride, cache_slot_stride, cache_dim_stride,
    table_stride, out_token_stride, out_head_stride, mem_token_stride, mem_head_stride,
    lse_head_stride, lse_token_stride,
    key_token_stride, key_head_stride, key_dim_stride,
    value_token_stride, value_head_stride, value_dim_stride,
    HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    DIM_BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr, WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    WRITE_KV: tl.constexpr,
):
    """Same-loop band attention of one decode query and head, merged with the final-loop partial.

    Query t reads its own loop at t - W < j <= t from the paged band cache and the final-loop
    partial (FlashAttention over j <= t - W) through one FP32 softmax. An empty final range
    (t < W) drops the partial. Replaces a band FlashAttention call and merge_attn_states.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, DIM_BLOCK)
    # Graph replay may have fewer live queries than the captured launch grid.
    token_valid = token < tl.load(query_start_loc + num_reqs)
    request = 0
    for r in range(num_reqs):
        request += (tl.load(query_start_loc + r + 1) <= token).to(tl.int32)
    request = tl.minimum(request, num_reqs - 1)
    begin = tl.load(query_start_loc + request)
    end = tl.load(query_start_loc + request + 1)
    position = tl.load(seq_lens + request) - (end - token)
    kv_head = head // (HEADS // KV_HEADS)
    q = tl.load(query + token * q_token_stride + head * q_head_stride + dims,
                mask=(dims < HEAD_DIM) & token_valid, other=0).to(tl.float32)
    keys = position - (WINDOW - 1) + tl.arange(0, WINDOW)
    valid = (keys >= 0) & token_valid
    if WRITE_KV:
        prefix_length = tl.load(seq_lens + request) - (end - begin)
        from_input = valid & (keys >= prefix_length)
        cached = valid & ~from_input
    else:
        cached = valid
    page = tl.load(block_table + request * table_stride + tl.maximum(keys, 0) // BLOCK_SIZE,
                   mask=cached, other=0)
    rows = (page.to(tl.int64) * cache_block_stride + kv_head * cache_head_stride
            + (tl.maximum(keys, 0) % BLOCK_SIZE) * cache_slot_stride)
    mask = cached[:, None] & (dims[None, :] < HEAD_DIM)
    k = tl.load(cache + rows[:, None] + dims[None, :] * cache_dim_stride, mask=mask, other=0)
    v = tl.load(cache + rows[:, None] + (dims[None, :] + HEAD_DIM) * cache_dim_stride, mask=mask, other=0)
    if WRITE_KV:
        input_rows = begin + keys - prefix_length
        mask = from_input[:, None] & (dims[None, :] < HEAD_DIM)
        input_k = tl.load(
            new_key + input_rows[:, None] * key_token_stride
            + kv_head * key_head_stride + dims[None, :] * key_dim_stride,
            mask=mask, other=0,
        )
        input_v = tl.load(
            new_value + input_rows[:, None] * value_token_stride
            + kv_head * value_head_stride + dims[None, :] * value_dim_stride,
            mask=mask, other=0,
        )
        k = tl.where(from_input[:, None], input_k, k)
        v = tl.where(from_input[:, None], input_v, v)
        # Only one query head writes each KV head. All current-step reads use
        # the input tensors, so programs never depend on another program's write.
        if head % (HEADS // KV_HEADS) == 0:
            slot = tl.load(slot_mapping + token)
            if (slot >= 0) & token_valid:
                row = (slot // BLOCK_SIZE) * cache_block_stride
                row += kv_head * cache_head_stride
                row += (slot % BLOCK_SIZE) * cache_slot_stride
                write_k = tl.load(
                    new_key + token * key_token_stride + kv_head * key_head_stride
                    + dims * key_dim_stride, mask=dims < HEAD_DIM, other=0,
                )
                write_v = tl.load(
                    new_value + token * value_token_stride
                    + kv_head * value_head_stride + dims * value_dim_stride,
                    mask=dims < HEAD_DIM, other=0,
                )
                tl.store(cache + row + dims * cache_dim_stride, write_k,
                         mask=dims < HEAD_DIM)
                tl.store(cache + row + (dims + HEAD_DIM) * cache_dim_stride,
                         write_v, mask=dims < HEAD_DIM)
    scores = tl.where(valid, tl.sum(k.to(tl.float32) * q[None, :], 1) * SCALE, float("-inf"))
    band_max = tl.max(scores, 0)
    weights = tl.exp(scores - band_max)
    band_sum = tl.sum(weights, 0)
    band = tl.sum(weights[:, None] * v.to(tl.float32), 0)
    has_memory = (position >= WINDOW) & token_valid
    memory_logsum = tl.load(
        memory_lse + head * lse_head_stride + token * lse_token_stride,
        mask=token_valid, other=float("-inf"),
    )
    memory_logsum = tl.where(has_memory, memory_logsum, float("-inf"))
    maximum = tl.maximum(memory_logsum, band_max)
    memory_weight = tl.where(has_memory, tl.exp(memory_logsum - maximum), 0.0)
    band_scale = tl.exp(band_max - maximum)
    memory = tl.load(memory_out + token * mem_token_stride + head * mem_head_stride + dims,
                     mask=(dims < HEAD_DIM) & has_memory, other=0).to(tl.float32)
    result = (memory_weight * memory + band_scale * band) / (memory_weight + band_scale * band_sum)
    result = tl.where(token_valid, result, 0.0)
    tl.store(output + token * out_token_stride + head * out_head_stride + dims,
             result.to(output.dtype.element_ty), mask=dims < HEAD_DIM)


def band_merge(q, cache, metadata, memory_out, memory_lse, output, window, scale,
               key=None, value=None):
    """Launch _band_merge for count decode queries; cache is the (B, H_kv, N, 2D) band cache."""
    count, heads, head_dim = q.shape
    if metadata.seq_lens.numel() == 0:
        output.zero_()
        return
    kv_heads, block_size = cache.shape[1], cache.shape[2]
    write_kv = key is not None
    if not write_kv:
        key = value = q
    _band_merge[(count, heads)](
        q, cache, metadata.block_table, metadata.seq_lens, metadata.query_start_loc,
        memory_out, memory_lse, output,
        key, value, metadata.slot_mapping if write_kv else metadata.seq_lens,
        metadata.seq_lens.shape[0],
        q.stride(0), q.stride(1),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        metadata.block_table.stride(0), output.stride(0), output.stride(1),
        memory_out.stride(0), memory_out.stride(1), memory_lse.stride(0), memory_lse.stride(1),
        *key.stride(), *value.stride(),
        HEADS=heads, KV_HEADS=kv_heads, HEAD_DIM=head_dim,
        DIM_BLOCK=triton.next_power_of_2(head_dim), BLOCK_SIZE=block_size, WINDOW=window,
        SCALE=scale, WRITE_KV=write_kv,
    )


def _shared_lengths(context, metadata, window):
    """Reuse lengths within a forward; graph capture records their computation."""
    readers = getattr(context, "loopedtf_shared_lengths", None)
    if readers is None:
        readers = context.loopedtf_shared_lengths = {}
    cache_key = (window, id(metadata.seq_lens))
    if cache_key not in readers:
        prefill = bool(context.loop_kv_prefill_active)
        if prefill:
            decode = context.loop_kv_decode_lengths[:len(metadata.seq_lens)] >= 0
            old = torch.where(
                decode, (metadata.seq_lens - window).clamp_min(0), 0
            ).to(torch.int32)
            current = torch.where(decode, metadata.seq_lens, 0).to(torch.int32)
        else:
            decode = None
            old = (metadata.seq_lens - window).clamp_min(0).to(torch.int32)
            current = metadata.seq_lens.to(torch.int32)
        readers[cache_key] = (old, current, decode)
    return readers[cache_key]


def shared_last_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          layer_names: str, loop: int, window: int,
                          max_context: int, sliding: int) -> torch.Tensor:
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

    names = layer_names.split("|")
    metadata, layer, cache = get_attention_context(names[loop])[:3]
    if metadata is None:
        return torch.zeros_like(query)  # Memory profiling, before caches exist.
    final_metadata, _, final_cache = get_attention_context(names[-1])[:3]
    count, d = metadata.num_actual_tokens, layer.head_size
    q = query[:count].reshape(count, layer.num_heads, d)
    k = key[:count].reshape(count, layer.num_kv_heads, d)
    v = value[:count].reshape(count, layer.num_kv_heads, d)
    context = get_forward_context()
    prefill_active = bool(context.loop_kv_prefill_active)
    fuse_write = (
        not prefill_active and window > 1 and window & (window - 1) == 0
        and cache.dtype == k.dtype
    )
    if not fuse_write:
        _write(layer, cache, metadata.slot_mapping[:count], k, v)
    old_lengths, current_lengths, decode = _shared_lengths(context, metadata, window)
    final_k, final_v = final_cache.transpose(1, 2).split(d, -1)
    version = get_flash_attn_version(head_size=d)
    scale = layer.impl.scale
    # Sliding layers (span S) clip both history sources to j >= t - S + 1.
    memory_span = {"window_size": (sliding - 1 - window, 0)} if sliding else {}
    native_span = {"window_size": (sliding - 1, 0)} if sliding else {}
    output = torch.empty_like(query).view(-1, layer.num_heads, d)
    if count < output.shape[0]:
        output[count:].zero_()
    # FlashAttention-2 writes the LSE of a query block with no visible key in the padded
    # [batch, heads, max_seqlen_q] layout, outside the unpadded [heads, tokens] buffer it
    # returns (observed with GQA and head_dim 256 on A100, overwriting query_start_loc
    # and positions). Every history call therefore gives each sequence at least one key
    # and only decode queries, whose blocks always contain a query that sees it; rows
    # without history are masked.
    if prefill_active:
        # Eager step: prompt queries read their current same-loop native K/V only.
        native_out, native_lse = flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=metadata.query_start_loc,
            cu_seqlens_k=metadata.query_start_loc,
            max_seqlen_q=metadata.max_query_len,
            max_seqlen_k=metadata.max_query_len,
            softmax_scale=scale, causal=True,
            return_softmax_lse=True, fa_version=version, **native_span,
        )
        memory_out = torch.zeros_like(native_out)
        memory_lse = torch.full_like(native_lse, float("-inf"))
        requests = decode.nonzero().squeeze(1)
        if requests.numel():
            starts = metadata.query_start_loc
            lengths = (starts[1:] - starts[:-1])[requests]
            cu = torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
            rows = torch.cat([torch.arange(int(starts[r]), int(starts[r + 1]), device=q.device)
                              for r in requests.tolist()])
            max_rows = int(lengths.max())
            decode_q = q[rows]
            history = old_lengths[requests]
            part_out, part_lse = flash_attn_varlen_func(
                q=decode_q, k=final_k, v=final_v, cu_seqlens_q=cu, max_seqlen_q=max_rows,
                seqused_k=history.clamp_min(1), max_seqlen_k=max_context,
                block_table=final_metadata.block_table[requests],
                softmax_scale=scale, causal=True,
                return_softmax_lse=True, fa_version=version, **memory_span,
            )
            owner = torch.bucketize(torch.arange(rows.numel(), device=q.device), cu[1:], right=True)
            present = history[owner] > 0
            memory_out[rows] = torch.where(present[:, None, None], part_out, 0)
            memory_lse[:, rows] = torch.where(present[None, :], part_lse, float("-inf"))
            if window == 1:
                band_out, band_lse = self_state(decode_q, k[rows], v[rows], scale)
            else:
                current_k, current_v = cache.transpose(1, 2).split(d, -1)
                band_out, band_lse = flash_attn_varlen_func(
                    q=decode_q, k=current_k, v=current_v, cu_seqlens_q=cu, max_seqlen_q=max_rows,
                    seqused_k=current_lengths[requests], max_seqlen_k=max_context,
                    block_table=metadata.block_table[requests],
                    softmax_scale=scale, causal=True, window_size=(window - 1, 0),
                    return_softmax_lse=True, fa_version=version,
                )
            native_out[rows] = band_out
            native_lse[:, rows] = band_lse
        merge_attn_states(output[:count], memory_out, memory_lse.contiguous(),
                          native_out, native_lse.contiguous())
        return output.view_as(query)
    # Decode (graph-captured): query t reads final-loop KV at j <= t - W. With the
    # queries of one request ending at its sequence length S and old_lengths = S - W,
    # bottom-right causal alignment gives exactly that boundary to every query, so a
    # speculative block of several tokens is verified like sequential decode.
    memory_out, memory_lse = flash_attn_varlen_func(
        q=q, k=final_k, v=final_v,
        cu_seqlens_q=metadata.query_start_loc,
        max_seqlen_q=metadata.max_query_len,
        seqused_k=old_lengths.clamp_min(1), max_seqlen_k=max_context,
        block_table=final_metadata.block_table,
        softmax_scale=scale, causal=True,
        return_softmax_lse=True, fa_version=version, **memory_span,
    )
    if window > 1 and window & (window - 1) == 0:
        # One kernel for the same-loop band and the merge; it ignores the history
        # partial of queries at t < W (no history).
        band_merge(q, cache, metadata, memory_out, memory_lse, output[:count], window,
                   scale, key=k if fuse_write else None,
                   value=v if fuse_write else None)
        return output.view_as(query)
    owner = torch.bucketize(torch.arange(count, device=q.device),
                            metadata.query_start_loc[1:], right=True)
    present = old_lengths[owner.clamp_max(old_lengths.numel() - 1)] > 0
    memory_out = torch.where(present[:, None, None], memory_out, 0)
    memory_lse = torch.where(present[None, :], memory_lse, float("-inf"))
    if window == 1:
        current_out, current_lse = self_state(q, k, v, scale)
    else:
        current_k, current_v = cache.transpose(1, 2).split(d, -1)
        current_out, current_lse = flash_attn_varlen_func(
            q=q, k=current_k, v=current_v,
            cu_seqlens_q=metadata.query_start_loc,
            max_seqlen_q=metadata.max_query_len,
            seqused_k=current_lengths, max_seqlen_k=max_context,
            block_table=metadata.block_table,
            softmax_scale=scale, causal=True,
            window_size=(window - 1, 0),
            return_softmax_lse=True, fa_version=version,
        )
    merge_attn_states(output[:count], memory_out, memory_lse.contiguous(),
                      current_out, current_lse.contiguous())
    return output.view_as(query)


def shared_last_attention_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    layer_names: str, loop: int, window: int, max_context: int, sliding: int,
) -> torch.Tensor:
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="loopedtf_shared_last_attention",
    op_func=shared_last_attention,
    mutates_args=[],
    fake_impl=shared_last_attention_fake,
)


class LoopKVSharing:
    """Shared last-loop history for one physical layer's per-loop caches.

    ``native`` lists (loop, physical_layer) readers that keep their own loop's full
    history (Ouro/Nanbeige: the lexical reader (0, 0)). ``sliding`` is the layer's
    attention span (0 for full attention).
    """

    def __init__(self, attentions, config, physical_layer, native=((0, 0),), sliding=0):
        self.names = "|".join(a.layer_name for a in attentions)
        self.loops = len(attentions)
        self.physical_layer = physical_layer
        self.native = frozenset(native)
        self.window = config.loopedtf_kv_window
        self.sliding = int(sliding or 0)
        if self.sliding and self.window >= self.sliding:
            raise ValueError("A shared sliding layer needs window < its span")
        from vllm.config import get_current_vllm_config

        # FlashAttention-2 sizes split-KV work from max_seqlen_k; a bound far above the
        # engine's max_model_len (Dory: 262144) faults in full-decode CUDA graph replay.
        self.max_context = min(config.max_position_embeddings,
                               get_current_vllm_config().model_config.max_model_len)
        for attention in attentions:
            if attention.attn_backend.get_name() != "FLASH_ATTN":
                raise ValueError("Shared-last requires FlashAttention")

    def attend(self, loop, query, key, value):
        if loop == self.loops - 1 or (loop, self.physical_layer) in self.native:
            return None
        return torch.ops.vllm.loopedtf_shared_last_attention(
            query, key, value, self.names, loop, self.window, self.max_context,
            self.sliding,
        )
