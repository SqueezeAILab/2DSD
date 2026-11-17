"""FlashAttention + CUDA graph path of exact_documents (CUDA, BF16/FP16).

Requires vllm.vllm_flash_attn, installed with the repository's vLLM fork: FlashAttention 3 on Hopper,
FlashAttention 2 on Ampere/Ada (flash_version). Without the module, use attention="sdpa" (the masked
reference with the same read rule).
"""
from __future__ import annotations

import importlib.util

import torch

from .rule import finish_layer, loop_count, pad_token, qkv, rotary_embeddings


def flash_available():
    """True if vllm.vllm_flash_attn (FlashAttention varlen) can be imported."""
    try:
        return importlib.util.find_spec("vllm.vllm_flash_attn") is not None
    except (ImportError, ValueError):
        return False


def flash_version(device):
    """FlashAttention version for a CUDA device: 3 on Hopper when vllm.vllm_flash_attn supports it, else 2.

    The vLLM fork's 2DSD runners choose the same kernels on the supported GPUs; Dory's teacher-forced
    features (training/hf_models/dory/native.py) use this function too.
    """
    from vllm.vllm_flash_attn import is_fa_version_supported

    return 3 if torch.cuda.get_device_capability(device)[0] == 9 and is_fa_version_supported(3) else 2


@torch.no_grad()
def _flash_varlen():
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func


FLASH_PAGE = 64


def _flash(q, keys, values, count, max_k, scale, *, version, pages=None, used=None, cu_k=None, window=None):
    """FA varlen of version 2 or 3, bottom-right causal (optionally a left window), count queries per document.

    Keys are either paged ([pages, FLASH_PAGE, heads, dim] with a per-document page table and used
    key count) or packed (cu_k offsets). Returns the output and the LSE as [heads, queries].
    """
    rows = q.shape[0] // count
    cu_q = torch.arange(0, (rows + 1) * count, count, device=q.device, dtype=torch.int32)
    out, lse = _flash_varlen()(q, keys, values, max_seqlen_q=count, cu_seqlens_q=cu_q, max_seqlen_k=max_k,
                               cu_seqlens_k=cu_k, seqused_k=used, block_table=pages, softmax_scale=scale,
                               causal=True, window_size=[window - 1, 0] if window else None,
                               return_softmax_lse=True, fa_version=version)
    return out, lse


def _flash_layer(layer, slot, q, k, v, buffers, rings, loops, index, inputs):
    """One layer's attention for a chunk of count queries per document (see exact_documents)."""
    batch, heads, count, dim = q.shape
    scale, keep, version = layer.self_attn.scaling, inputs["keep"], inputs["fa_version"]
    q = q.transpose(1, 2).reshape(batch * count, heads, dim)
    k, v = k.transpose(1, 2), v.transpose(1, 2)
    paged = lambda x: x.view(-1, FLASH_PAGE, *x.shape[2:])
    if slot in buffers:
        keys, values = buffers[slot]
        keys[inputs["rows"], inputs["positions"]] = k
        values[inputs["rows"], inputs["positions"]] = v
        out, _ = _flash(q, paged(keys), paged(values), count, inputs["capacity"], scale, version=version,
                        pages=inputs["pages"], used=inputs["native_used"])
    else:
        keys, values = buffers[(loops - 1, index)]
        past, past_lse = _flash(q, paged(keys), paged(values), count, inputs["capacity"], scale, version=version,
                                pages=inputs["pages"], used=inputs["history_used"])
        old_k, old_v = rings[slot]
        local_k, local_v = torch.cat([old_k, k], 1).flatten(0, 1), torch.cat([old_v, v], 1).flatten(0, 1)
        if keep:  # the ring keeps this loop's last W - 1 keys and values for the next chunk
            old_k.copy_(local_k.view(batch, -1, *k.shape[2:])[:, -keep:])
            old_v.copy_(local_v.view(batch, -1, *v.shape[2:])[:, -keep:])
        band_index, band_cu, band_max = inputs["band"]
        if band_index is not None:  # pack only the valid ring entries of each document
            local_k, local_v = local_k.index_select(0, band_index), local_v.index_select(0, band_index)
        band, band_lse = _flash(q, local_k, local_v, count, band_max, scale, version=version, cu_k=band_cu,
                                window=inputs["window"])
        # Queries at positions < W have no history; FA leaves those rows unspecified.
        valid = inputs["history_valid"]
        past_lse = torch.where(valid[None], past_lse, -torch.inf)
        total = torch.logaddexp(past_lse, band_lse)
        past = torch.where(valid[:, None, None], past.float(), 0.0)
        out = (past * (past_lse - total).exp().T[..., None]
               + band.float() * (band_lse - total).exp().T[..., None]).to(q.dtype)
    return out.view(batch, count, heads, dim).transpose(1, 2)


def _flash_chunks(model, documents, window, chunk, buffers, rings, capacity, prompts, max_response,
                  feature_layers, loop_states, outputs, feature_parts, loop_parts, use_graph):
    """Response chunks of exact_documents with FlashAttention, every chunk padded to full size.

    Once every document's history and ring are full, all chunks have the same shapes and buffer
    addresses, so one CUDA graph of the whole chunk step (all loops and layers) is captured and replayed.
    """
    device = prompts.device
    body, loops = model.model, loop_count(model)
    batch, keep = len(documents), window - 1
    prompt_lengths = [d["prompt_length"] for d in documents]
    steps = -(-max_response // chunk)
    response = torch.full((batch, steps * chunk), pad_token(model), dtype=torch.long)
    for row, doc in enumerate(documents):
        values = doc["input_ids"][doc["prompt_length"]:]
        response[row, :len(values)] = torch.tensor(values, dtype=torch.long)
    response = response.to(device)
    per_document = capacity // FLASH_PAGE
    offsets = torch.arange(chunk, device=device)[None]
    inputs = {
        "keep": keep, "window": window, "capacity": capacity, "fa_version": flash_version(device),
        "rows": torch.arange(batch, device=device)[:, None],
        "pages": (torch.arange(batch, device=device)[:, None] * per_document
                  + torch.arange(per_document, device=device)[None]).int(),
        "positions": prompts[:, None] + offsets,
        "native_used": torch.empty(batch, device=device, dtype=torch.int32),
        "history_used": torch.empty(batch, device=device, dtype=torch.int32),
        "history_valid": torch.empty(batch * chunk, device=device, dtype=torch.bool),
    }
    ids = response[:, :chunk].clone()
    full_band = (None, torch.arange(0, (batch + 1) * (keep + chunk), keep + chunk, device=device,
                                    dtype=torch.int32), keep + chunk)

    def band(start):
        if min(prompt_lengths) + start >= keep:
            return full_band
        ring_valid = [min(keep, p + start) for p in prompt_lengths]
        index = torch.cat([b * (keep + chunk) + keep - n + torch.arange(n + chunk) for b, n in enumerate(ring_valid)])
        cu = torch.tensor([0, *torch.tensor([n + chunk for n in ring_valid]).cumsum(0).tolist()], dtype=torch.int32)
        return index.to(device), cu.to(device), keep + chunk

    def set_inputs(start):
        ids.copy_(response[:, start:start + chunk])
        inputs["positions"].copy_(prompts[:, None] + start + offsets)
        inputs["native_used"].copy_(prompts + start + chunk)
        inputs["history_used"].copy_((prompts + start + chunk - window).clamp_min(0))
        inputs["history_valid"].copy_((inputs["positions"] >= window).reshape(-1))

    def step():
        hidden = body.embed_tokens(ids)
        rotary = rotary_embeddings(body, hidden, inputs["positions"])
        taps, states = {}, []
        for loop in range(loops):
            for index, layer in enumerate(body.layers):
                q, k, v = qkv(layer, hidden, rotary)
                hidden = finish_layer(layer, hidden, _flash_layer(layer, (loop, index), q, k, v, buffers, rings,
                                                                  loops, index, inputs))
                if index in feature_layers:
                    taps[(loop, index)] = hidden
            hidden = body.norm(hidden)
            if loop_states:
                states.append(hidden)
        features = (torch.stack([torch.cat([taps[(loop, layer)] for layer in feature_layers], -1)
                                 for loop in range(loops)]) if feature_layers else None)
        return hidden, features, torch.stack(states) if loop_states else None

    hidden_out = feature_out = loop_out = None
    graph, static, eager_full = None, None, 0
    for start in range(0, max_response, chunk):
        set_inputs(start)
        full = min(prompt_lengths) + start >= window
        if graph is not None:
            graph.replay()
            result = static
        elif use_graph and full and eager_full >= 1 and start + chunk < max_response:
            inputs["band"] = full_band
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static = step()
            graph.replay()
            result = static
        else:
            inputs["band"] = band(start)
            result = step()
            eager_full += full
        hidden, features, states = result
        if hidden_out is None:
            hidden_out = hidden.new_empty(batch, steps * chunk, hidden.shape[-1])
            if feature_layers:
                feature_out = features.new_empty(loops, batch, steps * chunk, features.shape[-1])
            if loop_states:
                loop_out = states.new_empty(loops, batch, steps * chunk, states.shape[-1])
        hidden_out[:, start:start + chunk] = hidden
        if feature_layers:
            feature_out[:, :, start:start + chunk] = features
        if loop_states:
            loop_out[:, :, start:start + chunk] = states
    del graph, static
    results = []
    for row, doc in enumerate(documents):
        n = len(doc["input_ids"]) - doc["prompt_length"]
        hidden = torch.cat([outputs[row][0], hidden_out[row, :n]], 0)
        if not (feature_layers or loop_states):
            results.append(hidden)
            continue
        result = {"hidden": hidden}
        if feature_layers:
            result["features"] = torch.cat([feature_parts[row][0], feature_out[:, row, :n]], 1)
        if loop_states:
            result["loops"] = torch.cat([loop_parts[row][0], loop_out[:, row, :n]], 1)
        results.append(result)
    return results
