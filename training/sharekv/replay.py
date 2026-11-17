"""Exact fixed-prefix replay of a shared-KV model: native one-pass prompt, then response chunks.

Response chunks run sequentially with chunk size C <= W, which guarantees that every last-loop
writer a query reads outside its window has finished. Chunking changes execution, not the model.
exact_document is the single-document reference; exact_documents batches documents of different
lengths (masked SDPA, or FlashAttention with CUDA graphs) and can return per-loop features.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .flash import FLASH_PAGE, _flash_chunks
from .rule import Layout, finish_layer, loop_count, native_forward, pad_token, qkv, rotary_embeddings


@torch.no_grad()
def exact_document(model, ids, prompt_length, window=64, chunk_size=None):
    """Exact HF fixed-prefix replay, native one-pass prompt, no generation.

    Response chunks are sequential; C <= W guarantees old shared writers have
    finished. Chunking changes execution, not the target model definition.
    """
    if ids.ndim != 2 or len(ids) != 1 or not 1 <= prompt_length <= ids.shape[1]:
        raise ValueError("exact replay expects one document with a nonempty prompt")
    chunk_size = window if chunk_size is None else chunk_size
    if not 1 <= chunk_size <= window:
        raise ValueError("exact chunk size must be <= W")
    body, device = model.model, ids.device
    p = prompt_length
    prefix_layout = Layout(torch.zeros((1, p), dtype=torch.long, device=device),
                           torch.arange(p, device=device)[None],
                           torch.ones((1, p), dtype=torch.bool, device=device))
    backend = "dense" if p <= 4096 else "flex"
    prefix, caches, _ = native_forward(model, ids[:, :p], prefix_layout,
                                       backend=backend, capture="all")
    loops = loop_count(model)
    for slot in list(caches):
        if slot[0] != loops - 1 and slot != (0, 0):
            keep = min(window - 1, p)
            caches[slot] = tuple(x[:, :, -keep:] if keep else x[:, :, :0]
                                 for x in caches[slot])
    outputs = [prefix]
    for start in range(p, ids.shape[1], chunk_size):
        stop = min(start + chunk_size, ids.shape[1])
        hidden = body.embed_tokens(ids[:, start:stop])
        rotary = rotary_embeddings(body, hidden, torch.arange(start, stop, device=device)[None])
        rows = torch.arange(start, stop, device=device)[:, None]
        for loop in range(loops):
            for index, layer in enumerate(body.layers):
                slot = loop, index
                q, k, v = qkv(layer, hidden, rotary)
                previous = caches[slot]
                local = tuple(torch.cat([old, now], 2) for old, now in zip(previous, (k, v)))
                local_start = start - previous[0].shape[2]
                columns = torch.arange(local_start, stop, device=device)[None]
                native = loop == loops - 1 or slot == (0, 0)
                if native:
                    keys, values = local
                    allowed = columns <= rows
                else:
                    old = caches[(loops - 1, index)]
                    old_columns = torch.arange(old[0].shape[2], device=device)[None]
                    keys, values = (torch.cat([a, b], 2) for a, b in zip(old, local))
                    allowed = torch.cat([old_columns <= rows - window,
                                         (columns > rows - window) & (columns <= rows)], -1)
                attention = F.scaled_dot_product_attention(
                    q, keys, values, attn_mask=allowed[None, None],
                    scale=layer.self_attn.scaling, enable_gqa=q.shape[1] != keys.shape[1])
                hidden = finish_layer(layer, hidden, attention)
                keep = local[0].shape[2] if native else min(window - 1, local[0].shape[2])
                caches[slot] = tuple(x[:, :, -keep:] if keep else x[:, :, :0] for x in local)
            hidden = body.norm(hidden)
        outputs.append(hidden)
    return torch.cat(outputs, 1)


def exact_documents(model, documents, window=64, chunk_size=None, *, feature_layers=(),
                    loop_states=False, attention="sdpa", cuda_graph=True):
    """Batched exact replay with variable prompt/response lengths and rings.

    Returns one hidden-state sequence per document. Padding never supplies
    memory to a valid query; batching is independent of the read rule.

    With feature_layers or loop_states, each document instead gets a dict:
    "hidden" as above, "features" [R, T, len(feature_layers) * H] holding the
    outputs of those 0-based layers in every loop (residual stream before the
    loop-end norm, concatenated in the given order), and "loops" [R, T, H]
    holding every loop's normed output, whose LM head is that loop's readout.

    attention="flash" (CUDA, BF16/FP16) keeps each document's KV contiguous and replaces the masked
    SDPA with FlashAttention varlen: a native slot attends causally to its own buffer; a shared slot
    merges its layer's final-loop history j <= t - W with its own-loop band (t - W, t] by LSE. The
    read rule is the same; only rounding differs. cuda_graph replays the chunk step from a graph.
    """
    if attention not in ("sdpa", "flash"):
        raise ValueError("attention must be sdpa or flash")
    flash = attention == "flash"
    device = next(model.parameters()).device
    body, loops = model.model, loop_count(model)
    chunk = window if chunk_size is None else chunk_size
    if not 1 <= chunk <= window or not documents:
        raise ValueError("exact batch requires documents and 1 <= chunk <= W")
    feature_layers = tuple(feature_layers)
    if any(not 0 <= layer < len(body.layers) for layer in feature_layers):
        raise ValueError("feature layers must index this model's physical layers")
    taps = {}

    def tap(loop, index, hidden):
        if index in feature_layers:
            taps[(loop, index)] = hidden

    def features():
        return torch.stack([torch.cat([taps[(loop, layer)] for layer in feature_layers], -1)
                            for loop in range(loops)])

    lengths = torch.tensor([len(d["input_ids"]) for d in documents], device=device)
    prompts = torch.tensor([d["prompt_length"] for d in documents], device=device)
    pmax = int(prompts.max())
    ids = torch.full((len(documents), pmax), pad_token(model), device=device, dtype=torch.long)
    for row, doc in enumerate(documents):
        ids[row, :doc["prompt_length"]] = torch.tensor(doc["input_ids"][:doc["prompt_length"]], device=device)
    positions = torch.arange(pmax, device=device)[None].expand(len(documents), -1)
    valid = positions < prompts[:, None]
    layout = Layout(torch.where(valid, 0, -1), positions, valid)
    prefix, caches, states = native_forward(model, ids, layout, backend="dense" if pmax <= 4096 else "flex",
                                            capture="all", return_loops=loop_states,
                                            layer_hook=tap if feature_layers else None)
    outputs = [[prefix[row, :doc["prompt_length"]]] for row, doc in enumerate(documents)]
    feature_parts = loop_parts = None
    if feature_layers:
        stacked = features()
        feature_parts = [[stacked[:, row, :doc["prompt_length"]]] for row, doc in enumerate(documents)]
    if loop_states:
        stacked = torch.stack(states)
        loop_parts = [[stacked[:, row, :doc["prompt_length"]]] for row, doc in enumerate(documents)]
    del states
    history_positions = torch.where(valid, positions, -1)
    keep = window - 1
    recent_positions = prompts[:, None] - keep + torch.arange(keep, device=device)[None]
    recent_valid = recent_positions >= 0
    recent_indices = recent_positions.clamp_min(0)
    max_response = int((lengths - prompts).max())
    # Full-history slots (final loop, layer-1/loop-1) live in preallocated buffers that grow by one
    # chunk per step. A shared slot reads its layer's final-loop buffer: its recent band and current
    # chunk are written into the scratch tail past the history, so the history is never copied.
    # Loop order makes this safe: the final loop appends its own chunk only after the shared loops.
    capacity = pmax + max_response + keep + chunk
    if flash:
        capacity = -(-capacity // FLASH_PAGE) * FLASH_PAGE
    buffers, rings = {}, {}
    for slot, pair in list(caches.items()):
        if slot[0] == loops - 1 or slot == (0, 0):
            if flash:  # [B, capacity, heads, dim]: position p of document b at row b, slot p
                # FlashAttention reads V rows past a sequence's length within a tile; keep unwritten rows finite.
                buffer = tuple(x.new_zeros(x.shape[0], capacity, x.shape[1], x.shape[3]) for x in pair)
                for target, source in zip(buffer, pair):
                    target[:, :pmax] = source.transpose(1, 2)
            else:
                buffer = tuple(x.new_empty(x.shape[0], x.shape[1], capacity, x.shape[3]) for x in pair)
                for target, source in zip(buffer, pair):
                    target[:, :, :pmax] = source
            buffers[slot] = buffer
        else:
            rings[slot] = tuple(x.gather(2, recent_indices[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[3]))
                                for x in pair)
            if flash:
                rings[slot] = tuple(x.transpose(1, 2).contiguous() for x in rings[slot])
        del caches[slot]
    if flash and max_response:
        return _flash_chunks(model, documents, window, chunk, buffers, rings, capacity, prompts, max_response,
                             feature_layers, loop_states, outputs, feature_parts, loop_parts, cuda_graph)
    recent_positions = torch.where(recent_valid, recent_positions, -1)
    history = pmax
    for start in range(0, max_response, chunk):
        count = min(chunk, max_response - start)
        query_positions = prompts[:, None] + torch.arange(start, start + count, device=device)[None]
        query_valid = query_positions < lengths[:, None]
        response_ids = torch.full((len(documents), count), pad_token(model), device=device, dtype=torch.long)
        for row, doc in enumerate(documents):
            values = doc["input_ids"][doc["prompt_length"] + start:doc["prompt_length"] + start + count]
            response_ids[row, :len(values)] = torch.tensor(values, device=device)
        hidden = body.embed_tokens(response_ids)
        rotary = rotary_embeddings(body, hidden, query_positions)
        new_positions = torch.where(query_valid, query_positions, -1)
        full_positions = torch.cat([history_positions, new_positions], 1)
        local_positions = torch.cat([recent_positions, new_positions], 1)
        # Both masks are the same for every layer and loop of this chunk.
        query = query_positions[:, :, None]
        native_mask = ((full_positions[:, None, :] >= 0) & (full_positions[:, None, :] <= query))[:, None]
        shared_mask = torch.cat([
            (history_positions[:, None, :] >= 0) & (history_positions[:, None, :] <= query - window),
            (local_positions[:, None, :] > query - window) & (local_positions[:, None, :] >= 0)
            & (local_positions[:, None, :] <= query)], -1)[:, None]
        chunk_loops = []
        for loop in range(loops):
            for index, layer in enumerate(body.layers):
                slot = loop, index
                q, k, v = qkv(layer, hidden, rotary)
                if slot in buffers:
                    keys, values = buffers[slot]
                    keys[:, :, history:history + count] = k
                    values[:, :, history:history + count] = v
                    end, mask = history + count, native_mask
                else:
                    keys, values = buffers[(loops - 1, index)]
                    ring = rings[slot]
                    for buffer, old, new in zip((keys, values), ring, (k, v)):
                        buffer[:, :, history:history + keep] = old
                        buffer[:, :, history + keep:history + keep + count] = new
                    end, mask = history + keep + count, shared_mask
                    rings[slot] = tuple(torch.cat([old, new], 2)[:, :, -keep:] if keep else old
                                        for old, new in zip(ring, (k, v)))
                attention = F.scaled_dot_product_attention(q, keys[:, :, :end], values[:, :, :end], attn_mask=mask,
                                    scale=layer.self_attn.scaling, enable_gqa=q.shape[1] != keys.shape[1])
                hidden = finish_layer(layer, hidden, attention)
                if feature_layers:
                    tap(loop, index, hidden)
            hidden = body.norm(hidden)
            if loop_states:
                chunk_loops.append(hidden)
        stacked_features = features() if feature_layers else None
        stacked_loops = torch.stack(chunk_loops) if loop_states else None
        for row, doc in enumerate(documents):
            n = max(0, min(count, len(doc["input_ids"]) - doc["prompt_length"] - start))
            if n:
                outputs[row].append(hidden[row, :n])
                if feature_layers:
                    feature_parts[row].append(stacked_features[:, row, :n])
                if loop_states:
                    loop_parts[row].append(stacked_loops[:, row, :n])
        history_positions = full_positions
        recent_positions = local_positions[:, -keep:] if keep else local_positions[:, :0]
        history += count
    if not (feature_layers or loop_states):
        return [torch.cat(chunks, 0) for chunks in outputs]
    results = []
    for row in range(len(documents)):
        result = {"hidden": torch.cat(outputs[row], 0)}
        if feature_layers:
            result["features"] = torch.cat(feature_parts[row], 1)
        if loop_states:
            result["loops"] = torch.cat(loop_parts[row], 1)
        results.append(result)
    return results
