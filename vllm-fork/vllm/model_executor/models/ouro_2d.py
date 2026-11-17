# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Batch-one Ouro execution across token positions and loop depths.

This runner reuses a loaded native vLLM model and its projection, RoPE, and
normalization kernels. Each physical layer processes concatenated cells once;
paged attention treats the loop spans as separate sequences. It owns its caches
and is called inside a worker, independently of the vLLM request scheduler.
"""

import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def select_fa_version() -> int:
    """FlashAttention version of the 2D target and drafter attention calls.

    FA3 where the GPU supports it (Hopper), FA2 otherwise. TWODSD_FA_VERSION=2 or 3
    forces one version, e.g. for paired timing on the same build.
    """
    import os

    from vllm.vllm_flash_attn import is_fa_version_supported

    forced = os.environ.get("TWODSD_FA_VERSION")
    if forced:
        version = int(forced)
        if not is_fa_version_supported(version):
            raise ValueError(f"FlashAttention {version} is not supported on this device")
        return version
    return 3 if is_fa_version_supported(3) else 2


@dataclass(frozen=True)
class Span:
    """Consecutive positions to advance through one zero-based loop."""

    loop: int
    start: int
    hidden: torch.Tensor


@dataclass(frozen=True)
class LoopResult:
    """Normalized loop output and selected pre-normalization layer features."""

    hidden: torch.Tensor
    features: torch.Tensor


@dataclass
class _Inputs:
    hidden: torch.Tensor
    metadata: torch.Tensor
    cpu_metadata: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    cu_query: torch.Tensor
    sequence_lengths: torch.Tensor
    block_table: torch.Tensor
    max_query: int
    lengths: tuple[int, ...]
    loops: tuple[int, ...]
    transfer_event: Any = None


@dataclass
class _Captured:
    inputs: _Inputs
    graph: Any
    output: LoopResult


class NativeOuro2DRunner:
    """Advance independent loop spans using one projection pass per layer.

    Args:
        model: Loaded ``OuroForCausalLM`` with tensor/pipeline parallel size one.
        max_len: Capacity of each loop's token history.
        feat_layers: Zero-based physical layer indices, in drafter feature order.
        use_cuda_graph: Capture and reuse small decode shapes with FlashAttention.
        attention_backend: ``flash_attn`` for CUDA or ``sdpa`` for reference checks.
        graph_max_query_len: Largest individual span eligible for CUDA capture.
        max_graph_shapes: Maximum captured shapes; additional shapes run eagerly.
        block_size: Paged attention block size; a multiple of 16.
        use_fused_norm: Use scoped vLLM CUDA normalization kernels. Otherwise
            inherit the process IR priority, which can be unfused in this path.
        use_fused_rope: Invoke CUDA rotary kernels explicitly. Otherwise inherit
            the loaded model's CustomOp dispatch; CPU references are unchanged.

    ``advance`` accepts at most one span per loop, and each span must begin at
    that loop's current history length. It copies every input before execution,
    so outputs of the previous call may be passed as inputs even with graphs.
    Returned graph output views remain valid until that graph is replayed again;
    callers must copy features or states retained beyond the next advance.
    """

    def __init__(
        self,
        model,
        max_len: int,
        feat_layers: list[int] | None = None,
        use_cuda_graph: bool = True,
        attention_backend: str = "flash_attn",
        graph_max_query_len: int = 64,
        block_size: int = 16,
        max_graph_shapes: int = 64,
        use_fused_norm: bool = True,
        use_fused_rope: bool = True,
        shared_prefill: bool = False,
        compact_shared_prefix: bool = False,
    ) -> None:
        # The tree runner uses this runner only for prompt prefill and prefix KV
        # storage. A shared-last prompt is native single-pass, so prefill is exact.
        self.shared_prefill = shared_prefill
        self.model = model
        self.backbone = model.model
        config = model.config
        self.R = self._loop_count(config)
        self.H = config.hidden_size
        self.L = config.num_hidden_layers
        self.max_len = max_len
        self.feat_layers = (
            [round(1 + j * (self.L - 4) / 4) for j in range(5)]
            if feat_layers is None
            else list(feat_layers)
        )
        self.nf = len(self.feat_layers)
        self._feature_slots = {layer: j for j, layer in enumerate(self.feat_layers)}
        self.layers = list(self.backbone.layers)
        parameter = next(self.backbone.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.attention_backend = attention_backend
        self.graph_max_query_len = graph_max_query_len
        self.max_graph_shapes = max_graph_shapes
        self.use_fused_norm = use_fused_norm
        self.use_fused_rope = use_fused_rope
        self.block_size = block_size
        self.blocks_per_loop = (max_len + block_size - 1) // block_size
        self.padded_len = self.blocks_per_loop * block_size
        self.length = [0] * self.R
        self._graphs: dict[tuple, _Captured] = {}
        self.stats = {
            "advances": 0,
            "cells": 0,
            "graph_replays": 0,
            "captures": 0,
            "capture_seconds": 0.0,
            "graph_cache_fallbacks": 0,
        }
        self._validate_model(config)
        attn = self.layers[0].self_attn
        self.heads, self.kv_heads = attn.num_heads, attn.num_kv_heads
        self.head_dim = attn.head_dim
        self.compact_prefix = None
        self._prefill_scratch = None
        if compact_shared_prefix:
            if not shared_prefill or (
                getattr(config, "loopedtf_kv_policy", "full") != "shared_last"
            ):
                raise ValueError("Compact prefix storage requires shared-last prefill")
            from .shared_prefix import SharedPrefixKV

            self.compact_prefix = SharedPrefixKV(
                self.L, self.R, self.padded_len, config.loopedtf_kv_window,
                block_size, self.kv_heads, self.head_dim,
                device=self.device, dtype=self.dtype,
            )
            self.k, self.v = self.compact_prefix.k, self.compact_prefix.v
        cache_shape = (
            self.L,
            self.R * self.blocks_per_loop,
            block_size,
            self.kv_heads,
            self.head_dim,
        )
        # FA3 reads V rows past a sequence's length within a tile; keep unwritten rows finite.
        if self.compact_prefix is None:
            self.k = torch.zeros(cache_shape, device=self.device, dtype=self.dtype)
            self.v = torch.zeros_like(self.k)
            self._block_table = torch.arange(
                self.R * self.blocks_per_loop, device=self.device, dtype=torch.int32
            ).view(self.R, self.blocks_per_loop)
        else:
            # Native prefill reuses one layer's scratch, independent of loop.
            self._block_table = torch.arange(
                self.blocks_per_loop, device=self.device, dtype=torch.int32
            )[None, :].repeat(self.R, 1)
        self._scale = torch.ones((), device=self.device, dtype=torch.float32)
        self._flash = None
        self.fa_version = None
        self._write_cache = None
        if attention_backend == "flash_attn":
            from vllm._custom_ops import reshape_and_cache_flash
            from vllm.vllm_flash_attn import (
                flash_attn_varlen_func,
                is_fa_version_supported,
            )

            if not is_fa_version_supported(2):
                raise ValueError("Native Ouro 2D requires supported FlashAttention 2")
            if self.device.type != "cuda" or self.dtype not in (
                torch.float16,
                torch.bfloat16,
            ):
                raise ValueError("FlashAttention requires CUDA float16 or bfloat16")
            self._flash = flash_attn_varlen_func
            self.fa_version = select_fa_version()
            self._write_cache = reshape_and_cache_flash
        self.use_cuda_graph = (
            use_cuda_graph and self.device.type == "cuda" and self._flash is not None
            and self.compact_prefix is None
        )

    @property
    def kv_storage(self):
        if self.compact_prefix is not None:
            return "shared-final-ring"
        return "full-loop-planes"

    @property
    def kv_bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    @staticmethod
    def _loop_count(config) -> int:
        return getattr(config, "total_ut_steps", 4)

    def _validate_model(self, config) -> None:
        if self.max_graph_shapes < 0:
            raise ValueError("max_graph_shapes must be nonnegative")
        if self.max_len < 1 or self.block_size < 16 or self.block_size % 16:
            raise ValueError(
                "Positive capacity and a block size multiple of 16 required"
            )
        if self.attention_backend not in ("flash_attn", "sdpa"):
            raise ValueError("attention_backend must be 'flash_attn' or 'sdpa'")
        if len(self.layers) != self.L or getattr(self.backbone, "start_layer", 0):
            raise ValueError("Pipeline parallel Ouro execution is unsupported")
        if getattr(self.backbone, "end_layer", self.L) != self.L:
            raise ValueError("Pipeline parallel Ouro execution is unsupported")
        if any(i < 0 or i >= self.L for i in self.feat_layers):
            raise ValueError(
                "feat_layers must contain zero-based physical layer indices"
            )
        if len(set(self.feat_layers)) != self.nf:
            raise ValueError("feat_layers must be distinct")
        policy = getattr(config, "loopedtf_kv_policy", "full")
        if policy != "full" and not (self.shared_prefill and policy == "shared_last"):
            raise ValueError(
                "The 2DSD target runner supports full per-loop KV or shared_last"
            )
        if not getattr(config, "is_causal", True):
            raise ValueError("Native Ouro 2D requires a causal target")
        if getattr(config, "dual_chunk_attention_config", None) is not None:
            raise ValueError("Dual chunk attention is unsupported")
        for layer in self.layers:
            attn = layer.self_attn
            rotary_cache = getattr(attn.rotary_emb, "cos_sin_cache", None)
            if rotary_cache is not None and self.max_len > rotary_cache.shape[0]:
                raise ValueError("Target position capacity exceeds the RoPE cache")
            if (
                self.use_fused_rope
                and self.device.type == "cuda"
                and not callable(getattr(attn.rotary_emb, "forward_cuda", None))
            ):
                raise ValueError("Fused RoPE requires a CUDA rotary implementation")
            if attn.num_heads != config.num_attention_heads:
                raise ValueError("Tensor parallel Ouro execution is unsupported")
            for loop_attn in attn.attn:
                # Shared-last marks its early-loop vLLM caches as W windows; this
                # runner owns its prefix storage and never reads those caches.
                if policy != "shared_last" and getattr(
                    loop_attn.impl, "sliding_window", None
                ) not in (None, (-1, -1)):
                    raise ValueError("Sliding window attention is unsupported")

    @torch.inference_mode()
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(token_ids)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.model.compute_logits(hidden)
        if logits is None:
            raise RuntimeError("Native Ouro 2D requires local target logits")
        return logits

    @torch.inference_mode()
    def readout(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.logits(hidden).argmax(dim=-1)

    def truncate(self, n: int) -> None:
        """Invalidate rejected positions without clearing their unreachable KV."""
        if not 0 <= n <= self.max_len:
            raise ValueError("Truncation position is outside cache capacity")
        if self.compact_prefix is not None and any(
            n < length for length in self.length
        ):
            raise ValueError(
                "Committed compact prefix cannot rewind; reset and prefill"
            )
        self.length = [min(length, n) for length in self.length]

    def reset(self) -> None:
        """Start a new sequence while retaining allocated buffers and CUDA graphs."""
        self.length = [0] * self.R

    @torch.inference_mode()
    def prefill(self, token_ids: torch.Tensor) -> LoopResult:
        """Run a fresh prompt through all loops and return final-loop features."""
        if token_ids.ndim != 1 or not 0 < token_ids.numel() <= self.max_len:
            raise ValueError("prefill expects a nonempty one-dimensional token tensor")
        self.reset()
        if self.compact_prefix is not None:
            shape = (
                self.blocks_per_loop, self.block_size, self.kv_heads, self.head_dim
            )
            scratch = torch.zeros(shape, device=self.device, dtype=self.dtype)
            self._prefill_scratch = scratch, torch.zeros_like(scratch)
        try:
            hidden = self.embed(token_ids.to(device=self.device, dtype=torch.long))
            for loop in range(self.R):
                result = self.run(loop, 0, hidden)
                hidden = result.hidden
            return result
        finally:
            self._prefill_scratch = None

    @torch.inference_mode()
    def run(self, loop: int, start: int, hidden: torch.Tensor) -> LoopResult:
        return self.advance([Span(loop, start, hidden)])[0]

    def _validate_spans(self, spans: list[Span]) -> None:
        if not spans:
            raise ValueError("advance requires at least one span")
        if len({span.loop for span in spans}) != len(spans):
            raise ValueError("advance accepts at most one span per loop")
        for span in spans:
            if not 0 <= span.loop < self.R:
                raise ValueError("Span loop is outside the model's loop range")
            if span.start != self.length[span.loop]:
                raise ValueError(
                    f"Loop {span.loop} holds {self.length[span.loop]} positions; "
                    f"span starts at {span.start}"
                )
            if span.hidden.ndim != 2 or span.hidden.shape[1] != self.H:
                raise ValueError(f"Span hidden must have shape [n, {self.H}]")
            if span.hidden.device != self.device or span.hidden.dtype != self.dtype:
                raise ValueError("Span hidden must match the model device and dtype")
            if not 0 < span.hidden.shape[0] <= self.max_len - span.start:
                raise ValueError("Span is empty or exceeds cache capacity")

    def _make_inputs(self, spans: list[Span]) -> _Inputs:
        lengths = tuple(span.hidden.shape[0] for span in spans)
        loops = tuple(span.loop for span in spans)
        total = sum(lengths)
        # One transfer updates all dynamic graph metadata. Byte views keep the
        # int64 positions/slots and int32 FA lengths in one contiguous buffer.
        cpu_metadata = torch.empty(
            total * 16 + len(spans) * 4,
            dtype=torch.uint8,
            pin_memory=self.use_cuda_graph,
        )
        metadata = torch.empty_like(cpu_metadata, device=self.device)
        positions, slots, sequence_lengths = self._metadata_views(
            metadata, total, len(spans)
        )
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        inputs = _Inputs(
            hidden=torch.empty(total, self.H, device=self.device, dtype=self.dtype),
            metadata=metadata,
            cpu_metadata=cpu_metadata,
            positions=positions,
            slots=slots,
            cu_query=torch.tensor(cumulative, device=self.device, dtype=torch.int32),
            sequence_lengths=sequence_lengths,
            block_table=self._block_table[list(loops)].contiguous(),
            max_query=max(lengths),
            lengths=lengths,
            loops=loops,
            transfer_event=torch.cuda.Event() if self.use_cuda_graph else None,
        )
        self._update_inputs(inputs, spans)
        return inputs

    @staticmethod
    def _metadata_views(metadata: torch.Tensor, total: int, batch: int):
        positions = metadata[: total * 8].view(torch.int64)
        slots = metadata[total * 8 : total * 16].view(torch.int64)
        lengths = metadata[total * 16 : total * 16 + batch * 4].view(torch.int32)
        return positions, slots, lengths

    def _update_inputs(self, inputs: _Inputs, spans: list[Span]) -> None:
        if inputs.transfer_event is not None:
            # A previous asynchronous copy may still read this pinned buffer.
            inputs.transfer_event.synchronize()
        positions, slots, sequence_lengths = self._metadata_views(
            inputs.cpu_metadata, sum(inputs.lengths), len(spans)
        )
        offset = 0
        for i, span in enumerate(spans):
            length = span.hidden.shape[0]
            end = offset + length
            torch.arange(span.start, span.start + length, out=positions[offset:end])
            torch.add(
                positions[offset:end],
                0 if self.compact_prefix is not None else span.loop * self.padded_len,
                out=slots[offset:end],
            )
            sequence_lengths[i] = span.start + length
            offset = end
        inputs.metadata.copy_(inputs.cpu_metadata, non_blocking=self.use_cuda_graph)
        if inputs.transfer_event is not None:
            inputs.transfer_event.record(torch.cuda.current_stream(self.device))
        torch.cat([span.hidden for span in spans], out=inputs.hidden)

    def _attention(self, layer_idx: int, q, k, v, inputs: _Inputs) -> torch.Tensor:
        attn = self.layers[layer_idx].self_attn
        q = q.view(-1, self.heads, self.head_dim)
        k = k.view(-1, self.kv_heads, self.head_dim)
        v = v.view(-1, self.kv_heads, self.head_dim)
        if self.compact_prefix is not None:
            if self._prefill_scratch is None or len(inputs.loops) != 1:
                raise ValueError(
                    "Compact prefix owner only supports native prompt prefill"
                )
            cache_k, cache_v = self._prefill_scratch
            self.compact_prefix.store_prefill(layer_idx, inputs.loops[0], k, v)
        else:
            cache_k, cache_v = self.k[layer_idx], self.v[layer_idx]
        if self._flash is not None:
            self._write_cache(
                k,
                v,
                cache_k,
                cache_v,
                inputs.slots,
                "auto",
                self._scale,
                self._scale,
            )
            return self._flash(
                q=q,
                k=cache_k,
                v=cache_v,
                cu_seqlens_q=inputs.cu_query,
                max_seqlen_q=inputs.max_query,
                seqused_k=inputs.sequence_lengths,
                max_seqlen_k=self.max_len,
                block_table=inputs.block_table,
                causal=True,
                softmax_scale=attn.scaling,
                fa_version=self.fa_version,
            ).view(-1, self.heads * self.head_dim)
        flat_k = cache_k.view(-1, self.kv_heads, self.head_dim)
        flat_v = cache_v.view_as(flat_k)
        flat_k.index_copy_(0, inputs.slots, k)
        flat_v.index_copy_(0, inputs.slots, v)
        result = []
        offset = 0
        for loop, length in zip(inputs.loops, inputs.lengths):
            end = self.length[loop] + length
            begin = 0 if self.compact_prefix is not None else loop * self.padded_len
            keys = flat_k[begin : begin + end].transpose(0, 1)[None]
            values = flat_v[begin : begin + end].transpose(0, 1)[None]
            query = q[offset : offset + length].transpose(0, 1)[None]
            visible = (
                torch.arange(end, device=self.device)[None, :]
                <= (inputs.positions[offset : offset + length, None])
            )
            output = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                attn_mask=visible[None, None],
                scale=attn.scaling,
                enable_gqa=self.heads != self.kv_heads,
            )
            result.append(output[0].transpose(0, 1))
            offset += length
        return torch.cat(result).reshape(-1, self.heads * self.head_dim)

    @contextmanager
    def _norm_context(self):
        # This runner bypasses the compiled model forward. A process priority
        # of "native" would capture individual PyTorch normalization kernels,
        # rather than the normalization fusion provided by that compiler.
        if not self.use_fused_norm or self.device.type != "cuda":
            yield
            return
        import vllm.kernels  # noqa: F401
        from vllm import ir

        with ExitStack() as stack:
            for op in (ir.ops.rms_norm, ir.ops.fused_add_rms_norm):
                stack.enter_context(op.set_priority(["vllm_c", "native"]))
            yield

    def _apply_rope(self, rotary_emb, positions, query, key):
        # The compiled worker selects PyTorch CustomOps for normal Inductor
        # forwards. This private graph bypasses Inductor, so call CUDA directly
        # without changing the model's global CustomOp dispatch. The kernel
        # updates the strided Q/K views of this layer's fresh packed QKV.
        if self.use_fused_rope and self.device.type == "cuda":
            return rotary_emb.forward_cuda(positions, query, key)
        return rotary_emb(positions, query, key)

    def _forward(self, inputs: _Inputs) -> LoopResult:
        with self._norm_context():
            return self._forward_layers(inputs)

    def _forward_layers(self, inputs: _Inputs) -> LoopResult:
        # Fused residual normalization mutates its residual buffer. Keep staging
        # inputs intact for warmup/capture and for subsequent graph replays.
        hidden = inputs.hidden.clone()
        residual = None
        features = torch.empty(
            hidden.shape[0], self.nf * self.H, device=self.device, dtype=self.dtype
        )
        for idx, layer in enumerate(self.layers):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(hidden)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q, k = self._apply_rope(attn.rotary_emb, inputs.positions, q, k)
            output = self._attention(idx, q, k, v, inputs)
            hidden, _ = attn.o_proj(output)
            hidden = layer.input_layernorm_2(hidden)
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.mlp(hidden)
            hidden = layer.post_attention_layernorm_2(hidden)
            slot = self._feature_slots.get(idx)
            if slot is not None:
                features[:, slot * self.H : (slot + 1) * self.H] = hidden + residual
        hidden, _ = self.backbone.norm(hidden, residual)
        return LoopResult(hidden, features)

    def _capture(self, inputs: _Inputs) -> _Captured:
        # Warmup resolves lazy kernels before capture. Repeated writes replace
        # identical positions; causal masking excludes all later stale slots.
        started = time.perf_counter()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward(inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self._forward(inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.stats["captures"] += 1
        self.stats["capture_seconds"] += time.perf_counter() - started
        return _Captured(inputs, graph, output)

    @torch.inference_mode()
    def advance(self, spans: list[Span]) -> list[LoopResult]:
        """Advance all spans together and update only the participating histories."""
        if self.compact_prefix is not None and self._prefill_scratch is None:
            raise ValueError("Compact shared prefix decode requires the tree runner")
        self._validate_spans(spans)
        lengths = tuple(span.hidden.shape[0] for span in spans)
        key = (tuple(span.loop for span in spans), lengths)
        wants_graph = self.use_cuda_graph and max(lengths) <= self.graph_max_query_len
        captured = self._graphs.get(key) if wants_graph else None
        if wants_graph and (
            captured is not None or len(self._graphs) < self.max_graph_shapes
        ):
            if captured is None:
                inputs = self._make_inputs(spans)
                captured = self._capture(inputs)
                self._graphs[key] = captured
            else:
                inputs = captured.inputs
                self._update_inputs(inputs, spans)
            captured.graph.replay()
            output = captured.output
            self.stats["graph_replays"] += 1
        else:
            self.stats["graph_cache_fallbacks"] += int(wants_graph)
            inputs = self._make_inputs(spans)
            output = self._forward(inputs)
        for span, length in zip(spans, inputs.lengths):
            self.length[span.loop] = span.start + length
        self.stats["advances"] += 1
        self.stats["cells"] += sum(inputs.lengths)
        results, offset = [], 0
        for length in inputs.lengths:
            results.append(
                LoopResult(
                    output.hidden[offset : offset + length],
                    output.features[offset : offset + length],
                )
            )
            offset += length
        return results
