# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Depth-conditioned DFlash drafter weights, caches and kernels for 2DSD.

The export loader, verified-prefix context projection and attention helpers
shared by the loop-context drafter (looped_dflash_context.py). Verified loop-R
features are projected into K/V once and cached incrementally.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def _with_empty_key_rows(flash):
    """FlashAttention varlen whose key-less sequences return a zero output and -inf LSE, as FA2 does.

    The adapters' prefix/span merges rely on that; FA3 leaves those rows undefined. The rows are masked
    on the GPU without a host sync, so the call stays capturable in CUDA graphs.
    """

    def call(q, k, v, *, cu_seqlens_q, cu_seqlens_k, return_softmax_lse=False, **kwargs):
        result = flash(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                       return_softmax_lse=return_softmax_lse, **kwargs)
        out, lse = result if return_softmax_lse else (result, None)
        rows = torch.arange(q.shape[0], device=q.device, dtype=cu_seqlens_q.dtype)
        sequence = torch.searchsorted(cu_seqlens_q[1:], rows, right=True).clamp_(max=cu_seqlens_q.numel() - 2)
        empty = (cu_seqlens_k[1:] == cu_seqlens_k[:-1])[sequence]
        out.masked_fill_(empty[:, None, None], 0)
        if lse is None:
            return out
        lse.masked_fill_(empty[None, :], float("-inf"))
        return out, lse

    return call


@dataclass
class DraftResult:
    token_ids: torch.Tensor
    confidence: torch.Tensor | None = None


@dataclass
class _LayerWeights:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o_proj: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor


class LoopedDFlashAdapter:
    """Drafter state sharing the target embedding and logits callable.

    This adapter uses contiguous per-layer K/V and native vLLM CUDA operators.
    It does not require the target's paged-attention metadata. Reset between
    requests and truncate before overwriting any cached verified features.

    Args:
        config: Exported depth-conditioned DFlash configuration.
        weights: Exported drafter tensors; the copied LM head is ignored.
        mask_embedding: Learned mask vector, or the reserved target embedding.
        embed: Target token embedding callable.
        lm_head: Target logits callable returning a tensor, not a linear tuple.
        max_len: Maximum token position plus one, including speculative slots.
        device: Drafter device.
        dtype: Drafter computation dtype.
        use_native_ops: Enable fused CUDA ops and FlashAttention on CUDA.
        use_cuda_graph: Capture and reuse drafting graphs.
        batched_graph_min_calls: Capture on this observed layout call (default 8).
        batched_graph_max_shapes: Maximum resident multi-leaf graph shapes.
        batched_graph_max_bytes: Conservative resident graph memory budget.
        batched_graph_max_shape_bytes: Oversized shapes execute eagerly.
    """

    def __init__(
        self,
        config: Mapping,
        weights: Mapping[str, torch.Tensor],
        mask_embedding: torch.Tensor,
        *,
        embed: Callable,
        lm_head: Callable,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        use_native_ops: bool = True,
        use_cuda_graph: bool = True,
        batched_graph_min_calls: int = 8,
        batched_graph_max_shapes: int = 64,
        batched_graph_max_bytes: int = 512 << 20,
        batched_graph_max_shape_bytes: int = 32 << 20,
    ):
        self.config = dict(config)
        self.device, self.dtype = torch.device(device), dtype
        self.embed, self.lm_head = embed, lm_head
        self.max_len = max_len
        self.hidden_size = int(config["hidden_size"])
        self.head_dim = int(config["head_dim"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.num_layers = int(config["num_hidden_layers"])
        self.block_size = int(config.get("block_size", 16))
        self.eps = float(config["rms_norm_eps"])
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.native = use_native_ops and self.device.type == "cuda"
        self.use_cuda_graph = use_cuda_graph and self.native
        if batched_graph_min_calls < 1:
            raise ValueError("Batched graph minimum calls must be positive")
        self.batched_graph_min_calls = batched_graph_min_calls
        if (
            min(
                batched_graph_max_shapes,
                batched_graph_max_bytes,
                batched_graph_max_shape_bytes,
            )
            < 1
        ):
            raise ValueError("Batched graph shape and memory budgets must be positive")
        self.batched_graph_max_shapes = batched_graph_max_shapes
        self.batched_graph_max_bytes = batched_graph_max_bytes
        self.batched_graph_max_shape_bytes = batched_graph_max_shape_bytes
        self.stats = {
            "graph_captures": 0,
            "capture_seconds": 0.0,
            "graph_replays": 0,
            "context_tokens_projected": 0,
            "context_weight_passes": 0,
            "context_projection_rows": 0,
            "query_weight_passes": 0,
            "query_rows": 0,
            "leaf_drafts": 0,
        }
        if config.get("attention_bias", False):
            raise ValueError("The 2DSD drafter requires bias-free attention")
        if config.get("use_sliding_window") or config.get("rope_scaling"):
            raise ValueError("Only full attention and default RoPE are supported")
        if any(x != "full_attention" for x in config.get("layer_types", [])):
            raise ValueError("The 2DSD drafter requires full attention")
        if config.get("dflash_config", {}).get("causal", False):
            raise ValueError("Draft blocks require bidirectional attention")
        if self.num_heads % self.num_kv_heads or max_len < 1:
            raise ValueError("Invalid head ratio or drafter capacity")

        def tensor(x):
            return x.detach().to(device=self.device, dtype=dtype).contiguous()

        def weight(name):
            return tensor(weights[name])

        self.fc = weight("fc.weight")
        self.hidden_norm = weight("hidden_norm.weight")
        self.norm = weight("norm.weight")
        self.mask_embedding = tensor(mask_embedding).reshape(self.hidden_size)
        self.layers = []
        context_weights = []
        for i in range(self.num_layers):
            p = f"layers.{i}."
            a = p + "self_attn."
            self.layers.append(
                _LayerWeights(
                    weight(p + "input_layernorm.weight"),
                    weight(p + "post_attention_layernorm.weight"),
                    tensor(
                        torch.cat(
                            [weights[a + x + "_proj.weight"] for x in ("q", "k", "v")]
                        )
                    ),
                    weight(a + "q_norm.weight"),
                    weight(a + "k_norm.weight"),
                    weight(a + "o_proj.weight"),
                    tensor(
                        torch.cat(
                            [
                                weights[p + "mlp." + x + "_proj.weight"]
                                for x in ("gate", "up")
                            ]
                        )
                    ),
                    weight(p + "mlp.down_proj.weight"),
                )
            )
            context_weights.extend(weights[a + x + "_proj.weight"] for x in ("k", "v"))
        self.context_kv = tensor(torch.cat(context_weights))
        self.context_k_norm = torch.stack([x.k_norm for x in self.layers])
        shape = (self.num_layers, max_len, self.num_kv_heads, self.head_dim)
        # FA3 reads V rows past a sequence's length within a tile; keep unwritten rows finite.
        self.k_cache = torch.zeros(shape, device=self.device, dtype=dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.cached_prefix = 0
        self.context_tokens_projected = 0
        self.positions = torch.arange(max_len, device=self.device)
        inv_freq = 1.0 / (
            float(config.get("rope_theta", 1e6))
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        freq = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.rope_cache = tensor(torch.cat((freq.cos(), freq.sin()), dim=-1))
        self._cu_seqlens: dict[int, torch.Tensor] = {}
        self.fa_version = None
        if self.native:
            from vllm import _custom_ops
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            from vllm.model_executor.models.ouro_2d import select_fa_version

            self._ops = _custom_ops
            self.fa_version = select_fa_version()
            self._flash_attention = (flash_attn_varlen_func if self.fa_version == 2
                                     else _with_empty_key_rows(flash_attn_varlen_func))

    @classmethod
    def from_export(
        cls,
        path: str | Path,
        *,
        embed: Callable,
        lm_head: Callable,
        max_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        use_native_ops: bool = True,
        use_cuda_graph: bool = True,
        batched_graph_min_calls: int = 8,
        batched_graph_max_shapes: int = 64,
        batched_graph_max_bytes: int = 512 << 20,
        batched_graph_max_shape_bytes: int = 32 << 20,
    ) -> "LoopedDFlashAdapter":
        """Load an existing export without loading its duplicate target LM head."""
        from safetensors import safe_open

        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        probe = embed(torch.zeros(1, dtype=torch.long, device=device))
        device = probe.device if device is None else device
        dtype = probe.dtype if dtype is None else dtype
        with safe_open(path / "model.safetensors", framework="pt") as handle:
            weights = {
                k: handle.get_tensor(k)
                for k in tuple(handle.keys())
                if k != "lm_head.weight"
            }
        mask_path = path / "mask_embedding.pt"
        if mask_path.exists():
            mask = torch.load(mask_path, map_location="cpu", weights_only=True)
            expected = config["dflash_config"]["mask_token_id"]
            if int(mask["mask_token_id"]) != expected:
                raise ValueError("Learned mask token disagrees with export config")
            mask_embedding = mask["embedding"]
        else:
            mask_id = config["dflash_config"]["mask_token_id"]
            mask_embedding = embed(torch.tensor([mask_id], device=device))[0]
        return cls(
            config,
            weights,
            mask_embedding,
            embed=embed,
            lm_head=lm_head,
            max_len=max_len,
            device=device,
            dtype=dtype,
            use_native_ops=use_native_ops,
            use_cuda_graph=use_cuda_graph,
            batched_graph_min_calls=batched_graph_min_calls,
            batched_graph_max_shapes=batched_graph_max_shapes,
            batched_graph_max_bytes=batched_graph_max_bytes,
            batched_graph_max_shape_bytes=batched_graph_max_shape_bytes,
        )

    def reset(self):
        """Start a request, retaining graphs, layout counts and cumulative stats."""
        self.cached_prefix = 0
        self.context_tokens_projected = 0

    def truncate(self, n: int):
        """Invalidate cached verified positions at and after ``n``."""
        if n < 0:
            raise ValueError("Prefix length cannot be negative")
        self.cached_prefix = min(self.cached_prefix, n)

    def _rms(self, x, weight):
        if self.native:
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            self._ops.rms_norm(out, x, weight, self.eps)
            return out
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        if weight.ndim == 2:
            weight = weight[:, None, None, :]
        return y.to(x.dtype) * weight

    def _rotate(self, x, positions):
        if self.native:
            shape = x.shape
            x = x.reshape(-1, shape[-2] * shape[-1]).contiguous()
            self._ops.rotary_embedding(
                positions, x, None, self.head_dim, self.rope_cache, True
            )
            return x.view(shape)
        cos, sin = self.rope_cache[positions].chunk(2, dim=-1)
        cos = torch.cat((cos, cos), dim=-1)[:, None]
        sin = torch.cat((sin, sin), dim=-1)[:, None]
        half1, half2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-half2, half1), dim=-1) * sin

    def _project_context(self, features, positions):
        self.stats["context_weight_passes"] += 1
        self.stats["context_projection_rows"] += features.shape[0]
        return self._project_context_forward(features, positions)

    def _project_context_forward(self, features, positions):
        """Tensor-only projection, also used inside the loop-context query graph."""
        ctx = self._rms(F.linear(features.to(self.dtype), self.fc), self.hidden_norm)
        kv = F.linear(ctx, self.context_kv).view(
            -1, self.num_layers, 2, self.num_kv_heads, self.head_dim
        )
        keys = kv[:, :, 0].permute(1, 0, 2, 3).contiguous()
        keys = self._rms(keys, self.context_k_norm)
        keys = self._rotate(
            keys.flatten(0, 1), positions.repeat(self.num_layers)
        ).view_as(keys)
        values = kv[:, :, 1].permute(1, 0, 2, 3)
        return keys, values

    def _cu(self, n):
        if n not in self._cu_seqlens:
            self._cu_seqlens[n] = torch.tensor(
                [0, n], device=self.device, dtype=torch.int32
            )
        return self._cu_seqlens[n]

    def _attention_lse(self, q, k, v, *, causal=False):
        if self.native:
            return self._flash_attention(
                q,
                k,
                v,
                max_seqlen_q=q.shape[0],
                cu_seqlens_q=self._cu(q.shape[0]),
                max_seqlen_k=k.shape[0],
                cu_seqlens_k=self._cu(k.shape[0]),
                causal=causal,
                return_softmax_lse=True,
                fa_version=self.fa_version,
            )
        if self.num_heads != self.num_kv_heads:
            groups = self.num_heads // self.num_kv_heads
            k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float())
        scores *= self.head_dim**-0.5
        if causal:
            mask = torch.ones(
                q.shape[0], k.shape[0], dtype=torch.bool, device=q.device
            ).tril(k.shape[0] - q.shape[0])
            scores.masked_fill_(~mask[None], -torch.inf)
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
        return out.to(q.dtype), scores.logsumexp(-1)

    def _many_graph_estimate(self, layouts):
        """Conservative graph workspace estimate; the full KV cache is shared.

        Reserve 16 hidden-width and 6 intermediate-width row buffers for QKV,
        attention, residuals and the MLP, plus logits, stable inputs and 1 MiB
        allocator slack. This is about 8 MiB for Q=40, four k=4 leaves in the
        current H=2048/I=5632 export. Actual resident deltas also charge the
        cache, so underestimation cannot silently grow the resident graph set.
        """
        rows = sum(inflight + k for _, inflight, k in layouts)
        predictions = sum(k - 1 for _, _, k in layouts)
        leaves = len(layouts)
        itemsize = self.mask_embedding.element_size()
        intermediate = self.layers[0].gate_up.shape[0] // 2
        vocab = int(self.config.get("vocab_size", 49152))
        workspace = rows * (16 * self.hidden_size + 6 * intermediate) * itemsize
        stable = rows * self.hidden_size * itemsize + rows * 8 * 8
        stable += 2 * self.num_layers * leaves * self.kv_size * itemsize
        return workspace + stable + predictions * vocab * itemsize + (1 << 20)
