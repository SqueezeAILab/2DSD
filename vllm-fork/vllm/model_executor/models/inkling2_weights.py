# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Inkling2 checkpoint loader with exact expert dequantization.

Reads a released looped Inkling2 checkpoint (``config.json``,
``hf_quant_config.json`` and the safetensors shards) into plain tensors without
vLLM's Inkling2 model class, which the fork does not have and which rejects dense
experts upstream. The layout below is derived from vLLM PR #69028 at b426b46e
(``PR/``; ``NV/`` is ``PR/vllm/models/inkling2/nvidia/``) and transformers PR
#51050 at 6173a2f (``TF/``, under ``src/transformers/``), and checked against the
header of ``thinkingmachines/Inkling-2-Mini`` at 1fb4134b. Nothing is imported
from either source.

Only the routed experts ``model.llm.layers.N.mlp.experts.*`` are quantized
(``PR/vllm/models/inkling2/quant_config.py:5-6``). With E experts, hidden size H
and intermediate size I, every tensor is ``[E, out, in]`` in row-major order as
stored; upstream shuffles rows and swizzles block scales only after loading, on
SM100 (``NV/moe.py:187-195``):

- ``w13_weight`` is the gate/up projection ``[E, 2I, H]``. Output rows are
  interleaved pairs: row ``2j`` is gate ``j`` and row ``2j + 1`` is up ``j``
  (``PR/tests/models/inkling2/test_moe.py:525``, ``NV/ops/activation.py:62-78``,
  ``Interleave(dim=1)`` in ``TF/conversion_mapping.py:2666``). ``w2_weight`` is the
  down projection ``[E, H, I]`` (``NV/moe.py:112``).
- NVFP4 layers: ``*_weight`` is U8 ``[E, out, in / 2]``. A byte holds two E2M1
  codes, element ``2k`` in the low nibble and ``2k + 1`` in the high nibble; a
  code is a sign (bit 3) and three bits indexing (0, 0.5, 1, 1.5, 2, 3, 4, 6)
  (``NV/ops/nvfp4.py:27-63`` and ``:132-139``,
  ``PR/tests/models/inkling2/test_moe.py:43-49``, ``TF/core_model_loading.py:777``).
  ``*_weight_scale`` is F8_E4M3 ``[E, out, in / 16]``, one block scale per 16
  consecutive input elements of a row (``NV/moe.py:109``). ``*_weight_amax`` is F32
  ``[E]`` and ``*_weight_outer_scale_denominator`` F32 ``[1]`` (2688 = 6 * 448); the
  outer scale of an expert is their FP32 quotient (``NV/moe.py:213``,
  ``PR/tests/models/inkling2/test_moe.py:505-510``)::

      W[e, r, c] = e2m1(code[e, r, c]) * e4m3(scale[e, r, c // 16])
                   * fl32(amax[e] / denominator)

- MXFP8 layers: ``*_weight`` is F8_E4M3 ``[E, out, in]`` and ``*_weight_scale`` U8
  ``[E, out, in / 32]``, a power-of-two exponent per 32 consecutive input
  elements of a row, with no outer scale (``NV/ops/mxfp8.py:3`` and ``:19-22``,
  ``PR/tests/models/inkling2/test_moe.py:52-54``, ``TF/core_model_loading.py:779``).
  ``hf_quant_config.json`` lists ``weight_block_shape`` [32, 32], but the scales
  are stored per row and differ between the rows of a 32-row group::

      W[e, r, c] = e4m3(value[e, r, c]) * 2 ** (scale[e, r, c // 32] - 127)

Five properties of the Mini weights agree with this reading. In every NVFP4
expert the largest element of the dense factor is exactly 2688, so the outer
scale restores the stored amax. Under the low-nibble-first order the column
norms of ``w13`` correlate with ``mlp_norm.weight`` (0.36 to 0.60 on three
experts) and not under the other order (below 0.05). Rows ``2j`` and ``2j + 1``
of ``w13`` are more aligned (mean absolute cosine 0.06 to 0.16) than rows ``j``
and ``I + j`` (0.017 to 0.019, the level of unrelated rows). Column ``j`` of
``w2`` is aligned with rows ``2j`` and ``2j + 1`` of ``w13`` (0.03 to 0.12 on four
experts) and at the level of unrelated directions when either projection is read
with the other nibble order or the rows are paired as ``j`` and ``I + j``, which
fixes the nibble order of ``w2`` too. The signed cosine of that column has a
positive mean with row ``2j + 1`` (5 to 64 standard errors on ten experts) and
none with row ``2j`` (below 2): the activation is odd in ``up`` only, so only
the up row can keep a signed alignment with its output column.

Exactness. ``e2m1 * e4m3`` has at most six significant bits, so the NVFP4 factor
without the outer scale is exact in BF16; the outer scale is a general FP32
number that does not fit BF16, and even an FP32 product with it is rounded. The
loader therefore returns the dense factor and the FP32 scale separately, to be
applied as ``(x @ W'.T) * scale`` on an FP32 accumulator; the Hopper path of the
PR also hands the quotient to its GEMM as a separate per-expert scale
(``NV/moe.py:207-224``). ``TF/core_model_loading.py:814-815`` folds the scale and
casts, which rounds every NVFP4 weight. MXFP8 products are exact in BF16 for
block exponents 3 to 246; the Mini checkpoint uses 113 to 118. Dequantization
raises when a product is not exactly representable in the requested dtype.

Vocabulary. ``embed.weight`` and ``unembed.weight`` have ``vocab_size`` rows
(201,024 for Mini), of which only the first ``unpadded_vocab_size`` (200,058) are
served (``PR/vllm/models/inkling2/configs.py:70-72``,
``PR/vllm/model_executor/layers/logits_processor.py:198``). The padding rows of
the Mini checkpoint are not zero, so a head over every row can predict an id
that does not exist; use ``Inkling2Checkpoint.embedding`` and ``unembedding``.
"""

import json
import math
import mmap
import struct
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch

LLM_PREFIX = "model.llm."
VISION_PREFIX = "model.visual."
EMBED_NAME = LLM_PREFIX + "embed.weight"
UNEMBED_NAME = LLM_PREFIX + "unembed.weight"
LOOP_STREAMS = ("attn", "k", "v", "mlp")
PROJECTIONS = ("w13", "w2")
GATE_ROWS = slice(0, None, 2)
UP_ROWS = slice(1, None, 2)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32

ExpertMode = Literal["dense", "packed", "none"]
QuantFormat = Literal["nvfp4", "mxfp8"]

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_CODES = (*_E2M1, *(-value for value in _E2M1))
_POW2 = tuple(math.ldexp(1.0, exponent - 127) for exponent in range(256))
# Block exponents for which every non-zero MXFP8 product is a normal BF16 / FP32
# number; other exponents are checked value by value.
_MXFP8_NORMAL_EXPONENTS = (10, 246)
_CHUNK_ELEMENTS = 1 << 24
_DENSE_DTYPES = (torch.bfloat16, torch.float32, torch.float64)
_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}
_QUANT_COMPUTE_DTYPES = ("float32", "bfloat16", "mxfp8", "nvfp4")
_REQUIRED_TEXT_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "d_rel",
    "rel_extent",
    "local_layer_ids",
    "sliding_window_size",
    "sconv_kernel_size",
    "n_layers_per_attn_res_block",
    "n_routed_experts",
    "num_experts_per_tok",
    "intermediate_size",
    "route_scale",
    "mlp_gate_softcap",
    "mlp_up_softcap",
    "final_logit_softcapping",
    "qk_l2_clip_threshold",
    "rel_l2_clip_threshold",
    "log_scaling_n_floor",
    "log_scaling_alpha",
    "rms_norm_eps",
    "model_max_length",
    "loop_config",
)
_LOOP_INT_KEYS = ("core_start", "core_end", "num_passes", "gate_rank")
Schema = dict[str, tuple[str, tuple[int, ...]]]


def _e4m3_codes() -> tuple[float, ...]:
    values = []
    for code in range(256):
        exponent, mantissa = (code >> 3) & 0xF, code & 0x7
        if exponent == 0xF and mantissa == 0x7:
            magnitude = math.nan
        elif exponent == 0:
            magnitude = math.ldexp(mantissa, -9)
        else:
            magnitude = math.ldexp(8 + mantissa, exponent - 10)
        values.append(-magnitude if code & 0x80 else magnitude)
    return tuple(values)


_E4M3_CODES = _e4m3_codes()


@dataclass(frozen=True)
class Inkling2Config:
    """Text configuration of a looped Inkling2 checkpoint.

    Fields come from ``text_config`` of ``config.json`` (the vision tower and the
    DSpark draft are ignored). ``padded_vocab_size`` is the row count of the
    embedding tables and ``served_vocab_size`` the number of ids the model can
    emit. The three trailing ``*_eps`` fields are constants of the PR code, not
    checkpoint fields: ``NV/ops/attn_res.py:252``, ``NV/ops/loop.py:57`` and
    ``NV/ops/sconv.py:862``.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    d_rel: int
    rel_extent: int
    local_layer_ids: tuple[int, ...]
    sliding_window_size: int
    sconv_kernel_size: int
    n_layers_per_attn_res_block: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    route_scale: float
    mlp_gate_softcap: float
    mlp_up_softcap: float
    final_logit_softcapping: float | None
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    log_scaling_n_floor: int | None
    log_scaling_alpha: float
    rms_norm_eps: float
    padded_vocab_size: int
    served_vocab_size: int
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    is_varloop: bool
    model_max_length: int
    eos_token_id: int | None = None
    attn_res_eps: float = 1e-6
    message_norm_eps: float = 1e-6
    l2_clip_eps: float = 1e-6

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "Inkling2Config":
        """Parses the content of ``config.json``.

        Raises:
            ValueError: If the checkpoint is not a looped Inkling2 model or uses
                an option the released checkpoints do not have.
        """
        if config.get("model_type") != "inkling2_loop":
            raise ValueError(
                f"model_type {config.get('model_type')!r} is not inkling2_loop"
            )
        text = config.get("text_config") or {}
        missing = [key for key in _REQUIRED_TEXT_KEYS if key not in text]
        if missing:
            raise ValueError(f"text_config lacks {missing}")
        loop = text["loop_config"]
        if (
            not isinstance(loop, Mapping)
            or set(loop) != {*_LOOP_INT_KEYS, "is_varloop"}
            or any(type(loop[key]) is not int for key in _LOOP_INT_KEYS)
            or type(loop["is_varloop"]) is not bool
        ):
            raise ValueError(
                f"loop_config needs integer {list(_LOOP_INT_KEYS)} and a boolean "
                "is_varloop"
            )
        unsupported = {
            "n_shared_experts": 0,
            "use_gate_bias": True,
            "use_value_norm_no_gamma": True,
            "attn_res_normalize_values": True,
            "q_bias": False,
            "o_bias": False,
            "pre_norm_pre_affine": False,
            "final_norm_pre_affine": False,
        }
        for key, value in unsupported.items():
            if text.get(key, value) != value:
                raise ValueError(f"text_config.{key}={text[key]!r} is not supported")
        vocab = text["vocab_size"]
        result = cls(
            **{
                key: text[key]
                for key in _REQUIRED_TEXT_KEYS
                if key not in ("vocab_size", "local_layer_ids", "loop_config")
            },
            local_layer_ids=tuple(text["local_layer_ids"] or ()),
            swa_num_attention_heads=text.get("swa_num_attention_heads")
            or text["num_attention_heads"],
            swa_num_key_value_heads=text.get("swa_num_key_value_heads")
            or text["num_key_value_heads"],
            swa_head_dim=text.get("swa_head_dim") or text["head_dim"],
            padded_vocab_size=vocab,
            served_vocab_size=text.get("unpadded_vocab_size") or vocab,
            **loop,
            eos_token_id=config.get("eos_token_id"),
        )
        result._validate()
        return result

    @classmethod
    def from_snapshot(cls, snapshot_dir: str | Path) -> "Inkling2Config":
        """Parses ``config.json`` of a snapshot directory."""
        path = Path(snapshot_dir) / "config.json"
        return cls.from_dict(json.loads(path.read_text()))

    def _validate(self) -> None:
        sizes = {f.name: getattr(self, f.name) for f in fields(self) if f.type is int}
        invalid = [
            name for name, value in sizes.items() if type(value) is not int or value < 1
        ]
        if invalid:
            raise ValueError(f"config.json needs positive integers for {invalid}")
        layers, block = self.num_hidden_layers, self.n_layers_per_attn_res_block
        if not 0 < self.core_start < self.core_end <= layers:
            raise ValueError("loop_config core layers are out of range")
        if self.core_start % block or self.core_end % block:
            raise ValueError(
                "loop_config core bounds must be multiples of "
                f"n_layers_per_attn_res_block={block}"
            )
        if self.num_passes < (1 if self.is_varloop else 2):
            raise ValueError(f"num_passes={self.num_passes} is too small")
        local = self.local_layer_ids
        if (
            any(type(layer) is not int for layer in local)
            or list(local) != sorted(set(local))
            or any(not 0 <= layer < layers for layer in local)
        ):
            raise ValueError("local_layer_ids must be sorted layer ids")
        if self.served_vocab_size > self.padded_vocab_size:
            raise ValueError("unpadded_vocab_size exceeds vocab_size")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok exceeds n_routed_experts")

    @property
    def core_layers(self) -> range:
        """Physical layers that run ``num_passes`` times."""
        return range(self.core_start, self.core_end)

    @property
    def num_layer_visits(self) -> int:
        """Layer executions per token; the final-norm output has this index."""
        return self.num_hidden_layers + (self.num_passes - 1) * len(self.core_layers)

    def is_local(self, layer_id: int) -> bool:
        """Whether ``layer_id`` uses sliding-window attention."""
        return layer_id in self.local_layer_ids

    def attention_dims(self, layer_id: int) -> tuple[int, int, int]:
        """Query heads, key/value heads and head dimension of ``layer_id``."""
        if self.is_local(layer_id):
            return (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            )
        return self.num_attention_heads, self.num_key_value_heads, self.head_dim

    def layer_rel_extent(self, layer_id: int) -> int:
        """Columns of the relative-bias table of ``layer_id``."""
        return self.sliding_window_size if self.is_local(layer_id) else self.rel_extent


def layer_prefix(layer_id: int) -> str:
    """Checkpoint name prefix of the tensors of physical layer ``layer_id``."""
    return f"{LLM_PREFIX}layers.{layer_id}."


def expert_prefix(layer_id: int, projection: str) -> str:
    """Checkpoint name of the packed weight of one expert projection."""
    return f"{layer_prefix(layer_id)}mlp.experts.{projection}_weight"


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8)
    raise TypeError(f"expected uint8 or float8_e4m3fn, got {tensor.dtype}")


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _DENSE_DTYPES:
        raise ValueError(f"dense dtype must be one of {_DENSE_DTYPES}, got {dtype}")
    return torch.float64 if dtype == torch.float64 else torch.float32


def _row_chunks(rows: int, row_elements: int) -> Iterator[slice]:
    step = max(1, _CHUNK_ELEMENTS // max(1, row_elements))
    for start in range(0, rows, step):
        yield slice(start, start + step)


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dense NVFP4 factor ``e2m1 * e4m3`` without the outer scale.

    Args:
        packed: ``[..., K / 2]`` bytes, two E2M1 codes each (low nibble first).
        block_scale: ``[..., K / 16]`` E4M3 scales, ``float8_e4m3fn`` or raw bytes.
        dtype: BF16, FP32 or FP64; every product is exact in each of them.

    Returns:
        ``[..., K]`` on the device of ``packed``, magnitudes at most 2688.

    Raises:
        ValueError: If the shapes disagree or a block scale is NaN.
    """
    work = _work_dtype(dtype)
    codes, scales = _raw_bytes(packed), _raw_bytes(block_scale)
    *lead, half = codes.shape
    blocks = 2 * half // NVFP4_BLOCK
    if 2 * half % NVFP4_BLOCK or tuple(scales.shape) != (*lead, blocks):
        raise ValueError(
            f"NVFP4 shapes disagree: {tuple(codes.shape)} and {tuple(scales.shape)}"
        )
    codes, scales = codes.reshape(-1, half), scales.reshape(-1, blocks)
    if bool(((scales & 0x7F) == 0x7F).any()):
        raise ValueError("NVFP4 block scales contain NaN")
    device = codes.device
    nibbles = torch.tensor(_E2M1_CODES, dtype=work, device=device)
    byte = torch.arange(256, device=device)
    pairs = torch.stack((nibbles[byte & 0xF], nibbles[byte >> 4]), dim=-1)
    e4m3 = torch.tensor(_E4M3_CODES, dtype=work, device=device)
    out = torch.empty((codes.shape[0], 2 * half), dtype=dtype, device=device)
    for rows in _row_chunks(codes.shape[0], 2 * half):
        values = pairs[codes[rows].int()].view(-1, blocks, NVFP4_BLOCK)
        values = values * e4m3[scales[rows].int()].unsqueeze(-1)
        out[rows] = values.flatten(1).to(dtype)
    return out.view(*lead, 2 * half)


def dequantize_mxfp8(
    values: torch.Tensor,
    block_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dense MXFP8 weights ``e4m3 * 2 ** (scale - 127)``.

    Args:
        values: ``[..., K]`` E4M3 values, ``float8_e4m3fn`` or raw bytes.
        block_scale: ``[..., K / 32]`` exponent bytes.
        dtype: BF16, FP32 or FP64.

    Returns:
        ``[..., K]`` on the device of ``values``.

    Raises:
        ValueError: If the shapes disagree, a value or scale is NaN, or a product
            is not exactly representable in ``dtype``.
    """
    work = _work_dtype(dtype)
    codes, exponents = _raw_bytes(values), _raw_bytes(block_scale)
    *lead, cols = codes.shape
    blocks = cols // MXFP8_BLOCK
    if cols % MXFP8_BLOCK or tuple(exponents.shape) != (*lead, blocks):
        raise ValueError(
            f"MXFP8 shapes disagree: {tuple(codes.shape)} and {tuple(exponents.shape)}"
        )
    codes, exponents = codes.reshape(-1, cols), exponents.reshape(-1, blocks)
    if bool(((codes & 0x7F) == 0x7F).any()) or bool((exponents == 0xFF).any()):
        raise ValueError("MXFP8 values or block scales contain NaN")
    device = codes.device
    e4m3 = torch.tensor(_E4M3_CODES, dtype=torch.float64, device=device)
    pow2 = torch.tensor(_POW2, dtype=torch.float64, device=device)
    low, high = _MXFP8_NORMAL_EXPONENTS
    verify = (
        work != torch.float64
        and exponents.numel() > 0
        and (int(exponents.min()) < low or int(exponents.max()) > high)
    )
    work_e4m3, work_pow2 = e4m3.to(work), pow2.to(work)
    out = torch.empty((codes.shape[0], cols), dtype=dtype, device=device)
    for rows in _row_chunks(codes.shape[0], cols):
        index = codes[rows].int().view(-1, blocks, MXFP8_BLOCK)
        shift = exponents[rows].int().unsqueeze(-1)
        out[rows] = (work_e4m3[index] * work_pow2[shift]).flatten(1).to(dtype)
        if verify:
            exact = (e4m3[index] * pow2[shift]).flatten(1)
            if not torch.equal(out[rows].double(), exact):
                raise ValueError(f"MXFP8 weights are not exact in {dtype}")
    return out.view(*lead, cols)


def nvfp4_outer_scale(amax: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """FP32 outer scale ``fl32(amax / denominator)`` of each NVFP4 expert.

    Raises:
        TypeError: If an input is not FP32.
        ValueError: If a scale is not finite.
    """
    if amax.dtype != torch.float32 or denominator.dtype != torch.float32:
        raise TypeError("NVFP4 amax and denominator must be float32")
    scale = amax / denominator
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("NVFP4 outer scales are not finite")
    return scale


def _global_schema(config: Inkling2Config) -> Schema:
    hidden, layers = config.hidden_size, config.num_hidden_layers
    table = (config.padded_vocab_size, hidden)
    core = 2 * len(config.core_layers)
    schema: Schema = {
        EMBED_NAME: ("BF16", table),
        UNEMBED_NAME: ("BF16", table),
        LLM_PREFIX + "embed_norm.weight": ("BF16", (hidden,)),
        LLM_PREFIX + "norm.weight": ("BF16", (hidden,)),
        LLM_PREFIX + "attn_res.output_query": ("F32", (hidden,)),
        LLM_PREFIX + "middle_cycle_cross_pass_queries": ("F32", (core, hidden)),
        LLM_PREFIX + "middle_cycle_cross_pass_kv_queries": ("F32", (core, hidden)),
    }
    for layer in range(layers):
        if layer:
            shape = (hidden,) if config.is_local(layer) else (3, hidden)
            schema[f"{LLM_PREFIX}attn_res.attn_queries.{layer - 1}"] = ("F32", shape)
        schema[f"{LLM_PREFIX}attn_res.mlp_queries.{layer}"] = ("F32", (hidden,))
    return schema


def _layer_schema(config: Inkling2Config, layer_id: int) -> Schema:
    hidden, kernel = config.hidden_size, config.sconv_kernel_size
    heads, kv_heads, head_dim = config.attention_dims(layer_id)
    q_size, kv_size = heads * head_dim, kv_heads * head_dim
    bf16 = {
        "attn.wq_du.weight": (q_size, hidden),
        "attn.wr_du.weight": (heads * config.d_rel, hidden),
        "attn.gate_attn.weight": (q_size, hidden),
        "attn.attn_sink_proj.weight": (heads, hidden),
        "attn.wk_dv.weight": (kv_size, hidden),
        "attn.wv_dv.weight": (kv_size, hidden),
        "attn.wo_ud.weight": (hidden, q_size),
        "attn.q_l2_clip.scale": (heads, head_dim),
        "attn.rel_l2_clip.scale": (heads, config.d_rel),
        "attn.rel_logits_proj.proj": (config.d_rel, config.layer_rel_extent(layer_id)),
        "attn_sconv.weight": (hidden, 1, kernel),
        "mlp_sconv.weight": (hidden, 1, kernel),
        "mlp_norm.weight": (hidden,),
        "mlp.gate.weight": (config.n_routed_experts, hidden),
    }
    if config.is_local(layer_id):
        bf16["attn_norm.weight"] = (hidden,)
        for stream in "kv":
            bf16[f"attn.{stream}_sconv.weight"] = (kv_size, 1, kernel)
    else:
        for stream in "qkv":
            bf16[f"{stream}_input_norm.weight"] = (hidden,)
        for stream in "kv":
            bf16[f"attn.{stream}_sconv.static_w"] = (kernel, kv_size)
            bf16[f"attn.{stream}_sconv.weight_proj"] = (kv_heads * kernel, hidden)
    if layer_id in config.core_layers:
        for stream in LOOP_STREAMS:
            prefix = f"controllers.{stream}."
            bf16[prefix + "down_weight"] = (config.gate_rank, 2 * hidden)
            bf16[prefix + "gain_weight"] = (hidden, config.gate_rank)
            bf16[prefix + "forget_weight"] = (hidden, config.gate_rank)
    prefix = layer_prefix(layer_id)
    schema: Schema = {prefix + name: ("BF16", shape) for name, shape in bf16.items()}
    schema[prefix + "mlp.gate.bias"] = ("F32", (config.n_routed_experts,))
    return schema


def _expert_schema(
    config: Inkling2Config, layer_id: int, quant_format: QuantFormat
) -> Schema:
    experts, hidden = config.n_routed_experts, config.hidden_size
    inter = config.intermediate_size
    nvfp4 = quant_format == "nvfp4"
    pack, block = (2, NVFP4_BLOCK) if nvfp4 else (1, MXFP8_BLOCK)
    weight_dtype, scale_dtype = ("U8", "F8_E4M3") if nvfp4 else ("F8_E4M3", "U8")
    schema: Schema = {}
    for projection, (rows, cols) in zip(
        PROJECTIONS, ((2 * inter, hidden), (hidden, inter))
    ):
        if cols % block:
            raise ValueError(
                f"{quant_format} needs input sizes divisible by {block}, got {cols}"
            )
        name = expert_prefix(layer_id, projection)
        schema[name] = (weight_dtype, (experts, rows, cols // pack))
        schema[name + "_scale"] = (scale_dtype, (experts, rows, cols // block))
        if nvfp4:
            schema[name + "_amax"] = ("F32", (experts,))
            schema[name + "_outer_scale_denominator"] = ("F32", (1,))
    return schema


def _expert_formats(directory: Path, num_layers: int) -> dict[int, QuantFormat]:
    """Format of each layer's experts, as ``Inkling2QuantConfig`` of the PR reads
    ``hf_quant_config.json``; without the file every layer is NVFP4."""
    mxfp8: set[str] = set()
    path = directory / "hf_quant_config.json"
    if path.is_file():
        for group in json.loads(path.read_text())["model"]["parameter_groups"]:
            compute = group["compute_dtype"]
            if compute not in _QUANT_COMPUTE_DTYPES:
                raise ValueError(f"Unsupported Inkling2 compute dtype: {compute}")
            if compute == "mxfp8":
                mxfp8.update(group["tensors"])
    return {
        layer: "mxfp8" if expert_prefix(layer, "w13") in mxfp8 else "nvfp4"
        for layer in range(num_layers)
    }


@dataclass(frozen=True)
class TensorEntry:
    """Location of one tensor in a safetensors shard."""

    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def _shard_paths(directory: Path) -> tuple[list[Path], set[str] | None]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
        return [directory / name for name in names], set(weight_map)
    paths = [directory / "model.safetensors"]
    if not paths[0].is_file():
        paths = sorted(directory.glob("model-*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no model safetensors shard in {directory}")
    return paths, None


def _read_entries(directory: Path) -> dict[str, TensorEntry]:
    paths, indexed = _shard_paths(directory)
    entries: dict[str, TensorEntry] = {}
    for path in paths:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            if 8 + length > size:
                raise ValueError(f"{path}: not a safetensors file")
            header = json.loads(handle.read(length))
        header.pop("__metadata__", None)
        base = 8 + length
        for name, info in header.items():
            dtype, shape = info["dtype"], tuple(info["shape"])
            start, stop = info["data_offsets"]
            if dtype not in _TORCH_DTYPES:
                raise ValueError(f"{name}: unsupported safetensors dtype {dtype}")
            expected = math.prod(shape) * _TORCH_DTYPES[dtype].itemsize
            if stop - start != expected or base + stop > size:
                raise ValueError(f"{name}: data range does not match its shape")
            if name in entries:
                raise ValueError(f"{name}: stored in more than one shard")
            entries[name] = TensorEntry(path, dtype, shape, base + start, expected)
    if indexed is not None and indexed != set(entries):
        difference = sorted(indexed ^ set(entries))
        raise ValueError(
            f"model.safetensors.index.json and the shard headers disagree on "
            f"{len(difference)} tensors, e.g. {difference[:4]}"
        )
    return entries


class Inkling2CheckpointIndex:
    """Header-level view of a snapshot: configuration, tensor locations and the
    strict accounting of every tensor name. No tensor data is read.

    Every name in the shard headers must be a text tensor that the configuration
    implies, with the implied dtype and shape, or a vision-tower tensor
    (``model.visual.*``); every implied text tensor must be present.

    Raises:
        ValueError: If a tensor is unknown, missing or has another dtype or shape.
    """

    def __init__(self, snapshot_dir: str | Path) -> None:
        if sys.byteorder != "little":
            raise RuntimeError("safetensors data is little-endian")
        self.directory = Path(snapshot_dir)
        self.config = config = Inkling2Config.from_snapshot(self.directory)
        layers = range(config.num_hidden_layers)
        self.formats = _expert_formats(self.directory, len(layers))
        self.entries = _read_entries(self.directory)
        self.global_schema = _global_schema(config)
        self.layer_schemas = [_layer_schema(config, layer) for layer in layers]
        self.expert_schemas = [
            _expert_schema(config, layer, self.formats[layer]) for layer in layers
        ]
        self.vision_names = tuple(
            name for name in self.entries if name.startswith(VISION_PREFIX)
        )
        self._maps: dict[Path, mmap.mmap] = {}
        self._check()

    @property
    def text_schema(self) -> Schema:
        """Name to (safetensors dtype, shape) of every text tensor."""
        schema = dict(self.global_schema)
        for layer_schema in (*self.layer_schemas, *self.expert_schemas):
            schema.update(layer_schema)
        return schema

    def _check(self) -> None:
        schema = self.text_schema
        problems = []
        for name, (dtype, shape) in schema.items():
            entry = self.entries.get(name)
            if entry is None:
                problems.append(f"missing {name}")
            elif (entry.dtype, entry.shape) != (dtype, shape):
                problems.append(
                    f"{name} is {entry.dtype} {list(entry.shape)}, expected "
                    f"{dtype} {list(shape)}"
                )
        known = schema.keys() | set(self.vision_names)
        problems += [f"unexpected {name}" for name in self.entries if name not in known]
        if problems:
            raise ValueError(
                f"{self.directory}: {len(problems)} tensors do not match the "
                "configuration: " + "; ".join(problems[:8])
            )

    def read(
        self, name: str, start: int | None = None, stop: int | None = None
    ) -> torch.Tensor:
        """Tensor ``name`` in its stored dtype, or rows ``[start, stop)`` of its
        first axis. The result aliases a private copy-on-write mapping of the
        shard, so only the pages that are used are read from disk."""
        entry = self.entries[name]
        dtype = _TORCH_DTYPES[entry.dtype]
        if start is None and stop is None:
            shape, offset, nbytes = entry.shape, entry.offset, entry.nbytes
        else:
            rows = range(entry.shape[0])[slice(start, stop)]
            row_bytes = entry.nbytes // max(1, entry.shape[0])
            shape = (len(rows), *entry.shape[1:])
            offset = entry.offset + rows.start * row_bytes
            nbytes = len(rows) * row_bytes
        if nbytes == 0:
            return torch.empty(shape, dtype=dtype)
        mapping = self._maps.get(entry.path)
        if mapping is None:
            with open(entry.path, "rb") as handle:
                mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
            self._maps[entry.path] = mapping
        raw = torch.frombuffer(mapping, dtype=torch.uint8, count=nbytes, offset=offset)
        if offset % dtype.itemsize:
            raw = raw.clone()
        return raw.view(dtype).reshape(shape)

    def account(
        self,
        layers: Iterable[int] | None = None,
        experts: ExpertMode = "dense",
        include_vision: bool = False,
    ) -> tuple[frozenset[str], dict[str, str]]:
        """Splits every header name into the names a load with these options
        consumes and the names it skips, each with the reason."""
        if experts not in ("dense", "packed", "none"):
            raise ValueError(f"experts must be dense, packed or none, got {experts!r}")
        selected = self.select_layers(layers)
        consumed = set(self.global_schema)
        skipped: dict[str, str] = {}
        for layer in range(self.config.num_hidden_layers):
            if layer not in selected:
                names = (*self.layer_schemas[layer], *self.expert_schemas[layer])
                skipped.update(dict.fromkeys(names, "layer not selected"))
                continue
            consumed.update(self.layer_schemas[layer])
            if experts == "none":
                skipped.update(
                    dict.fromkeys(self.expert_schemas[layer], "experts not requested")
                )
            else:
                consumed.update(self.expert_schemas[layer])
        if include_vision:
            consumed.update(self.vision_names)
        else:
            skipped.update(dict.fromkeys(self.vision_names, "vision tower"))
        if consumed & skipped.keys():
            raise AssertionError("a tensor is both consumed and skipped")
        if consumed | skipped.keys() != self.entries.keys():
            raise AssertionError("a header tensor is neither consumed nor skipped")
        return frozenset(consumed), skipped

    def select_layers(self, layers: Iterable[int] | None) -> tuple[int, ...]:
        """Sorted physical layer ids of a ``layers`` argument (None: all)."""
        count = self.config.num_hidden_layers
        if layers is None:
            return tuple(range(count))
        selected = tuple(sorted(set(layers)))
        if any(type(layer) is not int or not 0 <= layer < count for layer in selected):
            raise ValueError(f"layers must be ids in [0, {count}), got {selected}")
        return selected


@dataclass(frozen=True)
class Inkling2Experts:
    """Dense routed experts of one layer.

    The weight of expert ``e`` is ``w13[e] * w13_scale[e]`` (and likewise for
    ``w2``), with the factors kept apart because only the factored form is exact:
    apply the scale to the FP32 result of the product with the dense factor.
    ``w13`` is ``[E, 2I, H]`` with gate rows ``GATE_ROWS`` and up rows
    ``UP_ROWS``; ``w2`` is ``[E, H, I]``. The scales are FP32 ``[E]`` and are all
    ones for MXFP8 layers.
    """

    layer_id: int
    quant_format: QuantFormat
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


@dataclass(frozen=True)
class Inkling2PackedExperts:
    """Routed experts of one layer as stored: ``tensors`` maps the checkpoint
    names to the untouched packed weights, block scales and NVFP4 scalars."""

    layer_id: int
    quant_format: QuantFormat
    tensors: dict[str, torch.Tensor]

    def dequantize(
        self,
        projection: str,
        experts: torch.Tensor | list[int] | slice | int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense factor ``[n, out, in]`` and FP32 scale ``[n]`` of the selected
        experts of ``projection`` (``"w13"`` or ``"w2"``; None selects all)."""
        name = expert_prefix(self.layer_id, projection)
        if isinstance(experts, int):
            experts = [experts]
        elif isinstance(experts, torch.Tensor) and experts.ndim == 0:
            experts = experts.reshape(1)
        select = slice(None) if experts is None else experts
        weight = _raw_bytes(self.tensors[name])[select]
        block_scale = _raw_bytes(self.tensors[name + "_scale"])[select]
        if self.quant_format == "mxfp8":
            dense = dequantize_mxfp8(weight, block_scale, dtype)
            scale = torch.ones(len(dense), dtype=torch.float32, device=dense.device)
            return dense, scale
        scale = nvfp4_outer_scale(
            self.tensors[name + "_amax"],
            self.tensors[name + "_outer_scale_denominator"],
        )
        return dequantize_nvfp4(weight, block_scale, dtype), scale[select]


class Inkling2Checkpoint:
    """Tensors of an Inkling2 checkpoint (see ``load_inkling2_checkpoint``).

    Attributes:
        config: The parsed text configuration.
        tensors: Non-expert tensors under their checkpoint names. The embedding
            tables are stored here with their padding rows.
        layers: Physical layer ids whose tensors were requested.
        expert_mode: ``"dense"``, ``"packed"`` or ``"none"``.
        expert_formats: ``"nvfp4"`` or ``"mxfp8"`` of every physical layer.
        consumed: Header names this load reads, including the expert tensors
            that ``experts`` reads on first use.
        skipped: Header names this load leaves out, each with the reason.
    """

    def __init__(
        self,
        index: Inkling2CheckpointIndex,
        device: torch.device | str,
        dtype: torch.dtype,
        layers: Iterable[int] | None,
        experts: ExpertMode,
        include_vision: bool,
    ) -> None:
        _work_dtype(dtype)
        self.config = index.config
        self.device, self.dtype = torch.device(device), dtype
        self.layers = index.select_layers(layers)
        self.expert_mode: ExpertMode = experts
        self.expert_formats = dict(index.formats)
        accounting = index.account(self.layers, experts, include_vision)
        self.consumed, self.skipped = accounting
        self._index = index
        self._experts: dict[int, Inkling2Experts | Inkling2PackedExperts] = {}
        names = [*index.global_schema]
        for layer in self.layers:
            names += index.layer_schemas[layer]
        if include_vision:
            names += index.vision_names
        self.tensors = {name: self._place(index.read(name)) for name in names}

    def _place(self, tensor: torch.Tensor) -> torch.Tensor:
        dtype = self.dtype if tensor.dtype == torch.bfloat16 else tensor.dtype
        return tensor.to(device=self.device, dtype=dtype)

    @property
    def embedding(self) -> torch.Tensor:
        """Embedding rows of the served vocabulary (a view of the padded table)."""
        return self.tensors[EMBED_NAME][: self.config.served_vocab_size]

    @property
    def unembedding(self) -> torch.Tensor:
        """Unembedding rows of the served vocabulary (a view of the padded table);
        logits over these rows are the served distribution."""
        return self.tensors[UNEMBED_NAME][: self.config.served_vocab_size]

    @property
    def loaded_expert_layers(self) -> tuple[int, ...]:
        """Layers whose experts are currently held."""
        return tuple(sorted(self._experts))

    def experts(self, layer_id: int) -> Inkling2Experts | Inkling2PackedExperts:
        """Experts of ``layer_id``, read (and dequantized in dense mode) on the
        first call and held until ``drop_experts``.

        Raises:
            ValueError: If the layer or the experts were not requested.
        """
        if self.expert_mode == "none":
            raise ValueError("the checkpoint was loaded with experts='none'")
        if layer_id not in self.layers:
            raise ValueError(f"layer {layer_id} is not among the loaded {self.layers}")
        loaded = self._experts.get(layer_id)
        if loaded is None:
            loaded = self._experts[layer_id] = self._load_experts(layer_id)
        return loaded

    def drop_experts(self, layer_id: int | None = None) -> None:
        """Releases the held experts of ``layer_id`` (None: of every layer)."""
        if layer_id is None:
            self._experts.clear()
        else:
            self._experts.pop(layer_id, None)

    def _load_experts(self, layer_id: int) -> Inkling2Experts | Inkling2PackedExperts:
        index = self._index
        packed = Inkling2PackedExperts(
            layer_id,
            self.expert_formats[layer_id],
            {
                name: index.read(name).to(self.device)
                for name in index.expert_schemas[layer_id]
            },
        )
        if self.expert_mode == "packed":
            return packed
        w13, w13_scale = packed.dequantize("w13", dtype=self.dtype)
        w2, w2_scale = packed.dequantize("w2", dtype=self.dtype)
        return Inkling2Experts(
            layer_id, packed.quant_format, w13, w13_scale, w2, w2_scale
        )


def load_inkling2_checkpoint(
    snapshot_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    layers: Iterable[int] | None = None,
    experts: ExpertMode = "dense",
    include_vision: bool = False,
) -> Inkling2Checkpoint:
    """Loads an Inkling2 snapshot directory.

    Non-expert tensors of the selected layers are returned at once; on the CPU
    in their stored dtype they alias a copy-on-write mapping of the file and cost
    no memory until used. Experts are read per layer by
    ``Inkling2Checkpoint.experts``: all layers together take 45.6 GiB in BF16.

    Args:
        snapshot_dir: Directory with ``config.json`` and the safetensors shards.
        device: Device of the returned tensors.
        dtype: BF16, FP32 or FP64, for the tensors stored in BF16 and for the
            dense expert factors. Tensors stored in FP32 (AttnRes and message
            queries, router biases, expert scales) stay FP32.
        layers: Physical layer ids to load; None loads every layer. Tensors that
            belong to no layer are always loaded.
        experts: ``"dense"`` for ``Inkling2Experts``, ``"packed"`` for
            ``Inkling2PackedExperts``, ``"none"`` to leave the experts out.
        include_vision: Also load the ``model.visual.*`` tensors.

    Returns:
        The checkpoint container.

    Raises:
        ValueError: If the header does not match the configuration exactly, or
            an option is invalid.
    """
    index = Inkling2CheckpointIndex(snapshot_dir)
    return Inkling2Checkpoint(index, device, dtype, layers, experts, include_vision)
