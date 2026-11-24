# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Inkling2 checkpoint loader: exact dequantization and strict tensor accounting.

The oracle is a float64 definition of NVFP4 and MXFP8 written here from the format
specifications (bit fields of E2M1 and E4M3, nibble and block order), and raw
tensors read by safetensors' own reader; nothing of the loader is used to build
an expected value. Comparisons are on float64 bit patterns, so the sign of zero
counts. Three groups: synthetic packed tensors, a tiny checkpoint in the released
on-disk format (``inkling2_tiny``), and the released checkpoints when a local
snapshot exists.

The real-checkpoint tests look for ``INKLING2_SNAPSHOT`` (a snapshot directory),
then for the pinned revision in the Hugging Face cache (``HF_HOME`` /
``HF_HUB_CACHE``), and are skipped when neither has the files. They map the 19.5 GB
shard and read about 150 MB of it.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from inkling2_tiny import SHARD_NAME, tiny_config, tiny_layout, write_tiny_checkpoint
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from vllm.model_executor.models import inkling2_weights as weights

DENSE_DTYPES = [torch.bfloat16, torch.float32, torch.float64]
MINI = ("thinkingmachines/Inkling-2-Mini", "1fb4134b9a74f9681054f4ef82e4bf1f458a2337")
VARLOOP = ("thinkingmachines/TML-d24-varloop-research", "657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2")
LLM = "model.llm."


def e2m1_value(code):
    """4-bit E2M1: sign, two exponent bits (bias 1) and one mantissa bit; exponent 0 is subnormal."""
    sign, exponent, mantissa = code >> 3, (code >> 1) & 0x3, code & 0x1
    magnitude = mantissa / 2 if exponent == 0 else (1 + mantissa / 2) * 2.0 ** (exponent - 1)
    return -magnitude if sign else magnitude


def e4m3_value(code):
    """8-bit E4M3 (finite-only variant): sign, four exponent bits (bias 7), three mantissa bits;
    exponent 0 is subnormal and S.1111.111 is NaN."""
    sign, exponent, mantissa = code >> 7, (code >> 3) & 0xF, code & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return float("nan")
    magnitude = mantissa / 8 * 2.0**-6 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    return -magnitude if sign else magnitude


E2M1 = torch.tensor([e2m1_value(code) for code in range(16)], dtype=torch.float64)
E4M3 = torch.tensor([e4m3_value(code) for code in range(256)], dtype=torch.float64)
POW2 = torch.tensor([2.0 ** (byte - 127) for byte in range(256)], dtype=torch.float64)
FINITE_E4M3 = [code for code in range(256) if code & 0x7F != 0x7F]


def raw(tensor):
    return tensor.view(torch.uint8).long()


def nvfp4_definition(packed, block_scale):
    """Float64 NVFP4 factor: element 2k is the low nibble of byte k, element 2k + 1 the high
    nibble, and 16 consecutive elements share one E4M3 scale. No outer scale."""
    packed = raw(packed)
    elements = torch.stack((E2M1[packed & 0xF], E2M1[packed >> 4]), dim=-1).flatten(-2)
    return elements * E4M3[raw(block_scale)].repeat_interleave(16, dim=-1)


def mxfp8_definition(values, block_scale):
    """Float64 MXFP8 weight: an E4M3 value times 2 ** (exponent byte - 127), 32 per exponent."""
    return E4M3[raw(values)] * POW2[raw(block_scale)].repeat_interleave(32, dim=-1)


def outer_scale_definition(amax, denominator):
    """The FP32 quotient that the Hopper kernels apply to the FP32 accumulator, as float64."""
    quotient = amax.numpy().astype(np.float32) / denominator.numpy().astype(np.float32)
    assert quotient.dtype == np.float32
    return torch.from_numpy(quotient).double()


def assert_same_bits(actual, expected):
    assert actual.shape == expected.shape
    assert not torch.isnan(expected).any()
    actual_bits = actual.double().contiguous().view(torch.int64)
    assert torch.equal(actual_bits, expected.contiguous().view(torch.int64))


def u8(values):
    return torch.tensor(values, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Synthetic packed tensors


def test_reference_e4m3_agrees_with_torch_float8():
    """The oracle's E4M3 bit-field formula decodes all 256 codes as torch's float8_e4m3fn does."""
    decoded = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    finite = torch.tensor(FINITE_E4M3)
    assert_same_bits(decoded[finite], E4M3[finite])
    assert torch.isnan(decoded[[0x7F, 0xFF]]).all() and torch.isnan(E4M3[[0x7F, 0xFF]]).all()
    assert E4M3.nan_to_num(0).abs().max() == 448 and E2M1.abs().max() == 6


@pytest.mark.parametrize("dtype", DENSE_DTYPES, ids=str)
def test_nvfp4_every_byte_under_every_finite_scale_is_exact(dtype):
    """All 256 code pairs times all 254 finite block scales equal the definition bit for bit in
    BF16, FP32 and FP64: the dense factor never needs more than BF16 holds."""
    packed = torch.arange(256, dtype=torch.uint8).view(32, 8).expand(len(FINITE_E4M3), 32, 8)
    block_scale = u8(FINITE_E4M3)[:, None, None].expand(-1, 32, 1)
    dense = weights.dequantize_nvfp4(packed, block_scale.contiguous().view(torch.float8_e4m3fn), dtype)
    assert dense.dtype == dtype and dense.shape == (len(FINITE_E4M3), 32, 16)
    assert_same_bits(dense, nvfp4_definition(packed, block_scale))
    assert dense.double().abs().max() == 2688


def test_nvfp4_nibble_and_block_order_on_a_hand_computed_row():
    """Guards the packing order with values worked out by hand: the low nibble is the even
    element, and block scale b multiplies elements 16b..16b+15."""
    packed = u8([0x72, 0x1F, 0x08, 0x00, 0x00, 0x00, 0x00, 0x40] + [0x53] * 8)
    block_scale = u8([0x38, 0x30])  # 1.0 and 0.5
    dense = weights.dequantize_nvfp4(packed, block_scale, torch.float64)
    first = [1.0, 6.0, -6.0, 0.5, -0.0, 0.0] + [0.0] * 9 + [2.0]
    assert_same_bits(dense, torch.tensor(first + [0.75, 1.5] * 8, dtype=torch.float64))


@pytest.mark.parametrize("dtype", DENSE_DTYPES, ids=str)
def test_nvfp4_random_tensors_with_extreme_zero_and_signed_blocks(dtype):
    """Random packed experts, with the corner blocks written in: largest codes and scales of
    both signs, zero payloads, zero and negative-zero scales, and the negative-zero nibble."""
    generator = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (3, 5, 24), generator=generator, dtype=torch.uint8)
    choice = torch.randint(0, len(FINITE_E4M3), (3, 5, 3), generator=generator)
    block_scale = u8(FINITE_E4M3)[choice]
    packed[0, 0, :8], block_scale[0, 0, 0] = 0x77, 0x7E  # 6 * 448
    packed[0, 0, 8:16], block_scale[0, 0, 1] = 0xFF, 0xFE  # -6 * -448
    packed[0, 1, :8], block_scale[0, 1, 0] = 0x7F, 0x01  # +-6 * smallest subnormal scale
    packed[1, 0, :8], block_scale[1, 0, 0] = 0x00, 0x7E  # zero payload
    packed[1, 0, 8:16], block_scale[1, 0, 1] = 0x88, 0x38  # negative-zero nibbles
    block_scale[1, 1, 0], block_scale[1, 1, 1] = 0x00, 0x80  # zero and negative-zero scales
    dense = weights.dequantize_nvfp4(packed, block_scale, dtype)
    expected = nvfp4_definition(packed, block_scale)
    assert_same_bits(dense, expected)
    assert expected[0, 0, :32].tolist() == [2688.0] * 32
    assert torch.signbit(expected[1, 0, 16:32]).all() and (expected[1, 0, :32] == 0).all()
    # One row and a five-dimensional batch take the same path as [E, out, in].
    assert_same_bits(weights.dequantize_nvfp4(packed[2, 3], block_scale[2, 3], dtype), expected[2, 3])
    stacked = weights.dequantize_nvfp4(packed[None, None], block_scale[None, None], dtype)
    assert_same_bits(stacked, expected[None, None])


def test_nvfp4_rejects_nan_scales_wrong_shapes_and_narrow_dtypes():
    """A NaN block scale or a scale tensor of another block size must not produce weights, and
    float16 (which cannot hold BF16 checkpoint tensors) is not offered."""
    packed, block_scale = torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(2, 2, dtype=torch.uint8)
    for nan in (0x7F, 0xFF):
        scale = block_scale.clone()
        scale[1, 1] = nan
        with pytest.raises(ValueError, match="NaN"):
            weights.dequantize_nvfp4(packed, scale)
    with pytest.raises(ValueError, match="shapes"):
        weights.dequantize_nvfp4(packed, torch.zeros(2, 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="dtype"):
        weights.dequantize_nvfp4(packed, block_scale, torch.float16)
    with pytest.raises(TypeError):
        weights.dequantize_nvfp4(packed.float(), block_scale)


@pytest.mark.parametrize(
    "dtype, low, high",
    [(torch.bfloat16, 3, 246), (torch.float32, 0, 246), (torch.float64, 0, 254)],
    ids=["bfloat16", "float32", "float64"],
)
def test_mxfp8_every_value_under_every_exact_exponent(dtype, low, high):
    """All finite E4M3 values times every block exponent of the range that is exact in the
    dtype (BF16: 3..246, which includes subnormal results below exponent 10)."""
    exponents = torch.arange(low, high + 1, dtype=torch.uint8)
    values = torch.arange(256, dtype=torch.uint8).repeat(len(exponents), 1)
    values[:, [0x7F, 0xFF]] = 0x80
    block_scale = exponents[:, None].expand(-1, 8).contiguous()
    dense = weights.dequantize_mxfp8(values.view(torch.float8_e4m3fn), block_scale, dtype)
    assert dense.dtype == dtype
    assert_same_bits(dense, mxfp8_definition(values, block_scale))


def test_mxfp8_block_order_on_a_hand_computed_row():
    """Exponent byte b scales elements 32b..32b+31; 127 is the scale 1."""
    values = u8([0x38, 0x7E, 0xB8, 0x80] + [0x00] * 28 + [0x38, 0x7E, 0xB8, 0x01] + [0x40] * 28)
    dense = weights.dequantize_mxfp8(values, u8([127, 125]), torch.float64)
    expected = [1.0, 448.0, -1.0, -0.0] + [0.0] * 28 + [0.25, 112.0, -0.25, 2.0**-11] + [0.5] * 28
    assert_same_bits(dense, torch.tensor(expected, dtype=torch.float64))


def test_mxfp8_refuses_products_that_the_dtype_cannot_hold():
    """Outside the exact exponent range the result is checked value by value: a zero block with
    exponent 0 (what a quantizer writes for an all-zero block) loads, a product below the BF16
    subnormal step or above the largest float does not, and NaN bytes never do."""
    zero_block = torch.zeros(1, 32, dtype=torch.uint8)
    zero_block[0, 3] = 0x80
    dense = weights.dequantize_mxfp8(zero_block, u8([[0]]))
    assert_same_bits(dense, mxfp8_definition(zero_block, u8([[0]])))
    tiny = zero_block.clone()
    tiny[0, 0] = 0x01  # 2**-9 * 2**-125 = 2**-134
    with pytest.raises(ValueError, match="not exact"):
        weights.dequantize_mxfp8(tiny, u8([[2]]))
    in_fp32 = weights.dequantize_mxfp8(tiny, u8([[2]]), torch.float32)
    assert_same_bits(in_fp32, mxfp8_definition(tiny, u8([[2]])))
    huge = zero_block.clone()
    huge[0, 0] = 0x7E  # 448 * 2**120
    for dtype in (torch.bfloat16, torch.float32):
        with pytest.raises(ValueError, match="not exact"):
            weights.dequantize_mxfp8(huge, u8([[247]]), dtype)
    nan = zero_block.clone()
    nan[0, 5] = 0xFF
    for values, exponent in ((nan, 127), (zero_block, 255)):
        with pytest.raises(ValueError, match="NaN"):
            weights.dequantize_mxfp8(values, u8([[exponent]]))
    with pytest.raises(ValueError, match="shapes"):
        weights.dequantize_mxfp8(torch.zeros(1, 48, dtype=torch.uint8), u8([[127]]))


def test_gate_rows_are_the_even_rows_of_w13():
    """Output 2j of w13 is gate j and output 2j + 1 is up j (upstream's reference splits the
    product as gate_up[:, 0::2], gate_up[:, 1::2]). Exchanging the two selectors would feed the
    sigmoid branch with the up projection in every consumer."""
    w13 = torch.arange(6 * 3).view(6, 3)
    assert w13[weights.GATE_ROWS].tolist() == w13[[0, 2, 4]].tolist()
    assert w13[weights.UP_ROWS].tolist() == w13[[1, 3, 5]].tolist()


def test_dequantization_in_several_row_chunks_equals_the_definition(monkeypatch):
    """A layer is dequantized in chunks of rows (41 for a Mini layer, one for everything else
    in this file). With the chunk size cut down so that the rows span many chunks and the last
    one is partial, both formats still equal the definition, and an inexact MXFP8 product in
    the last chunk is still refused."""
    generator = torch.Generator().manual_seed(1)
    finite = u8(FINITE_E4M3)
    packed = torch.randint(0, 256, (7, 9, 40), generator=generator, dtype=torch.uint8)
    block_scale = finite[torch.randint(0, len(finite), (7, 9, 5), generator=generator)]
    values = finite[torch.randint(0, len(finite), (5, 7, 96), generator=generator)]
    exponents = torch.randint(100, 130, (5, 7, 3), generator=generator, dtype=torch.uint8)
    values[0, 0, :32], exponents[0, 0, 0] = 0, 0  # takes the value-by-value check
    inexact, low = values.clone(), exponents.clone()
    inexact[4, 6, 64], low[4, 6, 2] = 0x01, 1  # 2**-135, below the BF16 subnormal step
    for chunk_elements in (1, 320, 1 << 24):  # one row, four (three) rows, everything
        monkeypatch.setattr(weights, "_CHUNK_ELEMENTS", chunk_elements)
        for dtype in DENSE_DTYPES:
            dense = weights.dequantize_nvfp4(packed, block_scale, dtype)
            assert_same_bits(dense, nvfp4_definition(packed, block_scale))
            dense = weights.dequantize_mxfp8(values, exponents, dtype)
            assert_same_bits(dense, mxfp8_definition(values, exponents))
        with pytest.raises(ValueError, match="not exact"):
            weights.dequantize_mxfp8(inexact, low)


def test_outer_scale_is_the_fp32_quotient_and_does_not_fit_bf16():
    """The scale is fl32(amax / denominator), FP32 in and out. It is returned on its own
    because BF16 cannot hold it: folding it into the dense factor would round every weight."""
    amax = torch.tensor([0.10142918, 0.1211204081773758, 0.85205936, 0.0], dtype=torch.float32)
    denominator = torch.tensor([2688.0])
    scale = weights.nvfp4_outer_scale(amax, denominator)
    assert scale.dtype == torch.float32
    assert_same_bits(scale, outer_scale_definition(amax, denominator))
    assert (scale[:3].bfloat16().float() != scale[:3]).all()
    with pytest.raises(TypeError):
        weights.nvfp4_outer_scale(amax.double(), denominator)
    with pytest.raises(ValueError, match="finite"):
        weights.nvfp4_outer_scale(amax, torch.tensor([0.0]))


# ---------------------------------------------------------------------------
# Tiny checkpoint in the released on-disk format


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Directory of the default tiny checkpoint and its tensors as safetensors reads them."""
    directory = write_tiny_checkpoint(tmp_path_factory.mktemp("inkling2-tiny"))
    return directory, load_file(str(directory / SHARD_NAME))


def expected_schema(config, mxfp8_layers=(4, 5)):
    """Header entries (safetensors dtype, shape) implied by the independent tiny layout."""
    schema = {}
    for name, (kind, shape) in tiny_layout(config, mxfp8_layers).items():
        experts, rows, cols = shape if kind in ("nvfp4", "mxfp8") else (0, 0, 0)
        if kind == "nvfp4":
            schema[name] = ("U8", (experts, rows, cols // 2))
            schema[name + "_scale"] = ("F8_E4M3", (experts, rows, cols // 16))
            schema[name + "_amax"] = ("F32", (experts,))
            schema[name + "_outer_scale_denominator"] = ("F32", (1,))
        elif kind == "mxfp8":
            schema[name] = ("F8_E4M3", (experts, rows, cols))
            schema[name + "_scale"] = ("U8", (experts, rows, cols // 32))
        else:
            schema[name] = ({"bf16": "BF16", "f32": "F32"}[kind], shape)
    return schema


def rewrite(source, target, mutate, keep_index=False):
    """Copy of a tiny checkpoint whose shard tensors went through ``mutate``."""
    shutil.copytree(source, target)
    tensors = load_file(str(target / SHARD_NAME))
    mutate(tensors)
    save_file(tensors, str(target / SHARD_NAME), metadata={"format": "pt"})
    if not keep_index:
        (target / "model.safetensors.index.json").unlink()
    return target


def test_tiny_schema_equals_the_independent_layout_and_every_name_is_accounted(tiny):
    """The loader's name, dtype and shape rules agree with the layout written out separately in
    the tiny writer, and a default load consumes every text tensor and skips only the vision
    tower."""
    directory, stored = tiny
    index = weights.Inkling2CheckpointIndex(directory)
    expected = expected_schema(tiny_config())
    vision = {name for name in expected if name.startswith("model.visual.")}
    assert index.text_schema == {name: entry for name, entry in expected.items() if name not in vision}
    assert set(index.vision_names) == vision and len(vision) == 3
    checkpoint = weights.load_inkling2_checkpoint(directory)
    assert checkpoint.consumed == stored.keys() - vision
    assert checkpoint.skipped == dict.fromkeys(sorted(vision), "vision tower")
    formats = [checkpoint.expert_formats[layer] for layer in range(6)]
    assert formats == ["nvfp4"] * 4 + ["mxfp8"] * 2


def test_tiny_config_has_the_fields_of_config_json(tiny):
    """Parsed fields are the checkpoint's, with the padded and served vocabulary kept apart and
    the per-layer attention geometry derived from the local layer ids."""
    config = weights.Inkling2Config.from_snapshot(tiny[0])
    text = tiny_config()["text_config"]
    for name in ("hidden_size", "num_hidden_layers", "d_rel", "rel_extent", "sliding_window_size",
                 "sconv_kernel_size", "n_layers_per_attn_res_block", "n_routed_experts",
                 "num_experts_per_tok", "intermediate_size", "route_scale", "mlp_gate_softcap",
                 "mlp_up_softcap", "final_logit_softcapping", "qk_l2_clip_threshold",
                 "rel_l2_clip_threshold", "log_scaling_n_floor", "log_scaling_alpha", "rms_norm_eps"):
        assert getattr(config, name) == text[name], name
    assert (config.padded_vocab_size, config.served_vocab_size, config.eos_token_id) == (112, 100, 96)
    assert (config.core_start, config.core_end, config.num_passes, config.gate_rank) == (2, 4, 3, 4)
    assert config.is_varloop is False and config.core_layers == range(2, 4)
    assert config.num_layer_visits == 10
    assert config.local_layer_ids == (0, 2, 4) and config.is_local(2) and not config.is_local(3)
    assert config.layer_rel_extent(2) == 8 and config.layer_rel_extent(3) == 16
    with pytest.raises(AttributeError):
        config.hidden_size = 1


@pytest.mark.parametrize(
    "top, text, message",
    [
        (dict(model_type="inkling2"), {}, "inkling2_loop"),
        ({}, dict(loop_config=dict(core_start=1, core_end=4, num_passes=3, gate_rank=4, is_varloop=False)),
         "multiples"),
        ({}, dict(loop_config=dict(core_start=2, core_end=8, num_passes=3, gate_rank=4, is_varloop=False)),
         "out of range"),
        ({}, dict(loop_config=dict(core_start=2, core_end=4, num_passes=1, gate_rank=4, is_varloop=False)),
         "num_passes"),
        ({}, dict(loop_config=dict(core_start=2, core_end=4, num_passes=3, gate_rank=4)), "is_varloop"),
        ({}, dict(n_shared_experts=1), "n_shared_experts"),
        ({}, dict(attn_res_normalize_values=False), "attn_res_normalize_values"),
        ({}, dict(unpadded_vocab_size=113), "unpadded_vocab_size"),
        ({}, dict(local_layer_ids=[0, 2, 6]), "local_layer_ids"),
        ({}, dict(local_layer_ids=[0, "2", 4]), "local_layer_ids"),
        ({}, dict(hidden_size=0), "hidden_size"),
        ({}, dict(n_layers_per_attn_res_block=0), "n_layers_per_attn_res_block"),
        ({}, dict(num_hidden_layers=None), "num_hidden_layers"),
        ({}, dict(unpadded_vocab_size="100"), "served_vocab_size"),
    ],
    ids=["model-type", "core-not-on-blocks", "core-out-of-range", "one-pass", "no-varloop-flag",
         "shared-experts", "raw-attnres-values", "served-above-padded", "local-id-out-of-range",
         "local-id-not-an-integer", "zero-size", "zero-block-size", "null-size", "string-size"],
)
def test_config_rejects_what_the_loader_does_not_implement(top, text, message):
    """A configuration outside the released architecture stops at parsing instead of yielding
    tensors under wrong assumptions, and always as a ValueError: a zero, null or string size
    must not surface as a ZeroDivisionError or TypeError of the range checks."""
    config = {**tiny_config(**text), **top}
    with pytest.raises(ValueError, match=message):
        weights.Inkling2Config.from_dict(config)


@pytest.mark.parametrize(
    "layers, block, core, passes, varloop, visits",
    [
        (6, 2, (2, 4), 2, False, 8),
        (6, 2, (2, 4), 3, False, 10),
        (6, 2, (2, 4), 4, False, 12),
        (6, 2, (2, 4), 1, True, 6),
        (6, 2, (4, 6), 3, False, 10),
        (24, 6, (6, 18), 3, False, 48),
    ],
    ids=["two-passes", "three-passes", "four-passes", "varloop-one-pass", "core-ends-the-stack",
         "mini-shape"],
)
def test_layer_visits_count_the_core_once_per_pass(layers, block, core, passes, varloop, visits):
    """num_layer_visits, the index of the final-norm output, is the layer count plus the core
    once more per extra pass: 48 for the Mini shape (prelude 0-5, core 6-17 three times, coda
    18-23), and one pass is accepted only from a variable-loop checkpoint."""
    loop = dict(core_start=core[0], core_end=core[1], num_passes=passes, gate_rank=4, is_varloop=varloop)
    text = dict(num_hidden_layers=layers, n_layers_per_attn_res_block=block, loop_config=loop)
    config = weights.Inkling2Config.from_dict(tiny_config(**text))
    assert config.num_layer_visits == visits
    assert config.core_layers == range(*core)
    assert (config.num_passes, config.is_varloop) == (passes, varloop)


def test_config_needs_every_field_of_the_forward():
    """A missing text field is an error, not a default taken from another model size."""
    config = tiny_config()
    del config["text_config"]["rel_extent"]
    with pytest.raises(ValueError, match="rel_extent"):
        weights.Inkling2Config.from_dict(config)


@pytest.mark.parametrize("dtype", DENSE_DTYPES, ids=str)
def test_tiny_non_expert_tensors_keep_checkpoint_names_and_values(tiny, dtype):
    """Every non-expert tensor is under its checkpoint name with the stored values; BF16 tensors
    take the requested dtype and FP32 tensors (queries, router biases) are never narrowed."""
    directory, stored = tiny
    checkpoint = weights.load_inkling2_checkpoint(directory, dtype=dtype, experts="none")
    names = {name for name in stored if name.startswith(LLM) and ".experts." not in name}
    assert checkpoint.tensors.keys() == names
    for name in names:
        loaded, original = checkpoint.tensors[name], stored[name]
        assert loaded.dtype == (dtype if original.dtype == torch.bfloat16 else torch.float32), name
        assert_same_bits(loaded, original.double())


def test_tiny_dense_experts_equal_the_float64_definition(tiny):
    """For every layer and projection of both formats: the dense factor is the definition
    without the outer scale, the scale is the FP32 quotient (ones for MXFP8), and their float64
    product is the full weight. Shapes are [E, 2I, H] and [E, H, I]."""
    directory, stored = tiny
    checkpoint = weights.load_inkling2_checkpoint(directory)
    for layer in range(6):
        experts = checkpoint.experts(layer)
        assert isinstance(experts, weights.Inkling2Experts)
        assert experts.w13.shape == (8, 128, 32) and experts.w2.shape == (8, 32, 64)
        for projection in ("w13", "w2"):
            name = f"{LLM}layers.{layer}.mlp.experts.{projection}_weight"
            dense, scale = getattr(experts, projection), getattr(experts, projection + "_scale")
            assert dense.dtype == torch.bfloat16 and scale.dtype == torch.float32
            if experts.quant_format == "nvfp4":
                factor = nvfp4_definition(stored[name], stored[name + "_scale"])
                outer = outer_scale_definition(stored[name + "_amax"],
                                               stored[name + "_outer_scale_denominator"])
                assert factor.abs().amax(dim=(1, 2)).tolist() == [2688.0] * 8
            else:
                factor = mxfp8_definition(stored[name], stored[name + "_scale"])
                outer = torch.ones(8, dtype=torch.float64)
            assert_same_bits(dense, factor)
            assert_same_bits(scale, outer)
            assert_same_bits(dense.double() * scale.double()[:, None, None], factor * outer[:, None, None])
    assert [checkpoint.experts(layer).quant_format for layer in (3, 4)] == ["nvfp4", "mxfp8"]
    # The tiny writer zeroes one block per layer: zero scale (NVFP4) and exponent byte 0 (MXFP8).
    zero_block = checkpoint.experts(0).w13[0, 0, :16].float()
    assert not zero_block.any() and torch.signbit(zero_block).tolist() == [False, True] + [False] * 14
    assert stored[f"{LLM}layers.0.mlp.experts.w13_weight_scale"].view(torch.uint8)[0, 0, 0] == 0
    assert stored[f"{LLM}layers.4.mlp.experts.w13_weight_scale"][0, 0, 0] == 0


def test_tiny_packed_experts_are_untouched_and_dequantize_per_expert(tiny):
    """Packed mode returns the stored bytes and dtypes under the checkpoint names, and
    dequantizing a few experts from them gives the rows of the dense load."""
    directory, stored = tiny
    packed_checkpoint = weights.load_inkling2_checkpoint(directory, experts="packed")
    dense_checkpoint = weights.load_inkling2_checkpoint(directory, dtype=torch.float32)
    for layer in (1, 5):
        packed = packed_checkpoint.experts(layer)
        assert isinstance(packed, weights.Inkling2PackedExperts)
        names = {name for name in stored if name.startswith(f"{LLM}layers.{layer}.mlp.experts.")}
        assert packed.tensors.keys() == names and len(names) == (8 if layer == 1 else 4)
        for name in names:
            assert packed.tensors[name].dtype == stored[name].dtype
            assert torch.equal(packed.tensors[name].view(torch.uint8), stored[name].view(torch.uint8))
        dense = dense_checkpoint.experts(layer)
        for projection in ("w13", "w2"):
            factor, scale = packed.dequantize(projection, [5, 1], torch.float32)
            assert_same_bits(factor, getattr(dense, projection)[[5, 1]].double())
            assert_same_bits(scale, getattr(dense, projection + "_scale")[[5, 1]].double())
            everything, _ = packed.dequantize(projection, dtype=torch.float32)
            assert_same_bits(everything, getattr(dense, projection).double())
            # A single expert keeps its axis, as an integer or as a zero-dimensional tensor.
            for selector in (7, torch.tensor(7)):
                one, one_scale = packed.dequantize(projection, selector, torch.float32)
                assert_same_bits(one, getattr(dense, projection)[7:8].double())
                assert_same_bits(one_scale, getattr(dense, projection + "_scale")[7:8].double())


def test_experts_are_read_per_layer_on_first_use(tiny):
    """Nothing of the experts is held after loading; each layer is read on its first request,
    kept until dropped, and layers outside the selection are refused and listed as skipped."""
    directory, stored = tiny
    checkpoint = weights.load_inkling2_checkpoint(directory, layers=[4, 1])
    assert checkpoint.layers == (1, 4) and checkpoint.loaded_expert_layers == ()
    first = checkpoint.experts(4)
    assert checkpoint.loaded_expert_layers == (4,) and checkpoint.experts(4) is first
    checkpoint.experts(1)
    checkpoint.drop_experts(4)
    assert checkpoint.loaded_expert_layers == (1,)
    checkpoint.drop_experts()
    assert checkpoint.loaded_expert_layers == ()
    with pytest.raises(ValueError, match="layer 0"):
        checkpoint.experts(0)
    layer_names = {name for name in checkpoint.tensors if ".layers." in name}
    selected = (f"{LLM}layers.1.", f"{LLM}layers.4.")
    assert layer_names and all(name.startswith(selected) for name in layer_names)
    assert f"{LLM}attn_res.mlp_queries.3" in checkpoint.tensors  # tensors of no layer are always loaded
    reasons = set(checkpoint.skipped.values())
    assert reasons == {"layer not selected", "vision tower"}
    assert checkpoint.skipped[f"{LLM}layers.0.mlp.experts.w2_weight"] == "layer not selected"
    assert checkpoint.consumed | checkpoint.skipped.keys() == stored.keys()
    assert not checkpoint.consumed & checkpoint.skipped.keys()
    with pytest.raises(ValueError, match="layers"):
        weights.load_inkling2_checkpoint(directory, layers=[6])


def test_experts_none_and_vision_options(tiny):
    """experts='none' accounts the expert tensors as skipped and refuses to serve them;
    include_vision moves the vision tower from the skip list into the tensors."""
    directory, stored = tiny
    checkpoint = weights.load_inkling2_checkpoint(directory, experts="none", include_vision=True)
    with pytest.raises(ValueError, match="none"):
        checkpoint.experts(0)
    expert_names = {name for name in stored if ".experts." in name}
    assert checkpoint.skipped == dict.fromkeys(expert_names, "experts not requested")
    vision = "model.visual.proj_out.weight"
    assert_same_bits(checkpoint.tensors[vision], stored[vision].double())
    assert checkpoint.consumed == stored.keys() - expert_names
    with pytest.raises(ValueError, match="experts"):
        weights.load_inkling2_checkpoint(directory, experts="folded")
    with pytest.raises(ValueError, match="experts"):  # not accounted as if it were "dense"
        weights.Inkling2CheckpointIndex(directory).account(experts="None")


def test_tiny_served_vocabulary_views_exclude_the_padding_rows(tiny):
    """embedding and unembedding are views of the first unpadded_vocab_size rows of the padded
    tables: 100 of 112 rows here, so no logit exists for a padding id."""
    directory, stored = tiny
    checkpoint = weights.load_inkling2_checkpoint(directory, layers=[], experts="none")
    views = ((checkpoint.embedding, weights.EMBED_NAME), (checkpoint.unembedding, weights.UNEMBED_NAME))
    for view, name in views:
        padded = checkpoint.tensors[name]
        assert padded.shape == (112, 32) and view.shape == (100, 32)
        assert view.data_ptr() == padded.data_ptr()
        assert_same_bits(view, stored[name][:100].double())
        assert stored[name][100:].float().abs().sum() > 0


def test_sliding_and_full_layers_use_their_own_dimensions(tmp_path):
    """With sliding-window heads that differ from the full-attention ones, each layer's
    projections follow its own kind (the Mini checkpoint has equal dimensions, which would hide
    a mix-up), and other MXFP8 layers than the default are read from hf_quant_config.json."""
    config = tiny_config(swa_head_dim=4, swa_num_attention_heads=2, swa_num_key_value_heads=1)
    directory = write_tiny_checkpoint(tmp_path, config, seed=3, mxfp8_layers=(0, 3))
    index = weights.Inkling2CheckpointIndex(directory)
    expected = expected_schema(config, mxfp8_layers=(0, 3))
    assert index.text_schema == {name: entry for name, entry in expected.items() if ".visual." not in name}
    checkpoint = weights.load_inkling2_checkpoint(directory)
    assert checkpoint.tensors[f"{LLM}layers.2.attn.wq_du.weight"].shape == (8, 32)
    assert checkpoint.tensors[f"{LLM}layers.3.attn.wq_du.weight"].shape == (32, 32)
    assert checkpoint.tensors[f"{LLM}layers.2.attn.k_sconv.weight"].shape == (4, 1, 4)
    assert checkpoint.tensors[f"{LLM}layers.3.attn.k_sconv.static_w"].shape == (4, 16)
    formats = [checkpoint.experts(layer).quant_format for layer in range(4)]
    assert formats == ["mxfp8", "nvfp4", "nvfp4", "mxfp8"]


def test_same_seed_gives_the_same_checkpoint(tiny, tmp_path):
    """Tiny checkpoints are reproducible, so tests of other packages can pin values."""
    again = load_file(str(write_tiny_checkpoint(tmp_path) / SHARD_NAME))
    assert again.keys() == tiny[1].keys()
    for name, tensor in again.items():
        assert torch.equal(tensor.view(torch.uint8), tiny[1][name].view(torch.uint8)), name


def add_tensor(tensors):
    tensors[f"{LLM}layers.0.attn.extra.weight"] = torch.zeros(4, dtype=torch.bfloat16)


def drop_tensor(tensors):
    del tensors[f"{LLM}norm.weight"]


def reshape_tensor(tensors):
    name = f"{LLM}layers.2.controllers.k.down_weight"
    tensors[name] = tensors[name].t().contiguous()


def retype_tensor(tensors):
    tensors[f"{LLM}attn_res.output_query"] = tensors[f"{LLM}attn_res.output_query"].bfloat16()


def drop_amax(tensors):
    del tensors[f"{LLM}layers.1.mlp.experts.w2_weight_amax"]


def unpad_unembedding(tensors):
    tensors[f"{LLM}unembed.weight"] = tensors[f"{LLM}unembed.weight"][:100].contiguous()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (add_tensor, "unexpected model.llm.layers.0.attn.extra.weight"),
        (drop_tensor, "missing model.llm.norm.weight"),
        (reshape_tensor, r"controllers.k.down_weight is BF16 \[64, 4\], expected BF16 \[4, 64\]"),
        (retype_tensor, "output_query is BF16"),
        (drop_amax, "missing model.llm.layers.1.mlp.experts.w2_weight_amax"),
        (unpad_unembedding, r"unembed.weight is BF16 \[100, 32\], expected BF16 \[112, 32\]"),
    ],
    ids=["unknown", "missing", "shape", "dtype", "missing-scale", "unpadded-table"],
)
def test_a_header_that_does_not_match_the_configuration_is_refused(tiny, tmp_path, mutate, message):
    """Strict accounting: an unknown tensor, a missing one, or one of another shape or dtype
    raises at load time instead of being ignored."""
    directory = rewrite(tiny[0], tmp_path / "mutated", mutate)
    with pytest.raises(ValueError, match=message):
        weights.load_inkling2_checkpoint(directory, layers=[], experts="none")


def test_quant_config_and_shard_index_must_agree_with_the_header(tiny, tmp_path):
    """A layer declared MXFP8 whose tensors are NVFP4 is refused, as is a shard index that names
    a tensor the shards do not hold; without hf_quant_config.json every layer is NVFP4, as
    upstream assumes."""
    declared = tmp_path / "declared"
    shutil.copytree(tiny[0], declared)
    quant = json.loads((declared / "hf_quant_config.json").read_text())
    groups = {group["compute_dtype"]: group["tensors"] for group in quant["model"]["parameter_groups"]}
    moved = [name for name in groups["nvfp4"] if ".layers.0." in name]
    groups["nvfp4"][:] = [name for name in groups["nvfp4"] if name not in moved]
    groups["mxfp8"].extend(moved)
    (declared / "hf_quant_config.json").write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="layers.0.mlp.experts.w13_weight is U8"):
        weights.Inkling2CheckpointIndex(declared)

    stale = rewrite(tiny[0], tmp_path / "stale", drop_tensor, keep_index=True)
    with pytest.raises(ValueError, match="index.json"):
        weights.Inkling2CheckpointIndex(stale)

    undeclared = tmp_path / "undeclared"
    shutil.copytree(tiny[0], undeclared)
    (undeclared / "hf_quant_config.json").unlink()
    with pytest.raises(ValueError, match="layers.4.mlp.experts.w13_weight is F8_E4M3"):
        weights.Inkling2CheckpointIndex(undeclared)


def test_checkpoint_split_over_two_shards_loads_the_same_tensors(tiny, tmp_path):
    """Larger releases come in several shards. The tiny tensors divided between two shard files
    (so that packed weights and their scales fall into different files) load to the stored
    bytes through the shard index and, without it, through the model-*.safetensors pattern; a
    tensor that two shards both hold is refused."""
    directory, stored = tiny
    names = sorted(stored)
    for label, with_index in (("indexed", True), ("globbed", False)):
        target = tmp_path / label
        target.mkdir()
        for file in ("config.json", "hf_quant_config.json"):
            shutil.copy(directory / file, target / file)
        weight_map = {}
        for number, part in enumerate((names[0::2], names[1::2]), start=1):
            shard = f"model-{number:05d}-of-00002.safetensors"
            save_file({name: stored[name] for name in part}, str(target / shard), metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(part, shard))
        if with_index:
            index = dict(metadata={}, weight_map=weight_map)
            (target / "model.safetensors.index.json").write_text(json.dumps(index))
        checkpoint = weights.load_inkling2_checkpoint(target, experts="packed", include_vision=True)
        assert checkpoint.consumed == stored.keys() and not checkpoint.skipped
        loaded = dict(checkpoint.tensors)
        for layer in range(6):
            loaded.update(checkpoint.experts(layer).tensors)
        assert loaded.keys() == stored.keys()
        for name, tensor in loaded.items():
            assert tensor.dtype == stored[name].dtype, name
            assert torch.equal(tensor.view(torch.uint8), stored[name].view(torch.uint8)), name
    second = {name: stored[name] for name in names[1::2]} | {names[0]: stored[names[0]]}
    save_file(second, str(target / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="more than one shard"):
        weights.Inkling2CheckpointIndex(target)


def test_index_reads_rows_without_touching_the_rest(tiny):
    """Row ranges of the first axis come back in the stored dtype and equal safetensors' own
    lazy slices, so one expert can be read without its layer."""
    directory, stored = tiny
    index = weights.Inkling2CheckpointIndex(directory)
    with safe_open(str(directory / SHARD_NAME), framework="pt") as handle:
        for name in (f"{LLM}layers.0.mlp.experts.w13_weight", f"{LLM}layers.0.mlp.experts.w13_weight_scale",
                     f"{LLM}layers.5.mlp.experts.w2_weight", f"{LLM}unembed.weight",
                     f"{LLM}layers.0.mlp.gate.bias"):
            rows = index.read(name, 2, 5)
            reference = handle.get_slice(name)[2:5]
            assert rows.dtype == reference.dtype == stored[name].dtype and rows.shape == reference.shape
            assert torch.equal(rows.view(torch.uint8), reference.view(torch.uint8))
            assert torch.equal(index.read(name).view(torch.uint8), stored[name].view(torch.uint8))
    assert index.read(f"{LLM}unembed.weight", 100).shape == (12, 32)


def test_shard_whose_data_is_not_aligned_loads_the_same_values(tiny, tmp_path):
    """safetensors does not promise aligned data. With the header padded by one byte every BF16
    and FP32 tensor starts at an odd address; the values must still be the stored ones."""
    directory = tmp_path / "shifted"
    shutil.copytree(tiny[0], directory)
    content = (directory / SHARD_NAME).read_bytes()
    length = int.from_bytes(content[:8], "little")
    shifted = (length + 1).to_bytes(8, "little") + content[8 : 8 + length] + b" " + content[8 + length :]
    (directory / SHARD_NAME).write_bytes(shifted)
    index = weights.Inkling2CheckpointIndex(directory)
    assert index.entries[weights.UNEMBED_NAME].offset % 2 == 1
    checkpoint = weights.load_inkling2_checkpoint(directory, experts="packed")
    for name, tensor in checkpoint.tensors.items():
        assert torch.equal(tensor.view(torch.uint8), tiny[1][name].view(torch.uint8)), name
    for name, tensor in checkpoint.experts(0).tensors.items():
        assert torch.equal(tensor.view(torch.uint8), tiny[1][name].view(torch.uint8)), name


# ---------------------------------------------------------------------------
# Released checkpoints (skipped without a local snapshot)


def snapshot_path(variable, repo, revision):
    explicit = os.environ.get(variable)
    if explicit:
        path = Path(explicit)
    else:
        from huggingface_hub.constants import HF_HUB_CACHE

        path = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not ((path / "config.json").is_file() and (path / SHARD_NAME).is_file()):
        pytest.skip(f"no local snapshot of {repo} (set {variable})")
    return path


@pytest.fixture(scope="module")
def mini():
    return snapshot_path("INKLING2_SNAPSHOT", *MINI)


@pytest.fixture(scope="module")
def mini_index(mini):
    return weights.Inkling2CheckpointIndex(mini)


@pytest.mark.parametrize(
    "variable, pin, varloop",
    [("INKLING2_SNAPSHOT", MINI, False), ("INKLING2_VARLOOP_SNAPSHOT", VARLOOP, True)],
    ids=["mini", "varloop-research"],
)
def test_real_header_is_fully_accounted(variable, pin, varloop):
    """All 1,244 header names of a released checkpoint are consumed (822 text tensors, with the
    dtype and shape the configuration implies) or skipped (422 vision tensors); the experts are
    NVFP4 in layers 0-19 and MXFP8 in layers 20-23."""
    snapshot = snapshot_path(variable, *pin)
    index = weights.Inkling2CheckpointIndex(snapshot)
    config = index.config
    assert (config.hidden_size, config.num_hidden_layers, config.intermediate_size) == (2304, 24, 1152)
    assert (config.n_routed_experts, config.num_experts_per_tok) == (128, 8)
    assert (config.padded_vocab_size, config.served_vocab_size) == (201024, 200058)
    assert config.eos_token_id == 200006
    assert (config.core_start, config.core_end, config.num_passes, config.gate_rank) == (6, 18, 3, 64)
    assert config.is_varloop is varloop and config.num_layer_visits == 48
    assert [layer for layer in range(24) if not config.is_local(layer)] == [5, 11, 17, 23]
    assert (config.sliding_window_size, config.sconv_kernel_size) == (1024, 4)
    assert config.n_layers_per_attn_res_block == 6
    assert index.formats == {layer: "nvfp4" if layer < 20 else "mxfp8" for layer in range(24)}

    assert len(index.entries) == 1244
    consumed, skipped = index.account()
    assert len(consumed) == 822 and len(skipped) == 422
    assert consumed | skipped.keys() == index.entries.keys() and not consumed & skipped.keys()
    assert set(skipped.values()) == {"vision tower"}
    assert all(name.startswith("model.visual.") for name in skipped)
    assert consumed == index.text_schema.keys()
    entry = index.entries
    for layer, fmt in ((0, "nvfp4"), (23, "mxfp8")):
        prefix = f"{LLM}layers.{layer}.mlp.experts."
        found = {
            name[len(prefix) :]: (item.dtype, item.shape)
            for name, item in entry.items()
            if name.startswith(prefix)
        }
        assert found == {
            "nvfp4": {
                "w13_weight": ("U8", (128, 2304, 1152)), "w13_weight_scale": ("F8_E4M3", (128, 2304, 144)),
                "w13_weight_amax": ("F32", (128,)), "w13_weight_outer_scale_denominator": ("F32", (1,)),
                "w2_weight": ("U8", (128, 2304, 576)), "w2_weight_scale": ("F8_E4M3", (128, 2304, 72)),
                "w2_weight_amax": ("F32", (128,)), "w2_weight_outer_scale_denominator": ("F32", (1,)),
            },
            "mxfp8": {
                "w13_weight": ("F8_E4M3", (128, 2304, 2304)), "w13_weight_scale": ("U8", (128, 2304, 72)),
                "w2_weight": ("F8_E4M3", (128, 2304, 1152)), "w2_weight_scale": ("U8", (128, 2304, 36)),
            },
        }[fmt]
    assert entry[f"{LLM}attn_res.attn_queries.4"].shape == (3, 2304)  # the q/k/v queries of full layer 5
    assert entry[f"{LLM}layers.6.controllers.v.down_weight"].shape == (64, 4608)
    experts_none = index.account(layers=[0], experts="none")[1]
    assert len(experts_none) == 422 + 8 + (822 - 54 - 18 - 8)
    # The tiny writer's own layout rule, applied to the released configuration, gives the
    # released header: tiny checkpoints have the names, dtypes and packing of the real one.
    released = json.loads((snapshot / "config.json").read_text())
    layout = expected_schema(released, mxfp8_layers=range(20, 24))
    assert {name: item for name, item in layout.items() if name.startswith(LLM)} == {
        name: (item.dtype, item.shape) for name, item in entry.items() if name.startswith(LLM)
    }


@pytest.mark.parametrize("layer, expert", [(0, 37), (20, 5)], ids=["nvfp4", "mxfp8"])
def test_real_expert_equals_the_float64_definition(mini, layer, expert):
    """One released expert per format, read by safetensors' own slicing and decoded with the
    float64 definition, equals the loader's dense factor and scale bit for bit. The largest
    NVFP4 element of the expert is 6 * 448, i.e. the outer scale carries the expert's amax."""
    checkpoint = weights.load_inkling2_checkpoint(mini, layers=[layer], experts="packed")
    packed = checkpoint.experts(layer)
    with safe_open(str(mini / SHARD_NAME), framework="pt") as handle:
        for projection, shape in (("w13", (2304, 2304)), ("w2", (2304, 1152))):
            name = f"{LLM}layers.{layer}.mlp.experts.{projection}_weight"
            stored = handle.get_slice(name)[expert : expert + 1]
            scales = handle.get_slice(name + "_scale")[expert : expert + 1]
            dense, scale = packed.dequantize(projection, [expert])
            assert dense.dtype == torch.bfloat16 and dense.shape == (1, *shape)
            if packed.quant_format == "nvfp4":
                expected = nvfp4_definition(stored, scales)
                amax = handle.get_tensor(name + "_amax")
                outer = outer_scale_definition(amax, handle.get_tensor(name + "_outer_scale_denominator"))
                assert expected.abs().max() == 2688
                assert_same_bits(scale, outer[expert : expert + 1])
                # The outer scale restores the stored amax up to FP32 rounding of the quotient.
                assert abs(float(expected.abs().max() * outer[expert]) / float(amax[expert]) - 1) < 1e-6
            else:
                expected = mxfp8_definition(stored, scales)
                assert_same_bits(scale, torch.ones(1, dtype=torch.float64))
            assert_same_bits(dense, expected)
            # The weights have a standard deviation of about 1 / sqrt(fan_in).
            assert 0.5 < float(expected.std() * scale.double() * shape[1] ** 0.5) < 2


def test_real_w2_columns_align_with_their_w13_gate_and_up_rows(mini):
    """Checks the nibble order of both projections and the gate/up interleave on the released
    weights themselves, which a comparison against a definition with the same reading cannot.
    The output direction of intermediate unit j (column j of w2) is aligned with its two input
    directions (rows 2j and 2j + 1 of w13) well above the level of unrelated directions (0.017
    in 2,304 dimensions), and at that level as soon as either projection is read with the
    nibbles swapped or the rows are paired as (j, I + j).

    Which row of a pair is the gate follows from a symmetry: the activation is odd in up and
    not in gate, so negating up row j together with column j of w2 leaves the model unchanged.
    The signed cosine between the column and the up row is invariant under that and may have a
    mean; the one with the gate row changes sign and has to average out over the units."""
    packed = weights.load_inkling2_checkpoint(mini, layers=[7], experts="packed").experts(7)
    w13 = packed.dequantize("w13", 50, torch.float64)[0][0]
    w2 = packed.dequantize("w2", 50, torch.float64)[0][0]
    rows = torch.nn.functional.normalize(w13, dim=1)  # [2I, H] input directions
    columns = torch.nn.functional.normalize(w2, dim=0).T  # [I, H] output directions

    def alignment(outputs, inputs):
        return float((outputs * inputs).sum(dim=1).abs().mean())

    def mean_in_standard_errors(outputs, inputs):
        cosines = (outputs * inputs).sum(dim=1)
        return float(cosines.mean() / cosines.std() * len(cosines) ** 0.5)

    assert alignment(columns, rows[weights.GATE_ROWS]) > 0.06
    assert alignment(columns, rows[weights.UP_ROWS]) > 0.06
    assert abs(mean_in_standard_errors(columns, rows[weights.GATE_ROWS])) < 4
    assert mean_in_standard_errors(columns, rows[weights.UP_ROWS]) > 20
    # Reading a byte high nibble first swaps the elements 2k and 2k + 1 of a row.
    w13_swapped = rows.unflatten(1, (-1, 2)).flip(2).flatten(1)
    w2_swapped = columns.unflatten(0, (-1, 2)).flip(1).flatten(0, 1)
    half = len(columns)
    for outputs, inputs in (
        (columns, w13_swapped[weights.GATE_ROWS]),
        (columns, w13_swapped[weights.UP_ROWS]),
        (w2_swapped, rows[weights.GATE_ROWS]),
        (w2_swapped, rows[weights.UP_ROWS]),
        (columns, rows[:half]),
        (columns, rows[half:]),
    ):
        assert alignment(outputs, inputs) < 0.025


def test_real_outer_scale_denominators_are_2688(mini_index):
    """The 40 NVFP4 denominators are 6 * 448, the largest value of the dense factor, and the
    5,120 amax values are positive: every outer scale is a small finite FP32 number."""
    for layer in range(20):
        for projection in ("w13", "w2"):
            name = weights.expert_prefix(layer, projection)
            denominator = mini_index.read(name + "_outer_scale_denominator")
            assert denominator.dtype == torch.float32 and denominator.tolist() == [2688.0]
            amax = mini_index.read(name + "_amax")
            assert amax.shape == (128,) and 0.1 < float(amax.min()) and float(amax.max()) < 0.9
    scalars = [name for name in mini_index.entries if name.endswith(("_amax", "_outer_scale_denominator"))]
    assert len(scalars) == 80  # none in the MXFP8 layers 20-23


def test_real_mxfp8_block_exponents_are_in_the_exact_range(mini_index):
    """Every MXFP8 block exponent of the checkpoint (all 0.12 GiB of them) is 113..118, far
    inside the range where e4m3 * 2**(k - 127) is a normal BF16 number."""
    for layer in range(20, 24):
        for projection in ("w13", "w2"):
            exponents = mini_index.read(weights.expert_prefix(layer, projection) + "_scale")
            assert exponents.dtype == torch.uint8
            assert 113 <= int(exponents.min()) and int(exponents.max()) <= 118


def test_real_served_vocabulary_views_have_200058_rows(mini):
    """The served views drop the 966 padding rows of the 201,024-row tables. Those rows are not
    zero in the checkpoint (unit-norm unembedding rows), so a head over the padded table would
    give them real logits."""
    checkpoint = weights.load_inkling2_checkpoint(mini, layers=[], experts="none")
    views = ((checkpoint.embedding, weights.EMBED_NAME), (checkpoint.unembedding, weights.UNEMBED_NAME))
    for view, name in views:
        padded = checkpoint.tensors[name]
        assert padded.shape == (201024, 2304) and view.shape == (200058, 2304)
        assert view.dtype == torch.bfloat16 and view.data_ptr() == padded.data_ptr()
        assert torch.equal(view[-1], padded[200057])
    padding = checkpoint.tensors[weights.UNEMBED_NAME][200058:].float()
    assert padding.shape == (966, 2304)
    norms = padding.norm(dim=-1)
    assert 0.99 < float(norms.min()) and float(norms.max()) < 1.01
