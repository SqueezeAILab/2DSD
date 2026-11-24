# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Tiny Inkling2 checkpoint in the on-disk format of the released ones.

``write_tiny_checkpoint`` writes ``config.json``, ``hf_quant_config.json``,
``model.safetensors.index.json`` and one safetensors shard whose tensor names,
dtypes and packing are those of ``thinkingmachines/Inkling-2-Mini`` (header of
revision 1fb4134b): BF16 weights, FP32 pseudo-queries and router biases, NVFP4
experts (two E2M1 codes per byte, low nibble first, an E4M3 scale per 16 inputs,
a per-expert amax and the denominator 2688) and MXFP8 experts (E4M3 values with
an exponent byte per 32 inputs). The default model has 6 physical layers with
blocks of 2, the loop core 2..3 run 3 times (10 layer visits), full attention on
layers 1, 3 and 5, 8 experts with top-2 routing, NVFP4 experts in layers 0..3 and
MXFP8 experts in layers 4 and 5, and a 112-row vocabulary of which 100 ids are
served.

The layout is written out here independently of the loader, so loading a tiny
checkpoint also checks the loader's name and shape rules. The weights are seeded
random values with the scales of the released checkpoint (unit-RMS embeddings,
``1 / sqrt(fan_in)`` projections); the experts are drawn in FP32 and quantized,
so a tiny checkpoint has realistic nibbles and block scales. Tests compare
implementations on it, not checkpoint quality.
"""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

SHARD_NAME = "model-00001-of-00001.safetensors"
OUTER_SCALE_DENOMINATOR = 2688.0
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
LOOP_STREAMS = ("attn", "k", "v", "mlp")


def tiny_config(**text_overrides):
    """``config.json`` of the tiny model; ``text_overrides`` replace text fields."""
    text = dict(
        model_type="inkling2_loop_text",
        model_max_length=4096,
        torch_dtype="bfloat16",
        hidden_size=32,
        num_hidden_layers=6,
        vocab_size=112,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=4,
        rel_extent=16,
        q_bias=False,
        o_bias=False,
        log_scaling_n_floor=8,
        log_scaling_alpha=0.1,
        rms_norm_eps=1e-6,
        use_value_norm_no_gamma=True,
        qk_l2_clip_threshold=2 * 8**0.5,
        rel_l2_clip_threshold=4.0,
        local_layer_ids=[0, 2, 4],
        sconv_kernel_size=4,
        unpadded_vocab_size=100,
        final_logit_softcapping=30.0,
        swa_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        sliding_window_size=8,
        n_layers_per_attn_res_block=2,
        attn_res_normalize_values=True,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=0,
        intermediate_size=64,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        route_scale=2.0,
        use_gate_bias=True,
        loop_config=dict(
            core_start=2, core_end=4, num_passes=3, gate_rank=4, is_varloop=False
        ),
    )
    text.update(text_overrides)
    return dict(
        architectures=["Inkling2LoopForConditionalGeneration"],
        model_type="inkling2_loop",
        eos_token_id=96,
        text_config=text,
        vision_config=dict(
            model_type="inkling2_vision",
            text_hidden_size=text["hidden_size"],
            hidden_size=8,
            num_hidden_layers=1,
        ),
        quantization_config=dict(quant_method="modelopt", quant_algo="NVFP4"),
    )


def tiny_layout(config, mxfp8_layers=(4, 5)):
    """Name -> (kind, shape) of every tensor of the checkpoint.

    ``kind`` is ``"bf16"``, ``"f32"``, ``"nvfp4"`` or ``"mxfp8"``; an expert
    entry stands for the packed weight and its scale tensors, and its shape is
    the dense ``(experts, out, in)``.
    """
    text = config["text_config"]
    hidden, layers = text["hidden_size"], text["num_hidden_layers"]
    kernel, d_rel = text["sconv_kernel_size"], text["d_rel"]
    experts, inter = text["n_routed_experts"], text["intermediate_size"]
    loop = text["loop_config"]
    core = range(loop["core_start"], loop["core_end"])
    local = set(text["local_layer_ids"])
    llm = "model.llm."
    layout = {
        llm + "embed.weight": ("bf16", (text["vocab_size"], hidden)),
        llm + "unembed.weight": ("bf16", (text["vocab_size"], hidden)),
        llm + "embed_norm.weight": ("bf16", (hidden,)),
        llm + "norm.weight": ("bf16", (hidden,)),
        llm + "attn_res.output_query": ("f32", (hidden,)),
        llm + "middle_cycle_cross_pass_queries": ("f32", (2 * len(core), hidden)),
        llm + "middle_cycle_cross_pass_kv_queries": ("f32", (2 * len(core), hidden)),
    }
    for i in range(layers):
        full = i not in local
        if i:
            # Query i - 1 feeds the attention of layer i: one row for a sliding
            # layer, separate q/k/v rows for a full-attention layer.
            shape = (3, hidden) if full else (hidden,)
            layout[f"{llm}attn_res.attn_queries.{i - 1}"] = ("f32", shape)
        layout[f"{llm}attn_res.mlp_queries.{i}"] = ("f32", (hidden,))
        kind = "" if full else "swa_"
        heads = text[kind + "num_attention_heads"]
        kv_heads = text[kind + "num_key_value_heads"]
        head_dim = text[kind + "head_dim"]
        extent = text["rel_extent"] if full else text["sliding_window_size"]
        bf16 = {
            "attn.wq_du.weight": (heads * head_dim, hidden),
            "attn.wr_du.weight": (heads * d_rel, hidden),
            "attn.gate_attn.weight": (heads * head_dim, hidden),
            "attn.attn_sink_proj.weight": (heads, hidden),
            "attn.wk_dv.weight": (kv_heads * head_dim, hidden),
            "attn.wv_dv.weight": (kv_heads * head_dim, hidden),
            "attn.wo_ud.weight": (hidden, heads * head_dim),
            "attn.q_l2_clip.scale": (heads, head_dim),
            "attn.rel_l2_clip.scale": (heads, d_rel),
            "attn.rel_logits_proj.proj": (d_rel, extent),
            "attn_sconv.weight": (hidden, 1, kernel),
            "mlp_sconv.weight": (hidden, 1, kernel),
            "mlp_norm.weight": (hidden,),
            "mlp.gate.weight": (experts, hidden),
        }
        if full:
            for name in ("q_input_norm", "k_input_norm", "v_input_norm"):
                bf16[name + ".weight"] = (hidden,)
            for name in ("attn.k_sconv", "attn.v_sconv"):
                bf16[name + ".static_w"] = (kernel, kv_heads * head_dim)
                bf16[name + ".weight_proj"] = (kv_heads * kernel, hidden)
        else:
            bf16["attn_norm.weight"] = (hidden,)
            for name in ("attn.k_sconv", "attn.v_sconv"):
                bf16[name + ".weight"] = (kv_heads * head_dim, 1, kernel)
        if i in core:
            for stream in LOOP_STREAMS:
                name = f"controllers.{stream}."
                bf16[name + "down_weight"] = (loop["gate_rank"], 2 * hidden)
                bf16[name + "gain_weight"] = (hidden, loop["gate_rank"])
                bf16[name + "forget_weight"] = (hidden, loop["gate_rank"])
        prefix = f"{llm}layers.{i}."
        layout.update({prefix + name: ("bf16", shape) for name, shape in bf16.items()})
        layout[prefix + "mlp.gate.bias"] = ("f32", (experts,))
        fmt = "mxfp8" if i in mxfp8_layers else "nvfp4"
        layout[prefix + "mlp.experts.w13_weight"] = (fmt, (experts, 2 * inter, hidden))
        layout[prefix + "mlp.experts.w2_weight"] = (fmt, (experts, hidden, inter))
    vision = config["vision_config"]["hidden_size"]
    layout["model.visual.backbone.embeddings.class_embedding"] = ("bf16", (vision,))
    layout["model.visual.backbone.layers.0.ln_1.weight"] = ("bf16", (vision,))
    layout["model.visual.proj_out.weight"] = ("bf16", (hidden, vision))
    return layout


def quantize_nvfp4(weight):
    """FP32 ``[E, out, in]`` -> packed bytes, E4M3 block scales, amax, denominator.

    The outer scale of an expert is ``amax / 2688``; each 16-element block gets
    the E4M3 scale that maps its largest magnitude to the code 6, and each element
    the nearest E2M1 code of its value in units of block scale times outer scale.
    """
    amax = weight.abs().amax(dim=(1, 2))
    denominator = torch.tensor([OUTER_SCALE_DENOMINATOR])
    outer = (amax / denominator)[:, None, None]
    blocks = weight.unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(-1) / outer / 6.0).clamp(max=448.0)
    scale = scale.nan_to_num(0.0).to(torch.float8_e4m3fn)
    unit = (scale.float() * outer)[..., None]
    ratio = torch.where(unit > 0, blocks.abs() / unit, 0.0)
    grid = torch.tensor(E2M1_MAGNITUDES)
    codes = (ratio[..., None] - grid).abs().argmin(-1) | (blocks < 0).long() << 3
    codes = codes.flatten(-2)
    packed = (codes[..., 0::2] | codes[..., 1::2] << 4).to(torch.uint8)
    return packed, scale, amax, denominator


def quantize_mxfp8(weight):
    """FP32 ``[E, out, in]`` -> E4M3 values and exponent bytes per 32 inputs.

    The block scale is the smallest power of two that keeps the block within the
    E4M3 range (448); an all-zero block gets the exponent byte 0.
    """
    blocks = weight.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1)
    exponent = torch.ceil(torch.log2(amax / 448.0)) + 127
    exponent = torch.where(amax > 0, exponent, 0.0).clamp(0, 254)
    values = (blocks / torch.exp2(exponent - 127)[..., None]).clamp(-448.0, 448.0)
    return values.flatten(-2).to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def _draw(name, shape, generator):
    def normal(std, mean=0.0):
        return torch.randn(shape, generator=generator) * std + mean

    if name.endswith(".embed.weight"):
        return normal(1.0)
    if name.endswith("norm.weight"):
        return normal(0.2, 1.0)
    if name.endswith(("sconv.weight", "sconv.static_w")):
        return normal(0.5, 0.4)
    if name.endswith("l2_clip.scale"):
        return normal(0.4, 1.2)
    if "queries" in name or name.endswith("output_query"):
        return normal(0.7 if "cross_pass" in name else 0.35)
    if name.endswith(("gain_weight", "forget_weight")):
        return normal(0.1)
    if name.endswith("rel_logits_proj.proj"):
        return normal(0.3)
    if name.endswith("gate.bias"):
        return normal(0.03)
    return normal(shape[-1] ** -0.5)


def write_tiny_checkpoint(directory, config=None, seed=0, mxfp8_layers=(4, 5)):
    """Writes a tiny checkpoint to ``directory`` and returns the directory.

    ``config`` is a ``tiny_config()`` (the default) or a variant of it, and
    ``mxfp8_layers`` are the layers with MXFP8 experts. The same arguments give
    the same bytes. The first block of expert 0, row 0 of every ``w13`` (16 inputs
    for NVFP4, 32 for MXFP8) is zero with one negative-zero element, so that the
    zero-scale and zero-payload cases of both formats are on disk.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = tiny_config() if config is None else config
    generator = torch.Generator().manual_seed(seed)
    tensors, groups = {}, {"float32": [], "bfloat16": [], "nvfp4": [], "mxfp8": []}
    for name, (kind, shape) in tiny_layout(config, mxfp8_layers).items():
        value = _draw(name, shape, generator)
        if kind == "f32":
            tensors[name] = value
            groups["float32"].append(name)
        elif kind == "bf16":
            tensors[name] = value.bfloat16()
            groups["bfloat16"].append(name)
        elif kind == "nvfp4":
            if name.endswith("w13_weight"):
                value[0, 0, :16] = 0.0
            packed, scale, amax, denominator = quantize_nvfp4(value)
            if name.endswith("w13_weight"):
                packed[0, 0, 0] = 0x80
            tensors[name], tensors[name + "_scale"] = packed, scale
            tensors[name + "_amax"] = amax
            tensors[name + "_outer_scale_denominator"] = denominator
            groups["nvfp4"].append(name)
        else:
            if name.endswith("w13_weight"):
                value[0, 0, :32] = 0.0
            values, exponent = quantize_mxfp8(value)
            if name.endswith("w13_weight"):
                values.view(torch.uint8)[0, 0, 1] = 0x80
            tensors[name], tensors[name + "_scale"] = values, exponent
            groups["mxfp8"].append(name)
    shard = directory / SHARD_NAME
    partial = shard.with_suffix(".partial")
    save_file(tensors, str(partial), metadata={"format": "pt"})
    os.replace(partial, shard)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    quant = dict(
        model=dict(
            parameter_groups=[
                dict(compute_dtype=compute, tensors=names)
                for compute, names in groups.items()
                if names
            ]
        ),
        quant_method="inkling2",
    )
    (directory / "hf_quant_config.json").write_text(json.dumps(quant, indent=2))
    total = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    index = dict(
        metadata=dict(total_size=total),
        weight_map=dict.fromkeys(sorted(tensors), SHARD_NAME),
    )
    (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return directory
