# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Routed experts of Inkling 2 by Humming's packed expert GEMMs.

The served Hopper path (vLLM PR #69028, vllm/models/inkling2/nvidia/ops/moe_sm90.py,
HummingExperts) multiplies BF16 activations with the checkpoint's packed NVFP4 /
MXFP8 expert weights: nothing is dequantized ahead, so Mini's experts take 18 GiB
instead of 49 GiB and a call reads 4 or 8 MB per routed expert instead of 16 MB.
The kernels run in Humming's batch-invariant mode without stream-K, as served.

It needs humming-kernels >= 0.1.20 (humming.forward, humming.transform); the
release environment pins 0.1.10. ``available()`` tells whether the installed one
will do.
"""

import json

import torch


def available() -> bool:
    """Whether Humming's packed expert GEMM can be imported."""
    try:
        import humming.forward  # noqa: F401
        import humming.transform  # noqa: F401
    except ImportError:
        return False
    return True


def supported(model) -> bool:
    """Whether the routed experts of ``model`` can run on Humming: packed experts
    on a GPU, the served precision and a recent Humming."""
    p = model.precision
    return (model.device.type == "cuda" and model.checkpoint.expert_mode == "packed"
            and p.storage == torch.bfloat16 and p.dtype == torch.float32 and available())


FEW_PAIRS = 8  # (row, expert) pairs of one row: above it the throughput schedule packs two blocks per SM


class HummingExperts:
    """The two expert projections of one layer as Humming tensors and kernel configurations (HummingExperts
    of the served path), from the checkpoint's packed tensors."""

    def __init__(self, packed, shapes, batch_invariant=True):
        from humming.config import GemmType
        from humming.schema import HummingInputSchema
        from humming.schema.modelopt import ModeloptMxfp8WeightSchema, ModeloptNvfp4WeightSchema
        from humming.transform import prepare_and_transform_humming_tensors
        from humming.tune import get_heuristics_config
        from vllm.model_executor.models.inkling2_weights import expert_prefix

        nvfp4 = packed.quant_format == "nvfp4"
        self.configs, self.tensors, self.tunings, self.tuning_strs, self.few_tuning_strs = {}, {}, {}, {}, {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            key = expert_prefix(packed.layer_id, name)
            weight, scale = packed.tensors[key], packed.tensors[key + "_scale"]
            if nvfp4:  # two E2M1 codes per byte, an E4M3 scale per 16 inputs, a global scale per expert
                tensors = {"weight": weight.view(torch.uint8), "weight_scale": scale.view(torch.float8_e4m3fn),
                           "weight_scale_2": (packed.tensors[key + "_amax"].float()
                                              / packed.tensors[key + "_outer_scale_denominator"].float())}
                source = ModeloptNvfp4WeightSchema()
            else:  # E4M3 values, an exponent byte per 32 inputs
                tensors = {"weight": weight.view(torch.float8_e4m3fn), "weight_scale": scale.view(torch.uint8)}
                source = ModeloptMxfp8WeightSchema()
            num_experts = weight.shape[0]
            schema, tensors = source.convert_humming(tensors=tensors, shape_n_stacks=[n], shape_k_stacks=[k],
                                                     param_dtype=torch.bfloat16, num_experts=num_experts)
            config, tensors = prepare_and_transform_humming_tensors(
                tensors=tensors, shape_n=n, shape_k=k, pad_n_to_multiple=256, pad_k_to_multiple=128,
                input_schema=HummingInputSchema(), weight_schema=schema, num_experts=num_experts,
                torch_dtype=torch.bfloat16, device=weight.device)
            if (config.shape_n, config.shape_k) != (n, k):
                raise ValueError(f"Humming padded {name} to {(config.shape_n, config.shape_k)}")
            tuning = get_heuristics_config(layer_config=config, use_f16_accum=False,
                                           use_batch_invariant=batch_invariant, gemm_type=GemmType.INDEXED)
            for _, _, kernel in tuning:  # Stream-K adds K slices with BF16 atomics in an order that varies
                kernel["use_stream_k"] = False
            self.configs[name], self.tensors[name] = config, tensors
            self.tunings[name], self.tuning_strs[name] = tuning, json.dumps(tuning)
            # The same tiles (so the same sums) scheduled for throughput: more blocks per SM above one row's
            # experts. For calls of a few rows; see ``gemm``.
            few = []
            for lower, upper, kernel in tuning:
                if lower < FEW_PAIRS < upper:
                    few += [[lower, FEW_PAIRS, dict(kernel, num_ctas_per_sm=1)],
                            [FEW_PAIRS, upper, dict(kernel, num_ctas_per_sm=2)]]
                else:
                    few.append([lower, upper, dict(kernel, num_ctas_per_sm=1 if upper <= FEW_PAIRS else 2)])
            self.few_tuning_strs[name] = json.dumps(few)
        self.num_experts = num_experts
        self.compute_config = json.dumps({"use_batch_invariant": batch_invariant, "use_f16_accum": False,
                                          "gemm_type": GemmType.INDEXED.value})
        self.few_compute_config = json.dumps({"use_batch_invariant": False, "use_f16_accum": False,
                                              "gemm_type": GemmType.INDEXED.value})
        self.locks = torch.zeros(2048, dtype=torch.int32, device=weight.device)

    def block_rows(self, name, rows):
        for lower, upper, kernel in self.tunings[name]:
            if lower < rows <= upper:
                return kernel["block_shape"][0]
        return 64

    def align(self, ids):
        """Row order by expert for both projections: {block rows: (sorted ids, expert ids, padded count)}."""
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

        rows = ids.numel()
        return {size: moe_align_block_size(ids, size, self.num_experts, ignore_invalid_experts=True)[:3]
                for size in {self.block_rows(name, rows) for name in ("w13", "w2")}}

    def gemm(self, name, inputs, rows, alignments, top_k, few=False):
        """BF16 [rows, N] of projection `name`: inputs [rows / top_k, K], row r from expert ids.flat[r].
        ``few`` runs the same tiles without Humming's batch-invariant scheduling, which is faster for the
        handful of rows of a tree call."""
        from humming.forward import humming_forward

        config, tensors = self.configs[name], self.tensors[name]
        out = torch.empty((rows, config.shape_n), dtype=torch.bfloat16, device=inputs.device)
        sorted_ids, expert_ids, padded = alignments[self.block_rows(name, rows)]
        return humming_forward(config, inputs=inputs, weight=tensors["weight"],
                               weight_scale=tensors.get("weight_scale"),
                               weight_scale_2=tensors.get("weight_scale_2"), outputs=out, locks=self.locks,
                               sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded, top_k=top_k,
                               valid_shape_m=rows,
                               compute_config=self.few_compute_config if few else self.compute_config,
                               tuning_config=self.few_tuning_strs[name] if few else self.tuning_strs[name])

    def products(self, x, ids, weights, config, few=False):
        """Output BF16 [T, H] of the routed experts ``ids`` [T, top_k] with routing
        ``weights`` [T, top_k] on the BF16 normed inputs x [T, H]: the gate/up GEMM,
        the softcapped SwiGLU times the routing weight, the down GEMM and the sum
        of a row's experts in slot order."""
        from vllm.model_executor.models import inkling2_ops as ops

        tokens, top = ids.shape
        alignments = self.align(ids.to(torch.int32))
        gate_up = self.gemm("w13", x, tokens * top, alignments, top, few)
        hidden = ops.situ_rows(gate_up, weights.reshape(-1).contiguous(), config.mlp_gate_softcap,
                               config.mlp_up_softcap)
        down = self.gemm("w2", hidden, tokens * top, alignments, 1, few)
        return ops.sum_slots(down.view(tokens, top, -1))
