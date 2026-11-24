# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Worker-side 2DSD generation for Nanbeige (R=2) targets (batch size one).

With two loops there is no deeper readout, so 2DSD runs Stage 1 (early review
at loop 1 plus block drafting); k=1 is depth-axis speculation (LoopSpec d1).
"""

import json
from pathlib import Path

import torch

from .looped_2d_finalized import FinalizedTreeExecutor


def validate_nanbeige_schedule(k, stage, d1):
    if stage != "stage1":
        raise ValueError("Nanbeige R2 supports Stage 1; Stage 2 is undefined")
    if k not in (1, 4, 8, 16) or type(k) is not int or type(d1) is not int or d1 != 1:
        raise ValueError("Nanbeige Stage 1 requires k in {1,4,8,16} and d1=1")


def run_nanbeige_finalized(
    worker,
    requests,
    drafter_export=None,
    k=4,
    stage="stage1",
    d1=1,
    max_model_len=8192,
    initial_capacity=64,
    use_cuda_graph=True,
    result_callback=None,
    warmup=0,
    use_fused_norm=True,
    use_fused_rope=True,
    prewarm_tree_graphs=True,
    record_trace=False,
    context_cuda_graph=None,
    context_graph_min_calls=8,
    context_graph_grid=True,
    temperature=0.0,
    top_p=1.0,
    proposal_top_p=None,
    compact_shared_prefix=True,
):
    """Run Stage 1 or its k1 LoopSpec boundary on a loaded native R2 target.

    Output IDs, timings and physical counters follow the Ouro finalized RPC.
    Request cleanup runs on success and on exception. No vLLM global model or
    CustomOp state is replaced. This is a batch-one research worker path
    (greedy, or sampling with temperature > 0, optional top_p < 1 and a per-request
    "seed"), using a private full-precision prefix KV allocation.
    """
    from vllm.model_executor.models.nanbeige_2d import NativeNanbeigeTreeRunner
    from vllm.model_executor.models.looped_kv import check_tree_window
    from vllm.v1.spec_decode.looped_dflash_context import (
        ContextDFlashAdapter,
        drafting_grid,
    )

    validate_nanbeige_schedule(k, stage, d1)
    if warmup < 0:
        raise ValueError("Warmup must be nonnegative")
    if not requests:
        return []
    if k > 1 and not drafter_export:
        raise ValueError("Nanbeige blocks larger than one require a drafter export")
    layers, input_format = None, None
    if drafter_export and k > 1:
        config = json.loads((Path(drafter_export) / "config.json").read_text())
        metadata = config["loopedtf_drafter"]
        layers = metadata["feature_layers_per_loop"]
        input_format = metadata.get("input_format")
        if input_format != "loop-context":
            raise ValueError("2DSD drafters use the loop-context input format")
    model = worker.model_runner.get_model()
    policy = getattr(model.config, "loopedtf_kv_policy", None) or "full"
    check_tree_window(model.config, k)
    runner = NativeNanbeigeTreeRunner(
        model, max_model_len, max_nodes=initial_capacity, feat_layers=layers,
        use_cuda_graph=use_cuda_graph, max_graph_shapes=128,
        use_fused_norm=use_fused_norm, use_fused_rope=use_fused_rope,
        compact_shared_prefix=compact_shared_prefix,
    )
    adapter = None
    executor = None
    try:
        if k > 1:
            adapter = ContextDFlashAdapter.from_export(
                drafter_export, embed=runner.embed, lm_head=runner.logits,
                max_len=max_model_len, device=runner.device, dtype=runner.dtype,
                use_cuda_graph=use_cuda_graph and context_cuda_graph is not False,
                batched_graph_min_calls=context_graph_min_calls,
            )
        executor = FinalizedTreeExecutor(
            runner, adapter, k=k, stage="stage1", d1=1, record_trace=record_trace,
            temperature=temperature, top_p=top_p, proposal_top_p=proposal_top_p,
        )

        def generate(request):
            return executor.generate(
                request["prompt_ids"], request["max_new"],
                request.get("eos_token_ids", []), seed=request.get("seed"),
            )

        prewarm = runner.prewarm_graphs(
            min(initial_capacity, 16) if prewarm_tree_graphs else 0
        )
        prewarm["scope"] = "once_per_rpc_before_measured_requests"
        drafter_prewarm = (
            adapter.prewarm_grid(k, *drafting_grid(runner.R, k))
            # The padded grid reads the masks after the anchor; a DSpark drafter
            # whose anchor slot predicts drafts on the layout path.
            if context_graph_grid and adapter is not None and adapter.first_slot == 1
            else {"enabled": False}
        )
        for _ in range(warmup):
            request = dict(requests[0])
            request["max_new"] = min(64, request["max_new"])
            generate(request)
        results = []
        for request in requests:
            before = dict(runner.stats)
            draft_before = dict(adapter.stats) if adapter is not None else {}
            if runner.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runner.device)
            result = generate(request)
            result["target_runtime_stats"] = {
                name: value - before.get(name, 0)
                for name, value in runner.stats.items()
                if name not in ("capacity", "tree_kv_bytes", "peak_live_nodes")
            }
            result["target_runtime_state"] = dict(
                prefix_kv_bytes=runner.prefix.kv_bytes,
                prefix_kv_storage=runner.prefix.kv_storage,
                physical_capacity=runner.max_nodes, path_width=runner.path_width,
                tree_kv_bytes=runner.stats["tree_kv_bytes"],
                lifetime_peak_live_nodes=runner.stats["peak_live_nodes"],
            )
            result["drafter_runtime_stats"] = {
                name: value - draft_before.get(name, 0)
                for name, value in adapter.stats.items()
            } if adapter is not None else {}
            result["startup_target_graph_prewarm"] = prewarm
            result["startup_drafter_graph_prewarm"] = drafter_prewarm
            result["settings"] = dict(
                execution="2dsd-sampling" if temperature > 0 else "2dsd-greedy",
                **({"temperature": temperature, "seed": request.get("seed")}
                   if temperature > 0 else {}),
                **({"top_p": top_p} if top_p < 1 else {}),
                **({"proposal_top_p": proposal_top_p}
                   if proposal_top_p is not None and proposal_top_p != top_p else {}),
                context_graph_grid=drafter_prewarm.get("enabled", False),
                model_family="nanbeige",
                num_loops=runner.R, stage="stage1", d1=1, d2=None,
                block_size=k,
                initial_capacity=initial_capacity, physical_capacity=runner.max_nodes,
                max_graph_shapes=runner.max_graph_shapes, kv_policy=policy,
                kv_window=runner.window, kv_storage=runner.prefix.kv_storage,
                target_fa_version=getattr(runner.prefix, "fa_version", None),
                drafter_fa_version=getattr(adapter, "fa_version", None),
                drafter_input_format=input_format,
                cuda_graph=runner.use_cuda_graph,
                context_cuda_graph=bool(adapter and adapter.use_cuda_graph),
                context_graph_min_calls=(
                    context_graph_min_calls if adapter is not None else None
                ),
                fused_norm=use_fused_norm, fused_rope=use_fused_rope,
                prewarm_tree_graphs=prewarm_tree_graphs,
            )
            if runner.device.type == "cuda":
                result["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    runner.device
                )
                result["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                    runner.device
                )
            if result_callback is None:
                results.append(result)
            else:
                result_callback(result)
        return results
    finally:
        if executor is not None:
            executor._reset_candidates()
        runner.reset()
        if adapter is not None:
            adapter.reset()
