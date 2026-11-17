# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Finite batch-one Dory 2D worker RPC: LoopSpec and block Stage 1/2a/2b."""

import json
from pathlib import Path

import torch

from .looped_2d_finalized import FinalizedTreeExecutor

SHARED_WINDOW = 64
# Row and path bounds of the startup graphs: a LoopSpec tree holds one node per
# loop and fork. A block schedule is prepared up to the initial node capacity:
# frozen suffixes and forks can keep well over 8 * k nodes unfinished at once.
LOOPSPEC_ROWS, LOOPSPEC_PATH = 16, 16


def validate_dory_schedule(k, stage, d1, d2, loops):
    if stage not in ("stage1", "stage2a", "stage2b"):
        raise ValueError("Unknown finalized stage")
    if type(k) is not int or k not in (1, 4, 8, 16):
        raise ValueError("Dory schedules use k in {1, 4, 8, 16}")
    if not 1 <= d1 <= loops:
        raise ValueError("d1 lies in [1, R]")
    if stage != "stage1" and not d1 < d2 < loops:
        raise ValueError("Stage 2 needs d1 < d2 < R")


def run_dory_finalized(
    worker,
    requests,
    drafter_export=None,
    k=1,
    stage="stage2b",
    d1=1,
    d2=2,
    max_model_len=8192,
    initial_capacity=64,
    use_cuda_graph=True,
    result_callback=None,
    warmup=0,
    prewarm_tree_graphs=True,
    record_trace=False,
    context_cuda_graph=None,
    context_graph_min_calls=8,
    context_graph_grid=True,
    temperature=0.0,
    record_sampling_check=False,
    stage2_gate="argmax",
    top_p=1.0,
    proposal_top_p=None,
    compact_shared_prefix=True,
    compile_blocks=False,
    drafter_fast_grid=True,
):
    """Worker-side 2DSD generation for Dory targets (batch size one; R from the config).

    The target is the engine's model: its tensors are shared, and the KV policy
    of its config selects the runtime: per-loop KV, or last-loop sharing
    (``shared_last``) with W = 64 on the full-attention layers of the recurrent
    and output blocks. Readouts at d1, d2 and R run the output block inside the
    target call (``DoryGraphRunner``). Requests, results and counters follow
    ``run_ouro_finalized``.

    ``compact_shared_prefix`` selects the runner's compact prefix storage (rings
    for the sliding layers and the shared planes below R; ``False`` keeps every
    plane over the whole context, with the same values read).
    ``compile_blocks`` runs the nGPT glue around attention through
    torch.compile (the Python API always sets it, as in the paper's
    measurements); without it the glue runs eagerly inside the CUDA graphs.
    ``drafter_fast_grid`` runs the drafter's padded grid through its fused graphs
    with the scaled head inside the argmax (the same model; proposals may differ
    at BF16 rounding).
    """
    from vllm.model_executor.models.dory_2d import DoryGraphRunner, DoryWeights
    from vllm.v1.spec_decode.looped_dflash_context import (
        ContextDFlashAdapter,
        drafting_grid,
    )

    if warmup < 0:
        raise ValueError("Warmup must be nonnegative")
    if not requests:
        return []
    model = worker.model_runner.get_model()
    config = model.config
    loops = config.n_recurrent_loops
    validate_dory_schedule(k, stage, d1, d2, loops)
    if k > 1 and not drafter_export:
        raise ValueError("Dory blocks larger than one require a drafter export")
    if config.recurrent_kv_cache_mode != "per_loop":
        raise ValueError("The Dory tree runner keeps per-loop recurrent KV")
    policy = getattr(config, "loopedtf_kv_policy", None) or "full"
    window = None
    if policy == "shared_last":
        if (
            config.loopedtf_kv_window != SHARED_WINDOW
            or getattr(config, "loopedtf_kv_share_layers", "all") != "full_attention"
        ):
            raise ValueError(
                "Dory shares last-loop KV with W = 64 on the full-attention layers only"
            )
        window = SHARED_WINDOW
    elif policy != "full":
        raise ValueError(f"Unsupported Dory KV policy: {policy}")
    input_format = None
    if k > 1:
        metadata = json.loads((Path(drafter_export) / "config.json").read_text())[
            "loopedtf_drafter"
        ]
        input_format = metadata.get("input_format")
        if input_format != "loop-context" or metadata.get("target_loops") != loops:
            raise ValueError("Dory drafts with its loop-context drafter at the same R")
    depths = {d1, loops} if stage == "stage1" else {d1, d2, loops}
    runner = DoryGraphRunner(
        DoryWeights.from_vllm(model),
        loops,
        depths,
        window=window,
        max_len=max_model_len,
        max_nodes=initial_capacity,
        use_cuda_graph=use_cuda_graph,
        compile_blocks=compile_blocks,
        compact_prefix=compact_shared_prefix,
    )
    adapter = None
    executor = None
    try:
        if k > 1:
            adapter = ContextDFlashAdapter.from_export(
                drafter_export,
                embed=runner.embed,
                lm_head=runner.logits,
                max_len=runner.max_len,
                device=runner.device,
                dtype=runner.dtype,
                use_cuda_graph=use_cuda_graph and context_cuda_graph is not False,
                batched_graph_min_calls=context_graph_min_calls,
                fast_grid=(
                    dict(head_weight=runner.w.head, head_scale=runner.w.head_scale)
                    if drafter_fast_grid
                    else None
                ),
            )
        executor = FinalizedTreeExecutor(
            runner,
            adapter,
            k,
            stage,
            d1,
            d2,
            record_trace,
            temperature=temperature,
            record_sampling_check=record_sampling_check,
            stage2_gate=stage2_gate,
            top_p=top_p,
            proposal_top_p=proposal_top_p,
        )

        def generate(request):
            return executor.generate(
                request["prompt_ids"],
                request["max_new"],
                request.get("eos_token_ids", []),
                seed=request.get("seed"),
            )

        with torch.inference_mode():
            if not prewarm_tree_graphs:
                prewarm = runner.prewarm_graphs(0)
            elif k == 1:
                prewarm = runner.prewarm_graphs(LOOPSPEC_ROWS, LOOPSPEC_PATH)
            else:
                prewarm = runner.prewarm_graphs(initial_capacity)
        prewarm["scope"] = "once_per_rpc_before_measured_requests"
        drafter_prewarm = (
            adapter.prewarm_grid(k, *drafting_grid(runner.R, k))
            if context_graph_grid and adapter is not None
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
                if name != "capacity"
            }
            result["target_runtime_state"] = {
                "prefix_kv_bytes": runner.prefix_kv_bytes,
                "prefix_kv_storage": (
                    "compact" if runner.compact_prefix else "all-planes"
                ),
                "physical_capacity": runner.max_nodes,
                "path_width": runner.max_path,
                "graphs": len(runner._graphs),
            }
            result["drafter_runtime_stats"] = (
                {
                    name: value - draft_before.get(name, 0)
                    for name, value in adapter.stats.items()
                }
                if adapter is not None
                else {}
            )
            result["startup_target_graph_prewarm"] = prewarm
            result["startup_drafter_graph_prewarm"] = drafter_prewarm
            result["settings"] = dict(
                execution="native-vllm-dory-finalized-sampling"
                if temperature > 0
                else "native-vllm-dory-finalized-greedy",
                **(
                    {
                        "temperature": temperature,
                        "seed": request.get("seed"),
                        "stage2_gate": stage2_gate,
                    }
                    if temperature > 0
                    else {}
                ),
                **({"top_p": top_p} if top_p < 1 else {}),
                **(
                    {"proposal_top_p": proposal_top_p}
                    if proposal_top_p is not None and proposal_top_p != top_p
                    else {}
                ),
                model_family="dory",
                num_loops=loops,
                stage=stage,
                d1=d1,
                d2=d2 if stage != "stage1" else None,
                block_size=k,
                block_total=k,
                num_speculative_tokens=k - 1,
                control=(
                    "loopspec_d1_no_drafter"
                    if stage == "stage1"
                    else "loopspec_d1_d2_no_drafter"
                )
                if k == 1
                else None,
                initial_capacity=initial_capacity,
                physical_capacity=runner.max_nodes,
                node_budget=None,
                cut_threshold=None,
                kv_policy=policy,
                kv_window=window,
                kv_share_layers="full_attention" if window else None,
                kv_storage="compact" if runner.compact_prefix else "all-planes",
                readout="output-block-in-call",
                tree_attention=f"dory-{runner.attention}",
                compile_blocks=runner.compile_blocks,
                drafter_fast_grid=bool(adapter is not None and drafter_fast_grid),
                target_fa_version=getattr(runner, "fa_version", None),
                drafter_fa_version=getattr(adapter, "fa_version", None),
                drafter_input_format=input_format,
                cuda_graph=runner.use_cuda_graph,
                context_cuda_graph=bool(adapter and adapter.use_cuda_graph),
                context_graph_min_calls=(
                    context_graph_min_calls if adapter is not None else None
                ),
                context_graph_grid=drafter_prewarm.get("enabled", False),
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
