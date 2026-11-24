# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Worker-side 2DSD generation for Inkling 2 targets (batch size one).

The fork has no engine model class for Inkling 2, so the worker holds the
model (``NativeInkling2``, or ``FastInkling2`` on tensor cores) loaded straight
from the checkpoint. The tree runner is the static one (``runner="static"``:
every row of a call at once in padded shapes, replayed from CUDA graphs when
the routed experts run as one GEMM on a GPU), the batched one (``"batched"``:
the same computation in eager PyTorch) or the reference path (``"reference"``:
one token at a time through the model's layer function). ``k = 1`` runs without a
drafter: ``stage1`` with ``d1 = R`` is plain autoregressive decoding and the
other schedules speculate on depth only.
"""

import json
from pathlib import Path

import torch

from vllm.model_executor.models.inkling2_2d import Inkling2TreeRunner
from vllm.model_executor.models.inkling2_batch import Inkling2BatchRunner
from vllm.model_executor.models.inkling2_graph import Inkling2GraphRunner
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor
from vllm.v1.spec_decode.looped_dflash_context import ContextDFlashAdapter, drafting_grid


# Sampling with top-p: the cut of a target row is read from this many of its most probable tokens. A sort
# of the 200,058-token vocabulary for the rows of a call cost more than the rest of the executor's work
# on the call; a row that spreads top_p over more tokens than this still takes the sort.
NUCLEUS_WIDTH = 256


def validate_inkling2_schedule(k, stage, d1, d2, loops):
    if stage not in ("stage1", "stage2a", "stage2b"):
        raise ValueError("Unknown finalized stage")
    if type(k) is not int or not 1 <= k <= 16:
        raise ValueError("Inkling2 schedules use k in [1, 16]")
    if not 1 <= d1 <= loops:
        raise ValueError("d1 lies in [1, R]")
    if stage != "stage1" and not d1 < d2 < loops:
        raise ValueError("Stage 2 needs d1 < d2 < R")


def _expert_backend(path, model, runner):
    """How the routed experts of the tree calls ran: ``"humming"`` (packed, on
    Humming's expert GEMM), ``"fused"`` (dense, as one grouped GEMM) or
    ``"eager"`` (PyTorch products: the batched runner's own, the reference
    model's, or a static runner whose model offers neither GEMM)."""
    if path == "batched" or (path == "static" and not runner.fused_experts):
        return "eager"
    if getattr(model, "humming", False):
        return "humming"
    if path == "static" or getattr(model, "fused_experts", False):
        return "fused"
    return "eager"


def read_draft_vocabulary(path, vocab_size):
    """The sorted token ids of a drafting vocabulary file ({"ids": [...]}), checked on the host: integers
    of the served vocabulary, before any of them indexes a tensor on the device."""
    ids = json.loads(Path(path).read_text()).get("ids")
    if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError(f"{path}: a drafting vocabulary lists token ids, integers in [0, {vocab_size})")
    return sorted(set(ids))


def run_inkling2_finalized(
    worker,
    requests,
    drafter_export=None,
    k=1,
    stage="stage2b",
    d1=1,
    d2=2,
    max_model_len=8192,
    initial_capacity=64,
    result_callback=None,
    warmup=0,
    record_trace=False,
    temperature=0.0,
    record_sampling_check=False,
    stage2_gate="argmax",
    top_p=1.0,
    proposal_top_p=None,
    kv_window=None,
    runner="batched",
    drafter_cuda_graph=False,
    draft_confidence=None,
    frozen_suffix=True,
    draft_vocab=None,
    loops=None,
    drafter_of_other_loops=False,
):
    """Generate ``requests`` in order on the worker's Inkling 2 model.

    ``kv_window`` enables last-loop KV sharing with that window on the
    full-attention layers of the core and the coda; ``None`` keeps per-loop KV.
    ``drafter_cuda_graph`` lets the drafter adapter capture its drafting graphs.
    ``draft_confidence`` ends a drafted block before its first slot whose
    confidence (a DSpark drafter's confidence head) is below that value.
    ``frozen_suffix`` false drops the suffix that a greedy Stage 1 review
    rejects instead of keeping it in flight for the last loop.
    ``draft_vocab`` (a JSON file with ``ids``) restricts a DSpark drafter's
    drafts to those tokens; it changes drafts, never outputs.
    ``loops`` is the number of passes R the target runs (default: the
    checkpoint's): the verified prefix carries the features of pass R and the
    output is the readout of pass R. ``drafter_of_other_loops`` runs a drafter
    that was trained at another R; it reads the same feature layers per pass.
    """
    model = worker.model_runner.get_model().native
    loops = model.config.num_passes if loops is None else int(loops)
    validate_inkling2_schedule(k, stage, d1, d2, loops)
    if k > 1 and not drafter_export:
        raise ValueError("Inkling2 blocks larger than one require a drafter export")
    input_format = feature_layers = None
    final_feature = False
    if k > 1:
        metadata = json.loads((Path(drafter_export) / "config.json").read_text())[
            "loopedtf_drafter"
        ]
        input_format = metadata.get("input_format")
        if input_format != "loop-context":
            raise ValueError("Inkling2 drafts with its loop-context drafter")
        if metadata.get("target_loops") != loops and not drafter_of_other_loops:
            raise ValueError(
                f"The drafter was trained at R = {metadata.get('target_loops')} and "
                f"the target runs R = {loops}: Inkling2 drafts with its loop-context "
                "drafter at the same R (drafter_of_other_loops runs it anyway)"
            )
        # The runner hands the drafter the features it was trained on: the inputs
        # of these layers; num_hidden_layers at the end stands for the depth's
        # final-norm state.
        feature_layers = list(metadata["feature_layers_per_loop"])
        final_feature = feature_layers[-1] == model.config.num_hidden_layers
        if final_feature:
            feature_layers.pop()
    depths = {d1, loops} if stage == "stage1" else {d1, d2, loops}
    runners = {"batched": Inkling2BatchRunner, "static": Inkling2GraphRunner,
               "reference": Inkling2TreeRunner}
    if runner not in runners:
        raise ValueError("Inkling2 runners are 'batched', 'static' and 'reference'")
    target_path = runner
    runner = runners[runner](
        model,
        depths,
        loops=loops,
        window=kv_window,
        max_len=max_model_len,
        max_nodes=initial_capacity,
        feature_layers=feature_layers,
        final_feature=final_feature,
    )
    adapter = None
    executor = None
    try:
        if k > 1:
            # DSpark's Markov head adds its bias before the target's softcap.
            adapter = ContextDFlashAdapter.from_export(
                drafter_export,
                embed=runner.embed,
                lm_head=runner.raw_logits if metadata.get("dspark") else runner.logits,
                max_len=runner.max_len,
                device=runner.device,
                dtype=torch.bfloat16 if runner.device.type == "cuda" else torch.float32,
                use_cuda_graph=bool(drafter_cuda_graph),
                fast_grid=bool(drafter_cuda_graph) and runner.device.type == "cuda",
            )
            if drafter_cuda_graph and runner.device.type == "cuda" and adapter.first_slot == 1:
                # The padded drafting grid: every draft of the schedule replays a captured graph.
                adapter.prewarm_grid(k, *drafting_grid(runner.R, k))
            if draft_vocab is not None:
                if not metadata.get("dspark") or not hasattr(runner, "vocabulary_logits"):
                    raise ValueError("A drafting vocabulary needs a DSpark drafter and the static runner")
                ids = torch.tensor(read_draft_vocabulary(draft_vocab, model.config.served_vocab_size),
                                   dtype=torch.long, device=runner.device)
                adapter.configure_draft_vocabulary(ids, runner.vocabulary_logits(ids))
            if adapter.fc.shape[1] != runner.nf * runner.H:
                raise ValueError("The drafter reads features of another width than the runner hands out")
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
            draft_confidence=draft_confidence,
            frozen_suffix=frozen_suffix,
            nucleus_width=NUCLEUS_WIDTH,
            prefetch=True,
        )

        def generate(request):
            with torch.inference_mode():
                return executor.generate(
                    request["prompt_ids"],
                    request["max_new"],
                    request.get("eos_token_ids", []),
                    seed=request.get("seed"),
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
                if name != "peak_live_nodes"
            }
            result["drafter_runtime_stats"] = (
                {
                    name: value - draft_before.get(name, 0)
                    for name, value in adapter.stats.items()
                }
                if adapter is not None
                else {}
            )
            if k > 1:
                control = None
            elif stage == "stage1":
                control = "autoregressive" if d1 == loops else "loopspec_d1_no_drafter"
            else:
                control = "loopspec_d1_d2_no_drafter"
            sampled = temperature > 0
            result["settings"] = dict(
                execution=f"inkling2-{target_path}-finalized-"
                + ("sampling" if sampled else "greedy"),
                **(
                    {
                        "temperature": temperature,
                        "seed": request.get("seed"),
                        "stage2_gate": stage2_gate,
                    }
                    if sampled
                    else {}
                ),
                **({"top_p": top_p} if top_p < 1 else {}),
                **(
                    {"proposal_top_p": proposal_top_p}
                    if proposal_top_p is not None and proposal_top_p != top_p
                    else {}
                ),
                model_family="inkling2",
                num_loops=loops,
                stage=stage,
                d1=d1,
                d2=d2 if stage != "stage1" else None,
                block_size=k,
                num_speculative_tokens=k - 1,
                control=control,
                kv_policy="shared_last" if kv_window else "full",
                kv_window=kv_window,
                kv_share_layers=sorted(runner.share_layers) if kv_window else None,
                feature_layers=list(runner.feature_layers),
                readout="coda-in-call",
                target_path=target_path,
                precision=model.precision.name,
                drafter_input_format=input_format,
                drafter_heads=metadata.get("dspark") if k > 1 else None,
                **(
                    {"drafter_target_loops": metadata.get("target_loops")}
                    if k > 1 and metadata.get("target_loops") != loops
                    else {}
                ),
                draft_confidence=draft_confidence,
                frozen_suffix=bool(frozen_suffix),
                draft_vocab=None if draft_vocab is None else str(draft_vocab),
                cuda_graph=bool(getattr(runner, "use_cuda_graph", False)),
                # What ran, as the model and the runner report it: the measured path is
                # Humming's expert GEMM, the row kernels and the target's CUDA graphs.
                expert_backend=_expert_backend(target_path, model, runner),
                expert_storage=model.checkpoint.expert_mode,
                row_kernels=bool(getattr(runner, "row_kernels", False)),
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
