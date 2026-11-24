# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Python entry point for 2DSD generation (batch size one).

``TwoDSD`` loads a looped target (Ouro, Nanbeige or Dory) into a vLLM engine
that only holds the weights, then runs the 2DSD executor inside the worker
through the ``TwoDSDWorkerExtension`` collective RPC. The executor owns its own
prefix and tree KV storage and never uses vLLM's scheduler or paged KV cache.

Example::

    from vllm.v1.spec_decode.twodsd import TwoDSD

    engine = TwoDSD("path/to/Ouro-1.4B-Thinking-W64", drafter="path/to/drafter")
    [result] = engine.generate([prompt_ids], max_total_tokens=8192)
    print(result["output_ids"], result["decode_seconds"])

Method presets (paper Sec. 5.1; Nanbeige: App. E.1): ``2dsd`` is Stage 2b with (r1, r2) = (1, 2)
and block size 16 on R=4 targets (Ouro, Dory), and Stage 1 with r1 = 1 on R=2
targets (Nanbeige).

Dory1.0-4B shares without adaptation: its pinned checkpoint runs with R = 4
loops (an inference override of the checkpoint default of 2) and, by default,
last-loop KV sharing on the full-attention layers with W = 64
(``dory_kv_policy``). Dory runs with torch.compile (the engine's VLLM_COMPILE
and the 2DSD runner's compiled nGPT blocks); Ouro and
Nanbeige run CUDA graphs without torch.compile (the settings of the paper's
measurements).
``stage``, ``block_size``, ``d1`` and ``d2`` select the ablation variants
(Stage 1: early only; Stage 2a: plus deeper promotion; Stage 2b: plus deeper forking).
With ``block_size=1`` no drafter runs (depth-axis speculation only).

Inkling 2 (``model_type`` ``inkling2_loop``, R = 3 passes) runs without a vLLM
engine: the fork has no engine model class for it, so ``TwoDSD`` loads the
model from the checkpoint and runs the executor in this process on its own tree
runner (``inkling2_runtime``: on a GPU the static runner with CUDA graphs;
``runner="batched"`` and ``runner="reference"`` select the eager paths that it is
checked against). Its preset
is Stage 2b with (r1, r2) = (1, 2), the only Stage 2 schedule at R = 3, and block
size ``INKLING2_BLOCK_SIZE``; its EOS ids are fixed to
``INKLING2_EOS_TOKEN_IDS``. The block size and the EOS id are
``mtp_config.block_size`` and ``eos_token_id`` of the checkpoint's config.json
(thinkingmachines/Inkling-2-Mini at revision 1fb4134b). It keeps per-loop KV:
a ``kv_policy`` that asks for sharing is refused (``inkling2_kv_window``). Any
other ``model_type`` raises ``ValueError``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

WORKER_EXTENSION = "vllm.v1.spec_decode.twodsd.TwoDSDWorkerExtension"
ENGINE_MAX_LEN = 512  # The engine only holds weights; 2DSD owns its KV storage.
ENGINE_KV_BYTES = 1 << 30
GRAPH_ONLY = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
COMPILED = {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}  # VLLM_COMPILE
DORY_LOOPS = 4
DORY_WINDOW = 64
INKLING2 = "inkling2_loop"
MODEL_TYPES = ("ouro", "nanbeige", "dory", INKLING2)
# The block of TML's bundled drafter (mtp_config.block_size); 16, the block size
# of the other targets, is the other candidate.
INKLING2_BLOCK_SIZE = 8
INKLING2_EOS_TOKEN_IDS = (200006,)  # <|content_model_end_sampling|>


def dory_kv_policy(sharing: str = "w64", loops: int = DORY_LOOPS) -> dict[str, Any]:
    """Configuration overrides of a Dory checkpoint: ``"w64"`` shares last-loop
    KV on the full-attention layers with W = 64 (the paper's target), ``"native"``
    keeps per-loop KV. Both run ``loops`` recurrent loops over per-loop caches."""
    if sharing not in ("w64", "native"):
        raise ValueError(f"Dory KV sharing is 'w64' or 'native', not {sharing!r}")
    overrides = {
        "n_recurrent_loops": loops,
        "recurrent_kv_cache_mode": "per_loop",
        "loopedtf_kv_policy": "full",
    }
    if sharing == "w64":
        overrides.update(
            loopedtf_kv_policy="shared_last",
            loopedtf_kv_window=DORY_WINDOW,
            loopedtf_kv_share_layers="full_attention",
        )
    return overrides


def check_model_type(model_type: str | None) -> None:
    """Reject a ``model_type`` outside ``MODEL_TYPES``: no model runs as another."""
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type {model_type!r}: 2DSD supports "
            f"{', '.join(MODEL_TYPES)}"
        )


def model_type(model: str, revision: str | None = None) -> str | None:
    """``model_type`` of a local checkpoint or a Hugging Face repository (None if
    its config.json declares none)."""
    path = Path(model) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model, "config.json", revision=revision))
    return json.loads(path.read_text()).get("model_type")


def engine_options(model: str, revision: str | None, kv_policy) -> dict[str, Any]:
    """Model-specific vLLM settings shared by 2DSD and the native AR reference:
    the KV policy overrides and the paper's compilation setting."""
    family = model_type(model, revision)
    check_model_type(family)
    if family == "dory":
        if kv_policy is None or isinstance(kv_policy, str):
            overrides = dory_kv_policy(kv_policy or "w64")
        else:  # explicit keys override the named policy they extend (R = 4, per-loop caches)
            shared = kv_policy.get("loopedtf_kv_policy") == "shared_last"
            overrides = {**dory_kv_policy("w64" if shared else "native"), **kv_policy}
        options = {"hf_overrides": overrides, "compilation_config": dict(COMPILED)}
        if overrides.get("loopedtf_kv_policy") == "shared_last":
            options["disable_hybrid_kv_cache_manager"] = False
        return options
    if family == INKLING2:
        raise ValueError(
            "Inkling2 runs without a vLLM engine and keeps per-loop KV"
        )
    if isinstance(kv_policy, str):
        raise ValueError("Named KV policies apply to Dory; pass a dict of config keys")
    options = {"compilation_config": dict(GRAPH_ONLY)}
    if kv_policy:
        options["hf_overrides"] = dict(kv_policy)
    return options


def inkling2_kv_window(kv_policy: dict[str, Any] | str | None) -> None:
    """Check the KV policy of an Inkling2 target, which keeps per-loop KV.

    ``None``, ``"native"`` and a dict that names ``loopedtf_kv_policy`` ``"full"``
    are that policy; the result is the runner's sharing window, ``None``.

    Raises:
        ValueError: For any other policy: Inkling2 has no last-loop KV sharing.
    """
    if kv_policy in (None, "native", {}, {"loopedtf_kv_policy": "full"}):
        return None
    raise ValueError("Inkling2 runs with per-loop KV only: its KV policy is 'native'")


def check_block_size(drafter_dir, k: int) -> None:
    """Refuse a block that a drafter export cannot draft, before weights load.

    An export drafts blocks of up to ``block_size`` positions, the given token
    included (one more when a DSpark drafter's anchor slot predicts).

    Raises:
        ValueError: If ``k`` exceeds that.
    """
    config = json.loads((Path(drafter_dir) / "config.json").read_text())
    dspark = config.get("loopedtf_drafter", {}).get("dspark") or {}
    limit = config.get("block_size", 16) + bool(dspark.get("sample_from_anchor"))
    if k > limit:
        raise ValueError(
            f"block_size {k} exceeds the drafter's {limit} (the given token and "
            f"{limit - 1} drafted tokens)"
        )


def inkling2_runtime(engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Runner, arithmetic, expert storage and drafter graphs of an Inkling2 engine.

    On a GPU the defaults are the measured configuration: the static runner on
    ``FastInkling2`` (tensor cores, which run the served precision only), the
    drafter's CUDA graphs, and packed experts on Humming's expert GEMM when a
    recent Humming is installed (dense experts, 49 GiB of weights instead of
    18 GiB, otherwise and on the other runners). On a CPU: the batched runner on
    the reference model. Explicit ``runner``, ``tensor_cores``, ``experts`` and
    ``drafter_cuda_graph`` replace a default.
    """
    fast = str(engine_kwargs.get("device", "cuda")).startswith("cuda")
    runner = engine_kwargs.get("runner") or ("static" if fast else "batched")
    tensor_cores = engine_kwargs.get("tensor_cores")
    if tensor_cores is None:
        tensor_cores = fast and engine_kwargs.get("precision", "served") == "served"
    experts = engine_kwargs.get("experts")
    if experts is None:
        from vllm.model_executor.models.inkling2_humming import available

        packed = fast and tensor_cores and runner == "static" and available()
        experts = "packed" if packed else "dense"
    graphs = engine_kwargs.get("drafter_cuda_graph")
    return {
        "runner": runner,
        "tensor_cores": bool(tensor_cores),
        "experts": experts,
        "drafter_cuda_graph": fast if graphs is None else bool(graphs),
    }


def inkling2_expert_backend(runtime: dict[str, Any], device: str = "cuda") -> str:
    """How the routed experts of an ``inkling2_runtime`` run in the tree calls on
    ``device``, in the served precision.

    ``"humming"``: packed experts on Humming's expert GEMM, the measured path.
    ``"fused"``: dense experts in one grouped GEMM. ``"eager"``: PyTorch products
    expert by expert: a CPU, the batched runner, the reference runner without
    tensor cores, or packed experts without a recent Humming, which also leaves
    the static runner without its CUDA graphs.
    """
    runner = runtime["runner"]
    if not str(device).startswith("cuda") or runner == "batched":
        return "eager"
    if runner == "reference" and not runtime["tensor_cores"]:
        return "eager"
    if runtime["experts"] == "dense":
        return "fused"
    from vllm.model_executor.models.inkling2_humming import available

    return "humming" if runtime["tensor_cores"] and available() else "eager"


def inkling2_slow_path(engine_kwargs: dict[str, Any]) -> str | None:
    """Why an Inkling2 engine does not take the measured expert path, or None.

    The measured path is the static runner on tensor cores with packed experts
    on Humming's expert GEMM. Asking for dense experts, another runner or the
    reference arithmetic is a choice; a GPU run that falls off the path because
    no recent Humming can be imported is told so. On an H100 dense experts run
    at about half the speed and name other tokens at near-ties.
    """
    runtime = inkling2_runtime(engine_kwargs)
    device = str(engine_kwargs.get("device", "cuda"))
    wanted = (
        device.startswith("cuda")
        and runtime["runner"] == "static"
        and runtime["tensor_cores"]
        and engine_kwargs.get("experts") != "dense"
    )
    backend = inkling2_expert_backend(runtime, device)
    if not wanted or backend == "humming":
        return None
    return (
        "Inkling2 runs without Humming's packed expert GEMM (humming-kernels >= "
        "0.1.20 on the import path): "
        + (
            "dense experts take 49 GiB instead of 18 GiB and about twice the time."
            if backend == "fused"
            else "packed experts run one by one and the runner has no CUDA graphs."
        )
    )


def check_inkling2_drafter_target(drafter_dir, model_dir) -> None:
    """Refuse a drafter export that was trained against another checkpoint.

    Inkling 2 checkpoints of one architecture (Inkling-2-Mini and its variable-loop checkpoint) take each
    other's drafters without an error, at a lower acceptance. An export names its target: the hash of the
    checkpoint's ``config.json``, or in older exports the pinned revision in ``target_checkpoint``, which is
    compared with the name of a Hugging Face snapshot directory. An export or a directory that says neither
    is not checked.
    """
    import hashlib
    import re

    meta = json.loads((Path(drafter_dir) / "config.json").read_text()).get("loopedtf_drafter", {})
    recorded = meta.get("target_config_sha256")
    if recorded is not None:
        same = recorded == hashlib.sha256((Path(model_dir) / "config.json").read_bytes()).hexdigest()
    else:
        revision = str(meta.get("target_checkpoint") or "").partition("@")[2]
        snapshot = Path(model_dir).resolve().name
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", snapshot):
            return
        same = revision == snapshot
    if not same:
        raise ValueError(
            f"The drafter {drafter_dir} was trained against {meta.get('target_checkpoint') or meta.get('target')}, "
            f"not against the checkpoint {model_dir}. Pass drafter_of_other_target=True "
            "(scripts/benchmark.py --drafter-of-other-target) to run it anyway"
        )


class Inkling2Worker:
    """In-process stand-in of the vLLM worker for an Inkling2 target: it holds
    the reference model and answers ``collective_rpc`` in this process."""

    def __init__(self, native, loops=None):
        config = SimpleNamespace(
            model_type=INKLING2,
            text_config={"loop_config": {"num_passes": loops or native.config.num_passes}},
        )
        model = SimpleNamespace(config=config, native=native)
        self.model_runner = SimpleNamespace(get_model=lambda: model)

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if method != "run_2dsd":
            raise ValueError(f"Inkling2 worker has no method {method!r}")
        return [TwoDSDWorkerExtension.run_2dsd(self, *args, **(kwargs or {}))]


def inkling2_schedule(config, options: dict[str, Any]) -> dict[str, Any]:
    """Loop count and schedule of an Inkling2 request.

    R is ``text_config.loop_config.num_passes`` of the target config; ``options``
    override the preset's ``stage``, ``k``, ``d1`` and ``d2``.

    Raises:
        ValueError: If the config has no loop count or the schedule does not fit R.
    """
    from vllm.model_executor.models.looped_kv import loop_count

    loops = loop_count(config)
    settings = {**preset(INKLING2), **options}
    stage, d1, d2 = settings["stage"], settings["d1"], settings["d2"]
    if stage not in ("stage1", "stage2a", "stage2b"):
        raise ValueError("Unknown finalized stage")
    if not 1 <= d1 <= loops:
        raise ValueError(f"d1 lies in [1, R] (R = {loops})")
    if stage != "stage1" and not d1 < d2 < loops:
        raise ValueError(f"Stage 2 needs d1 < d2 < R (R = {loops})")
    return {"loops": loops, "stage": stage, "k": settings["k"], "d1": d1, "d2": d2}


class TwoDSDWorkerExtension:
    """Worker methods injected through ``LLM(worker_extension_cls=...)``."""

    def run_2dsd(
        self,
        requests: list[dict[str, Any]],
        options: dict[str, Any],
        warmup_requests: int = 2,
        warmup_max_new: int = 64,
        journal_path: str | None = None,
        keep_per_call: bool = False,
    ):
        """Generate ``requests`` in order after ``warmup_requests`` short warmups.

        Each request has ``prompt_ids``, ``max_new``, ``eos_token_ids``, an
        optional ``seed`` and an optional ``metadata`` dict copied into its
        result. Warmups repeat the first prompt with at most ``warmup_max_new``
        tokens and are not returned. Every result also records ``max_new`` and
        ``stopped_by_eos``. With ``journal_path``, each result is appended there
        as one JSON line as soon as it finishes and only the count is returned.
        """
        config = self.model_runner.get_model().config
        check_model_type(config.model_type)
        if config.model_type == "nanbeige":
            from vllm.v1.spec_decode.nanbeige_2d import run_nanbeige_finalized as run
            options = {k: v for k, v in options.items() if k not in ("d2", "stage2_gate")}
        elif config.model_type == "dory":
            from vllm.v1.spec_decode.dory_2d import run_dory_finalized as run

            options = {**options, "compile_blocks": True}
        elif config.model_type == INKLING2:
            from vllm.v1.spec_decode.inkling2_2d import run_inkling2_finalized as run

            options = {**options, **inkling2_schedule(config, options)}
        else:
            from vllm.v1.spec_decode.looped_2d_finalized import run_ouro_finalized as run
        if not requests:
            return []
        executed = [{k: v for k, v in r.items() if k != "metadata"} for r in requests]
        warm = {**executed[0], "max_new": min(warmup_max_new, executed[0]["max_new"])}
        schedule = [dict(warm) for _ in range(warmup_requests)] + executed
        results, emitted = [], 0
        journal = open(journal_path, "a") if journal_path else None  # noqa: SIM115

        def completed(result):
            nonlocal emitted
            emitted += 1
            if emitted <= warmup_requests:
                return
            request = requests[emitted - warmup_requests - 1]
            output = result["output_ids"]
            result["max_new"] = request["max_new"]
            result["stopped_by_eos"] = bool(output) and output[-1] in set(
                request.get("eos_token_ids", [])
            )
            result.update(request.get("metadata", {}))
            if not keep_per_call:
                result.get("stats", {}).pop("per_call", None)
            if journal is None:
                results.append(result)
            else:
                journal.write(json.dumps(result) + "\n")
                journal.flush()
                os.fsync(journal.fileno())

        try:
            run(self, schedule, **options, warmup=0, result_callback=completed)
        finally:
            if journal is not None:
                journal.close()
        if emitted != len(schedule):
            raise RuntimeError("2DSD did not complete every request")
        return results if journal is None else {"requests": len(requests)}


def preset(model_type: str, method: str = "2dsd") -> dict[str, Any]:
    """Paper settings for ``method`` on an Ouro or Dory (R=4) or Nanbeige (R=2) target.

    Inkling2 (R=3) takes the same Stage 2b schedule, (1, 2), with block size
    ``INKLING2_BLOCK_SIZE``.

    Raises:
        ValueError: If ``method`` or ``model_type`` is unknown.
    """
    if method != "2dsd":
        raise ValueError(f"Unknown method {method!r}; use stage/block_size/d1/d2")
    check_model_type(model_type)
    if model_type == "nanbeige":
        return {"stage": "stage1", "k": 16, "d1": 1}
    k = INKLING2_BLOCK_SIZE if model_type == INKLING2 else 16
    return {"stage": "stage2b", "k": k, "d1": 1, "d2": 2}


class TwoDSD:
    """Batch-one 2DSD generation on one GPU.

    Args:
        model: Hugging Face repository id or local path of the looped target.
            Shared-KV checkpoints carry ``loopedtf_kv_policy`` and
            ``loopedtf_kv_window`` in their config.json.
        drafter: Depth-conditioned DFlash drafter export: a directory, a
            Hugging Face repo id, or ``repo_id@revision``. Required unless
            ``block_size == 1``.
        max_model_len: Prompt plus output capacity of the 2DSD KV storage.
        stage, block_size, d1, d2: Override the ``2dsd`` preset (see module doc).
        initial_capacity: Initial candidate-node capacity (grows on demand).
        gpu_memory_utilization: vLLM memory budget of the weight-holding engine;
            2DSD allocates its own KV storage.
        stage2_gate: ``"argmax"`` (default) or a threshold in [0, 1).
        kv_policy: Dory: ``"w64"`` (default, the paper's target) or ``"native"``,
            or a dict of config keys applied over the matching named policy.
            Other models: a dict of config keys to force, e.g.
            ``{"loopedtf_kv_policy": ..., "loopedtf_kv_window": ...}`` for a
            checkpoint whose config lacks them (adapted checkpoints carry them).
            Inkling2: ``"native"`` only (per-loop KV); sharing is refused.
        revision: Hugging Face revision of ``model``.
        seed: Engine seed; per-request sampling seeds are passed to ``generate``.
        engine_kwargs: Extra ``vllm.LLM`` arguments. Inkling2 has no engine and
            takes ``device``, ``precision``, ``dtype``, ``plain``, ``loops``, the
            runtime of ``inkling2_runtime`` (``runner``: ``"static"``,
            ``"batched"`` or ``"reference"``; ``tensor_cores``; ``experts``;
            ``drafter_cuda_graph``), ``draft_confidence`` (a DSpark drafter's
            static confidence threshold; off by default), ``frozen_suffix``
            (false: greedy Stage 1 drops the suffix its review rejects),
            ``draft_vocab`` (JSON file with the token ``ids`` a DSpark drafter
            may draft) and ``drafter_of_other_target`` (true: run a drafter
            that was trained against another checkpoint).
    """

    def __init__(
        self,
        model: str,
        drafter: str | None = None,
        *,
        max_model_len: int = 8192,
        stage: str | None = None,
        block_size: int | None = None,
        d1: int | None = None,
        d2: int | None = None,
        initial_capacity: int = 256,
        stage2_gate: str | float = "argmax",
        gpu_memory_utilization: float = 0.5,
        kv_policy: dict[str, Any] | str | None = None,
        revision: str | None = None,
        seed: int = 0,
        **engine_kwargs,
    ) -> None:
        if model_type(model, revision) == INKLING2:
            self._init_inkling2(
                model,
                drafter,
                dict(
                    max_model_len=max_model_len,
                    initial_capacity=initial_capacity,
                    stage2_gate=stage2_gate,
                ),
                dict(stage=stage, k=block_size, d1=d1, d2=d2),
                kv_policy,
                revision,
                engine_kwargs,
            )
            return
        # The block must fit the drafter: checked here, before the engine loads the target.
        early_k = block_size if block_size is not None else preset(model_type(model, revision))["k"]
        if early_k > 1 and drafter is not None:
            if not Path(drafter).is_dir():
                from huggingface_hub import snapshot_download

                repo_id, _, drafter_revision = drafter.partition("@")
                drafter = snapshot_download(repo_id, revision=drafter_revision or None)
            check_block_size(drafter, early_k)
        # The executor runs on the V1 worker; keep vLLM's PyTorch sampler.
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        from vllm import LLM

        engine_len = min(max_model_len, ENGINE_MAX_LEN)
        options = dict(
            model=model,
            tokenizer=model,
            revision=revision,
            trust_remote_code=True,
            dtype="bfloat16",
            max_num_seqs=1,
            max_model_len=engine_len,
            kv_cache_memory_bytes=ENGINE_KV_BYTES,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_num_batched_tokens=engine_len,
            seed=seed,
            model_impl="vllm",
            generation_config="vllm",
            worker_extension_cls=WORKER_EXTENSION,
        )
        options.update(engine_options(model, revision, kv_policy))
        options.update(engine_kwargs)
        self.llm = LLM(**options)
        hf_config = self.llm.llm_engine.model_config.hf_config
        self.model_type = hf_config.model_type
        settings = preset(self.model_type)
        for name, value in (("stage", stage), ("k", block_size), ("d1", d1), ("d2", d2)):
            if value is not None:
                settings[name] = value
        if settings["k"] > 1 and drafter is None:
            raise ValueError("A drafter export is required for block_size > 1")
        if drafter is not None and not Path(drafter).is_dir():
            from huggingface_hub import snapshot_download

            repo_id, _, drafter_revision = drafter.partition("@")
            drafter = snapshot_download(repo_id, revision=drafter_revision or None)
        self.options = dict(
            settings,
            drafter_export=str(drafter) if drafter else None,
            max_model_len=max_model_len,
            initial_capacity=initial_capacity,
            stage2_gate=stage2_gate,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.model, self.revision = model, revision

    def _init_inkling2(
        self, model, drafter, options, schedule, kv_policy, revision, engine_kwargs
    ) -> None:
        """Engine-free construction: the model on ``device`` (default ``cuda``)
        with ``precision`` (``served``) and weight ``dtype`` (``bfloat16``), all
        taken from ``engine_kwargs``. The runner, the arithmetic
        (``tensor_cores``: ``FastInkling2``, the served arithmetic with matrix
        products on tensor cores, instead of the reference model), the storage
        of the routed experts (``dense``, or ``packed`` as the checkpoint
        stores them) and the drafter graphs are those of ``inkling2_runtime``:
        the measured configuration on a GPU.
        ``plain=True`` replaces the schedule by autoregressive decoding (Stage 1
        without a drafter, read out at the final loop only).
        ``loops`` runs a variable-loop checkpoint with that many passes instead
        of its own count (the model's output is then the readout of that pass);
        ``drafter_of_other_loops=True`` lets a drafter trained at another pass
        count of the same checkpoint draft for it."""
        import torch
        from transformers import AutoTokenizer

        from vllm.model_executor.models.inkling2_reference import NativeInkling2

        known = {"device", "precision", "dtype", "experts", "plain", "runner", "drafter_cuda_graph",
                 "draft_confidence", "tensor_cores", "frozen_suffix", "draft_vocab",
                 "drafter_of_other_target", "loops", "drafter_of_other_loops"}
        if set(engine_kwargs) - known:
            raise ValueError(
                f"Inkling2 takes the engine arguments {sorted(known)}, not "
                f"{sorted(set(engine_kwargs) - known)}"
            )
        window = inkling2_kv_window(kv_policy)
        settings = preset(INKLING2)
        settings.update({name: v for name, v in schedule.items() if v is not None})
        plain = bool(engine_kwargs.get("plain"))
        if plain:
            settings.update(stage="stage1", k=1)
            drafter = None
        if settings["k"] > 1 and drafter is None:
            raise ValueError("A drafter export is required for block_size > 1")
        # The executor's own checks, before a checkpoint of tens of GiB loads.
        if not engine_kwargs.get("frozen_suffix", True) and settings["stage"] != "stage1":
            raise ValueError("Only greedy Stage 1 can drop the suffix its review rejects")
        confidence = engine_kwargs.get("draft_confidence")
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError("draft_confidence is a probability")
        if not Path(model).is_dir():
            from huggingface_hub import snapshot_download

            model = snapshot_download(model, revision=revision)
        if drafter is not None and not Path(drafter).is_dir():
            from huggingface_hub import snapshot_download

            repo_id, _, drafter_revision = drafter.partition("@")
            drafter = snapshot_download(repo_id, revision=drafter_revision or None)
        if drafter is not None and not engine_kwargs.get("drafter_of_other_target"):
            check_inkling2_drafter_target(drafter, model)
        if drafter is not None:
            check_block_size(drafter, settings["k"])
        dtype = engine_kwargs.get("dtype", torch.bfloat16)
        runtime = inkling2_runtime(engine_kwargs)
        slow = inkling2_slow_path(engine_kwargs)
        if slow:
            import warnings

            warnings.warn(slow, stacklevel=2)
        if runtime["tensor_cores"]:
            from vllm.model_executor.models.inkling2_fast import FastInkling2 as NativeInkling2
        native = NativeInkling2.from_pretrained(
            model,
            device=engine_kwargs.get("device", "cuda"),
            precision=engine_kwargs.get("precision", "served"),
            dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
            experts=runtime["experts"],
        )
        loops = engine_kwargs.get("loops")
        if loops is not None and loops != native.config.num_passes:
            if type(loops) is not int or loops < 1:
                raise ValueError("loops is a positive number of passes")
            if not native.config.is_varloop:
                raise ValueError(
                    "Only a variable-loop Inkling2 checkpoint runs another number "
                    f"of passes than its own {native.config.num_passes}"
                )
        self.llm = Inkling2Worker(native, loops)
        self.model_type = INKLING2
        if plain:
            settings["d1"] = loops or native.config.num_passes
        inkling2_schedule(self.llm.model_runner.get_model().config, settings)
        self.options = dict(
            settings,
            drafter_export=str(drafter) if drafter else None,
            kv_window=window,
            runner=runtime["runner"],
            drafter_cuda_graph=runtime["drafter_cuda_graph"],
            draft_confidence=engine_kwargs.get("draft_confidence"),
            frozen_suffix=bool(engine_kwargs.get("frozen_suffix", True)),
            draft_vocab=engine_kwargs.get("draft_vocab"),
            drafter_of_other_loops=bool(engine_kwargs.get("drafter_of_other_loops", False)),
            **options,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model, self.revision = model, revision

    def eos_token_ids(self) -> list[int]:
        """Union of config.json, generation_config.json and tokenizer EOS ids.

        Inkling2 ends sampling with ``INKLING2_EOS_TOKEN_IDS`` only.
        """
        if self.model_type == INKLING2:
            return list(INKLING2_EOS_TOKEN_IDS)
        from transformers import GenerationConfig

        ids = set()
        # Dory's config.json has no eos_token_id; its generation_config.json does.
        values = [getattr(self.llm.llm_engine.model_config.hf_config, "eos_token_id", None),
                  self.tokenizer.eos_token_id]
        try:
            values.append(
                GenerationConfig.from_pretrained(self.model, revision=self.revision).eos_token_id
            )
        except OSError:
            pass  # No generation_config.json.
        for value in values:
            ids.update(value if isinstance(value, (list, tuple)) else [value])
        return sorted(v for v in ids if type(v) is int and v >= 0)

    def generate(
        self,
        prompts: list[list[int]] | list[str],
        *,
        max_total_tokens: int | None = None,
        max_new_tokens: int | list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        proposal_top_p: float | None = 1.0,
        seeds: list[int] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        eos_token_ids: list[int] | None = None,
        warmup_requests: int = 2,
        journal_path: str | None = None,
        keep_per_call: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate one sequence per prompt (token ids, or text tokenized as is).

        The output budget is ``max_new_tokens`` (one value or one per prompt), or
        ``max_total_tokens`` minus the prompt length (default: ``max_model_len``). Sampling uses
        ``temperature`` > 0 and ``top_p``; the early-loop proposals are not
        truncated (``proposal_top_p=1.0``), which keeps sampling lossless.
        Results hold ``output_ids``, ``prefill_seconds``, ``decode_seconds``,
        ``max_new``, ``stopped_by_eos``, executor statistics and the entries of
        the matching ``metadata`` dict (e.g. a question id).
        """
        eos = self.eos_token_ids() if eos_token_ids is None else list(eos_token_ids)
        total = max_total_tokens or self.options["max_model_len"]
        requests = []
        for index, prompt in enumerate(prompts):
            prompt_ids = (
                self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
            )
            if isinstance(max_new_tokens, (list, tuple)):
                budget = max_new_tokens[index]
            elif max_new_tokens is not None:
                budget = max_new_tokens
            else:
                budget = total - len(prompt_ids)
            request = {"prompt_ids": prompt_ids, "max_new": budget, "eos_token_ids": eos}
            if temperature > 0:
                request["seed"] = seeds[index] if seeds is not None else index
            if metadata is not None:
                request["metadata"] = dict(metadata[index])
            requests.append(request)
        options = dict(self.options)
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)
            if proposal_top_p is not None:
                options["proposal_top_p"] = proposal_top_p
        [reply] = self.llm.collective_rpc(
            "run_2dsd",
            timeout=None,
            kwargs=dict(
                requests=requests,
                options=options,
                warmup_requests=warmup_requests,
                journal_path=journal_path,
                keep_per_call=keep_per_call,
            ),
        )
        return reply
