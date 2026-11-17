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

Method presets (paper Sec. 5.1): ``2dsd`` is Stage 2b with (r1, r2) = (1, 2)
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
(Stage 1: early review only; Stage 2a: plus promotion; Stage 2b: plus forking).
With ``block_size=1`` no drafter runs (depth-axis speculation only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

WORKER_EXTENSION = "vllm.v1.spec_decode.twodsd.TwoDSDWorkerExtension"
ENGINE_MAX_LEN = 512  # The engine only holds weights; 2DSD owns its KV storage.
ENGINE_KV_BYTES = 1 << 30
GRAPH_ONLY = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
COMPILED = {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}  # VLLM_COMPILE
DORY_LOOPS = 4
DORY_WINDOW = 64


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


def model_type(model: str, revision: str | None = None) -> str:
    """``model_type`` of a local checkpoint or a Hugging Face repository."""
    path = Path(model) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model, "config.json", revision=revision))
    return json.loads(path.read_text())["model_type"]


def engine_options(model: str, revision: str | None, kv_policy) -> dict[str, Any]:
    """Model-specific vLLM settings shared by 2DSD and the native AR reference:
    the KV policy overrides and the paper's compilation setting."""
    family = model_type(model, revision)
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
    if isinstance(kv_policy, str):
        raise ValueError("Named KV policies apply to Dory; pass a dict of config keys")
    options = {"compilation_config": dict(GRAPH_ONLY)}
    if kv_policy:
        options["hf_overrides"] = dict(kv_policy)
    return options


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
        if config.model_type == "nanbeige":
            from vllm.v1.spec_decode.nanbeige_2d import run_nanbeige_finalized as run
            options = {k: v for k, v in options.items() if k not in ("d2", "stage2_gate")}
        elif config.model_type == "dory":
            from vllm.v1.spec_decode.dory_2d import run_dory_finalized as run

            options = {**options, "compile_blocks": True}
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
    """Paper settings for ``method`` on an Ouro or Dory (R=4) or Nanbeige (R=2) target."""
    if method != "2dsd":
        raise ValueError(f"Unknown method {method!r}; use stage/block_size/d1/d2")
    if model_type == "nanbeige":
        return {"stage": "stage1", "k": 16, "d1": 1}
    return {"stage": "stage2b", "k": 16, "d1": 1, "d2": 2}


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
        revision: Hugging Face revision of ``model``.
        seed: Engine seed; per-request sampling seeds are passed to ``generate``.
        engine_kwargs: Extra ``vllm.LLM`` arguments.
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

    def eos_token_ids(self) -> list[int]:
        """Union of config.json, generation_config.json and tokenizer EOS ids."""
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
