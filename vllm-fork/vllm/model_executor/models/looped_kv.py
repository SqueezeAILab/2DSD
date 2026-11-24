# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""KV-cache policies of looped models: native per-loop KV or last-loop sharing.

``loopedtf_kv_policy`` in the model config selects the policy:

* ``full`` (default): every loop keeps its own KV cache.
* ``shared_last``: in every loop except the last, a query at position t reads
  the last-loop KV of the same layer for positions j <= t - W and its own
  loop's KV for t - W < j <= t, where W = ``loopedtf_kv_window`` counts the
  query itself. The prompt is prefilled natively in one pass.

  On Ouro and Nanbeige (model-wise looping) every layer shares, except the last
  loop and the first layer of the first loop, which keep native KV. Checkpoints
  adapted for sharing carry the keys in their config.json.

  On Dory (block-wise looping) ``loopedtf_kv_share_layers = "full_attention"``
  limits sharing to full-attention layers; sliding-window layers keep per-loop
  KV, which their window already bounds, and the input block keeps its native
  KV. In autoregressive decoding the output block runs once per token, at loop
  R; the 2DSD runner, whose early readouts run the output block at several
  depths, applies the same rule to the output block's full-attention layers.
  Dory shares without adaptation, so the keys are passed as configuration
  overrides.

  Inkling2 (block-wise looping, ``model_type`` ``inkling2_loop``) keeps per-loop
  KV: ``shared_last`` is rejected for it. Its loop count is nested as
  ``text_config.loop_config.num_passes``, the fixed-loop case of
  ``Inkling2ModelConfig.num_loop_passes`` in vLLM PR #69028 (head b426b46e,
  vllm/models/inkling2/configs.py).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode

POLICIES = ("full", "shared_last")
SHARED_LAST_MODEL_TYPES = ("ouro", "nanbeige", "dory")
INKLING2_MODEL_TYPE = "inkling2_loop"


@dataclass(frozen=True)
class PolicyOptions:
    policy: str = "full"
    window: int | None = None
    # Shared layers of a model with mixed attention types (Dory): every recurrent
    # attention layer ("all") or the full-attention ones only.
    share_layers: str = "all"

    @classmethod
    def from_hf(cls, config):
        fields = cls.__dataclass_fields__
        unexpected = [
            name
            for name in vars(config)
            if name.startswith("loopedtf_kv_")
            and name.removeprefix("loopedtf_kv_") not in fields
        ]
        if unexpected:
            raise ValueError(f"Unsupported KV configuration fields: {unexpected}")
        return cls(
            **{
                name: getattr(config, f"loopedtf_kv_{name}", field.default)
                for name, field in fields.items()
            }
        )

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown loopedtf_kv_policy: {self.policy!r}")
        if self.share_layers not in ("all", "full_attention"):
            raise ValueError(
                f"Unknown loopedtf_kv_share_layers: {self.share_layers!r}"
            )
        if self.policy != "shared_last" and self.share_layers != "all":
            raise ValueError("loopedtf_kv_share_layers applies to shared_last only")
        if self.policy == "shared_last" and (
            type(self.window) is not int or self.window < 1
        ):
            raise ValueError(
                "shared_last needs loopedtf_kv_window >= 1 (the window includes "
                "the query itself)"
            )


def _field(config, name: str):
    """``name`` of a config given as an object or as a dict (None if absent)."""
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _inkling2_passes(config) -> int:
    text = _field(config, "text_config")
    passes = _field(_field(text, "loop_config"), "num_passes")
    if type(passes) is not int or passes < 1:
        raise ValueError(
            "An inkling2_loop config carries its loop count as an integer "
            f"text_config.loop_config.num_passes >= 1, not {passes!r}"
        )
    override = _field(text, "loop_num_passes")
    if override is not None and override != passes:
        raise ValueError(
            f"text_config.loop_num_passes = {override!r} differs from the "
            f"checkpoint's {passes} passes; 2DSD runs the checkpoint's loop count"
        )
    return passes


def loop_count(config) -> int:
    """Number of loops R of a looped model config.

    Ouro (``total_ut_steps``), Nanbeige (``num_loops``) and Dory
    (``n_recurrent_loops``) carry it at the top level. Inkling2 (``model_type``
    ``inkling2_loop``) nests it as ``text_config.loop_config.num_passes``; the
    config and its sub-configs may be objects or dicts.

    Raises:
        ValueError: If the config carries no loop count. A model of an unknown
            family is never treated as a dense (one-loop) model.
    """
    model_type = _field(config, "model_type")
    if model_type == INKLING2_MODEL_TYPE:
        return _inkling2_passes(config)
    loops = (
        _field(config, "total_ut_steps")
        or _field(config, "num_loops")
        or _field(config, "n_recurrent_loops")
    )
    if not loops:
        raise ValueError(
            f"No loop count in a config of model_type {model_type!r}: expected "
            "total_ut_steps (Ouro), num_loops (Nanbeige), n_recurrent_loops (Dory) "
            "or, for inkling2_loop, text_config.loop_config.num_passes"
        )
    return loops


def validate_config(config: VllmConfig) -> str:
    """Check that the engine configuration supports the model's KV policy."""
    hf = config.model_config.hf_config
    options = PolicyOptions.from_hf(hf)
    options.validate()
    if options.policy == "full":
        return options.policy
    if hf.model_type not in SHARED_LAST_MODEL_TYPES:
        raise ValueError("shared_last supports Ouro, Nanbeige and Dory models only")
    if options.share_layers != "all" and hf.model_type != "dory":
        raise ValueError("loopedtf_kv_share_layers applies to Dory only")
    if hf.model_type == "dory" and hf.recurrent_kv_cache_mode != "per_loop":
        raise ValueError("Dory shares last-loop KV over per-loop recurrent caches")
    compilation = config.compilation_config
    # Shared-last branches on per-step state only inside its registered attention
    # op, whose body runs on every call, so torch.compile is safe; prefill must
    # stay out of CUDA graphs (decode-only graphs).
    if not config.model_config.enforce_eager and (
        compilation.mode not in (CompilationMode.NONE, CompilationMode.VLLM_COMPILE)
        or compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        raise ValueError(
            "shared_last requires eager execution or FULL_DECODE_ONLY CUDA graphs "
            "(compilation mode NONE or VLLM_COMPILE)"
        )
    parallel = config.parallel_config
    if (
        any(
            getattr(parallel, name) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        )
        or parallel.enable_dbo
    ):
        raise ValueError("shared_last runs on one GPU without DBO")
    if config.use_v2_model_runner:
        raise ValueError("shared_last uses the V1 model runner")
    speculative = config.speculative_config
    if speculative is not None and 1 + speculative.num_speculative_tokens > options.window:
        # Every verified query must read its own loop's KV for the whole block;
        # a longer block would read last-loop KV this forward has not written.
        raise ValueError(
            "shared_last needs 1 + num_speculative_tokens <= loopedtf_kv_window "
            f"({1 + speculative.num_speculative_tokens} > {options.window})"
        )
    if (
        config.cache_config.enable_prefix_caching
        or config.scheduler_config.enable_chunked_prefill
    ):
        raise ValueError(
            "shared_last prefills the whole prompt natively: disable prefix "
            "caching and chunked prefill"
        )
    if (
        config.cache_config.cache_dtype not in ("auto", "bfloat16")
        or config.model_config.dtype != torch.bfloat16
    ):
        raise ValueError("shared_last requires BF16 activations and KV cache")
    return options.policy


def check_tree_window(config, block_size: int) -> None:
    """2DSD keeps up to R * k tokens in flight; all must fit in the window."""
    options = PolicyOptions.from_hf(config)
    options.validate()
    if options.policy != "shared_last":
        return
    model_type = getattr(config, "model_type", None)
    if model_type is not None and model_type not in SHARED_LAST_MODEL_TYPES:
        raise ValueError("shared_last supports Ouro, Nanbeige and Dory models only")
    loops = loop_count(config)
    if loops * block_size > options.window:
        raise ValueError(
            f"2DSD with shared_last needs R * k <= W ({loops} * {block_size} > "
            f"{options.window}); the in-flight span would leave the window"
        )


def requires_eager_prefill(computed, prompt_lengths, scheduled) -> bool:
    """Recognize real prompt work, including one-token and partial chunks."""
    return bool(np.any((scheduled > 0) & (computed < prompt_lengths)))
