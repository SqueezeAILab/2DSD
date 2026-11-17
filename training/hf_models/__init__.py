"""Transformers 5 ports of the Ouro and Nanbeige remote code, and model loading for training.

The KD trainer and the drafter targets call the decoder layers directly (training.sharekv); the
top-level causal-LM forward of these ports always computes native per-loop KV. Dory has no
Transformers model and needs no KD; training.hf_models.dory is a plain PyTorch implementation of the
original Dory for the drafter's teacher-forced features.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..common import registry

HERE = Path(__file__).resolve().parent
PORTS = ("ouro", "nanbeige")  # families with a Transformers port (and KD)
TOKENIZER_PATTERNS = ("tokenizer*", "special_tokens_map.json", "chat_template*", "added_tokens.json",
                      "merges.txt", "vocab.json")


def model_classes(family):
    if family == "ouro":
        from .ouro.configuration_ouro import OuroConfig
        from .ouro.modeling_ouro import OuroForCausalLM
        return OuroConfig, OuroForCausalLM
    if family == "nanbeige":
        from .nanbeige.configuration_nanbeige import NanbeigeConfig
        from .nanbeige.modeling_nanbeige import NanbeigeForCausalLM
        return NanbeigeConfig, NanbeigeForCausalLM
    raise ValueError(f"unsupported model family: {family}")


def adapter_files(family):
    """The remote-code files an exported checkpoint ships next to its weights (auto_map)."""
    if family not in PORTS:
        raise ValueError(f"unsupported model family: {family}")
    return {f"{kind}_{family}.py": HERE / family / f"{kind}_{family}.py" for kind in ("modeling", "configuration")}


def tokenizer_assets(model_dir):
    """Every tokenizer file of a checkpoint directory."""
    return sorted({p for pattern in TOKENIZER_PATTERNS for p in Path(model_dir).glob(pattern) if p.is_file()})


def validate_config(config, family):
    if family == "ouro":
        if config.total_ut_steps != 4 or any(t != "full_attention" for t in config.layer_types):
            raise ValueError("Ouro KD requires four fixed loops and full attention")
    else:
        unsupported = ("enable_double_loop_split", "loop_share_kv", "enable_hyper_connection",
                       "enable_depth_attention", "skip_loop_final_norm")
        if (config.num_loops != 2 or config.loop_loss_weights
                or any(getattr(config, flag, False) for flag in unsupported)
                or config.emb_neighbor_num is not None):
            raise ValueError("Nanbeige KD requires two complete loops, ordinary residuals and per-loop final norm")


def model_config(path, family):
    config_cls, _ = model_classes(family)
    config = config_cls.from_pretrained(path, local_files_only=True)
    config._attn_implementation = "sdpa"
    return config


def load_model(name, device, checkpoint_path=None, frozen=False, *, dtype=torch.bfloat16):
    """Model `name` (registry name or checkpoint directory; selects the family) from checkpoint_path,
    by default the registry's pinned original. The dynamic-exit gate is always frozen."""
    family = registry.family(name)
    config_cls, model_cls = model_classes(family)
    path = checkpoint_path or registry.resolve(name)
    config = config_cls.from_pretrained(path, local_files_only=True)
    config._attn_implementation = "sdpa"
    validate_config(config, family)
    model = model_cls.from_pretrained(path, config=config, dtype=dtype,
                                      local_files_only=True).to(device).eval()
    for parameter_name, parameter in model.named_parameters():
        # Fixed four-loop output never uses the dynamic-exit gate.
        parameter.requires_grad_(not frozen and "early_exit_gate" not in parameter_name)
    return model
