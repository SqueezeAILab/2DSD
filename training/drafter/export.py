"""Export a trained drafter as a vLLM DFlash checkpoint (config.json, model.safetensors, mask_embedding.pt).

vLLM's DFlash proposer reads the target's residual stream after the unrolled layers listed in
dflash_config.target_layer_ids (0-based over loop * L + layer); the loop-R copies of the training
feature layers reproduce the d = R inputs, so stock DFlash serves 1D decoding. The released vLLM fork
reads loopedtf_drafter (feature_layers_per_loop, input_format) for 2D decoding. The LM head is exported
as an exact copy of the target's, so vLLM shares it; a W64 drafter therefore carries its W64 target's
fine-tuned head. Embeddings are omitted and shared. The learned mask ships as mask_embedding.pt.

mask_token_id is the id whose input embedding vLLM replaces with the mask vector. By default it is the
first unused embedding row past the tokenizer's largest token id, which no prompt or generation can
contain; only a target without spare rows falls back to a reserved added token that the chat template
and generations never emit.

Dory: target_layer_ids index its unrolled input + R x recurrent + output path (the physical taps 3,
19/35/51 and 67 become [3, 139, 155, 171, 187] of 192 at R = 4), its head's post-GEMM FP32 scale ships
as target_head_scale.pt (logit_scale, initializer_range; read by the fork's qwen3_dflash.py when vLLM's
DFlash proposer loads the export; the Dory 2DSD runner applies the target's own head scale), and mask_token_id is the reserved <SPECIAL_999> that also initialised the mask.

  python -m training.drafter.export --checkpoint RUN/drafter/best.pt --out EXPORT [--target-path DIR] \\
      [--target-name ORG/NAME]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..common import registry
from ..hf_models.dory.native import FEATURE_LAYERS as DORY_FEATURE_LAYERS
from ..hf_models.dory.native import LOOPS as DORY_LOOPS
from ..hf_models.dory.native import feature_layer_ids, unrolled_layers
from .blocks import INPUT_FORMAT


def _token_ids(tokenizer):
    """All ids of a tokenizer.json: its model vocabulary (dict or Unigram list) and added tokens."""
    vocab = tokenizer.get("model", {}).get("vocab") or {}
    ids = set(vocab.values()) if isinstance(vocab, dict) else set(range(len(vocab)))
    return ids | {t["id"] for t in tokenizer.get("added_tokens", [])}


def default_mask_token_id(tokenizer, config):
    """(mask token id, rule) for a target's tokenizer.json and config.json dicts."""
    spare = max(_token_ids(tokenizer)) + 1
    if spare < config["vocab_size"]:
        return spare, "first-unused-embedding-row"
    # No spare row: a reserved added token that the chat template and generations never emit.
    names = {t["id"]: t["content"] for t in tokenizer.get("added_tokens", [])}
    used = set()
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = config.get(key)
        used.update(value if isinstance(value, list) else [value])
    candidates = sorted(i for i in names if i not in used and not names[i].startswith(("<|im", "<think", "</think")))
    if not candidates:
        raise ValueError("no unused embedding row or reserved added token; pass --mask-token-id")
    return candidates[-1], "last-reserved-added-token"


def target_config(model_dir):
    """config.json of a target, with bos/eos ids from generation_config.json where config.json has none
    (Dory keeps its EOS ids [2, 11] only there)."""
    config = json.loads((model_dir / "config.json").read_text())
    generation = model_dir / "generation_config.json"
    if generation.exists():
        for key, value in json.loads(generation.read_text()).items():
            if key in ("bos_token_id", "eos_token_id") and config.get(key) is None:
                config[key] = value
    return config


def target_tensor(model_dir, name):
    """One tensor of a (possibly sharded) safetensors checkpoint."""
    index = model_dir / "model.safetensors.index.json"
    shard = json.loads(index.read_text())["weight_map"][name] if index.exists() else "model.safetensors"
    with safe_open(model_dir / shard, "pt") as handle:
        return handle.get_tensor(name)


def target_layout(target, ck):
    """(loops R, unrolled target layers, target_layer_ids) of the drafter's feature layers."""
    if target.get("model_type") == "dory":
        if tuple(ck["feat_layers"]) != DORY_FEATURE_LAYERS:
            raise ValueError(f"Dory drafters use the physical feature taps {DORY_FEATURE_LAYERS}")
        R = ck.get("target_loops", DORY_LOOPS)  # the checkpoint's own n_recurrent_loops is 2
        return R, unrolled_layers(target, R), feature_layer_ids(target, R, ck["feat_layers"])
    L, R = target["num_hidden_layers"], target.get("total_ut_steps") or target.get("num_loops") or 1
    return R, R * L, [(R - 1) * L + layer for layer in ck["feat_layers"]]


def target_source(ck):
    """(spec, variant) of the target a checkpoint was trained on; older checkpoints recorded a path."""
    if "target_variant" in ck:
        return ck.get("target_spec"), ck["target_variant"]
    return ck.get("target_checkpoint"), "w64" if ck.get("target_checkpoint") else "original"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt or last.pt of a training run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-path", help="local directory or HF repo id of the target "
                                              "(default: as recorded at training, else configs/models.json)")
    parser.add_argument("--target-name", help="target identifier written to config.json, e.g. its HF repo id "
                                              "(default: the registry or HF id; required for a local target)")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mask-token-id", type=int, help="override the dflash_config mask id")
    args = parser.parse_args(argv)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("input_format") != INPUT_FORMAT:
        parser.error(f"only {INPUT_FORMAT} drafters are supported, not {ck.get('input_format')!r}")
    spec, variant = target_source(ck)
    path = args.target_path or spec
    model_dir = registry.resolve_target(ck["target"], variant, path)
    if args.target_name:
        target_name = args.target_name
    else:  # never a local path: the registry's repo id or the HF repo id the drafter was trained against
        target_name = registry.label(path) if path else registry.label(ck["target"], variant)
    if target_name is None:
        parser.error("the target is a local directory; pass --target-name (e.g. its Hugging Face repo id)")
    target = target_config(model_dir)
    dory = target.get("model_type") == "dory"
    R, num_target_layers, target_layer_ids = target_layout(target, ck)
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    names = {t["id"]: t["content"] for t in tokenizer["added_tokens"]}
    if args.mask_token_id is not None:
        mask_token_id, rule = args.mask_token_id, "explicit"
    else:
        mask_token_id, rule = default_mask_token_id(tokenizer, target)
    if dory and ck.get("mask_token", mask_token_id) != mask_token_id:
        parser.error(f"the Dory drafter's mask was initialised from token {ck['mask_token']}, "
                     f"not the exported mask_token_id {mask_token_id}")
    cfg = ck["cfg"]
    rope_theta = cfg["rope_parameters"]["rope_theta"] if dory else target.get("rope_theta", 1e6)
    config = {
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3", "attention_bias": False, "attention_dropout": 0.0,
        "block_size": args.block_size, "dtype": "bfloat16", "hidden_act": "silu",
        "hidden_size": cfg["hidden_size"], "intermediate_size": cfg["intermediate_size"], "head_dim": cfg["head_dim"],
        "num_attention_heads": cfg["num_attention_heads"], "num_key_value_heads": cfg["num_key_value_heads"],
        "num_hidden_layers": cfg["num_hidden_layers"], "layer_types": ["full_attention"] * cfg["num_hidden_layers"],
        "max_position_embeddings": target.get("max_position_embeddings", 65536), "max_window_layers": cfg["num_hidden_layers"],
        "rms_norm_eps": cfg["rms_norm_eps"], "rope_theta": rope_theta, "rope_scaling": None,
        "sliding_window": None, "use_sliding_window": False, "tie_word_embeddings": False, "use_cache": True,
        "vocab_size": target["vocab_size"], "bos_token_id": target.get("bos_token_id"), "eos_token_id": target.get("eos_token_id"),
        "num_target_layers": num_target_layers,
        "dflash_config": {"mask_token_id": mask_token_id, "target_layer_ids": target_layer_ids},
        "loopedtf_drafter": {"target": ck["target"], "target_variant": variant, "target_checkpoint": target_name,
                             "step": ck["step"], "anchor_depth_weights": ck["d_weights"],
                             "feature_layers_per_loop": ck["feat_layers"], "target_loops": R,
                             "input_format": INPUT_FORMAT,
                             "mask_token_rule": rule},
    }
    if dory:
        config["loopedtf_drafter"]["target_head"] = {"kind": "dory-scaled", "scale_file": "target_head_scale.pt"}
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in ck["model"].items()
             if not k.startswith("rotary.") and k != "mask_emb"}
    state["lm_head.weight"] = target_tensor(model_dir, "lm_head.weight").to(torch.bfloat16).contiguous()
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(state, args.out / "model.safetensors")
    if dory:  # kept apart from the BF16 weights so that it is applied in FP32 after the GEMM
        torch.save({"logit_scale": target_tensor(model_dir, "lm_head.logit_scale"),
                    "initializer_range": target["initializer_range"]}, args.out / "target_head_scale.pt")
    if ck.get("learn_mask", True):
        torch.save({"mask_token_id": mask_token_id, "embedding": ck["model"]["mask_emb"].to(torch.bfloat16)},
                   args.out / "mask_embedding.pt")
    elif ck.get("mask_token", mask_token_id) != mask_token_id:
        raise ValueError("Frozen mask token differs from the exported mask_token_id")
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "mask_token_id": mask_token_id, "mask_token_rule": rule,
                      "mask_token": names.get(mask_token_id, "<unused embedding row>"),
                      "target_layer_ids": config["dflash_config"]["target_layer_ids"], "tensors": len(state)}))


if __name__ == "__main__":
    main()
