"""Last-loop KV sharing adaptation: full fine-tuning by forward KL from the frozen original model.

Each microbatch is one packed 32K row and config["effective_batch_rows"] rows (16) across all ranks
form one update. The student runs the two-pass K2 forward (training.sharekv.rule.k2_forward): pass 1
is its own native no-grad forward, of which only the detached last-loop KV is kept; pass 2 applies the
shared rule with window W and reads pass-1 last-loop KV outside the window, with native prompts.
The frozen original model, run natively, gives the temperature-1 full-vocabulary teacher; the loss
is the forward KL averaged over response tokens, without CE.

  torchrun --standalone --nproc-per-node 4 -m training.kd.train --config configs/kd/ouro-1.4b-w64.json \\
      --data DATA --out RUN [--resume RUN/last]

Checkpoints: RUN/checkpoint-<input tokens>/model (an HF export whose config.json enables the shared
rule in the vLLM fork) plus per-rank optimizer state; RUN/last points to the newest one.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from ..common import registry
from ..common.io import atomic_json, digest, file_sha
from ..hf_models import adapter_files, load_model, tokenizer_assets
from ..sharekv.policy import config_keys, kd_policy
from ..sharekv.rule import k2_forward, native_forward, pad_token
from .data import open_store, pack_tensors
from .evaluate import evaluate
from .loss import chunked_full_kd_sum
from .optim import ShardedAdamW, restore_rng, rng_state, seed_all, token_lr


def run_identity(config, train, dev, world):
    """What a resume must match: the recipe, the world size and the exact data and packing."""
    return {"schema_version": 2, "config": config, "world_size": world,
            "data": train.manifest, "train_packs_sha256": digest(train.packs),
            "dev_packs_sha256": digest(dev.packs), "model_policy": kd_policy(config["window"])}


def environment(model_dir):
    """Recorded in run.json for provenance only; not part of the resume identity."""
    family = registry.family(model_dir)
    return {"allocator": os.environ.get("PYTORCH_ALLOC_CONF", "default"),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
            "hf_adapter_sha256": {name: file_sha(path) for name, path in adapter_files(family).items()}}


def save_checkpoint(model, optimizer, out, state, identity, rank, world, model_dir):
    final = out / f"checkpoint-{state['input_tokens']:09d}"
    temporary = final.with_name(final.name + ".tmp")
    if rank == 0:
        if final.exists() or temporary.exists():
            raise ValueError(f"checkpoint destination already exists: {final}")
        temporary.mkdir(parents=True)
    if world > 1:
        dist.barrier()
    payload = {"training": state, "identity_sha256": digest(identity),
               "optimizer": optimizer.state_dict(), "rng": rng_state()}
    torch.save(payload, temporary / f"rank{rank}.pt")
    del payload
    if rank == 0:
        model.config.loopedtf_kd_policy = identity["model_policy"]
        # The vLLM fork applies the shared rule to a checkpoint whose config declares it.
        for key, value in config_keys(identity["config"]["window"]).items():
            setattr(model.config, key, value)
        model.save_pretrained(temporary / "model", safe_serialization=True, max_shard_size="5GB")
        # The complete tokenizer inventory of the original model, and its generation config.
        for source in [*tokenizer_assets(model_dir), Path(model_dir) / "generation_config.json"]:
            if source.exists():
                shutil.copy2(source, temporary / "model" / source.name)
        for name, source in adapter_files(registry.family(model_dir)).items():
            shutil.copy2(source, temporary / "model" / name)
        atomic_json(temporary / "training.json", {**state, "identity_sha256": digest(identity),
                                                 "resume_world_size": world})
    if world > 1:
        dist.barrier()
    if rank == 0:
        files = {str(p.relative_to(temporary)): file_sha(p) for p in temporary.rglob("*") if p.is_file()}
        atomic_json(temporary / "receipt.json", {"files": files, "identity_sha256": digest(identity)})
        temporary.replace(final)
        link = out / "last.tmp"
        link.symlink_to(final.name, target_is_directory=True)
        link.replace(out / "last")
        # All requested milestone model weights remain available. Keep full
        # optimizer state only at the latest verified recovery boundary.
        # A new complete checkpoint must exist before pruning the previous one.
        retained = set(state.get("milestone_checkpoints", []))
        for previous in out.glob("checkpoint-*"):
            if previous == final or not previous.is_dir() or previous.name.endswith(".tmp"):
                continue
            receipt_path = previous / "receipt.json"
            if not receipt_path.is_file():
                continue
            if previous.name not in retained:
                shutil.rmtree(previous)
                continue
            old_receipt = json.loads(receipt_path.read_text())
            for path in previous.glob("rank*.pt"):
                path.unlink()
            old_receipt["files"] = {n: s for n, s in old_receipt["files"].items() if not (n.startswith("rank") and n.endswith(".pt"))}
            old_receipt["resumable"] = False
            atomic_json(receipt_path, old_receipt)
    if world > 1:
        dist.barrier()
    return final


def check_config(config):
    if (config["student_passes"], config["prompt"], config["temperature"], config["objective"]) != (
            2, "native-one-pass", 1.0, "forward-kl-response-token-mean-no-ce"):
        raise ValueError("this trainer implements detached student K2, native prompts and temperature-1 forward KL")
    if config.get("teacher_mode", "online_full") != "online_full":
        raise ValueError("only the online full-vocabulary teacher is supported")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="a checkpoint directory of this run, e.g. RUN/last")
    parser.add_argument("--model-path", help="local directory or HF repo id of the original model "
                                             "(teacher and student initialisation; default: configs/models.json)")
    parser.add_argument("--max-updates", type=int, help="stop after this many updates (pilot runs)")
    parser.add_argument("--skip-evaluation", action="store_true", help="skip the dev diagnostics")
    parser.add_argument("--checkpoint-layers", type=int,
                        help="decoder layers per activation-checkpoint group (default: the config's)")
    parser.add_argument("--activation-offload", action="store_true",
                        help="keep saved activations in pinned CPU memory (lower GPU memory, slower)")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    check_config(config)
    checkpoint_layers = config["checkpoint_layers"] if args.checkpoint_layers is None else args.checkpoint_layers
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if config["effective_batch_rows"] % world:
        raise ValueError("effective batch rows must be divisible by world size")
    seed_all(config["seed"])
    # Rank 0 downloads a registry model once; the other ranks then find it in the cache.
    model_spec = args.model_path or config["model"]
    if rank == 0:
        model_dir = registry.resolve(model_spec)
    if world > 1:
        dist.barrier()
    if rank != 0:
        model_dir = registry.resolve(model_spec)
    if registry.family(model_dir) != registry.family(config["model"]):
        raise ValueError("--model-path is a different model family than the config's model")
    train = open_store(args.data, "train", config["model"], config["sequence_length"], config["seed"], model_dir)
    dev = open_store(args.data, "dev", config["model"], config["sequence_length"], config["seed"], model_dir)
    for store in (train, dev):
        store.verify()
        if (store.manifest["model_tokenizer"] != config["model"]
                or store.manifest["sequence_length"] != config["sequence_length"]
                or store.manifest["seed"] != config["seed"]):
            raise ValueError("dataset model, sequence length or seed differs from training")
    if config.get("data_manifest_sha256") and file_sha(args.data / "manifest.json") != config["data_manifest_sha256"]:
        raise ValueError("prepared dataset identity differs from the execution config")
    if sum(p["input_tokens"] for p in train.packs) > config["input_token_budget"]:
        raise ValueError("training data exceeds configured input budget")
    identity = run_identity(config, train, dev, world)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "run.json"
    if rank == 0:
        if manifest_path.exists():
            if not args.resume or json.loads(manifest_path.read_text()).get("identity") != identity:
                raise ValueError("existing run needs an explicit compatible --resume")
        else:
            atomic_json(manifest_path, {"identity": identity, "identity_sha256": digest(identity),
                                        "environment": environment(model_dir)})
    if world > 1:
        dist.barrier()
    if json.loads(manifest_path.read_text()).get("identity") != identity:
        raise ValueError("distributed ranks disagree on run identity")
    if args.resume:
        receipt = json.loads((args.resume / "receipt.json").read_text())
        if receipt.get("resumable") is False:
            raise ValueError("this archived milestone retains weights only; resume from last")
        if receipt["identity_sha256"] != digest(identity):
            raise ValueError("checkpoint belongs to a different run")
        for name, expected in receipt["files"].items():
            if (name.startswith("model/") and rank == 0) or name == f"rank{rank}.pt":
                if file_sha(args.resume / name) != expected:
                    raise ValueError(f"resume checkpoint file hash mismatch: {name}")
        if world > 1:
            dist.barrier()
    student = load_model(config["model"], device, args.resume / "model" if args.resume else model_dir)
    teacher = load_model(config["model"], device, model_dir, frozen=True)
    optimizer = ShardedAdamW(student, config["optimizer"], rank, world)
    state = {"updates": 0, "input_tokens": 0, "target_tokens": 0, "next_pack": 0, "milestones_saved": []}
    if args.resume:
        name = f"rank{rank}.pt"
        payload = torch.load(args.resume / name, map_location="cpu", weights_only=False)
        if payload["identity_sha256"] != digest(identity):
            raise ValueError("optimizer state belongs to a different run")
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        state = payload["training"]
        del payload
    evaluation_config = {**config["evaluation"], "train_window": config["window"]}

    def diagnostics(label, exact=False):
        if not args.skip_evaluation and rank == 0:
            result = evaluate(student, teacher, dev, evaluation_config, exact=exact)
            atomic_json(args.out / "evaluation" / f"{label}.json", result)
            print(json.dumps({"event": "evaluation", "label": label,
                              "teacher_to_k2": result["teacher_to_k2_full_kl"],
                              "seconds": result["seconds"]}), flush=True)
        if world > 1:
            dist.barrier()

    if not args.resume:
        diagnostics("baseline", exact=True)
    started, batch_rows = time.monotonic(), config["effective_batch_rows"]
    milestones = [int(config["input_token_budget"] * f) for f in (.25, .5, .75)]
    while state["next_pack"] < len(train.packs) and (args.max_updates is None or state["updates"] < args.max_updates):
        begin = state["next_pack"]
        end = min(begin + batch_rows, len(train.packs))
        global_targets = sum(p["targets"] for p in train.packs[begin:end])
        global_inputs = sum(p["input_tokens"] for p in train.packs[begin:end])
        local_sum = torch.zeros((), dtype=torch.float64, device=device)
        update_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for pack_index in range(begin + rank, end, world):
            documents = train.pack(pack_index)
            ids, layout, queries, _ = pack_tensors(documents, config["sequence_length"], pad_token(student), device)
            with torch.no_grad():
                target_h, _, _ = native_forward(teacher, ids, layout, capture="none")
                target_h = target_h[0, queries]
            with torch.autograd.graph.save_on_cpu(pin_memory=True) if args.activation_offload else nullcontext():
                hidden, _ = k2_forward(student, ids, layout, window=config["window"],
                                        checkpoint_layers=checkpoint_layers)
                loss_sum = chunked_full_kd_sum(student.lm_head, hidden[0, queries], teacher.lm_head,
                                               target_h, chunk_size=config["head_chunk_size"])
            if not torch.isfinite(loss_sum).item():
                raise FloatingPointError("nonfinite KD loss before backward")
            local_sum += loss_sum.detach().double()
            (loss_sum / global_targets).backward()
            optimizer.accumulate()
            # Do not retain the last microbatch graph into the next P1 pass.
            del hidden, loss_sum, ids, layout, queries, target_h
        if world > 1:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        state["input_tokens"] += global_inputs
        state["target_tokens"] += global_targets
        state["updates"] += 1
        state["next_pack"] = end
        lr = token_lr(state["input_tokens"], config["input_token_budget"], config["schedule"])
        grad_norm = optimizer.step(lr)
        torch.cuda.synchronize()
        if rank == 0:
            progress = {"event": "update", **state, "loss_mean": float(local_sum) / global_targets,
                        "lr": lr, "unclipped_grad_norm": grad_norm,
                        "update_seconds": time.monotonic() - update_start,
                        "training_seconds": time.monotonic() - started,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            with (args.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(progress) + "\n")
            atomic_json(args.out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        crossed = [m for m in milestones if state["input_tokens"] >= m and m not in state["milestones_saved"]]
        final = end == len(train.packs)
        pilot_end = args.max_updates is not None and state["updates"] >= args.max_updates
        recovery = state["updates"] % config.get("recovery_every_updates", 10) == 0
        if crossed or final or pilot_end or recovery:
            state["milestones_saved"] += crossed
            if crossed or final:
                state.setdefault("milestone_checkpoints", []).append(f"checkpoint-{state['input_tokens']:09d}")
            save_checkpoint(student, optimizer, args.out, state, identity, rank, world, model_dir)
            diagnostics(f"tokens-{state['input_tokens']}", exact=bool(crossed or final or pilot_end))
        elif state["updates"] % config["evaluation"]["every_updates"] == 0:
            diagnostics(f"update-{state['updates']}")
        if pilot_end:
            break
    if rank == 0:
        atomic_json(args.out / "status.json", {"status": "complete" if state["next_pack"] == len(train.packs) else "pilot-complete",
                                               **state, "identity_sha256": digest(identity)})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
