"""Add the vLLM shared-KV keys to an already exported KD checkpoint (CPU, edits config.json only).

Exports written before training.kd.train recorded ``loopedtf_kv_policy`` / ``loopedtf_kv_window``
carry only ``loopedtf_kd_policy``; without the two keys the vLLM fork (and every other runtime)
serves them with native per-loop KV. The window defaults to the training window
(loopedtf_kd_policy.train_window).

  python -m training.kd.fix_config CHECKPOINT_DIR [--window 64] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..sharekv.policy import POLICY_KEY, WINDOW_KEY, config_keys


def fixed_config(config, window=None, force=False):
    """config with the shared-KV keys; raises on a conflict unless force."""
    trained = config.get("loopedtf_kd_policy", {}).get("train_window")
    if trained is None and not force:
        raise ValueError("config has no loopedtf_kd_policy: not a KD export (use --force to tag it anyway)")
    window = trained if window is None else window
    if window is None:
        raise ValueError("no training window recorded; pass --window")
    if trained is not None and window != trained and not force:
        raise ValueError(f"--window {window} differs from the training window {trained} (use --force)")
    keys = config_keys(window)
    for key, value in keys.items():
        if key in config and config[key] != value and not force:
            raise ValueError(f"config already sets {key}={config[key]!r} (use --force to replace it)")
    return {**config, **keys}


def fix(directory, window=None, force=False, dry_run=False):
    path = Path(directory) / "config.json"
    config = json.loads(path.read_text())
    updated = fixed_config(config, window, force)
    changed = updated != config
    if changed and not dry_run:
        tmp = path.with_name(path.name + ".tmp")
        # Transformers' own config.json format.
        tmp.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
    return {"config": str(path), "changed": changed, "dry_run": dry_run,
            POLICY_KEY: updated[POLICY_KEY], WINDOW_KEY: updated[WINDOW_KEY]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, nargs="+", help="exported checkpoint directories")
    parser.add_argument("--window", type=int, help="shared window W (default: the training window)")
    parser.add_argument("--force", action="store_true", help="replace conflicting values")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for directory in args.checkpoint:
        result = fix(directory, args.window, args.force, args.dry_run)
        if result["changed"] and (Path(directory) / "INFERENCE_MANIFEST.json").exists():
            result["note"] = "INFERENCE_MANIFEST.json still lists the previous config.json hash"
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
