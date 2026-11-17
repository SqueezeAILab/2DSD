"""configs/models.json: registry names -> Hugging Face repositories, family and loop count.

Every command that takes a model accepts a registry name, a local directory, or a Hugging Face repo
id (``org/name`` or ``org/name@revision``). The registry pins each base model to its Hugging Face
revision and lists the paper's shared-KV (W64) targets and drafters as plain repo ids.
Dory shares without adaptation: its W64 target is its pinned checkpoint, run with configuration
overrides (vllm.v1.spec_decode.twodsd.dory_kv_policy), so both variants resolve to that checkpoint.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_JSON = ROOT / "configs" / "models.json"
VARIANTS = ("original", "w64")
FAMILIES = ("ouro", "nanbeige", "dory")
# Families whose W64 target is the original checkpoint with configuration overrides (no KV-sharing adaptation).
TRAINING_FREE_SHARING = ("dory",)
HF_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
TOKENIZER_FILES = ["tokenizer*", "special_tokens_map.json", "chat_template*", "added_tokens.json",
                   "merges.txt", "vocab.json", "config.json", "generation_config.json"]


@lru_cache(maxsize=None)
def _registry(path=MODELS_JSON):
    return json.loads(Path(path).read_text())["models"]


def names():
    return tuple(_registry())


def entry(name):
    try:
        return _registry()[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; configs/models.json has {', '.join(names())}") from None


def _pin(value):
    """A registry slot (null, "org/name[@rev]" or {"repo_id", "revision"}) -> (repo_id, revision) or None."""
    if value is None:
        return None
    if isinstance(value, str):
        repo, _, revision = value.partition("@")
        return repo, revision or None
    return value["repo_id"], value.get("revision")


def pinned(name, variant="original"):
    """(repo_id, revision) of a registry target (revision None for a plain repo id), or None while it
    is unpublished."""
    if variant not in VARIANTS:
        raise ValueError(f"target variant must be one of {VARIANTS}")
    e = entry(name)
    if variant == "original" or e["family"] in TRAINING_FREE_SHARING:
        return e["repo_id"], e["revision"]
    return _pin(e.get("w64"))


def pinned_drafter(name, variant):
    return _pin(entry(name).get("drafters", {}).get(variant))


# Named KV policy of Dory's two targets, both of them its pinned checkpoint (vllm.v1.spec_decode.twodsd).
KV_POLICIES = {"w64": "w64", "original": "native"}


def serving(spec, target=None, *, revision=None, drafter=None, kv_policy=None, with_drafter=True):
    """Target, revision, drafter and KV policy that scripts/generate.py and scripts/benchmark.py run.

    A registry name runs its `target` (``"w64"`` by default, or ``"original"``): the target's
    repository and revision (pinned for a base model, else None), the registry's drafter of that
    target, and for Dory, whose targets are both its pinned checkpoint, the KV policy ``"w64"`` or
    ``"native"``. An explicit `revision`, `drafter` or `kv_policy` replaces the registry's value.
    Any other spec (a local directory or a Hugging Face id) is returned as given, and `target` must
    then be None. Without `with_drafter` (AR, or block size one) no drafter runs, and the drafter is
    None for every spec.
    Returns a dict with model, revision, drafter, kv_policy and family (None outside the registry).
    """
    spec = str(spec)
    if not with_drafter:
        drafter = None
    if spec not in _registry():
        if target is not None:
            raise ValueError(f"--target applies to registry names ({', '.join(names())}), not {spec!r}; "
                             "pass the checkpoint itself (with --revision, --drafter and --kv-policy) instead")
        return {"model": spec, "revision": revision, "drafter": drafter, "kv_policy": kv_policy, "family": None}
    target = target or "w64"
    pin = pinned(spec, target)
    if pin is None:
        raise ValueError(f"{spec} has no published {target} target in configs/models.json")
    family = entry(spec)["family"]
    if family in TRAINING_FREE_SHARING:
        kv_policy = kv_policy or KV_POLICIES[target]
    elif kv_policy is not None:
        raise ValueError(f"--kv-policy applies to Dory only, not {spec}")
    if drafter is None and with_drafter:
        pin_drafter = pinned_drafter(spec, target)
        drafter = "@".join(x for x in pin_drafter if x) if pin_drafter else None
    return {"model": pin[0], "revision": revision or pin[1], "drafter": drafter, "kv_policy": kv_policy,
            "family": family}


def revision(name):
    """Pinned revision of the original model, or None for a name outside the registry."""
    return entry(name)["revision"] if name in _registry() else None


def download(repo_id, revision=None, allow_patterns=None):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=allow_patterns))


def resolve(spec, *, variant="original", allow_patterns=None):
    """Local directory of a registry name (its pinned variant), a local path or an HF repo id."""
    spec = str(spec)
    if spec in _registry():
        pin = pinned(spec, variant)
        if pin is None:
            raise ValueError(f"{spec} has no published {variant} checkpoint in configs/models.json; "
                             "pass a local path instead")
        return download(*pin, allow_patterns=allow_patterns)
    path = Path(spec).expanduser()
    if path.is_dir():
        return path
    repo, _, rev = spec.partition("@")
    if HF_REPO.match(repo) and not path.exists():
        return download(repo, rev or None, allow_patterns)
    raise FileNotFoundError(f"{spec!r} is neither a registry name, a local directory nor a Hugging Face repo id")


def resolve_target(name, variant="original", path=None, allow_patterns=None):
    """Checkpoint of model `name`: an explicit path or repo id if given, else the registry's variant."""
    if path:
        return resolve(path, allow_patterns=allow_patterns)
    return resolve(name, variant=variant, allow_patterns=allow_patterns)


def label(spec, variant="original"):
    """Path-free identifier for metadata: the repo id of a registry name (repo@revision for a pinned
    base model), the repo id as given, or None for a local directory (the caller must then ask for
    an explicit name)."""
    spec = str(spec)
    if spec in _registry():
        pin = pinned(spec, variant)
        return None if pin is None else "@".join(x for x in pin if x)
    if Path(spec).expanduser().exists():
        return None
    return spec if HF_REPO.match(spec.partition("@")[0]) else None


def family(spec):
    """Model family of a registry name or of a checkpoint directory (config.json model_type)."""
    spec = str(spec)
    if spec in _registry():
        return entry(spec)["family"]
    if not (Path(spec) / "config.json").is_file():
        raise ValueError(f"{spec!r} is neither a registry name ({', '.join(names())}) nor a checkpoint directory")
    config = json.loads((Path(spec) / "config.json").read_text())
    if config.get("model_type") not in FAMILIES:
        raise ValueError(f"unsupported model_type {config.get('model_type')!r} in {spec}")
    return config["model_type"]
