"""configs/models.json: registry names -> Hugging Face repositories, family and loop count.

Every command that takes a model accepts a registry name, a local directory, or a Hugging Face repo
id (``org/name`` or ``org/name@revision``). The registry pins each base model to its Hugging Face
revision and lists the paper's shared-KV (W64) targets and drafters as plain repo ids.
Dory shares without adaptation: its W64 target is its pinned checkpoint, run with configuration
overrides (vllm.v1.spec_decode.twodsd.dory_kv_policy), so both variants resolve to that checkpoint.
Inkling2 runs per-loop KV only: its one target is the original checkpoint. The variable-loop checkpoint
has a DSpark drafter (`dspark_drafters`); Inkling-2-Mini has none yet, so a run names a drafter itself or
uses block size one.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_JSON = ROOT / "configs" / "models.json"
VARIANTS = ("original", "w64")
FAMILIES = ("ouro", "nanbeige", "dory", "inkling2")
# config.json model_type of each family. Inkling2 is the looped model only: the non-looped Inkling 2
# declares model_type "inkling2", which is not a family.
FAMILY_MODEL_TYPES = {"ouro": "ouro", "nanbeige": "nanbeige", "dory": "dory", "inkling2": "inkling2_loop"}
# Families with one checkpoint per registry name: no KV-sharing adaptation exists for them.
UNADAPTED = ("dory", "inkling2")
# Of those, the family whose W64 target is that checkpoint run with configuration overrides.
TRAINING_FREE_SHARING = ("dory",)
# Default target of a registry name: W64, except for Inkling2, which runs per-loop KV only.
DEFAULT_TARGET = {"inkling2": "original"}
# Families whose EOS ids are fixed here; the others read config.json, generation_config.json and the
# tokenizer. Inkling2 ends sampling with <|content_model_end_sampling|> only.
EOS_TOKEN_IDS = {"inkling2": (200006,)}
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


DRAFTER_KINDS = ("dflash", "dspark")


def pinned_drafter(name, variant, kind="dflash"):
    """(repo_id, revision) of the drafter of a registry target, or None for a model without one.

    `kind` "dflash" names the entry's `drafters` (the paper's), "dspark" its `dspark_drafters`. A run that
    names no kind takes the first of the two that the model has (`serving`)."""
    if kind not in DRAFTER_KINDS:
        raise ValueError(f"unknown drafter kind {kind!r}; choose from {', '.join(DRAFTER_KINDS)}")
    key = "drafters" if kind == "dflash" else f"{kind}_drafters"
    return _pin((entry(name).get(key) or {}).get(variant))


# Named KV policy of Dory's two targets, both of them its pinned checkpoint (vllm.v1.spec_decode.twodsd).
KV_POLICIES = {"w64": "w64", "original": "native"}


def serving(spec, target=None, *, revision=None, drafter=None, kv_policy=None, with_drafter=True,
            drafter_kind=None):
    """Target, revision, drafter and KV policy that scripts/generate.py and scripts/benchmark.py run.

    A registry name runs its `target` (``"w64"`` by default, or ``"original"``): the target's
    repository and revision (pinned for a base model, else None), the registry's drafter of that
    target, and for Dory, whose targets are both its pinned checkpoint, the KV policy ``"w64"`` or
    ``"native"``. An explicit `revision`, `drafter` or `kv_policy` replaces the registry's value.
    Inkling2 has the ``"original"`` target only (per-loop KV, policy ``"native"``). A name without a
    registry drafter needs an explicit `drafter` whenever a drafter runs.
    Any other spec (a local directory or a Hugging Face id) is returned as given, and `target` must
    then be None. Without `with_drafter` (AR, or block size one) no drafter runs, and the drafter is
    None for every spec.
    Returns a dict with model, revision, drafter, kv_policy and family (None outside the registry).
    """
    spec = str(spec)
    if not with_drafter:
        drafter = None
    if drafter_kind is not None and (drafter is not None or spec not in _registry()):
        # An export's own config.json says what it is; the kind only chooses among a registry name's drafters.
        raise ValueError("--drafter-kind chooses a registry name's drafter; with --drafter or a model outside "
                         "the registry the export itself tells its kind")
    if spec not in _registry():
        if target is not None:
            raise ValueError(f"--target applies to registry names ({', '.join(names())}), not {spec!r}; "
                             "pass the checkpoint itself (with --revision, --drafter and --kv-policy) instead")
        return {"model": spec, "revision": revision, "drafter": drafter, "kv_policy": kv_policy, "family": None}
    family = entry(spec)["family"]
    target = target or DEFAULT_TARGET.get(family, "w64")
    if family == "inkling2" and (target == "w64" or kv_policy not in (None, "native")):
        raise ValueError(f"{spec} runs with per-loop KV only: its target is original and its KV policy native")
    pin = pinned(spec, target)
    if pin is None:
        raise ValueError(f"{spec} has no published {target} target in configs/models.json")
    if family in TRAINING_FREE_SHARING:
        kv_policy = kv_policy or KV_POLICIES[target]
    elif family == "inkling2":
        kv_policy = "native"
    elif kv_policy is not None:
        raise ValueError(f"--kv-policy applies to Dory only, not {spec}")
    if drafter is None and with_drafter:
        # Without a kind: the paper's DFlash drafter, or the DSpark one of a model that has no other (Inkling2).
        kinds = (drafter_kind,) if drafter_kind else DRAFTER_KINDS
        pin_drafter = next((pin for pin in (pinned_drafter(spec, target, kind) for kind in kinds) if pin), None)
        if pin_drafter is None:
            what = "drafter" if drafter_kind in (None, "dflash") else f"{drafter_kind} drafter"
            raise ValueError(f"{spec} has no published {what} for its {target} target in configs/models.json; "
                             "pass --drafter, or --block-size 1 to run without a drafter")
        drafter = "@".join(x for x in pin_drafter if x)
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


def model_type_family(model_type):
    """Family of a config.json model_type. An unknown model_type raises: no model defaults to a family."""
    for name, value in FAMILY_MODEL_TYPES.items():
        if value == model_type:
            return name
    raise ValueError(f"unsupported model_type {model_type!r}; supported: {', '.join(FAMILY_MODEL_TYPES.values())}")


def eos_token_ids(family):
    """The fixed EOS ids of a family, or None for a family whose EOS ids are read from its checkpoint."""
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}; supported: {', '.join(FAMILIES)}")
    return list(EOS_TOKEN_IDS[family]) if family in EOS_TOKEN_IDS else None


def family(spec):
    """Model family of a registry name or of a checkpoint directory (config.json model_type)."""
    spec = str(spec)
    if spec in _registry():
        return entry(spec)["family"]
    if not (Path(spec) / "config.json").is_file():
        raise ValueError(f"{spec!r} is neither a registry name ({', '.join(names())}) nor a checkpoint directory")
    config = json.loads((Path(spec) / "config.json").read_text())
    if config.get("model_type") not in FAMILY_MODEL_TYPES.values():
        raise ValueError(f"unsupported model_type {config.get('model_type')!r} in {spec}")
    return model_type_family(config["model_type"])
