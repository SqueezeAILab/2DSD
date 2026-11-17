#!/usr/bin/env python3
"""Build the benchmark prompts of the 2DSD throughput evaluation.

Reproduces the prompt token IDs of the paper's Core and Hard suites (Sec. 5.1, App. D.2)
from the pinned benchmark sources in configs/benchmarks.json and the tokenizer and chat
template of the pinned model revision in configs/models.json.

  core  GSM8K test 1319, MATH-500 500, HumanEval+ 164 and MBPP+ 377 (Mbpp/255 excluded);
        prompt + output budget 8192 tokens.
  hard  AIME24 30, AIME25 30, and the first 100 questions of a fixed SHA ranking of
        GPQA-Diamond and of OlympiadBench OE_TO_maths_en_COMP; budget 32768 tokens.
        AIME26 (30) is optional (--include aime26); the paper reports AIME24/25 only.

usage: build_prompts.py --model MODEL --suite {core,hard} --out DIR
                        [--cache-dir DIR] [--include aime26] [--offline]

Writes DIR/SUITE.prompts.jsonl, one question per line sorted by id with the fields
id, benchmark, source, domain, problem_id, request_id, rendered_family, max_total_tokens,
prompt_tokens and prompt_ids, and DIR/SUITE.manifest.json. No benchmark text, answer or
reference solution is written. Manifest hashes:
  catalog_sha256  SHA-256 of the catalog serialization used for the paper's runs. Those
                  rows also carry the task text and the reference answer, which this script
                  computes in memory only. Equal to configs/benchmarks.json reference_catalogs
                  when the prompts, IDs and budgets are the paper's.
  prompt_digest   SHA-256 of the JSON list [[id, prompt_ids], ...]; recomputable from the
                  prompts file.
  prompts_sha256  SHA-256 of the written prompts file.
The exit status is 1 when a reference exists and either hash differs from it.

GPQA-Diamond is gated on Hugging Face: accept its terms on the dataset page and
authenticate (HF_TOKEN or `hf auth login`) before building the hard suite.
Dependencies: transformers, huggingface_hub, pyarrow (and the standard library).
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_CONFIG = ROOT / "configs" / "benchmarks.json"
MODELS_CONFIG = ROOT / "configs" / "models.json"

SEED = 20260930
SELECTION_METHOD = "sha256_rank_seed_benchmark_problem_id_v1"
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
GPQA_INSTRUCTION = "Reason step by step and put the single letter of your final answer within \\boxed{}."
# EvalPlus raw-chat prompt (evalplus.provider.utility.make_raw_chat_prompt, evalplus 0.3.1).
EVALPLUS_INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem "
                        "in a markdown code block:")
EVALPLUS_RESPONSE = ("Below is a Python script with a self-contained function that solves the problem "
                     "and passes corresponding tests:")
EVALPLUS_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"
# Explicit system turns. A family without one keeps its template's default turn
# (Ouro: "You are a helpful assistant.").
SYSTEM_PROMPTS = {
    "nanbeige": ("You are Nanbeige, a professional large language model independently developed "
                 "and trained by BOSS Zhipin."),
}
TEMPLATE_KWARGS = {"ouro": {}, "nanbeige": {"enable_thinking": True}, "dory": {"enable_thinking": True}}
# Files fetched from the model repository; weights are never downloaded.
TOKENIZER_PATTERNS = ["tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                      "chat_template*", "config.json"]
TOKENIZER_IDENTITY = ("tokenizer*", "special_tokens_map.json", "chat_template*")
PUBLIC_FIELDS = ("id", "benchmark", "source", "domain", "problem_id", "request_id", "rendered_family",
                 "max_total_tokens", "prompt_tokens", "prompt_ids")
DOWNLOAD_LIMIT = 64 * 1024 * 1024


class IntegrityError(RuntimeError):
    """A source, rendering or selection differs from its pinned definition."""


class SourceUnavailable(RuntimeError):
    """A pinned source is not cached and cannot be downloaded."""


# Hashing and selection

def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    """SHA-256 of canonical JSON; used for request IDs, selections and token lists."""
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def select_ids(seed, benchmark, identities, count):
    """The count IDs that rank first by sha256(json([seed, benchmark, id])), in their native order."""
    identities = list(identities)
    if len(identities) != len(set(identities)) or not 1 <= count <= len(identities):
        raise ValueError("Selection needs unique IDs and 1 <= count <= len(IDs)")
    chosen = set(sorted(identities, key=lambda identity: digest([seed, benchmark, identity]))[:count])
    return [identity for identity in identities if identity in chosen]


def catalog_bytes(rows):
    """JSON lines with sorted keys and ASCII escapes, the serialization the catalog hashes cover."""
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()


def prompt_digest(rows):
    return sha256_bytes(json.dumps([(row["id"], row["prompt_ids"]) for row in rows], separators=(",", ":")).encode())


# Task text

def gpqa_choice_order(identifier, seed=SEED):
    """Indices into [correct, incorrect 1, incorrect 2, incorrect 3] shown as A, B, C, D."""
    order = list(range(4))
    random.Random(int(digest([seed, identifier])[:16], 16)).shuffle(order)
    return order


def gpqa_task(row, identifier):
    choices = [row["Correct Answer"]] + [row[f"Incorrect Answer {i}"] for i in range(1, 4)]
    order = gpqa_choice_order(identifier)
    task = (row["Question"].strip() + "\n\n"
            + "\n".join(f"{letter}. {choices[index]}" for letter, index in zip("ABCD", order))
            + "\n\n" + GPQA_INSTRUCTION)
    return task, "ABCD"[order.index(0)]


def core_task(benchmark, row):
    """(task text, reference answer) of a Core row; code tasks are the bare EvalPlus prompt."""
    if benchmark == "gsm8k":
        return (row["question"].strip() + "\n\n" + MATH_INSTRUCTION,
                row["answer"].rsplit("####", 1)[-1].strip().replace(",", ""))
    if benchmark == "math500":
        return row["problem"] + "\n\n" + MATH_INSTRUCTION, str(row["answer"])
    if benchmark in ("humaneval_plus", "mbpp_plus"):
        return row["prompt"].strip() + "\n", row["canonical_solution"]
    raise ValueError(f"Not a Core benchmark: {benchmark}")


def hard_task(benchmark, row, identifier):
    """(task text, reference answer) of a Hard row."""
    if benchmark == "gpqa_diamond":
        return gpqa_task(row, identifier)
    if benchmark.startswith("aime"):
        task, answer = row["problem"].strip(), row["answer"]
    elif benchmark == "olympiad":
        task, answer = row["question"].strip(), row["final_answer"][0]
        if row.get("context"):
            task = str(row["context"]) + "\n\n" + task
    else:
        raise ValueError(f"Not a Hard benchmark: {benchmark}")
    return task + "\n\n" + MATH_INSTRUCTION, str(answer)


# Chat rendering

def system_messages(family):
    return [{"role": "system", "content": SYSTEM_PROMPTS[family]}] if family in SYSTEM_PROMPTS else []


def split_assistant_prefix(rendered, splitter=EVALPLUS_SPLITTER):
    """Cut a rendered conversation just before the splitter that stands in for the model's code."""
    if rendered.count(splitter) != 1:
        raise IntegrityError("The assistant-prefix splitter must occur exactly once in the rendered chat")
    return rendered.split(splitter)[0]


def render_prompt(apply_chat_template, family, task, mode):
    """Prompt text in the family's native chat template.

    user_chat: one user turn and the generation prompt. evalplus_chat: the EvalPlus raw-chat
    prompt, whose assistant turn is prefilled up to the opening of a Python code block.
    """
    if family not in TEMPLATE_KWARGS:
        raise ValueError(f"Unsupported model family: {family}")
    kwargs = TEMPLATE_KWARGS[family]
    if mode == "evalplus_chat":
        user = f"{EVALPLUS_INSTRUCTION}\n```\n{task.strip()}\n```\n"
        assistant = f"{EVALPLUS_RESPONSE}\n```python\n{EVALPLUS_SPLITTER}\n```\n"
        messages = [*system_messages(family), {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant}]
        return split_assistant_prefix(apply_chat_template(messages, tokenize=False, **kwargs))
    if mode == "user_chat":
        messages = [*system_messages(family), {"role": "user", "content": task}]
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    raise ValueError(f"Unknown prompt mode: {mode}")


def catalog_row(*, benchmark, problem_id, domain, task, target, prompt, ids, family, budget, extra=None):
    """A row of the reference catalog. Only PUBLIC_FIELDS of it are ever written."""
    if not 0 <= len(ids) < budget:
        raise IntegrityError(f"{benchmark}/{problem_id}: the prompt exhausts the {budget}-token budget")
    return {"problem_id": problem_id, "sample_id": 0, "target": target, "task_prompt": task,
            "task_prompt_sha256": sha256_text(task), "prompt": prompt, "prompt_sha256": sha256_text(prompt),
            "prompt_token_ids_sha256": digest(ids), "request_id": digest([benchmark, problem_id, 0]),
            "id": problem_id if problem_id.startswith(benchmark + "/") else f"{benchmark}/{problem_id}",
            "source": benchmark, "benchmark": benchmark, "domain": domain, "prompt_ids": ids,
            "prompt_token_ids": ids, "prompt_tokens": len(ids), "prompt_ids_sha256": digest(ids),
            "rendered_family": family, "max_total_tokens": budget, **(extra or {})}


# Pinned sources

def read_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "2dsd-build-prompts/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read(DOWNLOAD_LIMIT + 1)
            if len(content) > DOWNLOAD_LIMIT:
                raise IntegrityError(f"{url} exceeds the download size bound")
            return content
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise SourceUnavailable(f"HTTP {error.code} for {url}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise SourceUnavailable(f"Network error for {url}") from None
        time.sleep(attempt + 1)
    raise SourceUnavailable(f"Network error for {url}")


def verify_file(name, path, pin):
    if path.stat().st_size != pin["size_bytes"] or file_sha256(path) != pin["sha256"]:
        raise IntegrityError(f"{name}: {path} differs from the pinned size/SHA-256; remove it to download again")


def fetch(name, pin, cache_dir, offline=False):
    """Path of the verified pinned file, downloaded into CACHE_DIR/NAME/FILE when absent."""
    relative = Path(pin["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise IntegrityError(f"{name}: the pinned file path must be relative")
    path = Path(cache_dir) / name / relative
    if not path.is_file():
        if offline:
            raise SourceUnavailable(f"{name}: {path} is not cached and --offline forbids downloads")
        if pin["host"] == "huggingface":
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError
            try:
                hf_hub_download(pin["repo_id"], pin["file"], repo_type="dataset", revision=pin["revision"],
                                local_dir=Path(cache_dir) / name)
            except GatedRepoError:
                raise SourceUnavailable(
                    f"{name}: {pin['repo_id']} is gated. Accept its terms on "
                    f"https://huggingface.co/datasets/{pin['repo_id']} and authenticate with HF_TOKEN "
                    "or `hf auth login`.") from None
            except (HfHubHTTPError, LocalEntryNotFoundError) as error:
                raise SourceUnavailable(f"{name}: download of {pin['repo_id']}/{pin['file']} failed "
                                        f"({type(error).__name__})") from None
        elif pin["host"] == "github":
            content = read_url(pin["url"])
            if len(content) != pin["size_bytes"] or sha256_bytes(content) != pin["sha256"]:
                raise IntegrityError(f"{name}: the download differs from the pinned size/SHA-256")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        else:
            raise ValueError(f"{name}: unknown host {pin['host']}")
    verify_file(name, path, pin)
    return path


def read_rows(name, path, pin):
    kind = pin["format"]
    if kind == "parquet":
        import pyarrow.parquet as parquet
        rows = parquet.read_table(path).to_pylist()
    elif kind == "csv":
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif kind in ("jsonl", "jsonl.gz"):
        opener = gzip.open if kind == "jsonl.gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError(f"{name}: unknown format {kind}")
    if len(rows) != pin["rows"]:
        raise IntegrityError(f"{name}: expected {pin['rows']} rows, found {len(rows)}")
    return rows


def source_entry(pin, **extra):
    keys = ("host", "repo_id", "revision", "url", "file", "sha256", "size_bytes", "rows", "license", "access")
    return {**{key: pin[key] for key in keys if key in pin}, **extra}


# Suites

def build_core(config, encode, apply_chat_template, family, cache_dir, offline):
    suite = config["suites"]["core"]
    budget = suite["max_total_tokens"]
    excluded = {(entry["benchmark"], entry["problem_id"]) for entry in suite["excluded_questions"]}
    rows, sources, skipped = [], {}, 0
    for name in suite["benchmarks"]:
        pin = config["benchmarks"][name]
        data = read_rows(name, fetch(name, pin, cache_dir, offline), pin)
        identities, kept = [], 0
        for index, row in enumerate(data):
            native = row[pin["id_field"]] if pin["id_field"] else f"{pin['row_id_prefix']}{index}"
            if native is None or not str(native):
                raise IntegrityError(f"{name}: row {index} has no {pin['id_field']}")
            problem_id = str(native)
            identities.append(problem_id)
            if (name, problem_id) in excluded:
                skipped += 1
                continue
            task, target = core_task(name, row)
            mode = "evalplus_chat" if pin["domain"] == "code" else "user_chat"
            prompt = render_prompt(apply_chat_template, family, task, mode)
            extra = {"native_id": native, "prompt_mode": mode}
            if pin["domain"] == "code":
                extra["entry_point"] = row["entry_point"]
            rows.append(catalog_row(benchmark=name, problem_id=problem_id, domain=pin["domain"], task=task,
                                    target=target, prompt=prompt, ids=encode(prompt), family=family,
                                    budget=budget, extra=extra))
            kept += 1
        if len(set(identities)) != len(identities):
            raise IntegrityError(f"{name}: duplicate question IDs")
        sources[name] = source_entry(pin, questions=kept)
    if skipped != len(excluded):
        raise IntegrityError("The Core exclusions did not each match one question")
    selection = {"rule": "every question of each benchmark", "excluded_questions": suite["excluded_questions"]}
    return rows, sources, selection, budget


def build_hard(config, encode, apply_chat_template, family, cache_dir, offline, include=()):
    suite = config["suites"]["hard"]
    budget = suite["max_total_tokens"]
    unknown = set(include) - set(suite["optional_benchmarks"])
    if unknown:
        raise ValueError(f"Not optional Hard benchmarks: {sorted(unknown)}")
    names = [*suite["benchmarks"], *(name for name in suite["optional_benchmarks"] if name in include)]
    rows, sources = [], {}
    for name in names:
        pin = config["benchmarks"][name]
        data = read_rows(name, fetch(name, pin, cache_dir, offline), pin)
        identities = [f"{name}/{row[pin['id_field']]}" for row in data]
        if len(set(identities)) != len(identities):
            raise IntegrityError(f"{name}: duplicate question IDs")
        count = suite["subsets"].get(name)
        chosen = select_ids(SEED, name, identities, count) if count else identities
        extra = {"selected": count, "selected_problem_ids_sha256": digest(chosen)} if count else {}
        sources[name] = source_entry(pin, questions=len(chosen), **extra)
        chosen = set(chosen)
        for identifier, row in zip(identities, data):
            if identifier not in chosen:
                continue
            task, target = hard_task(name, row, identifier)
            prompt = render_prompt(apply_chat_template, family, task, "user_chat")
            rows.append(catalog_row(benchmark=name, problem_id=identifier, domain=pin["domain"], task=task,
                                    target=target, prompt=prompt, ids=encode(prompt), family=family,
                                    budget=budget))
    selection = {"complete": [name for name in names if name not in suite["subsets"]],
                 "subsets": {name: suite["subsets"][name] for name in names if name in suite["subsets"]},
                 "method": SELECTION_METHOD, "seed": SEED, "uses_model_outputs_for_selection": False}
    return rows, sources, selection, budget


# Model tokenizer

def load_tokenizer(model, offline=False):
    """AutoTokenizer of the pinned revision (tokenizer files only) and the SHA-256 of those files."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    from transformers import AutoTokenizer

    try:
        path = Path(snapshot_download(model["repo_id"], revision=model["revision"],
                                      allow_patterns=TOKENIZER_PATTERNS, local_files_only=offline))
    except (HfHubHTTPError, LocalEntryNotFoundError) as error:
        raise SourceUnavailable(f"Tokenizer of {model['repo_id']}@{model['revision']} is unavailable "
                                f"({type(error).__name__})") from None
    config = json.loads((path / "config.json").read_text()) if (path / "config.json").is_file() else {}
    options = {}
    if config.get("model_type") == "dory":
        # transformers has no Dory model; the vLLM fork's config class lets AutoTokenizer read config.json.
        from transformers import AutoConfig
        from vllm.transformers_utils.configs.dory import DoryConfig

        try:
            AutoConfig.register("dory", DoryConfig)
        except ValueError:
            pass  # already registered
        # The tokenizer as shipped, as vLLM loads it for the model; an explicit False also
        # silences transformers' warning that suggests its Mistral regex correction.
        options["fix_mistral_regex"] = False
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True, **options)
    if not tokenizer.chat_template:
        raise IntegrityError(f"{model['repo_id']}@{model['revision']} has no chat template")
    files = sorted({p for pattern in TOKENIZER_IDENTITY for p in path.glob(pattern) if p.is_file()})
    return tokenizer, {p.name: file_sha256(p) for p in files}


def package_versions():
    versions = {}
    for name in ("transformers", "tokenizers", "jinja2", "huggingface_hub", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_atomic(path, content):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build(model_name, suite, out, cache_dir, include=(), offline=False, config=None, models=None):
    config = config or json.loads(BENCHMARKS_CONFIG.read_text())
    models = models or json.loads(MODELS_CONFIG.read_text())["models"]
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name}; choose from {sorted(models)}")
    if include and suite != "hard":
        raise ValueError("--include applies to the hard suite only")
    model = models[model_name]
    family = model["family"]
    tokenizer, tokenizer_files = load_tokenizer(model, offline)

    def encode(prompt):
        return list(tokenizer.encode(prompt, add_special_tokens=False))

    arguments = (config, encode, tokenizer.apply_chat_template, family, cache_dir, offline)
    if suite == "core":
        rows, sources, selection, budget = build_core(*arguments)
    else:
        rows, sources, selection, budget = build_hard(*arguments, include)
    rows.sort(key=lambda row: row["id"])
    if len({row["id"] for row in rows}) != len(rows):
        raise IntegrityError("Duplicate question IDs in the catalog")

    public = [{key: row[key] for key in PUBLIC_FIELDS} for row in rows]
    content = catalog_bytes(public)
    reference_key = suite + "".join(f"+{name}" for name in sorted(include))
    catalog_sha256, digest_value = sha256_bytes(catalog_bytes(rows)), prompt_digest(rows)
    reference = config.get("reference_catalogs", {}).get(family, {}).get(reference_key)
    if reference:
        reference = {"name": reference_key, **reference,
                     "matches": (reference["catalog_sha256"] == catalog_sha256
                                 and reference["prompt_digest"] == digest_value
                                 and reference["questions"] == len(rows))}
    manifest = {
        "schema_version": 1, "name": suite, "model": model_name, "model_repo_id": model["repo_id"],
        "model_revision": model["revision"], "rendered_family": family, "tokenizer_files": tokenizer_files,
        "packages": package_versions(), "max_total_tokens": budget, "questions": len(rows),
        "by_benchmark": dict(sorted(collections.Counter(row["benchmark"] for row in rows).items())),
        "selection": selection, "sources": sources,
        "prompt_length": {"min": min(row["prompt_tokens"] for row in rows),
                          "max": max(row["prompt_tokens"] for row in rows)},
        "catalog_sha256": catalog_sha256, "prompt_digest": digest_value, "prompts_sha256": sha256_bytes(content),
        "reference": reference,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / f"{suite}.prompts.jsonl", content)
    write_atomic(out / f"{suite}.manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def default_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "2dsd" / "benchmarks"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a model name of configs/models.json")
    parser.add_argument("--suite", required=True, choices=("core", "hard"))
    parser.add_argument("--out", required=True, type=Path, help="directory for SUITE.prompts.jsonl/.manifest.json")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir(),
                        help="benchmark source cache, laid out as CACHE_DIR/BENCHMARK/FILE "
                             "(default: $XDG_CACHE_HOME/2dsd/benchmarks or ~/.cache/2dsd/benchmarks)")
    parser.add_argument("--include", action="append", default=[], choices=("aime26",),
                        help="add an optional Hard benchmark (repeatable)")
    parser.add_argument("--offline", action="store_true",
                        help="use only cached benchmark files and the local Hugging Face cache")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.model, args.suite, args.out, args.cache_dir, tuple(sorted(set(args.include))),
                         args.offline)
    except (IntegrityError, SourceUnavailable, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    reference = manifest["reference"]
    print(json.dumps({"suite": manifest["name"], "model": manifest["model"], "questions": manifest["questions"],
                      "by_benchmark": manifest["by_benchmark"], "catalog_sha256": manifest["catalog_sha256"],
                      "prompt_digest": manifest["prompt_digest"],
                      "matches_paper_catalog": reference["matches"] if reference else None}, indent=2))
    if reference and not reference["matches"]:
        print(f"warning: the {reference['name']} catalog differs from the paper's ({reference['catalog_sha256']}); "
              "check the transformers/tokenizers versions in the manifest", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
