# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""scripts/inkling2 on CPU: the journal and throughput conventions against scripts/benchmark.py,
native_baseline.py on a fake vLLM engine, and serve_official.sh with nvidia-smi and docker replaced through
PATH. Nothing starts a container or a GPU job."""

import ast
import hashlib
import importlib.util
import io
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "inkling2"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load("benchmark_script", ROOT / "scripts" / "benchmark.py")
native = load("inkling2_native_baseline", SCRIPTS / "native_baseline.py")
common = native.common

LOOP = [1, 2, 3, 4] * 1000  # a response that repeats itself


def result_row(row_id, benchmark_name, output_ids, seconds, max_new=10 ** 6, eos=True, **extra):
    return {"id": row_id, "benchmark": benchmark_name, "output_ids": output_ids, "decode_seconds": seconds,
            "max_new": max_new, "stopped_by_eos": eos, **extra}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def printed(function, *args):
    out = io.StringIO()
    with redirect_stdout(out):
        function(*args)
    return out.getvalue()


class Conventions(unittest.TestCase):
    """common.py must stay the stdlib copy of scripts/benchmark.py's definitions, not a second definition."""

    def test_repetition_filter_seed_and_cap_are_those_of_benchmark_py(self):
        self.assertEqual((common.LOOP_TAIL, common.LOOP_NGRAM, common.LOOP_DISTINCT),
                         (benchmark.LOOP_TAIL, benchmark.LOOP_NGRAM, benchmark.LOOP_DISTINCT))
        self.assertEqual(common.TABLE_BENCHMARK, benchmark.TABLE_BENCHMARK)
        rng = random.Random(0)
        half = list(range(1024)) + [7] * 1024  # about half of the tail's 16-grams are distinct
        cases = [LOOP, list(range(4000)), [], [3] * 15, [3] * 16, half, half[1:], LOOP[:2047] + [9],
                 *([rng.randrange(period) for _ in range(3000)] for period in (1, 2, 3, 50))]
        for output_ids in cases:
            for capped in (True, False):
                self.assertEqual(common.repetitive(output_ids, capped), benchmark.repetitive(output_ids, capped))
        self.assertTrue(common.repetitive(LOOP, True))
        self.assertFalse(common.repetitive(LOOP, False))
        for row in ({"output_ids": [1] * 8, "max_new": 8, "stopped_by_eos": False},
                    {"output_ids": [1] * 8, "max_new": 8, "stopped_by_eos": True},
                    {"output_ids": [1] * 5, "max_new": 8, "stopped_by_eos": False}):
            self.assertEqual(common.capped(row), benchmark.capped(row))
        self.assertEqual(common.request_seed(3, "gsm8k/main/test/row:0"),
                         benchmark.request_seed(3, "gsm8k/main/test/row:0"))

    def test_summary_pools_decode_tokens_and_drops_capped_repetition(self):
        rows = [result_row("a", "gsm8k", [0] * 11, 1.0),
                result_row("b", "gsm8k", [0] * 31, 3.0),
                result_row("c", "math500", LOOP, 10.0, max_new=len(LOOP), eos=False),  # capped and looping
                result_row("d", "math500", [0] * 21, 2.0),
                result_row("e", "math500", LOOP, 4.0, eos=True),  # looping but stopped by itself: kept
                result_row("f", "aime24", [0] * 11, 1.0),
                result_row("g", "aime25", [0] * 31, 1.0)]
        by_name = {group["benchmark"]: group for group in common.summary(rows)}
        self.assertEqual(list(by_name), ["aime24/25", "gsm8k", "math500", "all"])
        self.assertEqual(by_name["gsm8k"]["tok_s"], (10 + 30) / (1.0 + 3.0))
        self.assertEqual(by_name["aime24/25"]["tok_s"], (10 + 30) / 2.0)
        self.assertEqual(by_name["math500"]["tok_s"], (3999 + 20 + 3999) / 16.0)
        self.assertEqual(by_name["math500"]["clean_tok_s"], (20 + 3999) / 6.0)
        self.assertEqual((by_name["math500"]["n"], by_name["math500"]["clean_n"]), (3, 2))
        self.assertEqual(by_name["all"]["tok_s"], (40 + 8018 + 40) / (4.0 + 16.0 + 2.0))
        self.assertEqual(by_name["all"]["clean_tok_s"], (40 + 4019 + 40) / (4.0 + 6.0 + 2.0))
        with tempfile.TemporaryDirectory() as directory:
            path = write_rows(Path(directory) / "run.jsonl", rows)
            self.assertEqual(printed(common.summarize, [path]), printed(benchmark.summarize, [path]))

    def test_acceptance_is_pooled_over_verify_steps(self):
        engine_record = {"num_spec_steps": 4, "num_accepted_draft_tokens": 6, "num_draft_tokens": 28}
        counters = {"num_drafts": 6, "num_accepted_tokens": 3, "num_draft_tokens": 42}
        rows = [result_row("a", "x", [0] * 11, 1.0, spec_decode=engine_record, spec_decode_counters={"num_drafts": 9}),
                result_row("b", "x", [0] * 11, 1.0, spec_decode=None, spec_decode_counters=counters),
                result_row("c", "x", [0] * 11, 1.0)]  # no speculation recorded: not in the acceptance pool
        stats = common.summary(rows)[0]["acceptance"]
        self.assertEqual((stats["n"], stats["steps"]), (2, 10))
        self.assertEqual(stats["tokens_per_step"], 20 / 10)
        self.assertEqual(stats["acceptance_length"], 1 + 9 / 10)
        self.assertEqual(stats["acceptance_rate"], 9 / 70)
        self.assertIsNone(common.summary(rows[2:])[0]["acceptance"])
        with tempfile.TemporaryDirectory() as directory:
            path = write_rows(Path(directory) / "run.jsonl", rows)
            text, release = printed(common.summarize, [path]), printed(benchmark.summarize, [path])
        throughput, acceptance = text.split("\n\n")
        self.assertEqual(throughput + "\n", release)  # the first table is unchanged
        self.assertIn("accept len", acceptance)
        self.assertIn("1.90", acceptance)


class Journal(unittest.TestCase):
    def test_torn_tail_is_dropped_and_appends_stay_whole_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path, twin = Path(directory) / "run.jsonl", Path(directory) / "twin.jsonl"
            for before, after in (('{"a": 1}\n{"a": 2}', '{"a": 1}\n{"a": 2}\n'),
                                  ('{"a": 1}\n{"a": 2', '{"a": 1}\n'),
                                  ('{"a', ''),
                                  ('{"a": 1}\n', '{"a": 1}\n')):
                path.write_text(before)
                twin.write_text(before)
                common.prepare_append(path)
                benchmark.prepare_append(twin)
                self.assertEqual(path.read_text(), after)
                self.assertEqual(path.read_text(), twin.read_text())
                common.append_record(path, {"b": [1, 2]})
                self.assertEqual(path.read_text(), after + '{"b": [1, 2]}\n')
                self.assertEqual(common.read_jsonl(path)[-1], {"b": [1, 2]})

    def test_a_malformed_line_before_the_last_is_an_error_not_a_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
            with self.assertRaisesRegex(ValueError, "line 2 is not a valid JSON record"):
                common.prepare_append(path)
            self.assertEqual(path.read_text(), '{"a": 1}\n{"a": \n{"a": 3}\n')

    def test_identity_mismatch_names_what_changed(self):
        identity = {"method": "ar", "engine": {"image": "a"}}
        rows = [{"run_identity": {"method": "ar", "engine": {"image": "a"}}}]
        self.assertIsNone(common.identity_mismatch(rows, identity))
        self.assertEqual(common.identity_mismatch(rows, {**identity, "engine": {"image": "b"}}), "different engine")
        self.assertEqual(common.identity_mismatch(rows, {**identity, "seed": 0}), "different seed")
        self.assertEqual(common.identity_mismatch([{}], identity), "rows without a run identity")
        self.assertEqual(common.identity_mismatch(rows, identity), benchmark.identity_mismatch(rows, identity))


class FakeEngine:
    """A vllm module for native_baseline.py.

    LLM records its options and every request. A prompt whose first id is even runs into its budget; an
    odd one stops with EOS (9) after three tokens. Every request takes 0.5 s to its first token and 2 s
    from there to its last one, and, with a speculative config, two verify steps that accept three of
    fourteen draft tokens.
    """

    def __init__(self, per_request_metrics=True):
        self.options, self.requests, self.steps = [], [], 0
        engine = self
        names = ["model"] + (["per_request_spec_decode_metrics"] if per_request_metrics else [])

        class LLM:
            def __init__(self, **options):
                engine.options.append(options)
                self.speculative = "speculative_config" in options
                self.per_request = "per_request_spec_decode_metrics" in options

            def get_tokenizer(self):
                return SimpleNamespace(eos_token_id=9)

            def get_metrics(self):
                steps = engine.steps
                position = [SimpleNamespace(name="vllm:spec_decode_num_verified_tokens_per_pos",
                                            labels={"position": str(i)}, value=steps) for i in range(7)]
                return [SimpleNamespace(name="vllm:num_requests_running", labels={}, value=1),
                        SimpleNamespace(name="vllm:spec_decode_num_drafts", labels={}, value=steps),
                        SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", labels={}, value=7 * steps),
                        SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens", labels={}, value=steps * 3 // 2),
                        SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens_per_pos", labels={},
                                        values=[steps, steps // 2, 0, 0, 0, 0, 0]),
                        *position]

            def generate(self, prompts, params, use_tqdm):
                [prompt] = prompts  # one request at a time
                ids = prompt["prompt_token_ids"]
                engine.requests.append((ids, params))
                output = [5] * params.max_tokens if ids[0] % 2 == 0 else [5, 5, 9][: params.max_tokens]
                record = None
                if self.speculative:
                    engine.steps += 2
                    if self.per_request:
                        payload = {"num_spec_steps": 2, "num_accepted_draft_tokens": 3, "num_draft_tokens": 14,
                                   "mean_acceptance_length": 2.5}
                        record = SimpleNamespace(to_dict=lambda: payload)
                metrics = SimpleNamespace(queued_ts=9.75, scheduled_ts=10.0, first_token_ts=10.5,
                                          last_token_ts=12.5, num_preemptions=0)
                completion = SimpleNamespace(token_ids=output, stop_reason=None, spec_decode_metrics=record,
                                             finish_reason="length" if len(output) == params.max_tokens else "stop")
                return [SimpleNamespace(metrics=metrics, outputs=[completion])]

        vllm = types.ModuleType("vllm")
        vllm.LLM, vllm.SamplingParams = LLM, lambda **params: SimpleNamespace(**params)
        arg_utils = types.ModuleType("vllm.engine.arg_utils")
        arg_utils.EngineArgs = make_dataclass("EngineArgs", [(name, str, "") for name in names])
        self.modules = {"vllm": vllm, "vllm.engine": types.ModuleType("vllm.engine"),
                        "vllm.engine.arg_utils": arg_utils}


class NativeBaseline(unittest.TestCase):
    """native_baseline.py on four prompts (ids [i, 1, 2], budget 8) and a fake engine."""

    ENGINE = {"vllm": "0.1.dev1+gabc", "torch": "2.14.0", "python": "3.14.10", "image": "image@sha256:1",
              "cuda_arch_list": None, "gpu": "NVIDIA H100 NVL, 9.0"}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.model = self.root / "hub" / "models--org--name" / "snapshots" / "rev1"
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text(json.dumps({"eos_token_id": 9}))
        (self.model / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, 7]}))
        self.prompts = self.root / "core.prompts.jsonl"
        self.write_prompts(8)
        self.out = self.root / "runs" / "core.jsonl"
        self.engine = FakeEngine()

    def write_prompts(self, budget, count=4):
        write_rows(self.prompts, [{"id": f"q{i}", "benchmark": "x", "prompt_ids": [i, 1, 2],
                                   "max_total_tokens": budget} for i in range(count)])

    def run_native(self, *extra, method="ar", out=None, engine=None, warmup="0"):
        argv = ["--model", str(self.model), "--method", method, "--prompts", str(self.prompts),
                "--out", str(out or self.out), "--warmup", warmup, *extra]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with (patch.dict(sys.modules, self.engine.modules),
              patch.object(native, "engine_identity", return_value=dict(engine or self.ENGINE)),
              redirect_stdout(self.stdout), redirect_stderr(self.stderr)):
            native.main(argv)

    def rows(self, out=None):
        return {row["id"]: row for row in common.read_jsonl(out or self.out)}

    def test_rows_hold_the_tokens_and_the_engine_s_decode_interval(self):
        self.run_native()
        rows = self.rows()
        self.assertEqual(sorted(rows), ["q0", "q1", "q2", "q3"])
        capped, stopped = rows["q0"], rows["q1"]
        self.assertEqual((capped["output_ids"], capped["finish_reason"], capped["stopped_by_eos"]),
                         ([5] * 5, "length", False))
        self.assertEqual((stopped["output_ids"], stopped["finish_reason"], stopped["stopped_by_eos"]),
                         ([5, 5, 9], "stop", True))
        for row in rows.values():
            self.assertEqual((row["prefill_seconds"], row["decode_seconds"], row["queue_seconds"]), (0.5, 2.0, 0.25))
            self.assertEqual(row["decode_tokens"], len(row["output_ids"]) - 1)
            self.assertEqual((row["prompt_tokens"], row["max_new"], row["benchmark"]), (3, 5, "x"))
            self.assertEqual(row["prompt_ids"], [int(row["id"][1]), 1, 2])
            self.assertIsNone(row["spec_decode"])
            self.assertIsNone(row["spec_decode_counters"])
            self.assertEqual(row["run_identity"]["engine"], self.ENGINE)
        [options] = self.engine.options  # one engine, batch size one, per-request metrics on
        self.assertEqual((options["model"], options["max_num_seqs"], options["max_model_len"]), (str(self.model), 1, 8))
        self.assertIs(options["disable_log_stats"], False)
        self.assertIs(options["enable_prefix_caching"], False)
        self.assertNotIn("speculative_config", options)
        for _, params in self.engine.requests:
            self.assertEqual((params.temperature, params.seed, params.max_tokens, params.stop_token_ids,
                              params.detokenize), (0.0, None, 5, [7, 9], False))
        # (4 + 2 + 4 + 2) decode tokens over 4 x 2 s, and the same table from the release's own summary
        self.assertIn("1.5", [line for line in self.stdout.getvalue().splitlines() if " all " in line][0])
        self.assertEqual(printed(common.summarize, [self.out]), printed(benchmark.summarize, [self.out]))

    def test_shards_split_the_shuffled_order_and_share_the_run_identity(self):
        self.write_prompts(8, count=7)
        outs = [self.root / "runs" / f"core.s{i}.jsonl" for i in range(3)]
        for i, out in enumerate(outs):
            self.run_native("--shard", f"{i}/3", out=out)
        ids = [sorted(self.rows(out)) for out in outs]
        self.assertEqual([len(part) for part in ids], [3, 2, 2])
        self.assertEqual(sorted(sum(ids, [])), [f"q{i}" for i in range(7)])  # disjoint, complete
        identities = {json.dumps(row["run_identity"], sort_keys=True) for out in outs for row in self.rows(out).values()}
        self.assertEqual(len(identities), 1)
        self.run_native(out=self.root / "runs" / "whole.jsonl")  # the same order as the unsharded run
        order = [row["id"] for row in common.read_jsonl(self.root / "runs" / "whole.jsonl")]
        self.assertEqual([row["id"] for row in common.read_jsonl(outs[1])], order[1::3])
        for bad in ("3/3", "x", "1"):
            with self.assertRaises(SystemExit):
                self.run_native("--shard", bad, out=self.root / "runs" / "bad.jsonl")

    def test_warmup_requests_are_short_and_not_recorded(self):
        self.write_prompts(200)
        self.run_native(warmup="2")
        budgets = [params.max_tokens for _, params in self.engine.requests]
        self.assertEqual(budgets, [64, 64, 197, 197, 197, 197])
        self.assertEqual(self.engine.requests[0][0], self.engine.requests[2][0])  # the first measured prompt
        self.assertEqual(len(self.rows()), 4)

    def test_dspark_uses_the_bundled_draft_and_records_acceptance(self):
        self.run_native("--draft-sample-method", "probabilistic", method="dspark", warmup="1")
        [options] = self.engine.options
        self.assertEqual(options["speculative_config"], {"method": "dspark", "num_speculative_tokens": 7,
                                                         "draft_sample_method": "probabilistic"})
        self.assertEqual(options["per_request_spec_decode_metrics"], "summary")
        for row in self.rows().values():
            self.assertEqual(row["spec_decode"]["num_spec_steps"], 2)
            # what this request added, not the engine's running totals (the warmup advanced them too)
            self.assertEqual(row["spec_decode_counters"], {
                "num_drafts": 2, "num_draft_tokens": 14, "num_accepted_tokens": 3,
                "num_accepted_tokens_per_pos": [2, 1, 0, 0, 0, 0, 0], "num_verified_tokens_per_pos": [2] * 7})
            self.assertEqual(row["run_identity"]["draft_sample_method"], "probabilistic")
            self.assertEqual(row["run_identity"]["num_speculative_tokens"], 7)
        self.assertIn("accept len", self.stdout.getvalue())
        with self.assertRaises(SystemExit):  # a draft option without a draft
            self.run_native("--draft-sample-method", "greedy", method="ar", out=self.root / "ar.jsonl")

    def test_a_build_without_per_request_metrics_falls_back_to_the_counters(self):
        self.engine = FakeEngine(per_request_metrics=False)
        self.run_native(method="dspark")
        self.assertNotIn("per_request_spec_decode_metrics", self.engine.options[0])
        self.assertEqual(self.engine.options[0]["speculative_config"]["draft_sample_method"], "greedy")
        rows = self.rows()
        self.assertTrue(all(row["spec_decode"] is None for row in rows.values()))
        stats = common.summary(list(rows.values()))[0]["acceptance"]
        self.assertEqual((stats["steps"], stats["acceptance_length"]), (8, 1 + 12 / 8))

    def test_sampling_uses_the_release_s_request_seed(self):
        self.run_native("--temperature", "0.6", "--top-p", "0.95", "--seed", "3")
        for ids, params in self.engine.requests:
            self.assertEqual((params.temperature, params.top_p), (0.6, 0.95))
            self.assertEqual(params.seed, benchmark.request_seed(3, f"q{ids[0]}"))

    def test_resume_skips_finished_ids_and_repairs_a_torn_tail(self):
        self.run_native("--limit", "2")
        first = [ids[0] for ids, _ in self.engine.requests]
        with self.out.open("a") as handle:
            handle.write('{"output_ids": [5, 5')  # an interrupted write
        self.run_native()
        second = [ids[0] for ids, _ in self.engine.requests[2:]]
        self.assertEqual(sorted(first + second), [0, 1, 2, 3])
        self.assertEqual(len(second), 2)
        self.assertEqual(len(self.out.read_text().splitlines()), 4)  # every line is a whole record
        self.run_native()  # nothing pending: no engine is built
        self.assertEqual(len(self.engine.options), 2)

    def test_another_run_identity_is_refused(self):
        self.run_native("--limit", "2")
        before = self.out.read_bytes()
        other_image = {**self.ENGINE, "image": "image@sha256:2"}
        for extra, options, field in ((("--temperature", "0.6"), {}, "temperature"),
                                      ((), {"method": "dspark"}, "method"),
                                      (("--max-total-tokens", "6"), {}, "max_total_tokens"),
                                      (("--engine-args", '{"gpu_memory_utilization": 0.9}'), {}, "engine_args"),
                                      ((), {"engine": other_image}, "engine")):
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.run_native(*extra, **options)
            self.assertRegex(self.stderr.getvalue(), rf"another run \(different [^)]*\b{field}\b.*choose a new --out")
        self.write_prompts(8, count=5)  # another prompts file
        with self.assertRaises(SystemExit):
            self.run_native()
        self.assertIn("another run (different prompts_sha256)", self.stderr.getvalue())
        self.assertEqual(self.out.read_bytes(), before)
        self.assertEqual(len(self.engine.options), 1)  # refused before any engine is built

    def test_the_identity_names_a_cache_snapshot_by_repository_and_revision(self):
        self.run_native("--limit", "1")
        identity = next(iter(self.rows().values()))["run_identity"]
        self.assertEqual(identity["model"], "org/name@rev1")  # the same on the host and in the container
        self.assertEqual(identity["prompts_sha256"], hashlib.sha256(self.prompts.read_bytes()).hexdigest())
        self.assertEqual(native.model_name(self.root), str(self.root))

    def test_a_budget_cap_bounds_the_request_and_the_engine(self):
        self.run_native("--max-total-tokens", "6")
        self.assertEqual({params.max_tokens for _, params in self.engine.requests}, {3})
        self.assertEqual(self.engine.options[0]["max_model_len"], 6)
        with self.assertRaises(SystemExit):
            self.run_native("--max-total-tokens", "3", out=self.root / "none.jsonl")
        self.assertIn("leaves no output budget", self.stderr.getvalue())

    def test_engine_arguments_reach_the_engine(self):
        self.run_native("--engine-args", '{"gpu_memory_utilization": 0.9, "max_num_seqs": 1}')
        self.assertEqual(self.engine.options[0]["gpu_memory_utilization"], 0.9)

    def test_missing_request_metrics_stop_the_run(self):
        """disable_log_stats=True would leave no timestamps: fail instead of writing rows without timing."""
        output = SimpleNamespace(metrics=None, outputs=[SimpleNamespace(token_ids=[1])])
        with self.assertRaisesRegex(RuntimeError, "disable_log_stats=False"):
            native.measure(output, [9], 4)

    def test_summarize_needs_no_engine(self):
        self.run_native()
        with patch.dict(sys.modules, {"vllm": None}):  # importing vllm would raise
            text = printed(native.main, ["--summarize", str(self.out)])
        self.assertEqual(text, printed(benchmark.summarize, [self.out]))

    def test_engine_identity_reads_the_image_and_the_gpu(self):
        smi = self.root / "nvidia-smi"
        smi.write_text('#!/bin/sh\necho "$*" > "$0.args"\necho "NVIDIA H100 NVL, 9.0"\n')
        smi.chmod(0o755)
        environment = {"PATH": f"{self.root}{os.pathsep}{os.environ['PATH']}", "INKLING2_IMAGE": "repo@sha256:9",
                       "CUDA_VISIBLE_DEVICES": "5"}
        with patch.dict(os.environ, environment):
            identity = native.engine_identity()
        self.assertEqual((identity["image"], identity["gpu"]), ("repo@sha256:9", "NVIDIA H100 NVL, 9.0"))
        self.assertEqual((self.root / "nvidia-smi.args").read_text().split()[-2:], ["-i", "5"])
        self.assertEqual(set(identity), {"vllm", "torch", "python", "image", "cuda_arch_list", "gpu"})

    def test_the_released_snapshot_stops_at_the_contract_eos(self):
        """The fixture's EOS ids are invented. This reads the released checkpoint's own JSON files: the
        directory named by INKLING2_SNAPSHOT, else the pinned revision in the Hugging Face cache; skipped
        without a local copy."""
        repo, revision = "thinkingmachines/Inkling-2-Mini", "1fb4134b9a74f9681054f4ef82e4bf1f458a2337"
        explicit = os.environ.get("INKLING2_SNAPSHOT")
        if explicit:
            snapshot = Path(explicit)
        else:
            try:
                from huggingface_hub.constants import HF_HUB_CACHE
            except ImportError:
                self.skipTest("huggingface_hub is not installed")
            snapshot = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        if not (snapshot / "config.json").is_file():
            self.skipTest(f"no local snapshot of {repo} (set INKLING2_SNAPSHOT)")
        self.assertEqual(native.eos_ids(snapshot, SimpleNamespace()), [200006])
        self.assertEqual(native.eos_ids(snapshot, SimpleNamespace(eos_token_id=200006)), [200006])
        if snapshot.parent.name == "snapshots":  # a cache snapshot is named as the registry names it
            self.assertEqual(native.model_name(snapshot), f"{repo}@{snapshot.name}")

    def test_container_scripts_import_only_the_standard_library_vllm_and_torch(self):
        """The official image has neither this repository nor its dependencies."""
        for name in ("native_baseline.py", "common.py"):
            tree = ast.parse((SCRIPTS / name).read_text())
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.add(node.module.split(".")[0])
            self.assertLessEqual(modules - set(sys.stdlib_module_names), {"vllm", "torch"}, name)


class Containers(unittest.TestCase):
    """The container scripts with nvidia-smi and docker replaced by recorders that come first on PATH."""

    IMAGE = "vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"
    REVISION = "1fb4134b9a74f9681054f4ef82e4bf1f458a2337"
    CACHE = "models--thinkingmachines--Inkling-2-Mini"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        (self.root / "bin").mkdir()
        self.mock("nvidia-smi", 'printf "%s\\n" "$*" >> "$MOCK/nvidia-smi.calls"\ncat "$MOCK/compute-apps"\n'
                                'exit "$(cat "$MOCK/nvidia-smi.status")"\n')
        self.mock("docker", 'printf "%s\\n" "$@" > "$MOCK/docker.argv"\n')
        (self.root / "compute-apps").write_text("")
        (self.root / "nvidia-smi.status").write_text("0")
        (self.root / "hf" / "hub" / self.CACHE / "snapshots" / self.REVISION).mkdir(parents=True)
        self.prompts = self.root / "prompts" / "core.prompts.jsonl"
        self.prompts.parent.mkdir()
        self.prompts.write_text("{}\n")
        self.out = self.root / "runs" / "mini-core.jsonl"
        self.env = {**os.environ, "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}",
                    "MOCK": str(self.root), "HF_HOME": str(self.root / "hf")}
        for tool in ("docker", "nvidia-smi"):  # the real tools must be out of reach
            self.assertEqual(shutil.which(tool, path=self.env["PATH"]), str(self.root / "bin" / tool))

    def mock(self, name, body):
        path = self.root / "bin" / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)

    def serve(self, *args):
        return subprocess.run(["bash", str(SCRIPTS / "serve_official.sh"), *args], capture_output=True, text=True,
                              env=self.env)

    def started(self):
        argv = self.root / "docker.argv"
        return argv.read_text().splitlines() if argv.exists() else None


class ServeOfficial(Containers):
    """serve_official.sh: the vendor baseline in the official image."""

    def test_dry_run_prints_the_pinned_offline_unprivileged_command(self):
        done = self.serve("--dry-run", "3", str(self.prompts), str(self.out), "--method", "dspark", "--limit", "4")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        [line] = done.stdout.splitlines()
        argv = shlex.split(line)
        self.assertEqual(argv[:2], ["docker", "run"])
        image = argv.index(self.IMAGE)  # by digest, never by tag
        options, inside = argv[2:image], argv[image + 1:]
        self.assertEqual(inside, ["/opt/inkling2/native_baseline.py",
                                  "--model", f"/hf/hub/{self.CACHE}/snapshots/{self.REVISION}",
                                  "--prompts", "/prompts/core.prompts.jsonl", "--out", "/out/mini-core.jsonl",
                                  "--method", "dspark", "--limit", "4"])

        def values(flag):
            return [options[i + 1] for i, option in enumerate(options) if option == flag]

        self.assertEqual(values("--gpus"), ["device=3"])  # one GPU
        self.assertEqual(values("--network"), ["none"])
        self.assertEqual(values("--cap-drop"), ["ALL"])
        self.assertEqual(values("--security-opt"), ["no-new-privileges"])
        self.assertEqual(values("--pull"), ["never"])
        self.assertEqual(values("--entrypoint"), ["python3"])
        self.assertEqual(values("--user"), [f"{os.getuid()}:{os.getgid()}"])
        for forbidden in ("--privileged", "--cap-add", "--ipc", "--pid", "--device"):
            self.assertNotIn(forbidden, options)
        self.assertEqual(values("-v"), [f"{self.root}/hf/hub/{self.CACHE}:/hf/hub/{self.CACHE}:ro",
                                        f"{self.prompts.parent}:/prompts:ro", f"{SCRIPTS}:/opt/inkling2:ro",
                                        f"{self.out.parent}:/out"])
        environment = dict(value.split("=", 1) for value in values("-e"))
        self.assertEqual({key: environment[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME",
                                                            "VLLM_NO_USAGE_STATS", "CUDA_VISIBLE_DEVICES")},
                         {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/hf",
                          "VLLM_NO_USAGE_STATS": "1", "CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(environment["INKLING2_IMAGE"], self.IMAGE)  # recorded in the run identity
        self.assertIsNone(self.started())  # nothing ran, the GPU was not queried, nothing was created
        self.assertFalse((self.root / "nvidia-smi.calls").exists())
        self.assertFalse(self.out.parent.exists())

    def test_hopper_patches_are_mounted_over_the_image_s_files(self):
        patches = self.root / "hopper"
        missing = self.serve("--dry-run", "--hopper-patches", str(patches), "3", str(self.prompts), str(self.out))
        self.assertEqual(missing.returncode, 2)
        self.assertIn("conv_mix.py is missing", missing.stderr)
        patches.mkdir()
        for name in ("conv_mix.py", "moe_align_block_size.py", "_custom_ops.py"):
            (patches / name).write_text("")
        done = self.serve("--dry-run", "--hopper-patches", str(patches), "3", str(self.prompts), str(self.out))
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        argv = shlex.split(done.stdout)
        site = "/usr/local/lib/python3.14/dist-packages/vllm"
        self.assertIn(f"{patches}/conv_mix.py:{site}/models/inkling2/nvidia/ops/conv_mix.py:ro", argv)
        self.assertIn(f"{patches}/moe_align_block_size.py:"
                      f"{site}/model_executor/layers/fused_moe/moe_align_block_size.py:ro", argv)
        self.assertIn(f"{patches}/_custom_ops.py:{site}/_custom_ops.py:ro", argv)
        label = next(value for value in argv if value.startswith("INKLING2_IMAGE="))
        self.assertIn(self.IMAGE, label)
        self.assertIn("Hopper patches", label)  # the run identity differs from an unpatched image's

    def test_a_gpu_with_a_compute_process_is_refused(self):
        (self.root / "compute-apps").write_text("584705\n584702\n")
        done = self.serve("3", str(self.prompts), str(self.out))
        self.assertEqual(done.returncode, 3)
        self.assertIn("GPU 3 is busy (compute processes: 584705 584702)", done.stderr)
        self.assertIsNone(self.started())
        self.assertEqual((self.root / "nvidia-smi.calls").read_text().split(),
                         ["-i", "3", "--query-compute-apps=pid", "--format=csv,noheader"])

    def test_a_failing_gpu_query_is_refused(self):
        (self.root / "nvidia-smi.status").write_text("9")
        done = self.serve("3", str(self.prompts), str(self.out))
        self.assertEqual(done.returncode, 3)
        self.assertIn("could not query GPU 3", done.stderr)
        self.assertIsNone(self.started())

    def test_an_idle_gpu_starts_exactly_the_dry_run_command(self):
        dry = self.serve("--dry-run", "--cpuset", "24-47", "5", str(self.prompts), str(self.out), "--limit", "2")
        done = self.serve("--cpuset", "24-47", "5", str(self.prompts), str(self.out), "--limit", "2")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        self.assertEqual(["docker", *self.started()], shlex.split(dry.stdout))
        self.assertIn("--cpuset-cpus", self.started())
        self.assertTrue(self.out.parent.is_dir())  # created for the writable mount

    def test_the_default_checkpoint_is_the_registry_s_pin(self):
        """The script cannot read configs/models.json from the shell, so its copy of the pin is held to it."""
        pin = json.loads((ROOT / "configs" / "models.json").read_text())["models"]["inkling-2-mini"]
        argv = shlex.split(self.serve("--dry-run", "0", str(self.prompts), str(self.out)).stdout)
        cache = "models--" + pin["repo_id"].replace("/", "--")
        self.assertEqual(argv[argv.index("--model") + 1], f"/hf/hub/{cache}/snapshots/{pin['revision']}")
        self.assertEqual((cache, pin["revision"]), (self.CACHE, self.REVISION))

    def test_another_checkpoint_is_mounted_by_its_cache_directory(self):
        done = self.serve("--dry-run", "--repo", "thinkingmachines/TML-d24-varloop-research", "--revision", "657b",
                          "--hf-home", "/data/hf", "0", str(self.prompts), str(self.out))
        argv = shlex.split(done.stdout)
        cache = "models--thinkingmachines--TML-d24-varloop-research"
        self.assertIn(f"/data/hf/hub/{cache}:/hf/hub/{cache}:ro", argv)
        self.assertEqual(argv[argv.index("--model") + 1], f"/hf/hub/{cache}/snapshots/657b")

    def test_bad_arguments_start_nothing(self):
        for args, message in ((("0,1", str(self.prompts), str(self.out)), "one device index"),
                              (("all", str(self.prompts), str(self.out)), "one device index"),
                              (("3", str(self.root / "missing.jsonl"), str(self.out)), "does not exist"),
                              (("3", str(self.prompts)), "are required"),
                              (("--revision", "0000", "3", str(self.prompts), str(self.out)), "no snapshot 0000"),
                              (("--gpus", "all", "3", str(self.prompts), str(self.out)), "unknown option"),
                              (("--dry-run", "--repo"), "--repo needs a value"),
                              (("--hf-home", "", "3", str(self.prompts), str(self.out)), "--hf-home needs a value")):
            with self.subTest(args=args):
                done = self.serve(*args)
                self.assertEqual(done.returncode, 2)
                self.assertIn(message, done.stderr)
                self.assertIsNone(self.started())


class ServeRollout(Containers):
    """serve_rollout.sh: the trace servers of training.drafter.traces_remote."""

    def rollout(self, *args):
        return subprocess.run(["bash", str(SCRIPTS / "serve_rollout.sh"), *args], capture_output=True, text=True,
                              env=self.env)

    def test_dry_run_serves_the_pinned_image_under_the_name_and_budget_the_clients_expect(self):
        done = self.rollout("--dry-run", "2", "8102")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        argv = shlex.split(done.stdout)
        self.assertIn(self.IMAGE, argv)
        self.assertEqual(argv[argv.index("--served-model-name") + 1], "inkling-2-mini")
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "16384")
        self.assertEqual((argv[argv.index("--host") + 1], argv[argv.index("--port") + 1]), ("127.0.0.1", "8102"))
        self.assertEqual(argv[argv.index("--gpus") + 1], "device=2")
        self.assertFalse([value for value in argv if "dist-packages" in value])  # no patch without the option
        self.assertIsNone(self.started())

    def test_hopper_patches_are_mounted_over_the_image_s_files(self):
        patches = self.root / "hopper"
        missing = self.rollout("--dry-run", "--hopper-patches", str(patches), "2", "8102")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("conv_mix.py is missing", missing.stderr)
        patches.mkdir()
        for name in ("conv_mix.py", "moe_align_block_size.py", "_custom_ops.py"):
            (patches / name).write_text("")
        done = self.rollout("--dry-run", "--hopper-patches", str(patches), "2", "8102")
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        # The files and places of serve_official.sh --hopper-patches, which writes them.
        official = shlex.split(self.serve("--dry-run", "--hopper-patches", str(patches), "3", str(self.prompts),
                                          str(self.out)).stdout)
        mounts = [value for value in official if value.startswith(f"{patches}/")]
        self.assertEqual(len(mounts), 3)
        self.assertEqual([value for value in shlex.split(done.stdout) if value.startswith(f"{patches}/")], mounts)


if __name__ == "__main__":
    unittest.main()
