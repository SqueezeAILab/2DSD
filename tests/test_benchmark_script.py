"""scripts/benchmark.py on CPU: pure helpers, registry names, and resuming --out with fake vLLM engines."""

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_script", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
MODELS = json.loads((ROOT / "configs" / "models.json").read_text())["models"]


class Helpers(unittest.TestCase):
    def test_request_seed_is_the_paper_rule(self):
        # sha256("0:gsm8k/main/test/row:0:0")[:4], little endian.
        self.assertEqual(benchmark.request_seed(0, "gsm8k/main/test/row:0"), 1523373996)
        self.assertNotEqual(benchmark.request_seed(0, "a"), benchmark.request_seed(1, "a"))

    def test_repetition_needs_a_capped_response(self):
        loop = [1, 2, 3, 4] * 1000
        self.assertTrue(benchmark.repetitive(loop, capped=True))
        self.assertFalse(benchmark.repetitive(loop, capped=False))
        self.assertFalse(benchmark.repetitive(list(range(4000)), capped=True))

    def test_capped_means_budget_without_eos(self):
        self.assertTrue(benchmark.capped({"output_ids": [1] * 8, "max_new": 8, "stopped_by_eos": False}))
        self.assertFalse(benchmark.capped({"output_ids": [1] * 8, "max_new": 8, "stopped_by_eos": True}))
        self.assertFalse(benchmark.capped({"output_ids": [1] * 5, "max_new": 8, "stopped_by_eos": False}))

    def test_summary_pools_tokens_over_decode_time(self):
        rows = [{"id": "a", "benchmark": "x", "output_ids": [0] * 11, "decode_seconds": 1.0, "max_new": 99,
                 "stopped_by_eos": True},
                {"id": "b", "benchmark": "x", "output_ids": [0] * 31, "decode_seconds": 3.0, "max_new": 99,
                 "stopped_by_eos": True}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"torn')
            self.assertEqual(len(benchmark.read_jsonl(path)), 2)
            out = io.StringIO()
            with redirect_stdout(out):
                benchmark.summarize([path])
        line = [l for l in out.getvalue().splitlines() if " all " in l][0]
        self.assertIn("10.0", line)  # (10 + 30) tokens / (1 + 3) s

    def test_summary_reports_aime24_and_aime25_together(self):
        rows = [{"id": "a", "benchmark": "aime24", "output_ids": [0] * 11, "decode_seconds": 1.0, "max_new": 99,
                 "stopped_by_eos": True},
                {"id": "b", "benchmark": "aime25", "output_ids": [0] * 31, "decode_seconds": 1.0, "max_new": 99,
                 "stopped_by_eos": True}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with redirect_stdout(out):
                benchmark.summarize([path])
        lines = out.getvalue().splitlines()
        self.assertFalse(any(" aime24 " in l or " aime25 " in l for l in lines))
        line = [l for l in lines if " aime24/25 " in l][0]
        self.assertIn("    2 ", line)
        self.assertIn("20.0", line)  # (10 + 30) tokens / (1 + 1) s

    def test_prepare_append_drops_a_torn_tail_and_ends_the_last_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            for before, after in (('{"a": 1}\n{"a": 2}', '{"a": 1}\n{"a": 2}\n'),
                                  ('{"a": 1}\n{"a": 2', '{"a": 1}\n'),
                                  ('{"a', ''),
                                  ('{"a": 1}\n', '{"a": 1}\n')):
                path.write_text(before)
                benchmark.prepare_append(path)
                self.assertEqual(path.read_text(), after)

    def test_a_malformed_line_before_the_last_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
            with self.assertRaisesRegex(ValueError, "line 2 is not a valid JSON record"):
                benchmark.read_jsonl(path)
            with self.assertRaisesRegex(ValueError, "line 2"):
                benchmark.prepare_append(path)
            self.assertEqual(path.read_text(), '{"a": 1}\n{"a": \n{"a": 3}\n')


class FakeEngines:
    """vllm modules for scripts/benchmark.py: TwoDSD journals each result with its metadata, as the worker
    extension does, and LLM returns the whole budget; generated records each engine call (ids for TwoDSD,
    the number of prompts for LLM)."""

    def __init__(self, model_type="ouro"):
        self.generated, self.constructed, self.model_types = [], [], []
        engines = self

        class TwoDSD:
            def __init__(self, model, drafter, **options):
                engines.constructed.append({"model": model, "drafter": drafter, "revision": options["revision"],
                                            "kv_policy": options["kv_policy"]})

            def generate(self, prompts, *, max_new_tokens, metadata, journal_path, **options):
                engines.generated.append([m["id"] for m in metadata])
                with open(journal_path, "a") as journal:
                    for budget, extra in zip(max_new_tokens, metadata):
                        result = {"output_ids": [5] * budget, "prefill_seconds": 0.1, "decode_seconds": 1.0,
                                  "max_new": budget, "stopped_by_eos": False, "stats": {"calls": 1}}
                        result.update(extra)
                        journal.write(json.dumps(result) + "\n")

        class LLM:
            def __init__(self, **options):
                engines.constructed.append({"model": options["model"], "revision": options["revision"],
                                            "kv_policy": options["kv_policy"]})
                self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))

            def get_tokenizer(self):
                return SimpleNamespace(eos_token_id=2)

            def generate(self, prompts, params, use_tqdm):
                engines.generated.append(len(prompts))
                metrics = SimpleNamespace(scheduled_ts=0.0, first_token_ts=0.1, last_token_ts=1.1)
                return [SimpleNamespace(metrics=metrics, outputs=[SimpleNamespace(token_ids=[5] * params.max_tokens)])]

        twodsd = types.ModuleType("vllm.v1.spec_decode.twodsd")
        twodsd.TwoDSD = TwoDSD

        def fake_model_type(model, revision=None):
            engines.model_types.append(model)
            return model_type

        twodsd.model_type = fake_model_type
        twodsd.engine_options = lambda model, revision, kv_policy: {"kv_policy": kv_policy}
        vllm = types.ModuleType("vllm")
        vllm.LLM, vllm.SamplingParams = LLM, lambda **params: SimpleNamespace(**params)
        self.modules = {"vllm": vllm, "vllm.v1.spec_decode.twodsd": twodsd}


class BenchmarkRun(unittest.TestCase):
    """A prompts file of four questions, and scripts/benchmark.py runs on fake vLLM engines."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.prompts = self.root / "core.prompts.jsonl"
        self.prompts.write_text("".join(json.dumps({"id": f"q{i}", "benchmark": "x", "prompt_ids": [i, 1, 2],
                                                    "max_total_tokens": 8}) + "\n" for i in range(4)))
        self.out = self.root / "runs" / "core.jsonl"
        self.engines = FakeEngines()

    def run_benchmark(self, *extra, method="2dsd", out=None, model="m", drafter="d"):
        argv = ["--method", method, "--model", model, "--prompts", str(self.prompts), "--out", str(out or self.out),
                "--warmup", "0", *(["--drafter", drafter] if method == "2dsd" and drafter else []), *extra]
        self.stderr = io.StringIO()
        with (patch.dict(sys.modules, self.engines.modules), patch.dict(os.environ),
              patch.object(benchmark, "eos_ids", return_value=[2]),
              redirect_stdout(io.StringIO()), redirect_stderr(self.stderr)):
            benchmark.main(argv)

    def lines(self, path=None):
        return [json.loads(line) for line in (path or self.out).read_text().splitlines()]


class Resume(BenchmarkRun):
    def test_resume_after_a_torn_tail_reads_every_record(self):
        self.run_benchmark("--limit", "2")
        with self.out.open("a") as handle:
            handle.write('{"output_ids": [5, 5')
        self.run_benchmark()
        rows = self.lines()  # every line is a whole record
        self.assertEqual(sorted(r["id"] for r in rows), ["q0", "q1", "q2", "q3"])
        self.assertEqual(len(benchmark.read_jsonl(self.out)), 4)
        self.assertEqual([len(ids) for ids in self.engines.generated], [2, 2])

    def test_resume_after_a_record_without_newline(self):
        self.run_benchmark("--limit", "1")
        self.out.write_text(self.out.read_text().rstrip("\n"))
        self.run_benchmark("--limit", "3")
        self.assertEqual(len(self.lines()), 3)

    def test_a_malformed_middle_line_stops_the_resume(self):
        self.run_benchmark("--limit", "2")
        first, second = self.out.read_text().splitlines()
        self.out.write_text(first[:-1] + "\n" + second + "\n")
        with self.assertRaisesRegex(ValueError, "line 1 is not a valid JSON record"):
            self.run_benchmark()
        self.assertEqual(len(self.engines.generated), 1)

    def test_another_run_identity_is_rejected(self):
        self.run_benchmark("--limit", "2")
        before = self.out.read_bytes()
        for extra, method, field in ((("--temperature", "0.6"), "2dsd", "temperature"),
                                     (("--block-size", "8"), "2dsd", "block_size"),
                                     (("--model", "other"), "2dsd", "model"),
                                     ((), "ar", "method")):
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.run_benchmark(*extra, method=method)
            self.assertIn(field, self.stderr.getvalue())
            self.assertIn("choose a new --out", self.stderr.getvalue())
        with self.prompts.open("a") as handle:  # another prompts file
            handle.write(json.dumps({"id": "q4", "benchmark": "x", "prompt_ids": [4, 1, 2],
                                     "max_total_tokens": 8}) + "\n")
        with self.assertRaises(SystemExit):
            self.run_benchmark()
        self.assertIn("prompts_sha256", self.stderr.getvalue())
        self.assertEqual(self.out.read_bytes(), before)
        self.assertEqual(len(self.engines.generated), 1)

    def test_rows_without_a_run_identity_are_rejected(self):
        self.out.parent.mkdir()
        self.out.write_text(json.dumps({"id": "q0", "benchmark": "x", "output_ids": [5], "decode_seconds": 1.0,
                                        "max_new": 5, "stopped_by_eos": True}) + "\n")
        with self.assertRaises(SystemExit):
            self.run_benchmark()
        self.assertIn("rows without a run identity", self.stderr.getvalue())

    def test_the_same_run_with_a_larger_limit_resumes(self):
        self.run_benchmark("--limit", "2")
        self.run_benchmark("--limit", "4", "--warmup", "3", "--gpu-memory-utilization", "0.7")
        first, second = self.engines.generated
        self.assertEqual(sorted(first + second), ["q0", "q1", "q2", "q3"])
        self.assertEqual(len(second), 2)
        self.assertEqual(len(self.lines()), 4)
        self.run_benchmark()  # nothing pending
        self.assertEqual(len(self.engines.generated), 2)

    def test_another_spelling_of_the_same_local_checkpoint_resumes(self):
        checkpoint = self.root / "ckpt"
        checkpoint.mkdir()
        self.run_benchmark("--limit", "2", model=f"{checkpoint}/")
        self.run_benchmark(model=str(self.root / "ckpt" / ".." / "ckpt"))
        self.assertEqual(len(self.lines()), 4)
        self.assertEqual({row["run_identity"]["model"] for row in self.lines()}, {str(checkpoint.resolve())})

    def test_ar_and_2dsd_rows_carry_the_run_identity(self):
        sha = hashlib.sha256(self.prompts.read_bytes()).hexdigest()
        for method, drafter in (("2dsd", "d"), ("ar", None)):
            out = self.root / f"{method}.jsonl"
            self.run_benchmark("--temperature", "0.6", "--top-p", "0.95", "--seed", "3", method=method, out=out)
            expected = {"model": "m", "revision": None, "drafter": drafter, "method": method, "temperature": 0.6,
                        "top_p": 0.95, "seed": 3, "stage": None, "block_size": None, "d1": None, "d2": None,
                        "kv_policy": None, "prompts_sha256": sha}
            rows = self.lines(out)
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["run_identity"], expected)
                self.assertEqual((row["method"], row["temperature"], row["top_p"]), (method, 0.6, 0.95))
                self.assertEqual(row["prompt_tokens"], 3)


class RegistryNames(BenchmarkRun):
    """--model as a registry name of configs/models.json, with --target and explicit overrides."""

    def identity(self, out=None):
        rows = self.lines(out)
        identity = rows[0]["run_identity"]
        self.assertTrue(all(row["run_identity"] == identity for row in rows))
        return {key: identity[key] for key in ("model", "revision", "drafter", "kv_policy")}

    def expect(self, model, target, method="2dsd"):
        """The resolved model, revision, drafter and kv_policy of a registry name: the base model at its
        pinned revision, or the shared-KV target, a plain repo id (revision None)."""
        entry = MODELS[model]
        repo, revision = ((entry["repo_id"], entry["revision"]) if target == "original" or entry["w64"] is None
                          else (entry["w64"], None))
        drafter = entry["drafters"][target] if method == "2dsd" else None
        kv_policy = {"w64": "w64", "original": "native"}[target] if entry["family"] == "dory" else None
        return {"model": repo, "revision": revision, "drafter": drafter, "kv_policy": kv_policy}

    def test_ouro_targets_resolve_to_their_pins(self):
        for target, extra in (("w64", ()), ("w64", ("--target", "w64")), ("original", ("--target", "original"))):
            for method in ("2dsd", "ar"):
                with self.subTest(target=target, extra=extra, method=method):
                    out = self.root / f"{method}-{target}-{len(extra)}.jsonl"
                    self.run_benchmark(*extra, method=method, model="ouro-1.4b-thinking", drafter=None, out=out)
                    expected = self.expect("ouro-1.4b-thinking", target, method)
                    self.assertEqual(self.identity(out), expected)
                    built = dict(expected)
                    if method == "ar":
                        del built["drafter"]
                    self.assertEqual(self.engines.constructed[-1], built)
        self.assertEqual(self.engines.model_types, [])  # the registry gives the family
        self.assertEqual(self.expect("ouro-1.4b-thinking", "w64")["model"], "minseo25/2dsd-ouro-1.4b-thinking-shared-kv")
        self.assertEqual(self.expect("ouro-1.4b-thinking", "original")["model"], "ByteDance/Ouro-1.4B-Thinking")

    def test_dory_runs_its_pinned_checkpoint_with_the_target_kv_policy(self):
        entry = MODELS["dory1.0-4b"]
        for target, kv_policy in (("w64", "w64"), ("original", "native")):
            for method in ("2dsd", "ar"):
                with self.subTest(target=target, method=method):
                    out = self.root / f"dory-{method}-{target}.jsonl"
                    self.run_benchmark("--target", target, method=method, model="dory1.0-4b", drafter=None, out=out)
                    drafter = entry["drafters"]["w64"] if method == "2dsd" else None
                    self.assertEqual(self.identity(out), {"model": entry["repo_id"], "revision": entry["revision"],
                                                          "drafter": drafter, "kv_policy": kv_policy})
                    self.assertEqual(self.engines.constructed[-1]["kv_policy"], kv_policy)
                    self.assertEqual(self.lines(out)[0]["kv_policy"], kv_policy)
        self.assertEqual(entry["drafters"]["w64"], entry["drafters"]["original"])  # one drafter, both targets
        self.assertEqual(self.engines.model_types, [])

    def test_block_size_one_runs_no_drafter(self):
        self.run_benchmark("--block-size", "1", model="nanbeige4.2-3b", drafter=None)
        self.assertEqual(self.identity(), dict(self.expect("nanbeige4.2-3b", "w64"), drafter=None))
        self.assertIsNone(self.engines.constructed[-1]["drafter"])
        # A drafter that does not run is not part of the run: the same run resumes with or without it.
        self.prompts.write_text(self.prompts.read_text() + json.dumps(
            {"id": "q4", "benchmark": "x", "prompt_ids": [4, 1, 2], "max_total_tokens": 8}) + "\n")
        for extra, method, out in ((("--block-size", "1"), "2dsd", "block1.jsonl"), ((), "ar", "ar.jsonl")):
            with self.subTest(method=method):
                out = self.root / out
                self.run_benchmark(*extra, "--limit", "2", method=method, model="nanbeige4.2-3b", drafter=None,
                                   out=out)
                self.run_benchmark(*extra, "--drafter", "me/drafter@1", method=method, model="nanbeige4.2-3b",
                                   drafter=None, out=out)
                self.assertEqual(len(self.lines(out)), 5)
                self.assertIsNone(self.identity(out)["drafter"])
        self.assertIsNone(self.engines.constructed[-1].get("drafter"))

    def test_explicit_flags_override_the_registry(self):
        expected = self.expect("ouro-2.6b-thinking", "w64")
        for extra, changed in ((("--revision", "abc"), {"revision": "abc"}),
                               (("--drafter", "me/drafter@1"), {"drafter": "me/drafter@1"}),
                               (("--target", "original", "--revision", "abc"),
                                dict(self.expect("ouro-2.6b-thinking", "original"), revision="abc"))):
            with self.subTest(extra=extra):
                out = self.root / f"override-{len(self.engines.constructed)}.jsonl"
                self.run_benchmark(*extra, model="ouro-2.6b-thinking", drafter=None, out=out)
                self.assertEqual(self.identity(out), dict(expected, **changed))
        out = self.root / "dory-override.jsonl"
        self.run_benchmark("--kv-policy", "native", model="dory1.0-4b", drafter=None, out=out)
        self.assertEqual(self.identity(out), dict(self.expect("dory1.0-4b", "original")))

    def test_a_kv_policy_for_a_model_other_than_dory_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_benchmark("--kv-policy", "w64", model="ouro-1.4b-thinking", drafter=None)
        self.assertIn("applies to Dory only", self.stderr.getvalue())
        self.assertEqual(self.engines.constructed, [])

    def test_a_target_for_a_model_outside_the_registry_is_rejected(self):
        for model in ("m", "ByteDance/Ouro-1.4B-Thinking"):
            with self.subTest(model=model), self.assertRaises(SystemExit):
                self.run_benchmark("--target", "w64", model=model)
            self.assertIn("--target applies to registry names", self.stderr.getvalue())
        self.assertEqual(self.engines.constructed, [])
        self.assertFalse(self.out.exists())

    def test_a_model_outside_the_registry_runs_as_given(self):
        self.run_benchmark("--revision", "r1", model="org/looped")
        self.assertEqual(self.identity(), {"model": "org/looped", "revision": "r1", "drafter": "d", "kv_policy": None})
        self.assertEqual(self.engines.constructed[-1], self.identity())
        self.assertEqual(self.engines.model_types, ["org/looped"])
        with self.assertRaises(SystemExit):  # no drafter outside the registry
            self.run_benchmark(model="org/looped", drafter=None, out=self.root / "no-drafter.jsonl")
        self.assertIn("2DSD needs --drafter", self.stderr.getvalue())

    def test_a_dory_checkpoint_outside_the_registry_needs_a_kv_policy(self):
        self.engines = FakeEngines(model_type="dory")
        with self.assertRaises(SystemExit):
            self.run_benchmark(model="local/dory")
        self.assertIn("--kv-policy is required for Dory", self.stderr.getvalue())
        self.run_benchmark("--kv-policy", "native", method="ar", model="local/dory")
        self.assertEqual(self.identity()["kv_policy"], "native")

    def test_a_run_given_by_explicit_pins_resumes_by_name(self):
        for name, target, model_type in (("ouro-2.6b-thinking", "w64", "ouro"),
                                         ("ouro-2.6b-thinking", "original", "ouro"),
                                         ("dory1.0-4b", "w64", "dory"), ("dory1.0-4b", "original", "dory")):
            for method in ("2dsd", "ar"):
                with self.subTest(name=name, target=target, method=method):
                    self.engines = FakeEngines(model_type=model_type)
                    expected = self.expect(name, target, method)
                    explicit = ["--revision", expected["revision"]] if expected["revision"] else []
                    if expected["kv_policy"]:
                        explicit += ["--kv-policy", expected["kv_policy"]]
                    out = self.root / f"explicit-{name}-{target}-{method}.jsonl"
                    self.run_benchmark("--limit", "2", *explicit, method=method, model=expected["model"],
                                       drafter=expected["drafter"], out=out)
                    self.assertEqual(self.engines.model_types, [expected["model"]])  # family from config.json
                    self.run_benchmark("--target", target, method=method, model=name, drafter=None, out=out)
                    self.assertEqual([len(ids) if method == "2dsd" else ids for ids in self.engines.generated],
                                     [2, 2] if method == "2dsd" else [1, 1, 1, 1])
                    self.assertEqual(len(self.lines(out)), 4)
                    self.assertEqual(self.identity(out), expected)

    def test_a_run_given_by_name_resumes_with_the_explicit_pins(self):
        expected = self.expect("ouro-1.4b-thinking", "w64")
        self.run_benchmark("--limit", "2", model="ouro-1.4b-thinking", drafter=None)
        self.assertIsNone(expected["revision"])
        self.run_benchmark("--limit", "3", model=expected["model"], drafter=expected["drafter"])
        self.run_benchmark("--target", "w64", model="ouro-1.4b-thinking", drafter=None)
        self.assertEqual([len(ids) for ids in self.engines.generated], [2, 1, 1])
        self.assertEqual(self.identity(), expected)
        with self.assertRaises(SystemExit):  # the original target is another run
            self.run_benchmark("--target", "original", model="ouro-1.4b-thinking", drafter=None)
        self.assertIn("different drafter, model, revision", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
