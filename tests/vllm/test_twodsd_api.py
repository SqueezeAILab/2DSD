"""The 2DSD Python API builds requests and journals worker results correctly (CPU)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import vllm.v1.spec_decode.inkling2_2d as inkling2
import vllm.v1.spec_decode.looped_2d_finalized as finalized
from vllm.v1.spec_decode.twodsd import (INKLING2_BLOCK_SIZE, TwoDSD, TwoDSDWorkerExtension, dory_kv_policy,
                                        check_block_size, engine_options, inkling2_kv_window, inkling2_schedule, preset)

# text_config.loop_config of Inkling-2-Mini's config.json.
MINI_LOOP = {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}
SUPPORTED = "ouro, nanbeige, dory, inkling2_loop"


def fake_run(worker, requests, *, warmup, result_callback, **options):
    """Echo each executed request as a result, ending with EOS on even prompts."""
    worker.executed = requests
    worker.options = options
    for request in requests:
        output = list(range(request["max_new"]))
        if request["prompt_ids"][0] % 2 == 0:
            output[-1] = request["eos_token_ids"][0]
        result_callback({"output_ids": output, "prefill_seconds": 0.1, "decode_seconds": 1.0,
                         "stats": {"per_call": [1, 2], "calls": 3}})


class Worker(TwoDSDWorkerExtension):
    def __init__(self, model_type="ouro", **fields):
        config = SimpleNamespace(model_type=model_type, **fields)
        self.model_runner = SimpleNamespace(get_model=lambda: SimpleNamespace(config=config))


class WorkerExtension(unittest.TestCase):
    def requests(self):
        return [
            {"prompt_ids": [2, 5], "max_new": 4, "eos_token_ids": [99], "metadata": {"id": "a", "benchmark": "x"}},
            {"prompt_ids": [3, 5], "max_new": 6, "eos_token_ids": [99], "metadata": {"id": "b", "benchmark": "y"}},
        ]

    def test_warmups_are_dropped_and_metadata_is_copied(self):
        worker = Worker()
        with patch.object(finalized, "run_ouro_finalized", fake_run):
            results = worker.run_2dsd(self.requests(), {"k": 16}, warmup_requests=2, warmup_max_new=3)
        self.assertEqual(len(worker.executed), 4)
        self.assertEqual([r["max_new"] for r in worker.executed], [3, 3, 4, 6])
        self.assertTrue(all("metadata" not in r for r in worker.executed))
        self.assertEqual([r["id"] for r in results], ["a", "b"])
        self.assertEqual([r["benchmark"] for r in results], ["x", "y"])
        self.assertEqual([r["stopped_by_eos"] for r in results], [True, False])
        self.assertEqual([r["max_new"] for r in results], [4, 6])
        self.assertNotIn("per_call", results[0]["stats"])

    def test_journal_receives_every_result(self):
        worker = Worker()
        with tempfile.TemporaryDirectory() as directory, patch.object(finalized, "run_ouro_finalized", fake_run):
            path = Path(directory) / "out.jsonl"
            reply = worker.run_2dsd(self.requests(), {"k": 16}, warmup_requests=1, journal_path=str(path))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(reply, {"requests": 2})
        self.assertEqual([r["id"] for r in rows], ["a", "b"])

    def test_nanbeige_drops_second_readout_options(self):
        import vllm.v1.spec_decode.nanbeige_2d as nanbeige

        worker = Worker("nanbeige")
        with patch.object(nanbeige, "run_nanbeige_finalized", fake_run):
            worker.run_2dsd(self.requests(), {"k": 16, "d2": 2, "stage2_gate": "argmax"}, warmup_requests=0)
        self.assertNotIn("d2", worker.options)
        self.assertNotIn("stage2_gate", worker.options)

    def test_dory_runs_its_runner_with_compiled_blocks(self):
        import vllm.v1.spec_decode.dory_2d as dory

        worker = Worker("dory")
        with patch.object(dory, "run_dory_finalized", fake_run):
            worker.run_2dsd(self.requests(), {"k": 16, "d2": 2, "stage2_gate": "argmax"}, warmup_requests=0)
        self.assertTrue(worker.options["compile_blocks"])
        self.assertEqual(worker.options["d2"], 2)

    def test_inkling2_resolves_three_passes_and_runs_its_own_runner(self):
        """Guards the dispatch: an inkling2_loop worker reads R = 3 from the nested loop config (a dict, as in
        config.json, or config objects), accepts the preset's (1, 2) schedule and runs the Inkling2 entry
        point with it, never the Ouro runner."""
        options = dict(preset("inkling2_loop"), drafter_export="d", max_model_len=8192, kv_window=64)
        expected = {"loops": 3, "stage": "stage2b", "k": 8, "d1": 1, "d2": 2}
        text_configs = {"dicts": {"loop_config": dict(MINI_LOOP)},
                        "objects": SimpleNamespace(loop_config=SimpleNamespace(**MINI_LOOP))}
        for name, text_config in text_configs.items():
            worker = Worker("inkling2_loop", text_config=text_config)
            with self.subTest(text_config=name):
                self.assertEqual(inkling2_schedule(worker.model_runner.get_model().config, options), expected)
                with (patch.object(finalized, "run_ouro_finalized", side_effect=AssertionError("Ouro runner")),
                      patch.object(inkling2, "run_inkling2_finalized", fake_run)):
                    results = worker.run_2dsd(self.requests(), options, warmup_requests=0)
                self.assertEqual([r["id"] for r in results], ["a", "b"])
                self.assertEqual(worker.options, dict(options, loops=3))  # the schedule and R as resolved

    def test_inkling2_schedule_must_fit_the_passes_of_its_config(self):
        """Guards against an assumed R = 4: (2, 3) is a Stage 2 schedule at four passes but not at Mini's
        three, and a config without the nested count is an error rather than a default."""
        options = dict(preset("inkling2_loop"), d1=2, d2=3)

        def run(worker, **changes):
            with patch.object(inkling2, "run_inkling2_finalized", fake_run):
                worker.run_2dsd(self.requests(), dict(options, **changes), warmup_requests=0)
            return worker.options

        three = Worker("inkling2_loop", text_config={"loop_config": dict(MINI_LOOP)})
        with self.assertRaisesRegex(ValueError, r"d1 < d2 < R \(R = 3\)"):
            run(three)
        four = Worker("inkling2_loop", text_config={"loop_config": dict(MINI_LOOP, num_passes=4)})
        self.assertEqual(run(four), dict(options, loops=4))
        # At two passes there is no Stage 2 (as on Nanbeige): the preset is rejected, Stage 1 is not.
        two = Worker("inkling2_loop", text_config={"loop_config": dict(MINI_LOOP, num_passes=2)})
        with self.assertRaisesRegex(ValueError, r"d1 < d2 < R \(R = 2\)"):
            run(two, d1=1, d2=2)
        self.assertEqual(run(two, stage="stage1", d1=1)["stage"], "stage1")
        for fields in ({}, {"text_config": {}}, {"text_config": {"loop_config": {}}, "total_ut_steps": 4}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "loop_config.num_passes"):
                run(Worker("inkling2_loop", **fields))

    def test_an_unknown_model_type_does_not_run_as_ouro(self):
        """Guards the old fall-through: any model_type without a runner (a dense model, Inkling 1, the
        non-looped Inkling 2) raises with the supported types and never reaches the Ouro runner."""
        for model_type in ("qwen3", "inkling_mm_model", "inkling2", None):
            worker = Worker(model_type)
            with (self.subTest(model_type=model_type), patch.object(finalized, "run_ouro_finalized", fake_run),
                  self.assertRaisesRegex(ValueError, SUPPORTED)):
                worker.run_2dsd(self.requests(), {"k": 16}, warmup_requests=0)
            self.assertFalse(hasattr(worker, "executed"))


class GenerateRequests(unittest.TestCase):
    def engine(self):
        engine = TwoDSD.__new__(TwoDSD)
        engine.options = dict(preset("ouro"), drafter_export="d", max_model_len=32, initial_capacity=256,
                              stage2_gate="argmax")
        engine.tokenizer = SimpleNamespace(encode=lambda text: [7] * len(text))
        engine.eos_token_ids = lambda: [99]
        calls = []
        engine.llm = SimpleNamespace(collective_rpc=lambda name, timeout, kwargs: calls.append(kwargs) or [[]])
        return engine, calls

    def test_metadata_budgets_and_greedy_options(self):
        engine, calls = self.engine()
        engine.generate([[1], [2, 3, 4]], metadata=[{"id": "q1"}, {"id": "q2"}], warmup_requests=2)
        [kwargs] = calls
        self.assertEqual([r["prompt_ids"] for r in kwargs["requests"]], [[1], [2, 3, 4]])
        self.assertEqual([r["max_new"] for r in kwargs["requests"]], [31, 29])
        self.assertEqual([r["metadata"]["id"] for r in kwargs["requests"]], ["q1", "q2"])
        self.assertTrue(all("seed" not in r for r in kwargs["requests"]))
        self.assertNotIn("temperature", kwargs["options"])
        self.assertEqual(kwargs["warmup_requests"], 2)

    def test_sampling_uses_seeds_and_uncut_proposals(self):
        engine, calls = self.engine()
        engine.generate(["ab"], max_new_tokens=[5], temperature=0.6, top_p=0.95, seeds=[123])
        [kwargs] = calls
        [request] = kwargs["requests"]
        self.assertEqual((request["prompt_ids"], request["max_new"], request["seed"]), ([7, 7], 5, 123))
        self.assertEqual(
            {k: kwargs["options"][k] for k in ("temperature", "top_p", "proposal_top_p")},
            {"temperature": 0.6, "top_p": 0.95, "proposal_top_p": 1.0},
        )

    def test_inkling2_requests_end_at_its_end_of_sampling_token_only(self):
        """The Inkling2 EOS list is fixed to [200006]: no config, generation config or tokenizer is read
        (this engine has none), so <|end_message|> between its blocks can never stop a request."""
        engine, calls = self.engine()
        del engine.eos_token_ids  # the class method again
        engine.model_type = "inkling2_loop"
        engine.generate([[1, 2]])
        self.assertEqual(calls[0]["requests"][0]["eos_token_ids"], [200006])


class EngineOptions(unittest.TestCase):
    """Model-specific engine settings: KV overrides and the paper's compilation setting."""

    def checkpoint(self, config):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "config.json").write_text(json.dumps(config))
        return directory.name

    def test_dory_named_policies_set_four_loops_and_compile(self):
        dory = self.checkpoint({"model_type": "dory"})
        shared = engine_options(dory, None, None)
        self.assertEqual(shared["hf_overrides"], dory_kv_policy("w64"))
        self.assertEqual(shared["hf_overrides"]["n_recurrent_loops"], 4)
        self.assertEqual(shared["hf_overrides"]["loopedtf_kv_share_layers"], "full_attention")
        self.assertEqual(shared["compilation_config"]["mode"], 3)
        self.assertFalse(shared["disable_hybrid_kv_cache_manager"])
        native = engine_options(dory, None, "native")
        self.assertEqual(native["hf_overrides"]["loopedtf_kv_policy"], "full")
        self.assertNotIn("loopedtf_kv_window", native["hf_overrides"])
        self.assertNotIn("disable_hybrid_kv_cache_manager", native)
        with self.assertRaisesRegex(ValueError, "w64"):
            dory_kv_policy("w1")

    def test_dory_key_overrides_keep_the_named_policy(self):
        dory = self.checkpoint({"model_type": "dory"})
        options = engine_options(dory, None, {"loopedtf_kv_policy": "shared_last", "loopedtf_kv_window": 32})
        self.assertEqual(options["hf_overrides"]["n_recurrent_loops"], 4)
        self.assertEqual(options["hf_overrides"]["loopedtf_kv_window"], 32)
        self.assertEqual(options["hf_overrides"]["recurrent_kv_cache_mode"], "per_loop")

    def test_ouro_runs_graph_only_and_rejects_named_policies(self):
        ouro = self.checkpoint({"model_type": "ouro"})
        self.assertEqual(engine_options(ouro, None, None), {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}})
        forced = engine_options(ouro, None, {"loopedtf_kv_policy": "shared_last", "loopedtf_kv_window": 64})
        self.assertEqual(forced["hf_overrides"]["loopedtf_kv_window"], 64)
        with self.assertRaisesRegex(ValueError, "Dory"):
            engine_options(ouro, None, "w64")

    def test_dory_preset_is_the_r4_preset(self):
        self.assertEqual(preset("dory"), {"stage": "stage2b", "k": 16, "d1": 1, "d2": 2})

    def test_inkling2_preset_is_stage_2b_with_the_bundled_drafter_block(self):
        """R = 3 leaves (1, 2) as the only Stage 2 schedule; the block size is the one named constant."""
        self.assertEqual(INKLING2_BLOCK_SIZE, 8)
        self.assertEqual(preset("inkling2_loop"), {"stage": "stage2b", "k": INKLING2_BLOCK_SIZE, "d1": 1, "d2": 2})
        # The other families keep their presets.
        self.assertEqual(preset("ouro"), {"stage": "stage2b", "k": 16, "d1": 1, "d2": 2})
        self.assertEqual(preset("nanbeige"), {"stage": "stage1", "k": 16, "d1": 1})

    def test_inkling2_keeps_per_loop_kv(self):
        """Inkling2 has no vLLM engine and no KV sharing: the per-loop policy by its names is accepted, and a
        sharing policy is refused instead of running some other rule."""
        inkling = self.checkpoint({"model_type": "inkling2_loop", "text_config": {"loop_config": MINI_LOOP}})
        with self.assertRaisesRegex(ValueError, "without a vLLM engine"):
            engine_options(inkling, None, None)
        for kv_policy in (None, "native", {}, {"loopedtf_kv_policy": "full"}):
            self.assertIsNone(inkling2_kv_window(kv_policy))
        for kv_policy in ("w64", "w32", {"loopedtf_kv_policy": "shared_last"}, {"loopedtf_kv_window": 64},
                          {"loopedtf_kv_policy": "shared_last", "loopedtf_kv_window": 16}):
            with self.subTest(kv_policy=kv_policy), self.assertRaisesRegex(ValueError, "per-loop KV only"):
                inkling2_kv_window(kv_policy)

    def test_inkling2_runs_the_measured_runtime_on_a_gpu_by_default(self):
        """On a GPU the defaults are the static runner on the tensor-core model with drafter graphs, and
        packed experts when a recent Humming can run them (dense ones otherwise, and on the other runners,
        which do not call Humming); tensor cores run the served precision only; a CPU keeps the eager batched
        runner on the reference model; an explicit value replaces its default. The expert backend named for
        a runtime is the one the tree calls of its runner take."""
        from vllm.model_executor.models import inkling2_humming
        from vllm.v1.spec_decode.twodsd import inkling2_expert_backend, inkling2_runtime

        measured = {"runner": "static", "tensor_cores": True, "experts": "packed", "drafter_cuda_graph": True}
        with patch.object(inkling2_humming, "available", return_value=True):
            self.assertEqual(inkling2_runtime({}), measured)
            self.assertEqual(inkling2_runtime({"device": "cuda:1"}), measured)
            self.assertEqual(inkling2_runtime({"tensor_cores": False}),
                             dict(measured, tensor_cores=False, experts="dense"))
            self.assertEqual(inkling2_runtime({"runner": "reference", "experts": "dense", "drafter_cuda_graph": False}),
                             dict(measured, runner="reference", experts="dense", drafter_cuda_graph=False))
            self.assertEqual(inkling2_runtime({"device": "cpu"}),
                             {"runner": "batched", "tensor_cores": False, "experts": "dense",
                              "drafter_cuda_graph": False})
            self.assertEqual(inkling2_runtime({"runner": "batched"}), dict(measured, runner="batched", experts="dense"))
            self.assertEqual(inkling2_runtime({"precision": "exact"}),
                             dict(measured, tensor_cores=False, experts="dense"))
            for chosen, backend in (({}, "humming"), ({"experts": "dense"}, "fused"),
                                    ({"tensor_cores": False}, "fused"), ({"runner": "batched"}, "eager"),
                                    ({"runner": "batched", "experts": "packed"}, "eager"),
                                    ({"runner": "reference"}, "fused"),
                                    ({"runner": "reference", "experts": "packed"}, "humming"),
                                    ({"runner": "reference", "tensor_cores": False}, "eager")):
                with self.subTest(chosen=chosen):
                    self.assertEqual(inkling2_expert_backend(inkling2_runtime(chosen)), backend)
            self.assertEqual(inkling2_expert_backend(inkling2_runtime({"device": "cpu"}), "cpu"), "eager")
        with patch.object(inkling2_humming, "available", return_value=False):
            self.assertEqual(inkling2_runtime({}), dict(measured, experts="dense"))
            self.assertEqual(inkling2_runtime({"experts": "packed"}), measured)
            self.assertEqual(inkling2_expert_backend(inkling2_runtime({})), "fused")
            self.assertEqual(inkling2_expert_backend(inkling2_runtime({"experts": "packed"})), "eager")

    def test_inkling2_tells_a_gpu_run_that_falls_off_the_measured_expert_path(self):
        """Without a recent Humming a default GPU run takes dense experts (half the speed, 49 GiB) and a
        packed one loses its CUDA graphs: the engine says so. A run that asked for dense experts, another
        runner, the reference arithmetic or a CPU chose its path and is told nothing."""
        from vllm.model_executor.models import inkling2_humming
        from vllm.v1.spec_decode.twodsd import inkling2_slow_path

        with patch.object(inkling2_humming, "available", return_value=False):
            self.assertIn("dense experts take 49 GiB", inkling2_slow_path({}))
            self.assertIn("no CUDA graphs", inkling2_slow_path({"experts": "packed"}))
            for chosen in ({"experts": "dense"}, {"runner": "batched"}, {"tensor_cores": False}, {"device": "cpu"}):
                with self.subTest(chosen=chosen):
                    self.assertIsNone(inkling2_slow_path(chosen))
        with patch.object(inkling2_humming, "available", return_value=True):
            self.assertIsNone(inkling2_slow_path({}))

    def test_an_unknown_model_type_gets_no_engine_options_or_preset(self):
        """Guards the old defaults (Ouro's graph-only engine and R = 4 preset) for unsupported checkpoints."""
        for model_type in ("qwen3", "inkling2"):
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, SUPPORTED):
                    engine_options(self.checkpoint({"model_type": model_type}), None, None)
                with self.assertRaisesRegex(ValueError, SUPPORTED):
                    preset(model_type)
        with self.assertRaisesRegex(ValueError, SUPPORTED):  # a config.json that declares no model_type
            engine_options(self.checkpoint({"architectures": ["Other"]}), None, None)

    def test_twodsd_checks_an_inkling2_request_before_loading_weights(self):
        """TwoDSD on an Inkling2 checkpoint never touches the process environment or vLLM's engine, and
        refuses a block without a drafter, a schedule that does not fit R and unknown arguments first."""
        inkling = self.checkpoint({"model_type": "inkling2_loop", "text_config": {"loop_config": MINI_LOOP}})
        with patch.dict(os.environ):
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
            with self.assertRaisesRegex(ValueError, "drafter export is required"):
                TwoDSD(inkling)
            for kv_policy in ("w64", "w32"):
                with self.assertRaisesRegex(ValueError, "per-loop KV only"):
                    TwoDSD(inkling, block_size=1, kv_policy=kv_policy)
            with self.assertRaisesRegex(ValueError, "engine arguments"):
                TwoDSD(inkling, block_size=1, enforce_eager=True)
            # A block the drafter cannot draft: the export has 4 slots (5 when its anchor slot predicts).
            for dspark, block, fits in ((None, 5, False), (None, 4, True), ({"sample_from_anchor": True}, 5, True),
                                        ({"sample_from_anchor": True}, 6, False)):
                drafter = self.checkpoint({"block_size": 4, "loopedtf_drafter": {"dspark": dspark}})
                with self.subTest(dspark=dspark, block=block):
                    if fits:
                        check_block_size(drafter, block)
                    else:
                        with self.assertRaisesRegex(ValueError, "exceeds the drafter's"):
                            TwoDSD(inkling, drafter, block_size=block, drafter_of_other_target=True)
            self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", os.environ)


class Inkling2EndToEnd(unittest.TestCase):
    """TwoDSD on a tiny Inkling2 checkpoint in the released format, without an engine, on CPU."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        import torch
        from vllm.model_executor.models.inkling2_reference import Inkling2Cache, NativeInkling2

        spec = importlib.util.spec_from_file_location("inkling2_tiny_api", Path(__file__).with_name("inkling2_tiny.py"))
        tiny = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tiny)
        cls.directory, cls.tiny = tempfile.TemporaryDirectory(), tiny
        loop = dict(core_start=2, core_end=6, num_passes=3, gate_rank=4, is_varloop=False)
        config = tiny.tiny_config(num_hidden_layers=8, local_layer_ids=[0, 2, 4, 6], loop_config=loop)
        cls.model = str(tiny.write_tiny_checkpoint(Path(cls.directory.name) / "model", config, mxfp8_layers=(6, 7)))
        cls.prompt, cls.count = [5, 17, 3, 42, 9, 11, 2], 20
        native = NativeInkling2.from_pretrained(cls.model, precision="exact", dtype=torch.float32)
        cache, step, cls.want = Inkling2Cache(), cls.prompt, []
        while len(cls.want) < cls.count:
            cls.want.append(int(native.forward(step, cache=cache, pass_end=False).readouts[3].argmax[-1]))
            step = [cls.want[-1]]
        cls.kwargs = dict(block_size=1, max_model_len=64, device="cpu", precision="exact", dtype="float32")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def generate(self, model=None, **options):
        tokenizer = SimpleNamespace(encode=lambda text: [])
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            engine = TwoDSD(model or self.model, **{**self.kwargs, **options})
        self.assertEqual(engine.eos_token_ids(), [200006])
        [result] = engine.generate([self.prompt], max_new_tokens=self.count, warmup_requests=1,
                                   metadata=[{"id": "q"}])
        return result

    def test_depth_only_schedules_and_plain_decoding_name_the_reference_tokens(self):
        for options in ({}, {"stage": "stage1", "d1": 3}, {"stage": "stage1", "d1": 2}):
            with self.subTest(options=options):
                result = self.generate(**options)
                self.assertEqual(result["output_ids"], self.want)
                self.assertEqual((result["id"], result["max_new"], result["stopped_by_eos"]), ("q", 20, False))
                self.assertEqual(result["settings"]["num_loops"], 3)
                self.assertEqual(result["settings"]["kv_policy"], "full")
        self.assertEqual(self.generate(stage="stage1", d1=3)["settings"]["control"], "autoregressive")

    def test_a_variable_loop_checkpoint_runs_another_number_of_passes(self):
        """loops = 4 on a variable-loop checkpoint of three passes: plain decoding names the tokens of the
        reference forward with four passes (other tokens than three passes give), every depth schedule of
        R = 4 names them too, and a checkpoint that is not variable-loop refuses another count."""
        import torch
        from vllm.model_executor.models.inkling2_reference import Inkling2Cache, NativeInkling2

        loop = dict(core_start=2, core_end=6, num_passes=3, gate_rank=4, is_varloop=True)
        config = self.tiny.tiny_config(num_hidden_layers=8, local_layer_ids=[0, 2, 4, 6], loop_config=loop)
        model = str(self.tiny.write_tiny_checkpoint(Path(self.directory.name) / "varloop", config,
                                                    mxfp8_layers=(6, 7)))
        native = NativeInkling2.from_pretrained(model, precision="exact", dtype=torch.float32)
        cache, step, want = Inkling2Cache(), self.prompt, []
        while len(want) < self.count:
            out = native.forward(step, num_passes=4, cache=cache, pass_end=False)
            want.append(int(out.readouts[4].argmax[-1]))
            step = [want[-1]]
        self.assertNotEqual(want, self.generate(model)["output_ids"])  # three passes
        schedules = ({"plain": True}, {"stage": "stage1", "d1": 4}, {"stage": "stage1", "d1": 1}, {},
                     {"stage": "stage2b", "d1": 1, "d2": 3}, {"stage": "stage2a", "d1": 2, "d2": 3})
        for options in schedules:
            with self.subTest(options=options):
                result = self.generate(model, loops=4, **options)
                self.assertEqual(result["output_ids"], want)
                self.assertEqual(result["settings"]["num_loops"], 4)
        self.assertEqual(self.generate(model, loops=4, plain=True)["settings"]["control"], "autoregressive")
        with self.assertRaisesRegex(ValueError, r"d1 < d2 < R \(R = 4\)"):
            self.generate(model, loops=4, stage="stage2b", d1=1, d2=4)
        with self.assertRaisesRegex(ValueError, "variable-loop"):
            self.generate(loops=4)
        self.assertEqual(self.generate(loops=3, stage="stage1", d1=3)["output_ids"], self.want)

    def test_the_per_loop_policy_by_name_is_the_default_run(self):
        result = self.generate(stage="stage1", d1=3, kv_policy="native")
        self.assertEqual(result["output_ids"], self.want)
        self.assertEqual((result["settings"]["kv_policy"], result["settings"]["kv_window"]), ("full", None))
        with self.assertRaisesRegex(ValueError, "per-loop KV only"):
            self.generate(kv_policy="w64")


if __name__ == "__main__":
    unittest.main()
