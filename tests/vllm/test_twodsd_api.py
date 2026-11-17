"""The 2DSD Python API builds requests and journals worker results correctly (CPU)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import vllm.v1.spec_decode.looped_2d_finalized as finalized
from vllm.v1.spec_decode.twodsd import TwoDSD, TwoDSDWorkerExtension, dory_kv_policy, engine_options, preset


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
    def __init__(self, model_type="ouro"):
        config = SimpleNamespace(model_type=model_type)
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

if __name__ == "__main__":
    unittest.main()
