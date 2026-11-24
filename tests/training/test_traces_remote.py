"""The server client of the trace queue against a stub OpenAI-compatible server (CPU, no model), and its
copies of the queue and loop-stop rules against training.drafter.traces."""
import asyncio
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from training.drafter import traces, traces_remote
from training.drafter.traces import LOOP_STRIDE, loop_onset

EOS = 200006
MODEL = "inkling-2-mini"


def response_of(prompt, max_tokens):
    """(token IDs, finish reason) the stub generates; the prompt's first ID selects the case."""
    rng = random.Random(prompt[1])
    text = [rng.randrange(1000, 50000) for _ in range(300 + prompt[1] % 200)]
    case = prompt[0]
    if case == 1:  # stops at EOS and returns it
        return [*text, EOS], "stop"
    if case == 2:  # stops at EOS without returning it
        return text, "stop"
    if case == 3:  # runs into the budget
        return [rng.randrange(1000, 50000) for _ in range(max_tokens)], "length"
    return text + [rng.randrange(1000, 50000) for _ in range(120)] * 400, "length"  # case 4: loops


class Stub:
    def __init__(self):
        self.calls, self.failed, self.disconnects, self.bodies = 0, set(), 0, []

    async def completions(self, request):
        body = await request.json()
        self.calls += 1
        self.bodies.append(body)
        prompt = body["prompt"]
        if prompt[0] == 5 and prompt[1] not in self.failed:  # case 5: the first attempt fails
            self.failed.add(prompt[1])
            return web.Response(status=500, text="boom")
        ids, reason = response_of([1, *prompt[1:]] if prompt[0] == 5 else prompt, body["max_tokens"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            for start in range(0, len(ids), 37):
                last = start + 37 >= len(ids)
                choice = {"index": 0, "text": "", "token_ids": ids[start:start + 37],
                          "finish_reason": reason if last else None}
                chunk = {"id": "c", "choices": [choice]}
                if start == 0:
                    choice["prompt_token_ids"] = prompt
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await asyncio.sleep(0)
            await response.write(b'data: {"id": "c", "choices": [], "usage": {}}\n\n')
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            self.disconnects += 1
            raise
        return response

    def app(self):
        async def version(_):
            return web.json_response({"version": "0.test"})

        async def models(_):
            return web.json_response({"data": [{"id": "served-mini"}]})

        app = web.Application()
        app.add_routes([web.post("/v1/completions", self.completions), web.get("/version", version),
                        web.get("/v1/models", models)])
        return app


def write_queue(out, items, budget, served="served-mini"):
    """A queue as traces.init leaves it, without rendering prompts."""
    target = out / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))
    settings = {"model": MODEL, "variant": "original", "target_path": str(target), "target": "mini@test",
                "prompts": str(out), "budget": budget, "pools": ["train"], "sampled_fraction": 0.25,
                "kv_policy": "full", "window": None, "served_model": served, "gpu_lock_dir": "",
                "eos_token_ids": [EOS], "target_config_sha256": "abc"}
    queue = traces.queue_dir(out)
    queue.mkdir(parents=True)
    (queue / "init.json").write_text(json.dumps({"settings": settings}))
    (queue / "pending.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (queue / "requests.jsonl").write_text("".join(
        json.dumps({k: i[k] for k in ("id", "pool", "prompt_id", "sampling")}) + "\n" for i in items))
    (queue / "state.json").write_text(json.dumps({"next": 0}))


def item(index, case, sampled=False):
    name = f"p{index}"
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "seed": 7} if sampled else None
    return {"pool": "train", "source": "s", "domain": "math", "origin": "o", "prompt_id": name,
            "prompt_ids": [case, 100 + index, 3, 4], "id": name + ("#sampled" if sampled else ""),
            "sampling": sampling}


async def run_clients(out, stub, clients=1, concurrency=4):
    runner = web.AppRunner(stub.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        # Two clients are two servers in practice; here both names reach the stub.
        hosts = ["127.0.0.1", "localhost"][:clients]
        await asyncio.gather(*(traces_remote.serve(out, f"http://{host}:{port}", concurrency, 2) for host in hosts))
    finally:
        await runner.cleanup()


class ClientTests(unittest.TestCase):
    def run_queue(self, items, budget=4096, clients=1):
        stub = Stub()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        out = Path(self.directory.name)
        write_queue(out, items, budget)
        with patch.object(traces_remote, "BACKOFF_SECONDS", 0.01):
            asyncio.run(run_clients(out, stub, clients))
        rows = [r for path in sorted((out / "raw").glob("*.jsonl")) for r in traces.read_jsonl_partial(path)]
        return out, stub, {r["id"]: r for r in rows}

    def test_every_request_is_saved_once_in_the_trace_format(self):
        items = [item(i, 1 + i % 3, sampled=i % 4 == 0) for i in range(12)]
        out, stub, rows = self.run_queue(items, clients=2)
        self.assertEqual(set(rows), {i["id"] for i in items})
        self.assertEqual(stub.calls, len(items))
        for i in items:
            row = rows[i["id"]]
            want, reason = response_of(i["prompt_ids"], 4096 - len(i["prompt_ids"]))
            if i["prompt_ids"][0] == 2:
                want = [*want, EOS]  # a stop always ends with the EOS ID, as in-process traces do
            self.assertEqual((row["output_ids"], row["finish_reason"]), (want, reason))
            self.assertEqual(row["generated_tokens"], len(want))
            self.assertEqual({k: row[k] for k in i}, i)
            self.assertEqual((row["model"], row["target"], row["variant"], row["kv_policy"], row["max_total_tokens"],
                              row["loop_onset"], row["engine"], row["target_config_sha256"]),
                             (MODEL, "mini@test", "original", "full", 4096, None, "vllm 0.test (server)", "abc"))
        length = rows["p2"]
        self.assertEqual(len(length["prompt_ids"]) + len(length["output_ids"]), 4096)
        traces.collect(out)
        collected = traces.read_jsonl(out / "train.jsonl")
        self.assertEqual([r["id"] for r in collected], [i["id"] for i in items])
        self.assertFalse(any(r["degenerate"] for r in collected))

    def test_requests_carry_token_ids_and_the_sampling_of_the_item(self):
        _, stub, _ = self.run_queue([item(0, 1), item(1, 1, sampled=True)])
        greedy, sampled = sorted(stub.bodies, key=lambda b: b["prompt"][1])
        self.assertEqual(greedy, {"model": "served-mini", "prompt": [1, 100, 3, 4], "max_tokens": 4092,
                                  "temperature": 0.0, "top_p": 1.0, "stream": True, "return_token_ids": True,
                                  "skip_special_tokens": False, "add_special_tokens": False,
                                  "stop_token_ids": [EOS]})
        self.assertEqual((sampled["temperature"], sampled["top_p"], sampled["seed"]), (0.6, 0.95, 7))
        self.assertNotIn("top_k", sampled)

    def test_a_looping_response_is_cut_at_its_onset_and_aborted(self):
        out, stub, rows = self.run_queue([item(0, 4)], budget=60000)
        row = rows["p0"]
        full, _ = response_of([4, 100, 3, 4], 60000)
        self.assertEqual(row["finish_reason"], "loop")
        self.assertEqual(row["loop_onset"], loop_onset(full))
        self.assertEqual(row["loop_onset"] % LOOP_STRIDE, 0)
        self.assertEqual(row["output_ids"], full[:row["loop_onset"]])
        self.assertLess(row["generated_tokens"], 6000)  # stopped soon after the first looping window
        self.assertGreater(row["generated_tokens"], len(row["output_ids"]))

    def test_an_unfinished_queue_can_be_collected(self):
        out, _, _ = self.run_queue([item(0, 1), item(1, 1)])
        queue = traces.queue_dir(out)
        with (queue / "requests.jsonl").open("a") as handle:  # one more request that no client has run
            handle.write(json.dumps({"id": "p9", "pool": "train", "prompt_id": "p9", "sampling": None}) + "\n")
        with self.assertRaisesRegex(RuntimeError, "1 requests have no finished trace"):
            traces.collect(out)
        traces.collect(out, allow_missing=True)
        self.assertEqual([r["id"] for r in traces.read_jsonl(out / "train.jsonl")], ["p0", "p1"])
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual((manifest["train"]["traces"], manifest["train"]["missing"]), (2, 1))

    def test_a_server_that_does_not_name_the_model_is_refused(self):
        """Without --served-model the server must answer to the queue's model: another name may be another
        checkpoint, whose traces would be saved under this model."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        out = Path(self.directory.name)
        write_queue(out, [item(0, 1)], 4096, served=None)
        with self.assertRaisesRegex(RuntimeError, "serves .'served-mini'., not " + MODEL):
            asyncio.run(run_clients(out, Stub()))
        self.assertEqual(list((out / "raw").glob("*.jsonl")), [])

    def test_a_failed_attempt_is_retried(self):
        _, stub, rows = self.run_queue([item(0, 5), item(1, 1)])
        self.assertEqual(stub.calls, 3)
        self.assertEqual(rows["p0"]["finish_reason"], "stop")
        self.assertEqual(rows["p0"]["output_ids"][-1], EOS)

    def test_a_stop_file_ends_claiming(self):
        stub = Stub()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_queue(out, [item(i, 1) for i in range(6)], 4096)
            (traces.queue_dir(out) / "stop").touch()
            asyncio.run(run_clients(out, stub))
            self.assertEqual(stub.calls, 0)
            self.assertEqual(json.loads((traces.queue_dir(out) / "state.json").read_text()), {"next": 0})


class StandaloneTests(unittest.TestCase):
    """The client runs as a script without this repository, so it repeats a few rules of traces."""

    def test_the_loop_stop_is_that_of_the_in_process_generator(self):
        self.assertEqual((traces_remote.LOOP_WINDOW, traces_remote.LOOP_NGRAM, traces_remote.LOOP_MIN_UNIQUE,
                          traces_remote.LOOP_STRIDE),
                         (traces.LOOP_WINDOW, traces.LOOP_NGRAM, traces.LOOP_MIN_UNIQUE, traces.LOOP_STRIDE))
        rng = random.Random(3)
        for case in range(6):
            ids = [rng.randrange(50000) for _ in range(1500 + 400 * case)]
            if case % 2:
                ids += [rng.randrange(50000) for _ in range(90 + 40 * case)] * 40
            self.assertEqual(traces_remote.loop_onset(ids), loop_onset(ids))
            ours, theirs, onsets = traces_remote.LoopWatch(), traces.LoopWatch(), []
            for start in range(0, len(ids), 29):
                onsets.append((ours.extend(ids[start:start + 29]), theirs.extend(ids[start:start + 29])))
            self.assertTrue(all(a == b for a, b in onsets))
            self.assertEqual(any(a is not None for a, _ in onsets), bool(case % 2))

    def test_the_queue_is_shared_with_the_in_process_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            items = [item(i, 1) for i in range(5)]
            write_queue(out, items, 4096)
            queue = traces_remote.queue_dir(out)
            self.assertEqual(queue, traces.queue_dir(out))
            first = traces_remote.claim(queue, items, 2)
            second = traces.claim(queue, items, 2)
            self.assertEqual([i["id"] for i in first + second], ["p0", "p1", "p2", "p3"])
            self.assertEqual(traces_remote.NATIVE_KV, traces.NATIVE)
            self.assertEqual(traces_remote.POOL_ORDER, traces.POOL_ORDER)

    def test_status_runs_as_a_script_outside_the_package(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_queue(out, [item(0, 1)], 4096)
            (out / "raw").mkdir()
            record = {**item(0, 1), "output_ids": [7, EOS], "finish_reason": "stop"}
            (out / "raw" / "server-x.jsonl").write_text(json.dumps(record) + "\n")
            script = Path(traces_remote.__file__)
            done = subprocess.run([sys.executable, "-I", str(script), "status", "--out", str(out)], cwd=directory,
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout)["finished"],
                             {"train": {"traces": 1, "length": 0, "loop": 0, "tokens": 2}})


class HelperTests(unittest.TestCase):
    def test_eos_is_added_only_to_a_stop_that_lacks_it(self):
        self.assertEqual(traces_remote.finished_ids([5, 6], "stop", [EOS]), ([5, 6, EOS], True))
        self.assertEqual(traces_remote.finished_ids([5, EOS], "stop", [EOS]), ([5, EOS], False))
        self.assertEqual(traces_remote.finished_ids([5, 6], "length", [EOS]), ([5, 6], False))
        with self.assertRaisesRegex(RuntimeError, "several EOS"):
            traces_remote.finished_ids([5, 6], "stop", [1, 2])

    def test_server_tags_are_file_names(self):
        self.assertEqual(traces_remote.server_tag("http://127.0.0.1:8100"), "127-0-0-1-8100")


if __name__ == "__main__":
    unittest.main()
