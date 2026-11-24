"""Drafter training traces from OpenAI-compatible servers of the target, over the queue of
training.drafter.traces.

For a target the fork's engine cannot serve (Inkling 2): the official vLLM build serves the model
and one client per server claims requests from the same file-locked queue. Requests, the sampled
share and its settings, the prompt + output budget, the online loop stop, the raw records and
collect are those of training.drafter.traces; only the engine differs. A client sends the prompt
as token IDs to /v1/completions, reads the response as streamed token IDs (return_token_ids) and
closes the stream at a loop onset, which aborts the request on the server.

The queue is built where this repository is installed (init, or launch). Generation needs only
this file, the queue directory and aiohttp, so it also runs as a script on a machine without the
repository (inside the official image, next to its servers):

  python traces_remote.py join --out TRACES --servers http://127.0.0.1:8100 http://127.0.0.1:8101

  init     build the queue in --out (prompts rendered to token IDs) and start nothing
  launch   init, then start one detached client per server
  join     start clients on more servers for the existing queue
  worker   (internal) one server
  status   progress
  requeue  once every client of the queue has exited, re-queue claimed but unfinished requests
  collect  per-pool trace files (<pool>.jsonl, manifest.json) once the queue is drained
Touch OUT/queue/stop to stop claiming; clients finish their open requests and exit. A client that
loses its server exits with the requests it had open unfinished (requeue + join).

  python -m training.drafter.traces_remote launch --out TRACES --model inkling-2-mini \\
      --prompts PROMPTS_V2 --servers http://127.0.0.1:8100 http://127.0.0.1:8101
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

POOL_ORDER = ("dev", "eval", "train")
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None}
RETRIES, BACKOFF_SECONDS = 4, 10
NATIVE_KV = "full"  # training.sharekv.policy.NATIVE

# The queue and the loop stop below are those of training.drafter.traces, repeated so that this
# file runs without the repository; tests/training/test_traces_remote.py holds the two together.
LOOP_WINDOW, LOOP_NGRAM, LOOP_MIN_UNIQUE, LOOP_STRIDE = 1024, 16, 0.5, 64


def loop_onset(ids, start=0):
    grams = LOOP_WINDOW - LOOP_NGRAM + 1
    for p in range(start, len(ids) - LOOP_WINDOW + 1, LOOP_STRIDE):
        window = ids[p:p + LOOP_WINDOW]
        if len({tuple(window[i:i + LOOP_NGRAM]) for i in range(grams)}) < LOOP_MIN_UNIQUE * grams:
            return p
    return None


class LoopWatch:
    def __init__(self):
        self.ids, self.next = [], 0

    def extend(self, tokens):
        self.ids.extend(tokens)
        if len(self.ids) - self.next < LOOP_WINDOW:
            return None
        onset = loop_onset(self.ids, self.next)
        self.next += ((len(self.ids) - LOOP_WINDOW - self.next) // LOOP_STRIDE + 1) * LOOP_STRIDE
        return onset


def queue_dir(out):
    return Path(out) / "queue"


def read_rows(path):
    """Rows of a JSONL file that clients append to; a partial last line is skipped."""
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def claim(queue, pending, count):
    with (queue / "lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.loads((queue / "state.json").read_text())
        start = state["next"]
        state["next"] = min(len(pending), start + count)
        (queue / "state.json").write_text(json.dumps(state))
    return pending[start:state["next"]]


@contextmanager
def held(path):
    with Path(path).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def server_tag(server):
    return re.sub(r"[^A-Za-z0-9]+", "-", server.split("://", 1)[-1]).strip("-")


def request_body(model, item, budget, eos):
    """The /v1/completions request of one queue item: token IDs in, streamed token IDs out."""
    sampling = item["sampling"] or GREEDY
    body = {"model": model, "prompt": item["prompt_ids"], "max_tokens": budget - len(item["prompt_ids"]),
            "temperature": sampling["temperature"], "top_p": sampling["top_p"], "stream": True,
            "return_token_ids": True, "skip_special_tokens": False, "add_special_tokens": False,
            "stop_token_ids": list(eos)}
    if sampling["top_k"]:
        body["top_k"] = sampling["top_k"]
    if sampling["seed"] is not None:
        body["seed"] = sampling["seed"]
    return body


def finished_ids(ids, reason, eos):
    """Output IDs as the in-process engine reports them: a response that stopped at EOS ends with
    the EOS ID. Returns the IDs and whether the EOS had to be added."""
    if reason != "stop" or (ids and ids[-1] in eos):
        return ids, False
    if len(eos) != 1:
        raise RuntimeError("The server stopped without returning the stop token and the model has several EOS IDs")
    return [*ids, eos[0]], True


async def stream_one(session, server, body, eos):
    """One request: (output IDs, finish reason, loop onset or None, generated tokens, EOS added)."""
    watch, reason = LoopWatch(), None
    async with session.post(f"{server}/v1/completions", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
        async for line in response.content:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"server error: {str(chunk['error'])[:300]}")
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            onset = watch.extend(choice.get("token_ids") or [])
            if onset is not None:  # also in the final delta, as in training.drafter.traces
                return watch.ids[:onset], "loop", onset, len(watch.ids), False
            if choice.get("finish_reason"):
                reason = choice["finish_reason"]
    if reason is None:
        raise RuntimeError("the stream ended without a finish reason")
    ids, added = finished_ids(watch.ids, reason, eos)
    return ids, reason, None, len(ids), added


async def serve(out, server, concurrency, refill):
    import aiohttp

    queue, tag = queue_dir(out), server_tag(server)
    settings = json.loads((queue / "init.json").read_text())["settings"]
    model, budget, eos = settings["model"], settings["budget"], settings["eos_token_ids"]
    raw = Path(out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (queue / "workers").mkdir(exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=900)
    connector = aiohttp.TCPConnector(limit=0)
    with held(queue / "workers" / f"server-{tag}.lock"):
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.get(f"{server}/version") as response:
                engine = f"vllm {(await response.json())['version']} (server)"
            async with session.get(f"{server}/v1/models") as response:
                served = [m["id"] for m in (await response.json())["data"]]
            name = settings.get("served_model") or model
            if name not in served:
                raise RuntimeError(f"{server} serves {served}, not {name}. If one of them is {model}, say so at "
                                   "init with --served-model")
            pending = read_rows(queue / "pending.jsonl")
            print(json.dumps({"server": server, "model": model, "served": name, "engine": engine,
                              "budget": budget, "concurrency": concurrency}), flush=True)
            started = last = time.monotonic()
            counts = {"finished": 0, "loop_stopped": 0, "tokens": 0, "retries": 0, "eos_added": 0}

            with (raw / f"server-{tag}.jsonl").open("a") as sink:

                async def one(item):
                    body = request_body(name, item, budget, eos)
                    for attempt in range(RETRIES):
                        try:
                            ids, reason, onset, generated, added = await stream_one(session, server, body, eos)
                            break
                        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as error:
                            if attempt == RETRIES - 1:
                                raise RuntimeError(f"{item['id']}: {error}") from error
                            counts["retries"] += 1
                            print(json.dumps({"retry": item["id"], "error": str(error)[:200]}), flush=True)
                            await asyncio.sleep(BACKOFF_SECONDS * (attempt + 1))
                    record = {**item, "model": model, "target": settings["target"], "variant": settings["variant"],
                              "target_config_sha256": settings["target_config_sha256"], "kv_policy": NATIVE_KV,
                              "output_ids": ids, "finish_reason": reason, "max_total_tokens": budget,
                              "loop_onset": onset, "generated_tokens": generated, "engine": engine,
                              "compile_mode": None, "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    counts["finished"] += 1
                    counts["loop_stopped"] += reason == "loop"
                    counts["tokens"] += generated
                    counts["eos_added"] += added

                active, waiting, exhausted, stopping = set(), [], False, False
                while True:
                    if not stopping and (queue / "stop").exists():
                        stopping = True
                    while not stopping and len(active) < concurrency:
                        if not waiting and not exhausted:
                            waiting = claim(queue, pending, refill)
                            exhausted = not waiting
                        if not waiting:
                            break
                        active.add(asyncio.create_task(one(waiting.pop(0))))
                    if not active:
                        break
                    done, active = await asyncio.wait(active, timeout=30, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()  # a request that failed every retry ends the client
                    if time.monotonic() - last >= 60:
                        elapsed = time.monotonic() - started
                        print(json.dumps({"server": server, **counts, "active": len(active),
                                          "tok_per_s": round(counts["tokens"] / elapsed), "elapsed_s": round(elapsed)}),
                              flush=True)
                        last = time.monotonic()
            print(json.dumps({"server": server, **counts, "done": True,
                              "elapsed_s": round(time.monotonic() - started)}), flush=True)


def launch(out, servers, concurrency, refill):
    out = Path(out)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for server in servers:
        log = (logs / f"server-{server_tag(server)}.log").open("a")
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(out.resolve()),
                   "--server", server.rstrip("/"), "--concurrency", str(concurrency), "--refill", str(refill)]
        subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"{server} -> {log.name}", flush=True)


def running(out):
    busy = []
    for path in sorted((queue_dir(out) / "workers").glob("*.lock")):
        try:
            with held(path):
                pass
        except BlockingIOError:
            busy.append(path.stem)
    return busy


def status(out):
    """Progress from the queue and the raw records alone."""
    queue = queue_dir(out)
    state = json.loads((queue / "state.json").read_text())
    rows = [r for path in (Path(out) / "raw").glob("*.jsonl") for r in read_rows(path)]
    by_pool = {}
    for r in rows:
        key = r["pool"] + ("/sampled" if r.get("sampling") else "")
        p = by_pool.setdefault(key, {"traces": 0, "length": 0, "loop": 0, "tokens": 0})
        p["traces"] += 1
        p["length"] += r["finish_reason"] == "length"
        p["loop"] += r["finish_reason"] == "loop"
        p["tokens"] += len(r["output_ids"])
    print(json.dumps({"claimed": state["next"], "queued": len(read_rows(queue / "pending.jsonl")),
                      "running": running(out), "finished": by_pool}, indent=1))


def init(args):
    """Build the queue with this repository: prompts rendered by the target's chat template, and the
    EOS IDs and config hash a client needs written into the settings."""
    from ..common import registry
    from ..common.chat import eos_ids, load_tokenizer
    from ..common.io import file_sha
    from . import traces

    path = registry.resolve_target(args.model, "original", args.target_path,
                                   allow_patterns=registry.TOKENIZER_FILES)
    eos = registry.eos_token_ids(registry.family(args.model)) or eos_ids(load_tokenizer(path), path)
    settings = {"model": args.model, "variant": "original", "target_path": str(Path(path).resolve()),
                "target": args.target_name or registry.label(args.target_path or args.model, "original")
                or f"{args.model}-original",
                "prompts": str(args.prompts.resolve()), "budget": args.budget,
                "pools": [p for p in POOL_ORDER if p in args.pools], "sampled_fraction": args.sampled_fraction,
                "kv_policy": NATIVE_KV, "window": None, "ouro_tokenizer": args.ouro_tokenizer,
                "gpu_lock_dir": "", "served_model": args.served_model, "engine": "server",
                "eos_token_ids": list(eos), "target_config_sha256": file_sha(Path(path) / "config.json")}
    traces.init(args.out, settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    builders = [commands.add_parser("init", help="build the queue"),
                commands.add_parser("launch", help="build the queue and start clients")]
    for sub in builders:
        sub.add_argument("--model", required=True)
        sub.add_argument("--target-path", help="local directory or HF repo id with the target's tokenizer and "
                                               "config.json (default: those files of the registry's repository)")
        sub.add_argument("--target-name", help="target identifier recorded in every trace")
        sub.add_argument("--served-model", help="model name the servers answer to (default: --model)")
        sub.add_argument("--prompts", type=Path, required=True,
                         help="prompt pool directory (<pool>.jsonl, manifest.json)")
        sub.add_argument("--budget", type=int, default=16384, help="prompt + output tokens; the servers' "
                                                                   "--max-model-len must be at least this")
        sub.add_argument("--sampled-fraction", type=float, default=0.25)
        sub.add_argument("--pools", nargs="+", choices=POOL_ORDER, default=list(POOL_ORDER))
        sub.add_argument("--ouro-tokenizer", default="ouro-1.4b-thinking",
                         help="decodes the Ouro-rendered benchmark rows of the eval pool")
    for sub in (builders[1], commands.add_parser("join", help="add clients")):
        sub.add_argument("--servers", nargs="+", required=True, help="base URLs, e.g. http://127.0.0.1:8100")
        sub.add_argument("--concurrency", type=int, default=48, help="open requests per server")
        sub.add_argument("--refill", type=int, default=8)
    worker_p = commands.add_parser("worker", help="(internal) one server")
    worker_p.add_argument("--server", required=True)
    worker_p.add_argument("--concurrency", type=int, required=True)
    worker_p.add_argument("--refill", type=int, default=8)
    commands.add_parser("status", help="progress")
    commands.add_parser("requeue", help="re-queue unfinished requests (needs the repository)")
    collect_p = commands.add_parser("collect", help="write <pool>.jsonl and manifest.json (needs the repository)")
    collect_p.add_argument("--pools", nargs="+", choices=POOL_ORDER)
    collect_p.add_argument("--allow-missing", action="store_true", help="collect an unfinished queue")
    for sub in commands.choices.values():
        sub.add_argument("--out", type=Path, required=True, help="trace directory of one model")
    args = parser.parse_args(argv)
    if args.command in ("init", "launch"):
        init(args)
        if args.command == "launch":
            launch(args.out, args.servers, args.concurrency, args.refill)
    elif args.command == "join":
        launch(args.out, args.servers, args.concurrency, args.refill)
    elif args.command == "worker":
        asyncio.run(serve(args.out, args.server.rstrip("/"), args.concurrency, args.refill))
    elif args.command == "status":
        status(args.out)
    else:
        from . import traces

        if args.command == "requeue":
            traces.requeue(args.out)
        else:
            traces.collect(args.out, args.pools and [p for p in POOL_ORDER if p in args.pools], args.allow_missing)


if __name__ == "__main__":
    main()
