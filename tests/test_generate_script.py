"""scripts/generate.py on CPU: --model resolution (registry names, --target, overrides) with a fake TwoDSD."""

import importlib.util
import io
import json
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_script", ROOT / "scripts" / "generate.py")
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)
spec = importlib.util.spec_from_file_location("build_prompts", ROOT / "scripts" / "build_prompts.py")
build_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_prompts)
MODELS = json.loads((ROOT / "configs" / "models.json").read_text())["models"]


def resolved(*argv):
    args = generate.parse_args([*argv, "--prompt", "hi"])
    return {key: getattr(args, key) for key in ("model", "revision", "drafter", "kv_policy")}


def rejected(*argv):
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        try:
            generate.parse_args([*argv, "--prompt", "hi"])
        except SystemExit:
            return stderr.getvalue()
    raise AssertionError(f"{argv} was accepted")


class Resolution(unittest.TestCase):
    def test_only_the_base_models_are_pinned(self):
        for name, entry in MODELS.items():
            with self.subTest(name=name):
                self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
                for repo in (entry["w64"], *entry["drafters"].values()):
                    if repo is not None:
                        self.assertRegex(repo, r"^minseo25/2dsd-[\w.-]+$")

    def test_a_registry_name_runs_the_shared_kv_target_and_its_drafter(self):
        entry = MODELS["ouro-1.4b-thinking"]
        expected = {"model": entry["w64"], "revision": None, "drafter": entry["drafters"]["w64"], "kv_policy": None}
        self.assertEqual(resolved("--model", "ouro-1.4b-thinking"), expected)
        self.assertEqual(resolved("--model", "ouro-1.4b-thinking", "--target", "w64"), expected)

    def test_the_original_target_is_the_base_model_with_its_drafter(self):
        entry = MODELS["nanbeige4.2-3b"]
        self.assertEqual(resolved("--model", "nanbeige4.2-3b", "--target", "original"),
                         {"model": entry["repo_id"], "revision": entry["revision"],
                          "drafter": entry["drafters"]["original"], "kv_policy": None})

    def test_dory_targets_are_its_pinned_checkpoint_with_a_kv_policy(self):
        entry = MODELS["dory1.0-4b"]
        for target, kv_policy in (("w64", "w64"), ("original", "native")):
            with self.subTest(target=target):
                self.assertEqual(resolved("--model", "dory1.0-4b", "--target", target),
                                 {"model": entry["repo_id"], "revision": entry["revision"],
                                  "drafter": entry["drafters"]["w64"], "kv_policy": kv_policy})

    def test_block_size_one_runs_no_drafter(self):
        for argv in (("--model", "ouro-2.6b-thinking"), ("--model", "ouro-2.6b-thinking", "--drafter", "me/d"),
                     ("--model", "dory1.0-4b", "--target", "original"), ("--model", "/ckpt/m", "--drafter", "me/d")):
            with self.subTest(argv=argv):
                self.assertIsNone(resolved(*argv, "--block-size", "1")["drafter"])

    def test_explicit_flags_override_the_registry(self):
        pinned = resolved("--model", "ouro-2.6b-thinking")
        self.assertEqual(resolved("--model", "ouro-2.6b-thinking", "--revision", "abc", "--drafter", "me/d@1"),
                         dict(pinned, revision="abc", drafter="me/d@1"))
        self.assertEqual(resolved("--model", "dory1.0-4b", "--kv-policy", "native"),
                         resolved("--model", "dory1.0-4b", "--target", "original"))
        self.assertIn("applies to Dory only",
                      rejected("--model", "ouro-2.6b-thinking", "--kv-policy", "w64"))

    def test_a_model_outside_the_registry_runs_as_given(self):
        self.assertEqual(resolved("--model", "org/looped", "--drafter", "org/drafter@1", "--revision", "r"),
                         {"model": "org/looped", "revision": "r", "drafter": "org/drafter@1", "kv_policy": None})
        self.assertEqual(resolved("--model", "/ckpt/dory", "--drafter", "/ckpt/d", "--kv-policy", "native"),
                         {"model": "/ckpt/dory", "revision": None, "drafter": "/ckpt/d", "kv_policy": "native"})
        self.assertIsNone(resolved("--model", "/ckpt/m", "--block-size", "1")["drafter"])
        self.assertIn("2DSD needs --drafter", rejected("--model", "org/looped"))

    def test_a_target_for_a_model_outside_the_registry_is_rejected(self):
        self.assertIn("--target applies to registry names",
                      rejected("--model", "ByteDance/Ouro-1.4B-Thinking", "--target", "original", "--drafter", "d"))

    def test_inkling2_runs_its_pinned_checkpoint_without_sharing(self):
        """inkling-2-mini runs per-loop KV (the original target) and has no registry drafter; the shared
        target and policy are refused."""
        entry = MODELS["inkling-2-mini"]
        expected = {"model": entry["repo_id"], "revision": entry["revision"], "drafter": None,
                    "kv_policy": "native"}
        self.assertEqual(expected["revision"], "1fb4134b9a74f9681054f4ef82e4bf1f458a2337")
        self.assertEqual(resolved("--model", "inkling-2-mini", "--block-size", "1"), expected)
        self.assertEqual(resolved("--model", "inkling-2-mini", "--target", "original", "--drafter", "me/d@1"),
                         dict(expected, drafter="me/d@1"))
        self.assertEqual(resolved("--model", "inkling-2-mini", "--block-size", "1", "--kv-policy", "native"), expected)
        self.assertIn("inkling-2-mini has no published drafter", rejected("--model", "inkling-2-mini"))
        for extra in (("--target", "w64"), ("--kv-policy", "w64")):
            with self.subTest(extra=extra):
                self.assertIn("per-loop KV only", rejected("--model", "inkling-2-mini", "--block-size", "1", *extra))


class Main(unittest.TestCase):
    def run_main(self, *argv, model_type):
        """generate.main with a fake TwoDSD; returns (engine options, rendered chats, stdout, stderr) and
        keeps the options of the generate call in self.generated."""
        built, chats, self.generated = [], [], []
        generated = self.generated

        class TwoDSD:
            def __init__(self, model, drafter, **options):
                built.append({"model": model, "drafter": drafter, **options})
                self.model_type = model_type

                def apply_chat_template(messages, **kwargs):
                    chats.append((messages, kwargs))
                    return messages[-1]["content"]

                self.tokenizer = SimpleNamespace(apply_chat_template=apply_chat_template,
                                                 encode=lambda text, add_special_tokens: [1, 2, 3],
                                                 decode=lambda ids, skip_special_tokens: "answer")

            def generate(self, prompts, **options):
                generated.append({**options, "prompts": prompts})
                return [{"output_ids": [7] * 5, "decode_seconds": 2.0, "stats": {"calls": 2}}]

        def render_inkling2(tokenizer, messages):  # the shared Inkling2 renderer: (text, token IDs)
            chats.append((messages, "render_inkling2"))
            return "rendered", [8, 9]

        twodsd = types.ModuleType("vllm.v1.spec_decode.twodsd")
        twodsd.TwoDSD = TwoDSD
        out, err = io.StringIO(), io.StringIO()
        with (patch.dict(sys.modules, {"vllm.v1.spec_decode.twodsd": twodsd}),
              patch.object(generate, "render_inkling2", render_inkling2), redirect_stdout(out),
              redirect_stderr(err)):
            generate.main([*argv, "--prompt", "hi"])
        return built, chats, out.getvalue(), err.getvalue()

    def test_a_humming_directory_comes_first_on_the_import_path(self):
        """--humming DIR: the engine imports Humming (Inkling2's packed experts) from that directory."""
        with patch.object(sys, "path", list(sys.path)):
            self.run_main("--model", "inkling-2-mini", "--block-size", "1", "--humming", "/opt/humming",
                          model_type="inkling2_loop")
            self.assertEqual(sys.path[0], "/opt/humming")

    def test_the_engine_gets_the_resolved_arguments(self):
        built, _, out, err = self.run_main("--model", "dory1.0-4b", "--target", "original", model_type="dory")
        entry = MODELS["dory1.0-4b"]
        self.assertEqual(built, [{"model": entry["repo_id"], "drafter": entry["drafters"]["original"],
                                  "max_model_len": 8192, "block_size": None, "revision": entry["revision"],
                                  "kv_policy": "native"}])
        self.assertIn("decode 2.0 tok/s, 2 target calls, 2.00 tokens/call", out)
        self.assertIn(f"target {entry['repo_id']}@{entry['revision']}", err)
        self.assertIn("kv_policy native", err)

    def test_the_kv_policy_of_a_shared_kv_target_comes_from_its_config(self):
        _, _, _, err = self.run_main("--model", "ouro-2.6b-thinking", model_type="ouro")
        self.assertIn("kv_policy from config.json", err)
        self.assertNotIn("None", err)

    def test_the_prompt_is_rendered_as_the_benchmark_prompts(self):
        """Template defaults (no enable_thinking: Ouro's paper prompts have no <think>) and the
        benchmark's system turn, which only Nanbeige sets; --system replaces it."""
        nanbeige = {"role": "system", "content": build_prompts.SYSTEM_PROMPTS["nanbeige"]}
        user = {"role": "user", "content": "hi"}
        for argv, model_type, messages in (
            (("--model", "ouro-2.6b-thinking"), "ouro", [user]),
            (("--model", "dory1.0-4b"), "dory", [user]),
            (("--model", "nanbeige4.2-3b"), "nanbeige", [nanbeige, user]),
            (("--model", "nanbeige4.2-3b", "--system", "Be brief."), "nanbeige",
             [{"role": "system", "content": "Be brief."}, user]),
        ):
            with self.subTest(argv=argv):
                _, chats, _, _ = self.run_main(*argv, model_type=model_type)
                self.assertEqual(chats, [(messages, {"tokenize": False, "add_generation_prompt": True})])

    def test_inkling2_generates_with_its_fixed_eos_ids(self):
        """The engine reports model_type inkling2_loop, the family is inkling2: EOS [200006] is passed
        explicitly, there is no system turn, and the KV policy is reported as native (per-loop KV)."""
        _, chats, _, err = self.run_main("--model", "inkling-2-mini", "--block-size", "1", model_type="inkling2_loop")
        self.assertEqual(self.generated[0]["eos_token_ids"], [200006])
        # The token IDs are those of the renderer of the benchmark prompts, which encodes a message as text
        # (a tokenizer run over the rendered string reads a spelled-out framing token as the token).
        self.assertEqual(chats, [([{"role": "user", "content": "hi"}], "render_inkling2")])
        self.assertEqual(self.generated[0]["prompts"], [[8, 9]])
        self.assertIn("kv_policy native", err)
        # A checkpoint outside the registry is Inkling2 by the engine's model_type.
        self.run_main("--model", "/ckpt/inkling", "--drafter", "d", model_type="inkling2_loop")
        self.assertEqual(self.generated[0]["eos_token_ids"], [200006])
        # The other families leave EOS to the engine, which reads their checkpoints.
        for argv, model_type in ((("--model", "ouro-2.6b-thinking"), "ouro"), (("--model", "dory1.0-4b"), "dory"),
                                 (("--model", "nanbeige4.2-3b"), "nanbeige")):
            with self.subTest(argv=argv):
                self.run_main(*argv, model_type=model_type)
                self.assertIsNone(self.generated[0]["eos_token_ids"])

    def test_an_engine_of_an_unknown_model_type_is_an_error(self):
        """Guards the family lookup: no model_type is rendered or stopped as another family."""
        with self.assertRaisesRegex(ValueError, "unsupported model_type 'qwen3'"):
            self.run_main("--model", "org/dense", "--drafter", "d", model_type="qwen3")
        self.assertEqual(self.generated, [])


if __name__ == "__main__":
    unittest.main()
