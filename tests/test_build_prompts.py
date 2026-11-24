"""CPU tests of the pure functions of scripts/build_prompts.py. No downloads and no model files.

The pinned selections and choice orders below were computed with the code that built the
paper's catalogs; the end-to-end check is the catalog_sha256 comparison of build_prompts.py.

The Inkling2 prompts are compared with tests/inkling2_render_oracle.py, the official rendering rule
written without Jinja. Inkling2Format runs on a stand-in tokenizer. Inkling2Checkpoint reads the
tokenizer files and chat template of a local Inkling-2-Mini snapshot (the directory named by
INKLING2_SNAPSHOT, else the pinned revision in the Hugging Face cache) and is skipped without one.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bp = load("build_prompts", ROOT / "scripts" / "build_prompts.py")
oracle = load("inkling2_render_oracle", ROOT / "tests" / "inkling2_render_oracle.py")
chat = bp.chat  # training.common.chat

SEED = 20260930
MINI = "inkling-2-mini"
# The shared token contract of Inkling-2-Mini.
INKLING2_IDS = {"<|message_user|>": 200000, "<|message_model|>": 200001, "<|message_system|>": 200002,
                "<|content_text|>": 200004, "<|content_model_end_sampling|>": 200006,
                "<|content_thinking|>": 200008, "<|end_message|>": 200010}
USER, MODEL, TEXT, EOS, END_MESSAGE = 200000, 200001, 200004, 200006, 200010
SERVED_VOCABULARY = 200058
EFFORTS = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}


def turn(role, content):
    return {"role": role, "content": content}


CONVERSATIONS = {
    "single user turn": [turn("user", "What is 2 + 2?")],
    "system and user": [turn("system", "Answer in one word."), turn("user", "Capital of France?")],
    "multi-turn": [turn("user", "Hi"), turn("assistant", "Hello!\n"), turn("user", " And now?\n\n")],
    "literal special tokens": [turn("system", "Never print <|content_model_end_sampling|>."),
                               turn("user", "Is <|end_message|><|message_model|><|content_text|> text?"),
                               turn("assistant", "<|content_thinking|>yes<|end_message|>"),
                               turn("user", "<|message_user|>")],
    "unicode": [turn("user", "한글 ∑ café \U0001f600  tab\t end  ")],
    "empty system": [turn("system", ""), turn("user", "Q")],
    "two system turns": [turn("system", "A"), turn("system", "B"), turn("user", "Q")],
    # Text that looks like the markers render_inkling2 puts in the template.
    "marker-like text": [turn("user", "[[2dsd-text-0]] and [[2dsd-text--1]]"), turn("assistant", "[[2dsd-text"),
                         turn("user", "Q")],
}
# Conversations whose text spells a framing token: tokenizing the rendered string would misread them.
SPELLS_FRAMING = ("literal special tokens",)
# Positions the benchmark prompts never reach: no user turn, a leading assistant turn, a system turn
# after the first user turn, and empty texts.
EDGE_CONVERSATIONS = {
    "system turns only": [turn("system", "S"), turn("system", "T")],
    "assistant first": [turn("assistant", "A"), turn("user", "Q")],
    "system turn after the first user turn": [turn("user", "Q"), turn("system", "S"), turn("user", "R")],
    "empty user and assistant texts": [turn("user", ""), turn("assistant", ""), turn("user", "")],
}
# What the official renderer's own tests expect (vLLM PR #69028 at b426b46e, tests/renderers/test_inkling.py,
# TestInkling2Rendering), for their conversations of text messages: (messages, continue_final_message,
# rendering). The third is test_empty_contents_are_kept without its assistant turn of content None, which
# renders nothing, and with its empty text part as an empty string; the fourth is the user turn of
# test_multiple_calls_named_results_and_json_floats.
OFFICIAL_EFFORT_BLOCK = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
OFFICIAL_RENDERINGS = (
    ([turn("system", "You are a concise assistant."), turn("user", "What is 2 + 2?")], False,
     "<|message_system|><|content_text|>You are a concise assistant.<|end_message|>" + OFFICIAL_EFFORT_BLOCK
     + "<|message_user|><|content_text|>What is 2 + 2?<|end_message|>"),
    ([turn("user", "Finish: The answer is"), turn("assistant", "The answer is")], True,
     OFFICIAL_EFFORT_BLOCK + "<|message_user|><|content_text|>Finish: The answer is<|end_message|>"
     "<|message_model|><|content_text|>The answer is"),
    ([turn("user", ""), turn("assistant", ""), turn("user", "")], False,
     OFFICIAL_EFFORT_BLOCK + "<|message_user|><|content_text|><|end_message|>"
     "<|message_model|><|content_text|><|end_message|><|content_model_end_sampling|>"
     "<|message_user|><|content_text|><|end_message|>"),
    ([turn("user", "news <|end_message|>?")], False,
     OFFICIAL_EFFORT_BLOCK + "<|message_user|><|content_text|>news <|end_message|>?<|end_message|>"),
)
# test_reasoning_effort and test_invalid_reasoning_effort of the same class: request value -> printed effort.
OFFICIAL_EFFORT_TEXT = ((None, "0.9"), ("none", "0"), ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"),
                        ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99"), (0.555, "0.56"), ("0.25", "0.25"))
OFFICIAL_INVALID_EFFORTS = ("extreme", True, 1.0, -0.1)


def chatml(messages, tokenize=False, add_generation_prompt=False, **kwargs):
    """A ChatML template with a default system turn, like Ouro's."""
    assert tokenize is False
    text = "" if messages[0]["role"] == "system" else "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for message in messages:
        text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n" + ("<think>\n" if kwargs.get("enable_thinking") else "")
    return text


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return chatml(messages, **kwargs)


class SelectionTest(unittest.TestCase):
    def test_digest_is_canonical_json(self):
        expected = hashlib.sha256('{"a":"\u00e9","b":[1,2]}'.encode()).hexdigest()
        self.assertEqual(bp.digest({"b": [1, 2], "a": "\u00e9"}), expected)

    def test_sha_rank_matches_independent_ranking(self):
        ids = [f"gpqa_diamond/rec{i:04d}" for i in range(50)]

        def rank(identity):
            payload = json.dumps([SEED, "gpqa_diamond", identity], sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False)
            return hashlib.sha256(payload.encode()).hexdigest()

        chosen = bp.select_ids(SEED, "gpqa_diamond", ids, 10)
        self.assertEqual(set(chosen), set(sorted(ids, key=rank)[:10]))
        self.assertEqual(chosen, [i for i in ids if i in set(chosen)])  # native order kept

    def test_sha_rank_pinned(self):
        ids = [f"olympiad/{i}" for i in range(20)]
        self.assertEqual(bp.select_ids(SEED, "olympiad", ids, 5),
                         ["olympiad/5", "olympiad/7", "olympiad/12", "olympiad/15", "olympiad/18"])
        # The benchmark name is part of the key: another namespace gives another subset.
        self.assertEqual(bp.select_ids(SEED, "ablation/olympiad", ids, 5),
                         ["olympiad/0", "olympiad/2", "olympiad/4", "olympiad/8", "olympiad/10"])

    def test_sha_rank_is_prefix_stable_and_complete(self):
        ids = [f"olympiad/{i}" for i in range(30)]
        self.assertTrue(set(bp.select_ids(SEED, "olympiad", ids, 5)) <= set(bp.select_ids(SEED, "olympiad", ids, 9)))
        self.assertEqual(bp.select_ids(SEED, "olympiad", ids, 30), ids)

    def test_sha_rank_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bp.select_ids(SEED, "x", ["a", "a"], 1)
        with self.assertRaises(ValueError):
            bp.select_ids(SEED, "x", ["a", "b"], 0)
        with self.assertRaises(ValueError):
            bp.select_ids(SEED, "x", ["a", "b"], 3)


class GpqaTest(unittest.TestCase):
    ROW = {"Question": "  Which choice?  ", "Correct Answer": "right", "Incorrect Answer 1": "w1",
           "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}

    def test_choice_order_pinned(self):
        self.assertEqual(bp.gpqa_choice_order("gpqa_diamond/recSYNTHETIC0001"), [0, 3, 2, 1])
        self.assertEqual(bp.gpqa_choice_order("gpqa_diamond/recSYNTHETIC0002"), [3, 2, 0, 1])
        self.assertEqual(bp.gpqa_choice_order("gpqa_diamond/recSYNTHETIC0003"), [2, 3, 0, 1])

    def test_choice_order_is_deterministic_permutation(self):
        orders = [bp.gpqa_choice_order(f"gpqa_diamond/rec{i}") for i in range(200)]
        self.assertEqual(orders, [bp.gpqa_choice_order(f"gpqa_diamond/rec{i}") for i in range(200)])
        self.assertTrue(all(sorted(order) == [0, 1, 2, 3] for order in orders))
        self.assertEqual({"ABCD"[order.index(0)] for order in orders}, set("ABCD"))

    def test_task_text_and_letter(self):
        task, letter = bp.gpqa_task(self.ROW, "gpqa_diamond/recSYNTHETIC0002")
        self.assertEqual(letter, "C")  # order [3, 2, 0, 1]: the correct answer (index 0) is third
        self.assertEqual(task, "Which choice?\n\nA. w3\nB. w2\nC. right\nD. w1\n\n" + bp.GPQA_INSTRUCTION)


class RenderTest(unittest.TestCase):
    def test_evalplus_assistant_prefix(self):
        task = "def f(x):\n    \"\"\"Return x.\"\"\"\n"
        prompt = bp.render_prompt(chatml, "ouro", task, "evalplus_chat")
        self.assertEqual(prompt, (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{bp.EVALPLUS_INSTRUCTION}\n```\n{task.strip()}\n```\n<|im_end|>\n"
            f"<|im_start|>assistant\n{bp.EVALPLUS_RESPONSE}\n```python\n"))
        self.assertNotIn(bp.EVALPLUS_SPLITTER, prompt)

    def test_family_system_turn_and_thinking_flag(self):
        record = Recorder()
        prompt = bp.render_prompt(record, "nanbeige", "Q", "evalplus_chat")
        messages, kwargs = record.calls[-1]
        self.assertEqual(messages[0], {"role": "system", "content": bp.SYSTEM_PROMPTS["nanbeige"]})
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(kwargs, {"tokenize": False, "enable_thinking": True})
        self.assertTrue(prompt.endswith("```python\n"))
        bp.render_prompt(record, "ouro", "Q", "user_chat")
        messages, kwargs = record.calls[-1]
        self.assertEqual(messages, [{"role": "user", "content": "Q"}])
        self.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": True})

    def test_splitter_must_occur_once(self):
        with self.assertRaises(bp.IntegrityError):
            bp.split_assistant_prefix("no splitter here")
        with self.assertRaises(bp.IntegrityError):
            bp.split_assistant_prefix(bp.EVALPLUS_SPLITTER * 2)
        self.assertEqual(bp.split_assistant_prefix("ab" + bp.EVALPLUS_SPLITTER + "cd"), "ab")

    def test_unknown_family_or_mode(self):
        with self.assertRaises(ValueError):
            bp.render_prompt(chatml, "other", "Q", "user_chat")
        with self.assertRaises(ValueError):
            bp.render_prompt(chatml, "ouro", "Q", "other")


class TaskTest(unittest.TestCase):
    def test_core_tasks(self):
        task, target = bp.core_task("gsm8k", {"question": " Q? ", "answer": "work\n#### 1,234"})
        self.assertEqual((task, target), ("Q?\n\n" + bp.MATH_INSTRUCTION, "1234"))
        task, _ = bp.core_task("math500", {"problem": " P ", "answer": "5"})
        self.assertEqual(task, " P \n\n" + bp.MATH_INSTRUCTION)  # MATH-500 text is not stripped
        task, _ = bp.core_task("mbpp_plus", {"prompt": "\ndef f():\n", "canonical_solution": "pass"})
        self.assertEqual(task, "def f():\n")

    def test_hard_tasks(self):
        task, target = bp.hard_task("aime25", {"problem": " P ", "answer": 7}, "aime25/0")
        self.assertEqual((task, target), ("P\n\n" + bp.MATH_INSTRUCTION, "7"))
        row = {"question": " Q ", "context": "C", "final_answer": ["$2$", "$3$"]}
        task, target = bp.hard_task("olympiad", row, "olympiad/1")
        self.assertEqual((task, target), ("C\n\nQ\n\n" + bp.MATH_INSTRUCTION, "$2$"))
        task, _ = bp.hard_task("olympiad", {**row, "context": None}, "olympiad/1")
        self.assertEqual(task, "Q\n\n" + bp.MATH_INSTRUCTION)


class CatalogTest(unittest.TestCase):
    def row(self, benchmark, problem_id, ids=(1, 2, 3), budget=8192):
        return bp.catalog_row(benchmark=benchmark, problem_id=problem_id, domain="math", task="t", target="secret",
                              prompt="p", ids=list(ids), family="ouro", budget=budget)

    def test_ids_and_budget(self):
        self.assertEqual(self.row("humaneval_plus", "HumanEval/0")["id"], "humaneval_plus/HumanEval/0")
        self.assertEqual(self.row("aime24", "aime24/60")["id"], "aime24/60")
        self.assertEqual(self.row("aime24", "aime24/60")["request_id"], bp.digest(["aime24", "aime24/60", 0]))
        with self.assertRaises(bp.IntegrityError):
            self.row("gsm8k", "x", ids=range(4), budget=4)

    def test_public_fields_hold_no_reference_data(self):
        self.assertFalse({"target", "task_prompt", "prompt", "entry_point"} & set(bp.PUBLIC_FIELDS))
        public = {key: self.row("gsm8k", "x")[key] for key in bp.PUBLIC_FIELDS}
        self.assertNotIn("secret", bp.catalog_bytes([public]).decode())

    def test_serialization(self):
        content = bp.catalog_bytes([{"b": "\u2028", "a": 1}])
        self.assertEqual(content, b'{"a":1,"b":"\\u2028"}\n')
        rows = [{"id": "x/1", "prompt_ids": [1, 2]}, {"id": "x/2", "prompt_ids": [3]}]
        self.assertEqual(bp.prompt_digest(rows), hashlib.sha256(b'[["x/1",[1,2]],["x/2",[3]]]').hexdigest())


class ConfigTest(unittest.TestCase):
    def test_pins_are_complete(self):
        config = json.loads(bp.BENCHMARKS_CONFIG.read_text())
        named = {name for suite in config["suites"].values()
                 for name in suite["benchmarks"] + suite.get("optional_benchmarks", [])}
        self.assertEqual(named, set(config["benchmarks"]))
        for name, pin in config["benchmarks"].items():
            self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$", name)
            self.assertGreater(pin["size_bytes"], 0, name)
            self.assertIn(pin["host"], ("huggingface", "github"), name)
            self.assertIn(pin["access"], ("open", "gated"), name)
            self.assertIn(pin["format"], ("parquet", "csv", "jsonl", "jsonl.gz"), name)
            self.assertTrue(pin["license"], name)
        self.assertEqual(config["benchmarks"]["gpqa_diamond"]["access"], "gated")
        self.assertEqual(config["suites"]["hard"]["optional_benchmarks"], ["aime26"])
        models = json.loads(bp.MODELS_CONFIG.read_text())["models"]
        families = {model["family"] for model in models.values()}
        for family in families:
            bp.prompt_renderer(None, family)  # every registered family has a prompt renderer
        # Inkling2 is not one of the paper's models: it has no reference catalog.
        self.assertEqual(families - set(config["reference_catalogs"]), {"inkling2"})
        self.assertEqual(families - {"inkling2"}, set(bp.TEMPLATE_KWARGS))
        for family in bp.TEMPLATE_KWARGS:
            # The paper's Dory runs used the Hard suite without AIME26 only.
            expected = {"core", "hard"} if family == "dory" else {"core", "hard", "hard+aime26"}
            self.assertEqual(set(config["reference_catalogs"][family]), expected)


# Inkling2

class StandInTokenizer:
    """Stand-in for the Inkling2 tokenizer: a framing token encodes to its id and any other character to
    its code point. Its template writes text messages in the format of the checkpoint's
    chat_template.jinja and records the keyword arguments of each rendering."""

    def __init__(self):
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return INKLING2_IDS.get(token)

    def encode(self, text, add_special_tokens=False, split_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        while text:
            token = None if split_special_tokens else next((t for t in INKLING2_IDS if text.startswith(t)), None)
            ids.append(INKLING2_IDS[token] if token else ord(text[0]))
            text = text[len(token) if token else 1:]
        return ids

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert tokenize is False
        self.calls.append(kwargs)
        effort = kwargs.get("reasoning_effort", 0.9)
        effort = f"<|message_system|><|content_text|>Thinking effort level: {effort or 0}<|end_message|>"
        text = ""
        for message in messages:
            if effort and message["role"] != "system":
                text, effort = text + effort, ""
            text += f"{oracle.ROLE_TOKENS[message['role']]}<|content_text|>{message['content']}<|end_message|>"
            if message["role"] == "assistant":
                text += "<|content_model_end_sampling|>"
        return text + effort


SYNTHETIC_ROWS = {
    "gsm8k": [{"question": "What is 2 + 3?", "answer": "2 + 3 = 5\n#### 5"},
              {"question": "Is <|end_message|> one token? ∑ café", "answer": "#### 1"}],
    "humaneval_plus": [{"task_id": "HumanEval/0", "prompt": 'def add(a, b):\n    """Add."""\n',
                        "canonical_solution": "    return a + b\n", "entry_point": "add"}],
}


def synthetic_suite(cache_dir):
    """benchmarks.json of a Core suite of SYNTHETIC_ROWS, whose pinned files are written to cache_dir."""
    benchmarks = {}
    for name, rows in SYNTHETIC_ROWS.items():
        content = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (cache_dir / name).mkdir(parents=True)
        (cache_dir / name / "rows.jsonl").write_bytes(content)
        code = "task_id" in rows[0]
        benchmarks[name] = {"domain": "code" if code else "math", "host": "github", "file": "rows.jsonl",
                            "format": "jsonl", "sha256": hashlib.sha256(content).hexdigest(),
                            "size_bytes": len(content), "rows": len(rows), "id_field": "task_id" if code else None,
                            "row_id_prefix": "row:", "license": "test", "access": "open"}
    return {"suites": {"core": {"max_total_tokens": 8192, "benchmarks": list(SYNTHETIC_ROWS),
                                "excluded_questions": []}}, "benchmarks": benchmarks}


def inkling2_snapshot():
    """The local Inkling-2-Mini snapshot (INKLING2_SNAPSHOT, else the pinned revision in the Hugging Face
    cache), or None without its tokenizer files."""
    explicit = os.environ.get("INKLING2_SNAPSHOT")
    if explicit:
        path = Path(explicit)
    else:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
        except ImportError:
            return None
        model = json.loads(bp.MODELS_CONFIG.read_text())["models"][MINI]
        path = Path(HF_HUB_CACHE) / ("models--" + model["repo_id"].replace("/", "--")) / "snapshots" / model["revision"]
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    return path if all((path / name).is_file() for name in files) else None


class Inkling2Effort(unittest.TestCase):
    def test_level_names_map_to_the_documented_numbers(self):
        """Guards the level table and the default against drifting from the documented ones (EFFORTS,
        0.9), which the oracle states independently."""
        self.assertEqual(chat.INKLING2_EFFORTS, EFFORTS)
        self.assertEqual(oracle.NAMED_EFFORT, EFFORTS)
        for name, number in EFFORTS.items():
            self.assertEqual(chat.inkling2_effort(name), number)
            self.assertEqual(oracle.resolve_effort(name), number)
        self.assertEqual((chat.inkling2_effort(), oracle.resolve_effort(None)), (0.9, 0.9))
        self.assertEqual([chat.inkling2_effort(value) for value in (0, -0.0, "0.25", 0.5, 0.99)],
                         [0.0, 0.0, 0.25, 0.5, 0.99])
        self.assertEqual(str(chat.inkling2_effort(-0.0)), "0.0")  # the manifest never holds -0.0

    def test_out_of_range_and_unknown_efforts_are_refused(self):
        """An effort outside [0, 0.99], an unknown name or a non-number must not reach the template."""
        for value in ("ultra", "", " high", True, 1, 1.0, -0.01, 0.995, float("nan"), float("inf"), [0.5],
                      10 ** 400, "1e400"):  # beyond the float range: still a ValueError, which the CLI reports
            with self.subTest(value=value), self.assertRaises(ValueError):
                chat.inkling2_effort(value)
        for value in (1.0, -0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                oracle.effort_text(oracle.resolve_effort(value))

    def test_an_effort_the_two_renderers_print_differently_is_refused(self):
        """The official renderer rounds to two decimals (0.555 -> 0.56) and the template prints the number
        as given, so a finer effort would make prompts that differ from the served ones."""
        self.assertEqual(oracle.effort_text(0.555), "Thinking effort level: 0.56")
        with self.assertRaises(ValueError):
            chat.inkling2_effort(0.555)

    def test_the_oracle_prints_efforts_as_the_official_test_suite_expects(self):
        """The oracle and the builder come from one reading of the official renderer; its own tests state
        the printed efforts independently of both, and the values it refuses."""
        for value, printed in OFFICIAL_EFFORT_TEXT:
            with self.subTest(value=value):
                self.assertEqual(oracle.effort_text(oracle.resolve_effort(value)), f"Thinking effort level: {printed}")
        for value in OFFICIAL_INVALID_EFFORTS:
            with self.subTest(value=value), self.assertRaises(ValueError):
                oracle.effort_text(oracle.resolve_effort(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                chat.inkling2_effort(value)


class Inkling2Cases:
    """Checks that hold on the stand-in tokenizer (Inkling2Format) and on the checkpoint's own tokenizer
    and chat template (Inkling2Checkpoint); subclasses set self.tokenizer."""

    def test_conversations_render_as_the_official_rule(self):
        """Guards the prompt ids against any difference from the official renderer: effort block, empty
        blocks, the end-of-sampling token after assistant turns, and text that spells framing tokens."""
        tokenizer = self.tokenizer
        for name, conversation in CONVERSATIONS.items():
            for effort in (None, "none", "low", 0.25, "max"):
                with self.subTest(conversation=name, effort=effort):
                    text, ids = chat.render_inkling2(tokenizer, conversation, effort)
                    self.assertEqual(ids, oracle.render(tokenizer, conversation, effort))
                    # The text is the template's own rendering of the conversation.
                    self.assertEqual(text, tokenizer.apply_chat_template(
                        conversation, tokenize=False, reasoning_effort=chat.inkling2_effort(effort)))
                    # Tokenizing that text gives the same ids unless a message spells a framing token.
                    whole = tokenizer.encode(text, add_special_tokens=False)
                    self.assertEqual(ids == whole, name not in SPELLS_FRAMING)

    def test_edge_conversations_render_as_the_official_rule(self):
        """Guards the positions the benchmark prompts never reach: the effort block ends a conversation of
        system turns, precedes a leading assistant turn and is not repeated before a later system turn;
        empty texts keep their blocks; an open assistant text follows each of them."""
        tokenizer = self.tokenizer
        for name, conversation in EDGE_CONVERSATIONS.items():
            with self.subTest(conversation=name):
                text, ids = chat.render_inkling2(tokenizer, conversation, "low")
                self.assertEqual(ids, oracle.render(tokenizer, conversation, "low"))
                self.assertEqual(text, tokenizer.apply_chat_template(conversation, tokenize=False,
                                                                     reasoning_effort=0.2))
                self.assertEqual(text.count("Thinking effort level: 0.2"), 1)
            for prefix in ("", "A:", "<|end_message|>"):
                with self.subTest(conversation=name, prefix=prefix):
                    opened = [*conversation, turn("assistant", prefix)]
                    text, ids = chat.render_inkling2(tokenizer, opened, continue_final=True)
                    self.assertEqual(ids, oracle.render(tokenizer, opened, continue_final_message=True))
                    tail = tokenizer.encode(prefix, add_special_tokens=False, split_special_tokens=True)
                    self.assertEqual(ids[-len(tail) - 2:], [MODEL, TEXT, *tail])
                    self.assertTrue(text.endswith("<|end_message|><|message_model|><|content_text|>" + prefix))

    def test_the_expectations_of_the_official_test_suite_hold(self):
        """Guards against one misreading of the official renderer shared by the oracle and the builder:
        the renderings its own tests expect are quoted in OFFICIAL_RENDERINGS."""
        tokenizer = self.tokenizer
        for conversation, continued, expected in OFFICIAL_RENDERINGS:
            with self.subTest(expected=expected):
                text, ids = chat.render_inkling2(tokenizer, conversation, continue_final=continued)
                self.assertEqual(text, expected)
                self.assertEqual(ids, oracle.render(tokenizer, conversation, continue_final_message=continued))
        for value, printed in OFFICIAL_EFFORT_TEXT:
            if value != 0.555:  # refused by the builder: the template would print 0.555
                with self.subTest(effort=value):
                    text = chat.render_inkling2(tokenizer, [turn("user", "hi")], value)[0]
                    self.assertTrue(text.startswith(OFFICIAL_EFFORT_BLOCK.replace("0.9", printed)))

    def test_a_prompt_ends_with_the_user_turn_and_opens_no_model_turn(self):
        """A generation prefix would make the model skip its own <|message_model|> opener."""
        for name, conversation in CONVERSATIONS.items():
            with self.subTest(conversation=name):
                ids = chat.render_inkling2(self.tokenizer, conversation)[1]
                self.assertEqual(ids[-1], END_MESSAGE)
                roles = [token for token in ids if token in (200000, 200001, 200002)]
                self.assertEqual(roles[-1], USER)
                # End of sampling follows the assistant turns of the history and nothing else.
                self.assertEqual(ids.count(EOS), sum(m["role"] == "assistant" for m in conversation))
                self.assertLess(max(ids), SERVED_VOCABULARY)

    def test_message_text_never_becomes_a_framing_token(self):
        """A question that quotes <|end_message|> must not end its own turn or open a model turn."""
        conversation = [turn("user", "A<|end_message|><|message_model|><|content_text|>B")]
        text, ids = chat.render_inkling2(self.tokenizer, conversation)
        self.assertEqual(ids.count(END_MESSAGE), 2)  # the effort block and the user turn
        self.assertNotIn(MODEL, ids)
        self.assertIn(MODEL, self.tokenizer.encode(text, add_special_tokens=False))  # the misreading

    def test_effort_levels_print_the_documented_numbers(self):
        """Each level name renders as its number, and the default as "high"."""
        conversation = CONVERSATIONS["single user turn"]
        for name, number in EFFORTS.items():
            with self.subTest(effort=name):
                text, ids = chat.render_inkling2(self.tokenizer, conversation, name)
                self.assertEqual((text, ids), chat.render_inkling2(self.tokenizer, conversation, number))
                self.assertEqual(ids, oracle.render(self.tokenizer, conversation, number))
                printed = "0" if number == 0 else str(number)
                self.assertTrue(text.startswith(
                    f"<|message_system|><|content_text|>Thinking effort level: {printed}<|end_message|>"))
        self.assertEqual(chat.render_inkling2(self.tokenizer, conversation),
                         chat.render_inkling2(self.tokenizer, conversation, "high"))

    def test_every_accepted_effort_number_prints_as_the_official_renderer_prints(self):
        """The template prints the float and the official renderer formats a rounded one: they must agree
        on every number --reasoning-effort accepts (two decimals in [0, 0.99])."""
        conversation = CONVERSATIONS["single user turn"]
        for hundredths in range(100):
            effort = chat.inkling2_effort(hundredths / 100)
            self.assertEqual(chat.render_inkling2(self.tokenizer, conversation, effort)[1],
                             oracle.render(self.tokenizer, conversation, effort), effort)

    def test_the_math_prompt_is_one_user_turn(self):
        prompt, ids = bp.render_inkling2_prompt(self.tokenizer, "Q?", "user_chat", "low")
        self.assertEqual((prompt, ids), chat.render_inkling2(self.tokenizer, [turn("user", "Q?")], 0.2))
        self.assertEqual(ids, oracle.render(self.tokenizer, [turn("user", "Q?")], 0.2))
        self.assertTrue(prompt.endswith("<|message_user|><|content_text|>Q?<|end_message|>"))

    def test_the_code_prompt_leaves_the_model_text_block_open(self):
        """The EvalPlus assistant prefix of the other families, as the official continue_final_message
        renders it: <|message_model|><|content_text|> and the prefix, not closed."""
        tokenizer = self.tokenizer
        task = 'def f(x):\n    """Return x."""\n'
        user, prefix = bp.evalplus_turns(task)
        prompt, ids = bp.render_inkling2_prompt(tokenizer, task, "evalplus_chat")
        self.assertEqual(ids, oracle.render(tokenizer, [turn("user", user), turn("assistant", prefix)],
                                            continue_final_message=True))
        tail = tokenizer.encode(prefix, add_special_tokens=False)
        self.assertEqual(ids[-len(tail) - 2:], [MODEL, TEXT, *tail])
        self.assertNotIn(EOS, ids)
        self.assertTrue(prefix.endswith("```python\n") and prompt.endswith("<|content_text|>" + prefix))
        # The splitter cut that renders the prefix of the other families gives the same prompt.
        closed = [turn("user", user), turn("assistant", f"{prefix}{bp.EVALPLUS_SPLITTER}\n```\n")]
        rendered = tokenizer.apply_chat_template(closed, tokenize=False, reasoning_effort=0.9)
        self.assertEqual(prompt, bp.split_assistant_prefix(rendered))
        self.assertEqual(ids, tokenizer.encode(prompt, add_special_tokens=False))

    def test_trace_prompts_use_the_same_rendering(self):
        """training.common.chat.render_native (trace generation) renders Inkling2 as the builder does."""
        tokenizer = self.tokenizer
        self.assertEqual(chat.render_native(tokenizer, "inkling2", "Q?"),
                         oracle.render(tokenizer, [turn("user", "Q?")]))
        self.assertEqual(chat.render_native(tokenizer, "inkling2", "Q?", "A:", effort="medium"),
                         oracle.render(tokenizer, [turn("user", "Q?"), turn("assistant", "A:")], 0.7,
                                       continue_final_message=True))

    def test_the_builder_writes_the_release_format_and_records_the_effort(self):
        """End to end through main() on a synthetic Core suite: the prompts file keeps the fields
        scripts/benchmark.py reads, its ids are the official ones, and the manifest names the effort."""
        tokenizer = self.tokenizer
        models = json.loads(bp.MODELS_CONFIG.read_text())["models"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmarks.json").write_text(json.dumps(synthetic_suite(root / "cache")))
            stdout = io.StringIO()
            with (patch.object(bp, "BENCHMARKS_CONFIG", root / "benchmarks.json"),
                  patch.object(bp, "load_tokenizer", return_value=(tokenizer, {"tokenizer.json": "0" * 64})) as loader,
                  contextlib.redirect_stdout(stdout)):
                status = bp.main(["--model", MINI, "--suite", "core", "--out", str(root / "out"), "--offline",
                                  "--cache-dir", str(root / "cache"), "--reasoning-effort", "medium"])
            content = (root / "out" / "core.prompts.jsonl").read_bytes()
            manifest = json.loads((root / "out" / "core.manifest.json").read_text())
        self.assertEqual(status, 0)  # no reference catalog to differ from
        # The tokenizer and template come from the pinned revision of the registry.
        loader.assert_called_once_with(models[MINI], True)
        self.assertEqual(models[MINI]["revision"], "1fb4134b9a74f9681054f4ef82e4bf1f458a2337")

        expected = {}
        for index, row in enumerate(SYNTHETIC_ROWS["gsm8k"]):
            task = bp.core_task("gsm8k", row)[0]
            expected[f"gsm8k/row:{index}"] = oracle.render(tokenizer, [turn("user", task)], 0.7)
        user, prefix = bp.evalplus_turns(bp.core_task("humaneval_plus", SYNTHETIC_ROWS["humaneval_plus"][0])[0])
        expected["humaneval_plus/HumanEval/0"] = oracle.render(
            tokenizer, [turn("user", user), turn("assistant", prefix)], 0.7, continue_final_message=True)
        rows = [json.loads(line) for line in content.decode().splitlines()]
        self.assertEqual([row["id"] for row in rows], sorted(expected))
        self.assertEqual({row["id"]: row["prompt_ids"] for row in rows}, expected)
        for row in rows:
            self.assertEqual(set(row), set(bp.PUBLIC_FIELDS))
            self.assertEqual((row["rendered_family"], row["max_total_tokens"], row["prompt_tokens"]),
                             ("inkling2", 8192, len(row["prompt_ids"])))
            if row["domain"] == "math":
                self.assertEqual(row["prompt_ids"][-1], END_MESSAGE)
                self.assertNotIn(MODEL, row["prompt_ids"])
        self.assertEqual((manifest["rendered_family"], manifest["reasoning_effort"], manifest["questions"]),
                         ("inkling2", 0.7, 3))
        self.assertEqual((manifest["model_repo_id"], manifest["model_revision"]),
                         (models[MINI]["repo_id"], models[MINI]["revision"]))
        self.assertIsNone(manifest["reference"])
        self.assertEqual(manifest["prompt_digest"], bp.prompt_digest(rows))
        self.assertEqual(manifest["prompts_sha256"], hashlib.sha256(content).hexdigest())
        summary = json.loads(stdout.getvalue())
        self.assertEqual((summary["reasoning_effort"], summary["matches_paper_catalog"]), (0.7, None))


class Inkling2Format(Inkling2Cases, unittest.TestCase):
    def setUp(self):
        self.tokenizer = StandInTokenizer()

    def test_template_gets_the_effort_and_neither_thinking_switch_nor_generation_prompt(self):
        """The template knows no enable_thinking and no add_generation_prompt: passing them would hide
        that the prompt carries neither."""
        bp.render_inkling2_prompt(self.tokenizer, "Q", "user_chat")
        bp.render_inkling2_prompt(self.tokenizer, "Q", "evalplus_chat", "low")
        chat.render_native(self.tokenizer, "inkling2", "Q")
        self.assertEqual(self.tokenizer.calls, [{"reasoning_effort": 0.9}, {"reasoning_effort": 0.2},
                                                {"reasoning_effort": 0.9}])

    def test_the_oracle_reproduces_the_renderings_of_the_official_test_suite(self):
        """Every other Inkling2 test trusts the oracle. Its ids, read back one character per text id as the
        official tests read theirs, must spell the renderings those tests expect, and quoted framing
        tokens must stay text."""
        spelling = {token_id: token for token, token_id in INKLING2_IDS.items()}
        for conversation, continued, expected in OFFICIAL_RENDERINGS:
            with self.subTest(expected=expected):
                ids = oracle.render(self.tokenizer, conversation, continue_final_message=continued)
                self.assertEqual("".join(spelling.get(token) or chr(token) for token in ids), expected)
                quoted = sum(message["content"].count("<|end_message|>") for message in conversation)
                self.assertEqual(ids.count(END_MESSAGE), expected.count("<|end_message|>") - quoted)

    def test_messages_outside_plain_text_are_refused(self):
        """Tools, content parts and reasoning have rendering rules the builder does not implement."""
        cases = ([turn("tool", "x")], [turn("user", [{"type": "text", "text": "x"}])], [turn("user", None)],
                 [turn("user", "x"), {"role": "assistant", "content": "y", "reasoning_content": "z"}])
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                chat.render_inkling2(self.tokenizer, messages)
        with self.assertRaises(ValueError):  # only an assistant text can be left open
            chat.render_inkling2(self.tokenizer, [turn("user", "x")], continue_final=True)
        with self.assertRaises(ValueError):
            bp.render_inkling2_prompt(self.tokenizer, "Q", "other")

    def test_a_template_that_changes_the_text_layout_is_refused(self):
        """The ids are assembled around each message text, so a template that drops or repeats a text,
        or closes an assistant text differently, must fail instead of yielding other prompts."""
        tokenizer = self.tokenizer
        conversation = [turn("user", "x"), turn("assistant", "y")]
        for change in (lambda text: "", lambda text: text + text, lambda text: text + "<|end_message|>"):
            render = tokenizer.apply_chat_template
            with (self.subTest(result=change("T")),
                  patch.object(tokenizer, "apply_chat_template", lambda *a, **k: change(render(*a, **k))),
                  self.assertRaises(ValueError)):
                chat.render_inkling2(tokenizer, conversation, continue_final=True)

    def test_an_effort_is_refused_for_the_other_families(self):
        """--reasoning-effort on a paper model would be silently ignored by its template."""
        with (patch.object(bp, "load_tokenizer", side_effect=AssertionError("not reached")),
              self.assertRaises(ValueError)):
            bp.build("ouro-1.4b-thinking", "core", "unused", "unused", reasoning_effort="high")
        with self.assertRaises(ValueError):
            chat.render_native(self.tokenizer, "ouro", "Q", effort=0.5)
        with (patch.object(bp, "load_tokenizer", side_effect=AssertionError("not reached")),
              self.assertRaises(ValueError)):
            bp.build(MINI, "core", "unused", "unused", reasoning_effort=1.5)


class Inkling2Checkpoint(Inkling2Cases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = inkling2_snapshot()
        if cls.snapshot is None:
            raise unittest.SkipTest("no local snapshot of Inkling-2-Mini (set INKLING2_SNAPSHOT)")
        cls.tokenizer = chat.load_tokenizer(cls.snapshot)

    def test_the_tokenizer_serves_200058_ids_and_stops_at_200006(self):
        """The served vocabulary is the tokenizer's 200,058 ids (the 966 padding rows of the embedding
        table have no token), and the only EOS is <|content_model_end_sampling|>."""
        from training.common import registry

        tokenizer = self.tokenizer
        self.assertEqual(len(tokenizer), SERVED_VOCABULARY)
        self.assertEqual(max(tokenizer.get_vocab().values()), SERVED_VOCABULARY - 1)
        self.assertEqual((tokenizer.eos_token, tokenizer.eos_token_id), ("<|content_model_end_sampling|>", EOS))
        self.assertEqual(registry.eos_token_ids("inkling2"), [EOS])
        self.assertEqual(chat.eos_ids(tokenizer, self.snapshot), [EOS])
        for token, token_id in INKLING2_IDS.items():
            self.assertEqual(tokenizer.convert_tokens_to_ids(token), token_id, token)
            self.assertEqual(tokenizer.encode(token, add_special_tokens=False), [token_id], token)

    def test_the_files_the_builder_fetches_load_the_same_tokenizer_and_template(self):
        """build_prompts.py downloads the files of TOKENIZER_PATTERNS only, while the repository also holds
        tiktoken/ and processor files: guards its Hub path against a pattern list that misses a file."""
        import fnmatch

        with tempfile.TemporaryDirectory() as directory:
            for path in self.snapshot.iterdir():
                if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in bp.TOKENIZER_PATTERNS):
                    (Path(directory) / path.name).symlink_to(path.resolve())
            tokenizer = chat.load_tokenizer(directory)
        self.assertEqual((len(tokenizer), tokenizer.eos_token_id), (SERVED_VOCABULARY, EOS))
        self.assertEqual(tokenizer.chat_template, self.tokenizer.chat_template)
        for conversation in CONVERSATIONS.values():
            self.assertEqual(chat.render_inkling2(tokenizer, conversation),
                             chat.render_inkling2(self.tokenizer, conversation))

    def test_the_template_has_no_thinking_switch_generation_prompt_or_other_default(self):
        """Guards the premises of the builder against a changed template: enable_thinking and
        add_generation_prompt change nothing, and no effort means 0.9."""
        tokenizer = self.tokenizer
        for name in ("enable_thinking", "add_generation_prompt"):
            self.assertNotIn(name, tokenizer.chat_template)
        for conversation in CONVERSATIONS.values():
            text = tokenizer.apply_chat_template(conversation, tokenize=False)
            self.assertEqual(text, chat.render_inkling2(tokenizer, conversation)[0])
            self.assertEqual(text, tokenizer.apply_chat_template(conversation, tokenize=False, enable_thinking=True,
                                                                 add_generation_prompt=True))

    def test_the_template_maps_the_level_names_itself_to_the_same_numbers(self):
        """The builder resolves a level name before rendering; the template's own table must agree."""
        conversation = CONVERSATIONS["single user turn"]
        for name in EFFORTS:
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, reasoning_effort=name)
            self.assertEqual(text, chat.render_inkling2(self.tokenizer, conversation, name)[0], name)


if __name__ == "__main__":
    unittest.main()
