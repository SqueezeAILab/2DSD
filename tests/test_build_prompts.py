"""CPU tests of the pure functions of scripts/build_prompts.py. No downloads and no model files.

The pinned selections and choice orders below were computed with the code that built the
paper's catalogs; the end-to-end check is the catalog_sha256 comparison of build_prompts.py.
"""
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("build_prompts", ROOT / "scripts" / "build_prompts.py")
bp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bp)

SEED = 20260930


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
        self.assertTrue(families <= set(bp.TEMPLATE_KWARGS))
        for family in families:
            # The paper's Dory runs used the Hard suite without AIME26 only.
            expected = {"core", "hard"} if family == "dory" else {"core", "hard", "hard+aime26"}
            self.assertEqual(set(config["reference_catalogs"][family]), expected)


if __name__ == "__main__":
    unittest.main()
