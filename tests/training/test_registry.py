# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""training.common.registry on CPU: the Inkling2 entry of configs/models.json (pinned, per-loop KV only,
no drafter yet), the model_type -> family lookup, and the constants it shares with the vLLM fork."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.common import registry

MINI = "inkling-2-mini"
MINI_PIN = ("thinkingmachines/Inkling-2-Mini", "1fb4134b9a74f9681054f4ef82e4bf1f458a2337")
VARLOOP_PIN = ("thinkingmachines/TML-d24-varloop-research", "657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2")
# text_config.loop_config of Inkling-2-Mini's config.json.
MINI_LOOP = {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}


class Inkling2Entry(unittest.TestCase):
    def test_the_name_resolves_to_the_pinned_revision(self):
        """Every lookup carries the pinned revision: the repository's main branch can change the chat
        template, and with it every prompt."""
        self.assertEqual(registry.pinned(MINI), MINI_PIN)
        self.assertEqual(registry.revision(MINI), MINI_PIN[1])
        self.assertEqual(registry.label(MINI), "@".join(MINI_PIN))
        self.assertEqual(registry.family(MINI), "inkling2")
        self.assertEqual(registry.entry(MINI)["loops"], 3)

    def test_serving_runs_the_pinned_original_target(self):
        """Its default target is the original one (per-loop KV), not w64."""
        expected = {"model": MINI_PIN[0], "revision": MINI_PIN[1], "drafter": None, "kv_policy": "native",
                    "family": "inkling2"}
        self.assertEqual(registry.serving(MINI, with_drafter=False), expected)
        self.assertEqual(registry.serving(MINI, "original", with_drafter=False), expected)
        self.assertEqual(registry.serving(MINI, revision="abc", with_drafter=False), dict(expected, revision="abc"))

    def test_an_unknown_target_is_not_served_as_the_original_one(self):
        """Guards the single-target default: only a missing target means "original"; any other name is
        rejected as it is for the other families, not replaced."""
        for name in (MINI, "ouro-1.4b-thinking"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "target variant must be one of"):
                registry.serving(name, "shared", with_drafter=False)

    def test_inkling2_serves_its_per_loop_model_only(self):
        """Inkling2 has no KV sharing: the original target with the native policy is the only one served, and
        a request for the shared target or policy is an error rather than another run."""
        pin = registry.pinned(MINI)
        base = {"model": pin[0], "revision": pin[1], "drafter": None, "family": "inkling2", "kv_policy": "native"}
        for options in ({}, {"target": "original"}, {"kv_policy": "native"}):
            with self.subTest(options=options):
                self.assertEqual(registry.serving(MINI, with_drafter=False, **options), base)
        for options in ({"target": "w64"}, {"kv_policy": "w64"}, {"target": "w64", "kv_policy": "native"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "per-loop KV only"):
                registry.serving(MINI, with_drafter=False, **options)
        self.assertEqual(registry.serving("dory1.0-4b", with_drafter=False)["kv_policy"], "w64")
        with self.assertRaisesRegex(ValueError, "applies to Dory only"):
            registry.serving("ouro-1.4b-thinking", kv_policy="w64", with_drafter=False)

    def test_a_drafter_is_required_but_not_registered(self):
        """A model without drafters is a valid entry; a run that drafts must name its drafter."""
        for variant in registry.VARIANTS:
            self.assertIsNone(registry.pinned_drafter(MINI, variant))
        for target in (None, "original"):
            with (self.subTest(target=target),
                  self.assertRaisesRegex(ValueError, f"{MINI} has no published drafter for its original target.*"
                                         "--drafter, or --block-size 1")):
                registry.serving(MINI, target)
        self.assertEqual(registry.serving(MINI, drafter="me/drafter@1")["drafter"], "me/drafter@1")

    def test_an_entry_may_omit_its_drafters(self):
        """drafters absent, null or empty all mean "none": the drafter-free run works and a drafting run
        gets the registry's error, not a KeyError or an AttributeError."""
        base = {"repo_id": "org/looped", "revision": "0" * 40, "family": "ouro", "loops": 4, "w64": "org/shared"}
        for drafters in ({}, {"drafters": None}, {"drafters": {}}, {"drafters": {"w64": None}}):
            models = {"m": {**base, **drafters}}
            with self.subTest(drafters=drafters), patch.object(registry, "_registry", lambda: models):
                self.assertIsNone(registry.pinned_drafter("m", "w64"))
                self.assertEqual(registry.serving("m", with_drafter=False)["model"], "org/shared")
                with self.assertRaisesRegex(ValueError, "m has no published drafter for its w64 target"):
                    registry.serving("m")


class DrafterKinds(unittest.TestCase):
    def test_a_registry_name_serves_the_drafter_of_the_kind_asked_for(self):
        """`drafters` are a model's DFlash drafters (the default kind) and `dspark_drafters` its DSpark ones;
        a kind without a published drafter is an error that names it, and an explicit drafter or a model
        outside the registry takes no kind: the export tells what it is."""
        models = {"m": {"repo_id": "org/looped", "revision": "0" * 40, "family": "ouro", "loops": 4,
                        "w64": "org/shared", "drafters": {"w64": "org/dflash", "original": "org/dflash-base"},
                        "dspark_drafters": {"w64": "org/dspark"}}}
        with patch.object(registry, "_registry", lambda: models):
            self.assertEqual(registry.serving("m")["drafter"], "org/dflash")
            self.assertEqual(registry.serving("m", drafter_kind="dflash")["drafter"], "org/dflash")
            self.assertEqual(registry.serving("m", drafter_kind="dspark")["drafter"], "org/dspark")
            self.assertEqual(registry.pinned_drafter("m", "original", "dspark"), None)
            with self.assertRaisesRegex(ValueError, "m has no published dspark drafter for its original target"):
                registry.serving("m", "original", drafter_kind="dspark")
            self.assertIsNone(registry.serving("m", drafter_kind="dspark", with_drafter=False)["drafter"])
            with self.assertRaisesRegex(ValueError, "the export itself tells its kind"):
                registry.serving("m", drafter="local/export", drafter_kind="dspark")
            with self.assertRaisesRegex(ValueError, "unknown drafter kind"):
                registry.pinned_drafter("m", "w64", "eagle")
        with self.assertRaisesRegex(ValueError, "the export itself tells its kind"):
            registry.serving("org/elsewhere", drafter_kind="dspark", with_drafter=False)
        # A model whose only drafter is a DSpark one (Inkling2) runs it without a kind; asking for DFlash is an error.
        only = {"i": {"repo_id": "org/inkling", "revision": "0" * 40, "family": "inkling2", "loops": 3, "w64": None,
                      "drafters": {}, "dspark_drafters": {"original": "org/dspark@abc"}}}
        with patch.object(registry, "_registry", lambda: only):
            self.assertEqual(registry.serving("i")["drafter"], "org/dspark@abc")
            self.assertEqual(registry.serving("i", drafter_kind="dspark")["drafter"], "org/dspark@abc")
            with self.assertRaisesRegex(ValueError, "i has no published drafter for its original target"):
                registry.serving("i", drafter_kind="dflash")
        # The released drafters are DFlash drafters, but for the variable-loop Inkling2 checkpoint's DSpark one,
        # which is that model's drafter without a kind.
        published = {(name, variant): registry.pinned_drafter(name, variant, "dspark")
                     for name in registry.names() for variant in registry.VARIANTS}
        varloop = ("minseo25/2dsd-inkling-2-mini-varloop-dspark-drafter", None)
        self.assertEqual({key: pin for key, pin in published.items() if pin},
                         {("inkling-2-mini-varloop", "original"): varloop})
        self.assertEqual(registry.serving("inkling-2-mini-varloop")["drafter"], varloop[0])


class FamilyLookup(unittest.TestCase):
    def checkpoint(self, config):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "config.json").write_text(json.dumps(config))
        return directory.name

    def test_inkling2_loop_checkpoints_are_the_inkling2_family(self):
        config = {"model_type": "inkling2_loop", "text_config": {"loop_config": MINI_LOOP}}
        self.assertEqual(registry.family(self.checkpoint(config)), "inkling2")
        self.assertEqual(registry.model_type_family("inkling2_loop"), "inkling2")

    def test_the_other_families_are_named_by_their_model_type(self):
        for name in ("ouro", "nanbeige", "dory"):
            with self.subTest(family=name):
                self.assertEqual(registry.family(self.checkpoint({"model_type": name})), name)
                self.assertEqual(registry.model_type_family(name), name)

    def test_an_unknown_model_type_has_no_family(self):
        """Never a default family: a dense model, Inkling 1, and the non-looped Inkling 2 (whose
        model_type is the family name "inkling2") all raise."""
        for model_type in ("qwen3", "inkling_mm_model", "inkling2", None):
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, "unsupported model_type"):
                    registry.family(self.checkpoint({"model_type": model_type}))
                with self.assertRaisesRegex(ValueError, "supported: ouro, nanbeige, dory, inkling2_loop"):
                    registry.model_type_family(model_type)

    def test_every_registry_entry_has_a_known_family(self):
        self.assertEqual(tuple(registry.FAMILY_MODEL_TYPES), registry.FAMILIES)
        for name in registry.names():
            with self.subTest(name=name):
                self.assertIn(registry.family(name), registry.FAMILIES)

    def test_the_variable_loop_checkpoint_is_a_model_of_its_own(self):
        """inkling-2-mini-varloop shares Inkling-2-Mini's family, loops and tokenizer but is another pinned
        checkpoint under another name: traces and drafters name the model they belong to."""
        name = "inkling-2-mini-varloop"
        mini, varloop = registry.entry(MINI), registry.entry(name)
        self.assertEqual((registry.family(name), varloop["loops"]), (registry.family(MINI), mini["loops"]))
        self.assertEqual(registry.pinned(name, "original"),
                         ("thinkingmachines/TML-d24-varloop-research", "657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2"))
        self.assertNotEqual(registry.pinned(name, "original"), registry.pinned(MINI, "original"))
        self.assertEqual(registry.label(name, "original"),
                         "thinkingmachines/TML-d24-varloop-research@657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2")

    def test_only_inkling2_has_fixed_eos_ids(self):
        """[200006] is <|content_model_end_sampling|>; the other families read their checkpoints."""
        self.assertEqual(registry.eos_token_ids("inkling2"), [200006])
        for name in ("ouro", "nanbeige", "dory"):
            self.assertIsNone(registry.eos_token_ids(name))
        for unknown in ("inkling2_loop", "qwen3", None):
            with self.subTest(family=unknown), self.assertRaisesRegex(ValueError, "unknown model family"):
                registry.eos_token_ids(unknown)


class ForkContract(unittest.TestCase):
    def test_the_registry_and_the_vllm_fork_agree_on_inkling2(self):
        """The fork (executor) cannot import the registry (training, scripts), so each holds its own
        copy of the model types, the EOS ids and the loop count; the copies must not drift apart."""
        from vllm.model_executor.models.looped_kv import INKLING2_MODEL_TYPE, loop_count
        from vllm.v1.spec_decode import twodsd

        self.assertEqual(tuple(registry.FAMILY_MODEL_TYPES.values()), twodsd.MODEL_TYPES)
        self.assertEqual(registry.FAMILY_MODEL_TYPES["inkling2"], twodsd.INKLING2)
        self.assertEqual(twodsd.INKLING2, INKLING2_MODEL_TYPE)
        self.assertEqual(registry.eos_token_ids("inkling2"), list(twodsd.INKLING2_EOS_TOKEN_IDS))
        config = {"model_type": "inkling2_loop", "text_config": {"loop_config": MINI_LOOP}}
        self.assertEqual(loop_count(config), registry.entry(MINI)["loops"])


class ReleasedCheckpoints(unittest.TestCase):
    """The constants above are copied by hand, so tests built on them agree with the code even when both
    are wrong. These read the JSON files of the released checkpoints instead: the snapshot directory
    named by an environment variable, else the pinned revision in the Hugging Face cache (HF_HOME /
    HF_HUB_CACHE); skipped without a local copy."""

    def snapshot(self, variable, repo, revision, files=("config.json",)):
        explicit = os.environ.get(variable)
        if explicit:
            path = Path(explicit)
        else:
            from huggingface_hub.constants import HF_HUB_CACHE

            path = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        if not all((path / name).is_file() for name in files):
            self.skipTest(f"no local snapshot of {repo} (set {variable})")
        return path

    def test_mini_carries_the_registered_loop_count_eos_id_and_block_size(self):
        from vllm.model_executor.models.looped_kv import loop_count
        from vllm.v1.spec_decode import twodsd

        path = self.snapshot("INKLING2_SNAPSHOT", *MINI_PIN, files=(
            "config.json", "generation_config.json", "special_tokens_map.json", "tokenizer.json"))
        config = json.loads((path / "config.json").read_text())
        self.assertEqual(config["text_config"]["loop_config"], MINI_LOOP)
        self.assertEqual(registry.family(path), "inkling2")
        self.assertEqual(twodsd.model_type(str(path)), twodsd.INKLING2)
        self.assertEqual(loop_count(config), registry.entry(MINI)["loops"])
        self.assertEqual(twodsd.inkling2_schedule(config, {}),
                         {"loops": 3, "stage": "stage2b", "k": config["mtp_config"]["block_size"], "d1": 1, "d2": 2})
        # One EOS id in config.json, generation_config.json and the tokenizer: the end-of-sampling token.
        eos = registry.eos_token_ids("inkling2")
        self.assertEqual([config["eos_token_id"]], eos)
        self.assertEqual([json.loads((path / "generation_config.json").read_text())["eos_token_id"]], eos)
        name = json.loads((path / "special_tokens_map.json").read_text())["eos_token"]
        added = json.loads((path / "tokenizer.json").read_text())["added_tokens"]
        self.assertEqual(name, "<|content_model_end_sampling|>")
        self.assertEqual([token["id"] for token in added if token["content"] == name], eos)

    def test_the_variable_loop_checkpoint_is_the_same_family_at_three_passes(self):
        from vllm.model_executor.models.looped_kv import loop_count

        path = self.snapshot("INKLING2_VARLOOP_SNAPSHOT", *VARLOOP_PIN)
        config = json.loads((path / "config.json").read_text())
        self.assertEqual(config["text_config"]["loop_config"], dict(MINI_LOOP, is_varloop=True))
        self.assertEqual(registry.family(path), "inkling2")
        self.assertEqual(loop_count(config), 3)


if __name__ == "__main__":
    unittest.main()
