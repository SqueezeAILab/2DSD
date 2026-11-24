"""Inkling 2 drafter: the training target against the tree runner, the DSpark recipe (speculators' heads,
loss and optimizers), the export and 2DSD with the exported drafter (CPU, tiny checkpoint in the released
format)."""
import copy
import importlib.util
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch.nn.attention.flex_attention import create_mask
from vllm.model_executor.models import inkling2_weights as weights
from vllm.model_executor.models.inkling2_batch import Inkling2BatchRunner
from vllm.model_executor.models.inkling2_reference import Inkling2Cache, NativeInkling2, drafter_feature_layers
from vllm.v1.spec_decode.looped_dflash_context import ContextDFlashAdapter
from vllm.v1.spec_decode.twodsd import TwoDSD

from training.drafter import export
from speculators.losses import resolve_loss_config
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead

from training.drafter.blocks import Blocks, block_labels, build_context_inputs, context_mask_mod
from training.drafter.data import build_packed
from training.drafter.model import DSPARK, DepthDFlash, drafter_config, drafter_forward, dspark_loss, softcap
from training.drafter.targets import Inkling2Target, native_target, target_feature_layers
from training.drafter.data import load_traces
from training.drafter.train import (build_optimizers, build_parser, load_resume_entry, parse_drafter, resume_entry,
                                    trace_expectation)
from training.common.io import file_sha
import shutil

from vllm.v1.spec_decode.twodsd import check_inkling2_drafter_target

spec = importlib.util.spec_from_file_location("inkling2_tiny_drafter",
                                              Path(__file__).parents[1] / "vllm" / "inkling2_tiny.py")
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)

PROMPT, R = [5, 17, 3, 42, 9, 11, 2], 3
TOLERANCE = dict(rtol=0, atol=1e-9)


def setUpModule():
    global DIRECTORY, MODEL_DIR, WANT
    DIRECTORY = tempfile.TemporaryDirectory()
    loop = dict(core_start=2, core_end=6, num_passes=R, gate_rank=4, is_varloop=False)
    config = tiny.tiny_config(num_hidden_layers=8, local_layer_ids=[0, 2, 4, 6], loop_config=loop)
    MODEL_DIR = tiny.write_tiny_checkpoint(Path(DIRECTORY.name) / "model", config, mxfp8_layers=(6, 7))
    # The export reads the tokenizer for the first unused embedding row: 100 served ids, 112 rows.
    (MODEL_DIR / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(100)}},
                                                          "added_tokens": []}))
    model = NativeInkling2(weights.load_inkling2_checkpoint(MODEL_DIR), "exact")
    cache, step, WANT = Inkling2Cache(), PROMPT, []
    while len(WANT) < 24:
        WANT.append(int(model.forward(step, cache=cache, pass_end=False).readouts[R].argmax[-1]))
        step = [WANT[-1]]


def tearDownModule():
    DIRECTORY.cleanup()


def load(precision="exact", dtype=torch.bfloat16):
    return NativeInkling2(weights.load_inkling2_checkpoint(MODEL_DIR, dtype=dtype), precision)


class TargetTests(unittest.TestCase):
    def test_features_are_what_the_tree_runner_hands_the_drafter(self):
        """Teacher forcing gives, for every position and depth, the features the runner holds for that token
        at that depth, verified or in flight behind unverified tokens: the drafter is trained on its
        run-time inputs."""
        model = load()
        target = Inkling2Target.from_model(model)
        self.assertEqual((target.feat_layers, target.R, target.H), ([2, 3, 4, 5, 6], R, 32))
        ids = PROMPT + WANT[:16]
        feats, states = target.features(torch.tensor(ids))
        self.assertEqual((tuple(feats.shape), tuple(states.shape)), ((R, len(ids), 5 * 32), (R, len(ids), 32)))
        # The last feature is the input of coda layer 6 at each depth: it differs by depth and is not
        # the final-norm state, which is no feature.
        self.assertFalse(torch.allclose(feats[0, :, -32:], feats[R - 1, :, -32:]))
        self.assertFalse(torch.allclose(feats[..., -32:], states))
        runner = Inkling2BatchRunner(model, (1, 2, 3), max_len=64)
        self.assertEqual(runner.feature_layers, tuple(target.feat_layers))
        prefill = runner.prefill(torch.tensor(PROMPT))
        torch.testing.assert_close(prefill.features, feats[R - 1, :len(PROMPT)], **TOLERANCE)
        depth, checked = {}, 0
        tokens = ids[len(PROMPT):]
        committed = torch.empty(len(tokens), feats.shape[-1], dtype=feats.dtype)
        for step in range(len(tokens) + R - 1):  # up to three tokens in flight, one pass apart
            if step < len(tokens):
                runner.register_many([(step, step - 1 if step - 1 in depth else None, tokens[step])])
                depth[step] = 0
            live = sorted(depth)
            rows, argmax, logits = runner.advance_nodes(live, [depth[node] for node in live])
            for node in live:
                depth[node] += 1
            for i, row in enumerate(rows):
                node = live[row]
                position = len(PROMPT) + node
                got = runner.span_features([(depth[node], [node])])[0][0]
                torch.testing.assert_close(got, feats[depth[node] - 1, position], **TOLERANCE)
                torch.testing.assert_close(logits[i], target.readout_logits(states[depth[node] - 1, position][None])[0],
                                           **TOLERANCE)
                checked += 1
            if depth[live[0]] == R:
                runner.commit([live[0]], features=committed[live[0]:live[0] + 1])
                del depth[live[0]]
        self.assertEqual(checked, R * len(tokens))
        torch.testing.assert_close(committed, feats[R - 1, len(PROMPT):], **TOLERANCE)

    def test_embedding_head_and_labels_are_the_models(self):
        model = load()
        target = Inkling2Target.from_model(model)
        ids = torch.tensor(PROMPT)
        torch.testing.assert_close(target.embed(ids), model.embed_tokens(ids))
        self.assertEqual(tuple(target.embed.weight[100].shape), (32,))  # a spare row past the served vocabulary
        self.assertEqual(tuple(target.lm_head.weight.shape), (100, 32))
        states = target.features(ids)[1]
        cap = target.lm_head.softcap
        self.assertEqual(cap, model.config.final_logit_softcapping)
        raw = target.lm_head(states[R - 1])
        torch.testing.assert_close(cap * torch.tanh(raw / cap), target.readout_logits(states[R - 1]), **TOLERANCE)
        self.assertEqual(target.readout_argmax(states[R - 1]).tolist(), raw.argmax(-1).tolist())
        self.assertEqual(int(target.readout_argmax(states[R - 1])[-1]), WANT[0])

    def test_factory_feature_layers_and_drafter_config(self):
        config = json.loads((MODEL_DIR / "config.json").read_text())
        # DFlash's rule on a depth's 8 layers: the states after layers 1..5, read as the inputs of layers
        # 2..6 (four core layers and the first coda layer).
        self.assertEqual(target_feature_layers(config), [2, 3, 4, 5, 6])
        self.assertEqual(drafter_feature_layers(24), [2, 7, 12, 17, 22])  # Mini: prelude, core x 3, coda
        with patch("training.drafter.targets.registry.family", return_value="inkling2"):
            target = native_target("inkling-2-mini", [2, 3, 4, 5, 6], "cpu", path=MODEL_DIR, experts="packed")
        self.assertIsInstance(target, Inkling2Target)
        self.assertEqual(target.model.checkpoint.expert_mode, "packed")
        for layers in ([2, 3, 3], [2, 9], []):
            with self.assertRaisesRegex(ValueError, "distinct layers below 8"):
                Inkling2Target.from_model(target.model, layers)
        # 8 at the end adds the depth's final-norm state as a feature.
        with_final = Inkling2Target.from_model(target.model, [1, 3, 5, 8])
        feats, states = with_final.features(torch.tensor(PROMPT))
        torch.testing.assert_close(feats[..., -32:], states)
        cfg = drafter_config(2, target.model.config)
        # The target's intermediate_size (64) is one expert's width; the drafter's MLP is 4H.
        self.assertEqual((cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                          cfg.head_dim), (32, 128, 4, 2, 8))

    def test_drafter_specs(self):
        self.assertEqual(parse_drafter("a:5:1=0.5,2=0.25,3=0.25"), ("a", 5, {1: .5, 2: .25, 3: .25}, "ce-dpace"))
        self.assertEqual(parse_drafter("a:3:3=1:dspark")[3], "dspark")
        for spec in ("a:3:3=1:ce-tv", "a:3:3=1:dspark:markov"):
            with self.assertRaisesRegex(ValueError, "ce-dpace, dspark"):
                parse_drafter(spec)


def dense_mask(p, b, loops, k, anchor_predicts=False):
    mask_mod, q_len, kv_len = context_mask_mod(p, b, loops, k, anchor_predicts)
    return create_mask(mask_mod, None, None, q_len, kv_len, "cpu"), mask_mod


def heads(kind, rank=6):
    return {"vocab": 100, "rank": rank, "type": kind}


class OneModelTests(unittest.TestCase):
    """Traces train the drafter of the model that wrote them, and a drafter drafts for the checkpoint it was
    trained against: two checkpoints of one architecture would otherwise take each other's without an error."""

    @staticmethod
    def row(**fields):
        return dict({"id": "p0", "degenerate": False, "prompt_ids": [1, 2], "output_ids": list(range(3, 12)),
                     "finish_reason": "stop", "domain": "math", "source": "s", "sampling": None,
                     "model": "inkling-2-mini", "target_config_sha256": "a" * 64}, **fields)

    def test_traces_of_another_model_or_checkpoint_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.jsonl"
            path.write_text(json.dumps(self.row()) + "\n")
            expect = {"model": "inkling-2-mini", "target_config_sha256": "a" * 64}
            self.assertEqual(len(load_traces(path, 64, expect=expect)[0]), 1)
            self.assertEqual(len(load_traces(path, 64)[0]), 1)  # --allow-other-target: nothing is compared
            for other in ({"model": "inkling-2-mini-varloop"}, {"target_config_sha256": "b" * 64}):
                with self.assertRaisesRegex(ValueError, "traces of another model"):
                    load_traces(path, 64, expect=dict(expect, **other))

    def test_what_a_trace_must_say_follows_the_family(self):
        """Inkling 2 and Dory pin one checkpoint per name: its config hash, and a --target-path must be a copy
        of it. Ouro and Nanbeige train on the shared-KV checkpoint's traces: the model name only."""
        with tempfile.TemporaryDirectory() as root:
            copy_dir, other = Path(root) / "copy", Path(root) / "other"
            shutil.copytree(MODEL_DIR, copy_dir)
            shutil.copytree(MODEL_DIR, other)
            (other / "config.json").write_text((other / "config.json").read_text() + "\n")
            sha = file_sha(MODEL_DIR / "config.json")
            self.assertEqual(trace_expectation("inkling-2-mini", "inkling2", MODEL_DIR),
                             {"model": "inkling-2-mini", "target_config_sha256": sha})
            self.assertEqual(trace_expectation("ouro-1.4b-thinking", "ouro", MODEL_DIR, str(other)),
                             {"model": "ouro-1.4b-thinking"})
            with patch("training.common.registry.resolve", return_value=MODEL_DIR):
                self.assertEqual(trace_expectation("inkling-2-mini", "inkling2", copy_dir, str(copy_dir)),
                                 {"model": "inkling-2-mini", "target_config_sha256": sha})
                with self.assertRaisesRegex(ValueError, "is not the checkpoint of inkling-2-mini"):
                    trace_expectation("inkling-2-mini", "inkling2", other, str(other))

    def test_a_drafter_of_another_checkpoint_is_refused(self):
        """By the config hash of the export, or in older exports by the pinned revision against the
        snapshot directory; an export or a directory that says neither is not checked."""
        revision = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            snapshot, elsewhere, local = root / revision, root / ("f" * 40), root / "local"
            for directory in (snapshot, elsewhere, local):
                directory.mkdir()
                (directory / "config.json").write_text(json.dumps({"dir": directory.name}))
            def drafter(name, **metadata):
                (root / name).mkdir()
                (root / name / "config.json").write_text(json.dumps({"loopedtf_drafter": metadata}))
                return root / name
            hashed = drafter("hashed", target_config_sha256=file_sha(snapshot / "config.json"),
                             target_checkpoint="org/model@" + revision)
            labelled = drafter("labelled", target_checkpoint="org/model@" + revision)
            unnamed = drafter("unnamed", target_checkpoint="org/model")
            check_inkling2_drafter_target(hashed, snapshot)
            check_inkling2_drafter_target(labelled, snapshot)
            check_inkling2_drafter_target(labelled, local)    # not a snapshot directory: nothing to compare
            check_inkling2_drafter_target(unnamed, elsewhere)  # no pinned revision: nothing to compare
            for export_dir, model_dir in ((hashed, elsewhere), (hashed, local), (labelled, elsewhere)):
                with self.assertRaisesRegex(ValueError, "was trained against org/model@" + revision):
                    check_inkling2_drafter_target(export_dir, model_dir)


class ResumeTests(unittest.TestCase):
    """resume.pt entries: the release's format still resumes, another recipe's entry is refused whole."""

    ARGS = SimpleNamespace(lr=1e-2, wd=0.01, muon_lr=1e-2)

    def drafter(self, optimizers=1, seed=0):
        torch.manual_seed(seed)
        module = torch.nn.Linear(4, 3)
        opts = build_optimizers(module, "ce-dpace", self.ARGS)
        opts += [torch.optim.SGD(module.parameters(), lr=1e-3) for _ in range(optimizers - 1)]
        scheds = [torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (s + 1)) for opt in opts]
        return {"name": "d", "module": module, "opts": opts, "scheds": scheds, "best": -1.0}

    @staticmethod
    def update(d, seed):
        torch.manual_seed(seed)
        d["module"](torch.randn(8, 4)).square().mean().backward()
        for opt, sched in zip(d["opts"], d["scheds"]):
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

    def test_a_release_entry_and_a_current_one_continue_like_the_uninterrupted_run(self):
        run = self.drafter()
        for seed in (1, 2):
            self.update(run, seed)
        current = copy.deepcopy(resume_entry(run))
        release = {"model": current["model"], "opt": current["opts"][0], "sched": current["scheds"][0], "best": 0.25}
        before_optimizers = {k: v for k, v in current.items() if k != "optimizers"}  # written before the kinds were
        self.update(run, 3)
        for entry in (release, current, before_optimizers):
            resumed = self.drafter(seed=9)
            load_resume_entry(resumed, copy.deepcopy(entry))
            self.assertEqual(resumed["best"], entry["best"])
            self.update(resumed, 3)
            for a, b in zip(run["module"].parameters(), resumed["module"].parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(resumed["scheds"][0].get_last_lr(), run["scheds"][0].get_last_lr())

    def test_an_entry_of_another_recipe_is_refused_before_anything_is_loaded(self):
        saved = resume_entry(self.drafter())
        for target, entry in ((self.drafter(optimizers=2, seed=5), saved),
                              (self.drafter(seed=5), dict(saved, optimizers=["Muon"])),
                              (self.drafter(optimizers=2, seed=5), {"model": saved["model"], "opt": saved["opts"][0],
                                                                    "sched": saved["scheds"][0], "best": 0.0})):
            kept = copy.deepcopy(target["module"].state_dict())
            with self.assertRaisesRegex(ValueError, "another recipe"):
                load_resume_entry(target, entry)
            for name, value in target["module"].state_dict().items():
                self.assertTrue(torch.equal(value, kept[name]))


class DSparkRecipeTests(unittest.TestCase):
    def setUp(self):
        self.target = Inkling2Target.from_model(load("exact-fp32", torch.float32))
        ids = PROMPT[:2]
        while len(ids) < 10:  # a greedy trace of the target, so that every drafted slot has a label
            ids.append(int(self.target.readout_argmax(self.target.features(torch.tensor(ids))[1][-1])[-1]))
        docs = [{"id": "d", "domain": "math", "source": "fixture", "resp_start": 2, "ids": ids}]
        self.p = build_packed(self.target, docs, [0], len(ids), "cpu")
        self.p.feats, self.p.loop_out = self.p.feats.float(), self.p.loop_out.float()
        self.blocks = Blocks(torch.tensor([3] * 3), torch.tensor([1, 2, 3]), torch.tensor([1, 1, 0]),
                             torch.tensor([4] * 3), torch.tensor([0] * 3), torch.tensor([True] * 3))

    def drafter(self, kind):
        cfg = drafter_config(2, self.target.model.config)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(0)
        drafter = DepthDFlash(cfg, 5, heads=heads(kind) if kind else None)
        with torch.no_grad():
            drafter.mask_emb.copy_(self.target.embed.weight[100].float())
        return drafter

    def forward(self, drafter, p=None, **options):
        with patch("training.drafter.model.build_context_mask", dense_mask), \
                patch.object(torch, "autocast", lambda *a, **k: nullcontext()):
            return drafter_forward(drafter, self.target, p or self.p, self.blocks, R, 4, **options)

    def test_the_heads_are_speculators(self):
        drafter = self.drafter("vanilla")
        self.assertIsInstance(drafter.markov_head, MarkovHead)
        self.assertIsInstance(drafter.confidence_head, ConfidenceHead)
        self.assertEqual(tuple(drafter.markov_head.markov_w1.weight.shape), (100, 6))
        self.assertEqual(tuple(drafter.markov_head.markov_w2.weight.shape), (100, 6))
        self.assertEqual(tuple(drafter.confidence_head.proj.weight.shape), (1, 32 + 6))  # [state, Markov embedding]
        self.assertLess(drafter.markov_head.markov_w1.weight.std().item(), 0.02)  # speculators' init, std 0.01
        plain = self.drafter(None)
        self.assertIsNone(plain.markov_head)
        self.assertIsNone(plain.confidence_head)

    def test_the_loss_is_speculators_dspark_loss(self):
        """The loss is 0.1 CE + 0.9 TV + 1.0 BCE with the slot weights exp(-slot / 4), divided by the number
        of valid slots; speculators gives the same value for blocks whose first slot does not predict when
        that slot is put in front (its sample_from_anchor = False layout shifts the weights by one slot)."""
        torch.manual_seed(5)
        blocks, slots, vocab = 3, 3, 11
        logits, target = torch.randn(blocks, slots, vocab) * 3, torch.randn(blocks, slots, vocab) * 3
        confidence = torch.randn(blocks, slots)
        valid = torch.tensor([[True, True, True], [True, True, False], [False, False, False]])
        loss, metrics = dspark_loss(logits, target, confidence, valid)

        pad = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x], 1).reshape(1, blocks * (slots + 1), *x.shape[2:])
        want, _ = compute_metrics(pad(logits), pad(target), pad(confidence), pad(valid), slots + 1,
                                  loss_config=resolve_loss_config(json.dumps(DSPARK["loss"]), "eager"),
                                  tv_loss_fn=resolve_loss_config("tv", "eager")["tv"][0], gamma=DSPARK["gamma"],
                                  confidence_head_alpha=DSPARK["confidence_alpha"],
                                  per_position_loss_weight="fixed-exp-decay", sample_from_anchor=False)
        torch.testing.assert_close(loss, want)

        q, t = torch.softmax(logits, -1), torch.softmax(target, -1)
        tv = 1 - torch.minimum(q, t).sum(-1)
        ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.argmax(-1).flatten(),
                                               reduction="none").view(blocks, slots)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(confidence, 1 - tv, reduction="none")
        weight = torch.exp(-torch.arange(slots) / 4.0) * valid
        torch.testing.assert_close(loss, ((0.1 * ce + 0.9 * tv + bce) * weight).sum() / valid.sum(), rtol=1e-4,
                                   atol=1e-5)
        torch.testing.assert_close(metrics["accept_rate_sum"], ((1 - tv) * valid).sum(), rtol=1e-4, atol=1e-5)

    def test_every_slot_of_a_block_is_trained_without_the_chain_mask(self):
        """The DFlash recipe drops the slots after a trace token that is not the target's argmax; DSpark's
        keeps them, as the target's distribution at a position is its label whatever came before."""
        p = copy.copy(self.p)
        p.final_argmax = self.p.final_argmax.clone()
        p.final_argmax[4] = (p.final_argmax[4] + 1) % 100  # the trace leaves the argmax chain at position 5
        self.assertEqual(int(block_labels(p, self.blocks, 4)[1].sum()), 3)
        self.assertEqual(int(block_labels(p, self.blocks, 4, chain=False)[1].sum()), 9)
        drafter = self.drafter("vanilla")
        metrics = self.forward(drafter, p, recipe="dspark")[1]
        self.assertEqual(int(metrics["n_valid"]), 9)
        self.assertEqual(int(metrics["valid"].sum()), 3)  # acceptance is still counted on the argmax chain
        self.assertEqual(int(self.forward(drafter, p)[1]["n_valid"]), 3)

    def test_a_step_trains_the_backbone_and_both_heads_against_the_targets_distribution(self):
        """For every kind of head and both slot layouts (the given token and 3 masks; 3 slots with the given
        token's slot drafting the first token)."""
        for kind, anchor_predicts in (("vanilla", False), ("gated", False), ("rnn", False), ("vanilla", True)):
            with self.subTest(kind=kind, anchor_predicts=anchor_predicts):
                drafter = self.drafter(kind)
                loss, metrics = self.forward(drafter, recipe="dspark", anchor_predicts=anchor_predicts)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(int(metrics["n_valid"]), 9)
                self.assertTrue(0.0 <= float(metrics["overlap"]) <= 9.0)  # summed 1 - TV of the 9 slots
                self.assertTrue(0.0 <= float(metrics["confidence_error"]) <= 9.0)
                loss.backward()
                for weight in (drafter.fc.weight, drafter.markov_head.markov_w2.weight,
                               drafter.confidence_head.proj.weight):
                    self.assertGreater(weight.grad.abs().sum().item(), 0)
                # Slot i reads the row of the trace token before it: only rows of tokens that precede a slot move.
                used = {int(token) for a in self.blocks.a.tolist() for token in self.p.input_ids[a + 1:a + 4].tolist()}
                moved = set(drafter.markov_head.markov_w1.weight.grad.abs().sum(-1).nonzero().flatten().tolist())
                self.assertEqual(moved, used)
        with self.assertRaisesRegex(ValueError, "unknown drafter recipe"):
            self.forward(drafter, recipe="ce-tv")
        with self.assertRaisesRegex(ValueError, "only the dspark recipe"):
            self.forward(self.drafter(None), anchor_predicts=True)

    def test_the_two_slot_layouts_read_the_same_tokens_and_labels(self):
        """A block of 4 drafts positions a+2 .. a+4 either way: 4 slots whose masks predict their own
        positions, or 3 slots that each predict the next one."""
        own = build_context_inputs(self.p, self.blocks, R, 4, chain=False)
        nxt = build_context_inputs(self.p, self.blocks, R, 4, chain=False, anchor_predicts=True)
        self.assertEqual((own["S"], nxt["S"]), (4, 3))
        self.assertEqual(own["q_index"][0].tolist(), [1, 2, 3])
        self.assertEqual(nxt["q_index"][0].tolist(), [0, 1, 2])
        self.assertEqual(own["q_is_mask"].view(-1, 4)[0].tolist(), [False, True, True, True])
        self.assertEqual(nxt["q_is_mask"].view(-1, 3)[0].tolist(), [False, True, True])
        torch.testing.assert_close(own["labels"], nxt["labels"])
        self.assertEqual(own["q_tok"].view(-1, 4)[:, :3].tolist(), nxt["q_tok"].view(-1, 3).tolist())
        self.assertEqual(own["q_pos"].view(-1, 4)[:, :3].tolist(), nxt["q_pos"].view(-1, 3).tolist())

    def test_the_dflash_recipe_is_unchanged_and_has_no_heads(self):
        plain = self.drafter(None)
        loss, metrics = self.forward(plain)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(metrics["n_valid"]), 9)
        self.assertNotIn("overlap", metrics)
        loss.backward()
        self.assertGreater(plain.fc.weight.grad.abs().sum().item(), 0)

    def test_the_softcap_and_its_gradient_are_the_formulas(self):
        torch.manual_seed(2)
        x = torch.randn(3, 7, dtype=torch.float64) * 40
        torch.testing.assert_close(softcap(x.float(), 30.0), 30.0 * torch.tanh(x.float() / 30.0))
        half = softcap(x.to(torch.bfloat16), 30.0)
        self.assertEqual(half.dtype, torch.float32)
        torch.testing.assert_close(half, 30.0 * torch.tanh(x.to(torch.bfloat16).float() / 30.0))
        leaf = x.float().requires_grad_()
        softcap(leaf, 30.0).sum().backward()
        torch.testing.assert_close(leaf.grad, 1 - torch.tanh(x.float() / 30.0) ** 2, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(leaf, x.float()))  # the input is not written

    def test_the_bias_is_added_before_the_softcap(self):
        drafter = self.drafter("vanilla")
        with torch.no_grad():
            drafter.markov_head.markov_w2.weight.fill_(1e3)  # a bias far beyond the cap of 30
            drafter.markov_head.markov_w1.weight.fill_(1.0)
        loss, _ = self.forward(drafter, recipe="dspark")
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), 30.0)  # every logit is capped alike, so the loss stays bounded

    def test_optimizers_follow_the_recipe(self):
        args = build_parser().parse_args([])
        muon, adamw = build_optimizers(self.drafter("vanilla"), "dspark", args)
        self.assertIsInstance(muon, torch.optim.Muon)
        self.assertIsInstance(adamw, torch.optim.AdamW)
        self.assertEqual((muon.defaults["lr"], muon.defaults["momentum"], muon.defaults["weight_decay"]),
                         (3e-3, 0.95, 0.1))
        self.assertEqual((adamw.defaults["lr"], adamw.defaults["weight_decay"], adamw.defaults["betas"]),
                         (6e-4, 0.0, (0.9, 0.999)))
        drafter = self.drafter("vanilla")
        muon, adamw = build_optimizers(drafter, "dspark", args)
        names = {id(p): n for n, p in drafter.named_parameters()}
        group = lambda opt: {names[id(p)] for g in opt.param_groups for p in g["params"]}
        self.assertTrue(all(drafter.get_parameter(n).ndim == 2 for n in group(muon)))
        self.assertIn("fc.weight", group(muon))
        self.assertTrue({"markov_head.markov_w1.weight", "markov_head.markov_w2.weight", "mask_emb",
                         "confidence_head.proj.weight", "norm.weight"} <= group(adamw))
        self.assertEqual(group(muon) | group(adamw), set(names.values()))
        [single] = build_optimizers(self.drafter(None), "ce-dpace", args)
        self.assertEqual((single.defaults["betas"], single.defaults["weight_decay"]), ((0.9, 0.95), 0.01))


class ExportAndRuntimeTests(unittest.TestCase):
    def checkpoint(self, root, kind, anchor_predicts=False):
        target = Inkling2Target.from_model(load())
        cfg = drafter_config(2, target.model.config)
        torch.manual_seed(1 + len(kind or ""))
        module = DepthDFlash(cfg, 5, heads=heads(kind) if kind else None)
        with torch.no_grad():
            module.mask_emb.normal_()
            if kind:  # large enough to change drafts, and exactly what the BF16 export holds
                module.markov_head.markov_w1.weight.normal_(std=1.0)
                for parameter in (*module.markov_head.parameters(), *module.confidence_head.parameters()):
                    parameter.copy_(parameter.to(torch.bfloat16).float())
        ck = {"model": module.state_dict(), "cfg": cfg.to_dict(), "feat_layers": target.feat_layers,
              "d_weights": {1: .5, 2: .25, 3: .25}, "recipe": "dspark" if kind else "ce-dpace",
              "heads": {"rank": 6, "type": kind, "sample_from_anchor": anchor_predicts} if kind else None,
              "k_max": 4, "learn_mask": True, "mask_token": 100, "input_format": "loop-context",
              "target": "inkling-2-mini", "target_variant": "original", "target_spec": str(MODEL_DIR),
              "target_loops": R, "step": 7}
        path = root / f"ck-{kind}-{int(anchor_predicts)}.pt"
        torch.save(ck, path)
        return path, module

    def export(self, root, kind, anchor_predicts=False):
        path, module = self.checkpoint(root, kind, anchor_predicts)
        out = root / f"export-{kind}-{int(anchor_predicts)}"
        export.main(["--checkpoint", str(path), "--out", str(out), "--target-name", "org/mini@rev"])
        return out, module

    def test_an_export_is_against_the_target_of_training(self):
        """Training records the config hash of its target in the checkpoint: an export against another
        checkpoint (a wrong --target-path would otherwise get that checkpoint's hash) is refused."""
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path, _ = self.checkpoint(root, "vanilla")
            ck = torch.load(path, weights_only=False)
            for name, sha in (("same", file_sha(MODEL_DIR / "config.json")), ("other", "0" * 64)):
                torch.save(dict(ck, target_config_sha256=sha), root / f"{name}.pt")
            arguments = ["--out", str(root / "export"), "--target-name", "org/mini@rev"]
            export.main(["--checkpoint", str(root / "same.pt"), *arguments])
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                export.main(["--checkpoint", str(root / "other.pt"), *arguments])
            export.main(["--checkpoint", str(root / "other.pt"), "--allow-other-target", *arguments])

    def test_export_layout(self):
        with tempfile.TemporaryDirectory() as root:
            out, module = self.export(Path(root), "vanilla")
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["num_target_layers"], 16)  # 2 + 3 x 4 + 2 layer visits
            self.assertEqual(config["dflash_config"],
                             {"mask_token_id": 100, "target_layer_ids": [10, 11, 12, 13, 14]})
            # Trained on blocks of the given token and 3 masks.
            self.assertEqual((config["block_size"], config["vocab_size"], config["intermediate_size"]), (4, 100, 128))
            drafter = config["loopedtf_drafter"]
            self.assertEqual(drafter["target_config_sha256"], file_sha(MODEL_DIR / "config.json"))
            self.assertEqual((drafter["target_loops"], drafter["input_format"], drafter["target_checkpoint"],
                              drafter["feature_layers_per_loop"], drafter["mask_token_rule"]),
                             (R, "loop-context", "org/mini@rev", [2, 3, 4, 5, 6], "first-unused-embedding-row"))
            self.assertEqual(drafter["dspark"], {"markov_rank": 6, "markov_head_type": "vanilla",
                                                 "confidence_head_with_markov": True, "sample_from_anchor": False})
            self.assertEqual(drafter["target_head"], {"kind": "inkling2-shared", "softcap": 30.0})
            self.assertNotIn(str(MODEL_DIR), (out / "config.json").read_text())
            with safe_open(out / "model.safetensors", "pt") as handle:
                names = set(handle.keys())
                torch.testing.assert_close(handle.get_tensor("markov_head.markov_w2.weight").float(),
                                           module.markov_head.markov_w2.weight.detach())
            self.assertTrue({"markov_head.markov_w1.weight", "confidence_head.proj.weight",
                             "confidence_head.proj.bias"} <= names)
            self.assertNotIn("lm_head.weight", names)  # the target's head is not copied
            plain, _ = self.export(Path(root), None)
            config = json.loads((plain / "config.json").read_text())
            self.assertNotIn("dspark", config["loopedtf_drafter"])
            self.assertEqual(config["block_size"], 4)
            # The DSpark paper's block: 3 slots, the given token's slot drafting the first of 3 tokens.
            config = json.loads((self.export(Path(root), "vanilla", True)[0] / "config.json").read_text())
            self.assertEqual((config["block_size"], config["loopedtf_drafter"]["dspark"]["sample_from_anchor"]),
                             (3, True))

    def test_drafts_and_confidences_are_the_training_heads_decided_left_to_right(self):
        """For each kind of sequential head: the adapter's drafts and confidences of a block are those of
        the training module's heads applied slot by slot to the tokens drafted so far."""
        runner = Inkling2BatchRunner(load(), (1, 2, 3), max_len=64)
        for kind in ("vanilla", "gated", "rnn"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                out, module = self.export(Path(root), kind)
                adapter = ContextDFlashAdapter.from_export(out, embed=runner.embed, lm_head=runner.raw_logits,
                                                           max_len=64, device="cpu", dtype=torch.float32,
                                                           use_cuda_graph=False)
                self.assertEqual(adapter.dspark.kind, kind)
                torch.manual_seed(3)
                states = [torch.randn(3, 32), None, torch.randn(2, 32), torch.randn(3, 32)]
                requests = [{"head": 4}, {"head": 9}, {"head": 31}, {"head": 77}]
                with patch.object(adapter, "_context_outputs", return_value=(states, False)):
                    drafts = adapter.draft_context_many(requests)
                    logits = adapter.context_logits(requests)
                plain = []
                for request, hidden, draft, rows in zip(requests, states, drafts, logits):
                    if hidden is None:
                        self.assertEqual((draft.token_ids.numel(), draft.confidence, rows), (0, None, None))
                        continue
                    tokens, confidences = [], []
                    for slot in range(hidden.shape[0]):
                        previous = torch.tensor([[request["head"], *tokens]])
                        with torch.no_grad():
                            corrected, confidence = module.head_outputs(hidden[None, :slot + 1], previous,
                                                                        runner.raw_logits)
                        torch.testing.assert_close(rows[slot], corrected[0, slot], rtol=1e-5, atol=1e-5)
                        tokens.append(int(corrected[0, slot].argmax()))
                        confidences.append(torch.sigmoid(confidence[0, slot]))
                    self.assertEqual(draft.token_ids.tolist(), tokens)
                    torch.testing.assert_close(draft.confidence, torch.stack(confidences), rtol=1e-5, atol=1e-6)
                    plain.append(runner.raw_logits(hidden).argmax(-1).tolist() == tokens)
                self.assertFalse(all(plain))  # the bias changes drafts

    def test_2dsd_with_an_exported_drafter_names_the_reference_tokens(self):
        """Block drafting through TwoDSD with each kind of export: whatever an untrained drafter proposes,
        the output is plain decoding's, and drafts are made and verified."""
        tokenizer = SimpleNamespace(encode=lambda text: [])
        engine_options = dict(max_model_len=64, device="cpu", precision="exact", dtype="float32")
        with tempfile.TemporaryDirectory() as root:
            for kind, anchor_predicts in (("vanilla", False), ("vanilla", True), (None, False)):
                out, _ = self.export(Path(root), kind, anchor_predicts)
                for stage, d1 in (("stage2b", 1), ("stage1", 1), ("stage1", 3)):
                    with self.subTest(kind=kind, anchor_predicts=anchor_predicts, stage=stage, d1=d1), \
                            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                        engine = TwoDSD(str(MODEL_DIR), str(out), block_size=4, stage=stage, d1=d1, **engine_options)
                        [result] = engine.generate([PROMPT], max_new_tokens=len(WANT), warmup_requests=0)
                        self.assertEqual(result["output_ids"], WANT)
                        settings = result["settings"]
                        self.assertEqual((settings["block_size"], settings["target_path"], settings["control"]),
                                         (4, "batched", None))
                        self.assertEqual(settings["drafter_heads"] is not None, kind is not None)
                        self.assertGreater(result["drafter_runtime_stats"]["leaf_drafts"], 0)
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                engine = TwoDSD(str(MODEL_DIR), str(out), block_size=4, runner="static", **engine_options)
                [result] = engine.generate([PROMPT], max_new_tokens=len(WANT), warmup_requests=0)
                self.assertEqual(result["output_ids"], WANT)  # the static runner, eager on the CPU
                self.assertEqual((result["settings"]["target_path"], result["settings"]["cuda_graph"]),
                                 ("static", False))
            vocabulary = Path(root) / "vocabulary.json"  # host-side checks before any id reaches the device
            static = dict(block_size=4, runner="static", **engine_options)
            dspark = Path(root) / "export-vanilla-0"
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                vocabulary.write_text(json.dumps({"ids": list(range(0, 100, 2))}))
                [result] = TwoDSD(str(MODEL_DIR), str(dspark), draft_vocab=str(vocabulary), **static).generate(
                    [PROMPT], max_new_tokens=len(WANT), warmup_requests=0)
                self.assertEqual(result["output_ids"], WANT)
                for bad in ([], [1.5, 2], [3, 100], [-1, 2], "1,2"):
                    vocabulary.write_text(json.dumps({"ids": bad}))
                    with self.assertRaisesRegex(ValueError, "integers in"):
                        TwoDSD(str(MODEL_DIR), str(dspark), draft_vocab=str(vocabulary), **static).generate(
                            [PROMPT], max_new_tokens=4, warmup_requests=0)
            other = Path(root) / "other-checkpoint"  # the same weights under another config.json
            shutil.copytree(MODEL_DIR, other)
            (other / "config.json").write_text((other / "config.json").read_text() + "\n")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                with self.assertRaisesRegex(ValueError, "was trained against org/mini@rev"):
                    TwoDSD(str(other), str(out), block_size=4, **engine_options)
                engine = TwoDSD(str(other), str(out), block_size=4, drafter_of_other_target=True, **engine_options)
                [result] = engine.generate([PROMPT], max_new_tokens=len(WANT), warmup_requests=0)
                self.assertEqual(result["output_ids"], WANT)
            with self.assertRaisesRegex(ValueError, "block"), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                TwoDSD(str(MODEL_DIR), str(out), block_size=8, **engine_options).generate(
                    [PROMPT], max_new_tokens=8, warmup_requests=0)

    def test_the_confidence_threshold_ends_blocks_and_keeps_the_output(self):
        """draft_confidence drops the drafted slots from the first one whose confidence is below it: at 1.0
        no drafted token is kept, at 0.0 every one is, and the output is plain decoding's either way."""
        tokenizer = SimpleNamespace(encode=lambda text: [])
        options = dict(block_size=4, stage="stage1", d1=1, max_model_len=64, device="cpu", precision="exact",
                       dtype="float32")
        with tempfile.TemporaryDirectory() as root, \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            out, _ = self.export(Path(root), "vanilla")
            slots = {}
            for threshold in (None, 0.0, 0.5, 1.0):
                engine = TwoDSD(str(MODEL_DIR), str(out), draft_confidence=threshold, **options)
                [result] = engine.generate([PROMPT], max_new_tokens=len(WANT), warmup_requests=0)
                self.assertEqual(result["output_ids"], WANT)
                self.assertEqual(result["settings"]["draft_confidence"], threshold)
                slots[threshold] = (result["stats"]["draft_slots"], result["stats"]["draft_slots_cut"])
            self.assertEqual(slots[0.0], slots[None])
            self.assertEqual(slots[None][1], 0)
            self.assertEqual(slots[1.0][0], 0)
            self.assertGreater(slots[1.0][1], 0)
            self.assertLessEqual(slots[0.5][0], slots[None][0])
            plain, _ = self.export(Path(root), None)
            with self.assertRaisesRegex(RuntimeError, "confidence head"):
                TwoDSD(str(MODEL_DIR), str(plain), draft_confidence=0.5, **options).generate(
                    [PROMPT], max_new_tokens=8, warmup_requests=0)


if __name__ == "__main__":
    unittest.main()
