# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Inkling 2 tree runners against the reference forward, on a tiny checkpoint in the released format.

The model has Mini's layout in small: prelude layers 0-1, core layers 2-5 run three times, coda layers 6-7,
full attention on every second layer (1 in the prelude, 3 and 5 in the core, 7 in the coda), a sliding window
of 8 and a relative extent of 16, so sequences of 40 tokens leave the window and the extent. Everything runs
in the exact precision policy (FP64, no rounding), where a runner must agree with the reference to rounding
error of the summation order and name the same tokens. Every test runs on the reference path (one node at a
time through the reference forward's layer function), on the batched runner (all rows of a call at once) and
on the static runner (the batched computation in padded, data-independent shapes).
"""
from types import SimpleNamespace

import inkling2_tiny as tiny
import pytest
import torch
from vllm.model_executor.models import inkling2_weights as weights
from vllm.model_executor.models.inkling2_2d import Inkling2TreeRunner
from vllm.model_executor.models.inkling2_batch import ExpertBank, Inkling2BatchRunner
from vllm.model_executor.models.inkling2_fast import FastInkling2, tensor_cores
from vllm.model_executor.models.inkling2_graph import Inkling2GraphRunner
from vllm.model_executor.models.inkling2_reference import Inkling2Cache, Inkling2Plane, NativeInkling2
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor as Executor

PROMPT = [5, 17, 3, 42, 9, 11, 2]
R = 3
TOLERANCE = dict(rtol=0, atol=1e-9)


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory):
    loop = dict(core_start=2, core_end=6, num_passes=R, gate_rank=4, is_varloop=False)
    config = tiny.tiny_config(num_hidden_layers=8, local_layer_ids=[0, 2, 4, 6], loop_config=loop)
    return tiny.write_tiny_checkpoint(tmp_path_factory.mktemp("inkling2"), config, mxfp8_layers=(6, 7))


@pytest.fixture(scope="module")
def model(checkpoint_dir):
    return NativeInkling2(weights.load_inkling2_checkpoint(checkpoint_dir), "exact")


@pytest.fixture(params=[Inkling2TreeRunner, Inkling2BatchRunner, Inkling2GraphRunner],
                ids=["reference", "batched", "static"])
def Runner(request):
    return request.param


def reference_decode(model, count, depths=(R,)):
    """Greedy tokens of the reference forward, one token at a time, with every depth's logits."""
    cache, step, output, logits = Inkling2Cache(), PROMPT, [], []
    while len(output) < count:
        out = model.forward(step, depths=depths, cache=cache, return_logits=True, pass_end=False)
        logits.append({depth: out.readouts[depth].logits[-1] for depth in depths})
        output.append(int(logits[-1][R].argmax()))
        step = [output[-1]]
    return output, logits


class SharedSequence:
    """Token-by-token decoding under last-loop KV sharing, written without the tree runner: every position's
    rows are kept per plane, the prompt is run without sharing, and a later token at position t in a plane
    other than the final one reads the final plane's keys and values for positions <= t - window."""

    def __init__(self, model, depths, window, share_layers):
        self.model, self.depths, self.window, self.share = model, tuple(sorted(depths)), window, set(share_layers)
        self.rows = {}  # (layer, index) -> list of (keys, values, conv inputs)
        self.length = 0
        self.sharing = False

    def plane(self, layer, index):
        rows = self.rows.setdefault((layer, index), [])
        source = list(rows)
        if self.sharing and layer in self.share and index != R:
            cut = max(0, self.length - self.window + 1)
            source[:cut] = self.rows[(layer, R)][:cut]
        if not source:
            self.view = Inkling2Plane()
            return self.view
        kernel = self.model.config.sconv_kernel_size
        conv = {}
        for stream in rows[-1][2]:
            inputs = [row[2][stream] for row in rows[-(kernel - 1):]]
            zeros = [torch.zeros_like(inputs[0])] * (kernel - 1 - len(inputs))
            conv[stream] = torch.cat((*zeros, *inputs))
        self.view = Inkling2Plane(torch.cat([row[0] for row in source]), torch.cat([row[1] for row in source]),
                                  0, conv)
        return self.view

    def visit(self, layer, index, blocks, partial, bank):
        blocks, partial, _ = self.model._layer(self, layer, index, blocks, partial, bank)
        view = self.view
        self.rows[(layer, index)].append((view.keys[-1:].clone(), view.values[-1:].clone(),
                                          {stream: history[-1:].clone() for stream, history in view.conv.items()}))
        return blocks, partial

    def step(self, token):
        """Logits of every depth for the token after `token`."""
        model, core = self.model, self.model.config.core_layers
        self.start, self.positions = self.length, torch.tensor([self.length])
        blocks, partial = [], model.embed_tokens(torch.tensor([token]))
        for layer in range(core.start):
            blocks, partial = self.visit(layer, 0, blocks, partial, None)
        prelude, bank, logits = (blocks, partial), None, {}
        for depth in range(1, R + 1):
            blocks, partial = prelude
            for layer in core:
                blocks, partial = self.visit(layer, depth, blocks, partial, bank)
            bank = [*blocks[len(prelude[0]) + 1:], partial]
            if depth in self.depths:
                state = (blocks, partial)
                for layer in range(core.stop, model.config.num_hidden_layers):
                    state = self.visit(layer, depth, *state, None)
                logits[depth] = model.readout_logits(model.readout_hidden(*state))[0]
        self.length += 1
        return logits

    def decode(self, count):
        for token in PROMPT:
            logits = self.step(token)
        self.sharing = True
        output = [int(logits[R].argmax())]
        while len(output) < count:
            output.append(int(self.step(output[-1])[R].argmax()))
        return output


class ScriptedDraft:
    """Proposes the known continuation with every fifth position replaced by another token."""

    input_format = "loop-context"

    def __init__(self, sequence, vocab):
        self.sequence, self.vocab, self.stats = sequence, vocab, {}

    def reset(self):
        pass

    def draft_context_many(self, requests):
        results = []
        for request in requests:
            span = request["span_features"]
            start = request["prefix_len"] + (0 if span is None else len(span)) + 1
            slots = []
            for position in range(start, start + request["k"] - 1):
                token = self.sequence[position] if position < len(self.sequence) else 1
                slots.append((token + 1) % self.vocab if position % 5 == 0 else token)
            results.append(SimpleNamespace(token_ids=torch.tensor(slots, dtype=torch.long)))
        return results


def schedule_depths(stage, d1, d2):
    return {d1, R} if stage == "stage1" else {d1, d2, R}


def run_schedule(Runner, model, want, stage, k, d1, d2=2, window=None, **options):
    target = Runner(model, schedule_depths(stage, d1, d2), window=window, max_len=128)
    draft = ScriptedDraft(PROMPT + want, model.config.served_vocab_size) if k > 1 else None
    result = Executor(target, draft, k, stage, d1, d2, **options).generate(PROMPT, len(want), [])
    return result, target


def assert_planes_equal(planes, expected):
    assert planes.keys() == expected.keys()
    for name, plane in planes.items():
        other = expected[name]
        assert plane.first == other.first, name
        torch.testing.assert_close(plane.keys, other.keys, **TOLERANCE, msg=str(name))
        torch.testing.assert_close(plane.values, other.values, **TOLERANCE, msg=str(name))
        assert plane.conv.keys() == other.conv.keys()
        for stream, history in plane.conv.items():
            torch.testing.assert_close(history, other.conv[stream], **TOLERANCE, msg=f"{name} {stream}")


SCHEDULES = [
    ("stage1", 1, 1),
    ("stage1", 1, 2),
    ("stage1", 1, 3),
    ("stage1", 4, 1),
    ("stage1", 4, 2),
    ("stage1", 8, 3),
    ("stage2a", 4, 1),
    ("stage2b", 1, 1),
    ("stage2b", 4, 1),
    ("stage2b", 8, 1),
    ("stage2b", 16, 1),
]


def test_sequential_readouts_and_features_match_the_reference_at_every_depth(Runner, model):
    """One node at a time through the node interface: every depth's logits are the reference's, and a
    committed token's features are the reference's taps and final-norm state at depth R."""
    count, depths = 14, (1, 2, 3)
    want, logits = reference_decode(model, count, depths)
    target = Runner(model, depths, max_len=64)
    prefill = target.prefill(torch.tensor(PROMPT))
    torch.testing.assert_close(target.logits(prefill.hidden[-1:])[0], logits[0][R], **TOLERANCE)
    features = torch.empty(count - 1, prefill.features.shape[-1], dtype=prefill.features.dtype)
    for step in range(1, count):
        target.register_many([(step, None, want[step - 1])])
        for depth in depths:
            rows, argmax, got = target.advance_nodes([step], [depth - 1])
            assert rows == [0] and int(argmax[0]) == int(logits[step][depth].argmax())
            torch.testing.assert_close(got[0], logits[step][depth], **TOLERANCE)
        final = target.span_features([(R, [step])])[0][0].clone()
        assert not torch.equal(target.span_features([(2, [step])])[0][0], final)
        target.commit([step], features=features[step - 1:step])
        assert torch.equal(features[step - 1], final)  # a commit hands out the depth-R span features
    assert target.prefix_length == len(PROMPT) + count - 1 and not target.nodes
    stats = target.stats  # every token ran the prelude once, three passes and a readout at each depth
    assert (stats["calls"], stats["cells"], stats["prelude_calls"], stats["prelude_rows"], stats["readout_calls"],
            stats["readout_rows"]) == (3 * (count - 1), 3 * (count - 1), count - 1, count - 1, 3 * (count - 1),
                                       3 * (count - 1))

    sequence = (PROMPT + want)[:target.prefix_length]
    core = model.config.core_layers
    visits = [layer if layer < core.start else layer + (R - 1) * len(core) for layer in target.feature_layers]
    # DFlash's rule on the 8 layers of a depth: the core's four layers and the first coda layer.
    assert target.feature_layers == (2, 3, 4, 5, 6) and not target.final_feature
    cache = Inkling2Cache()
    full = model.forward(sequence, depths=depths, taps=visits, cache=cache, pass_end=False)
    expected = torch.cat([full.features[visit] for visit in visits], -1)
    torch.testing.assert_close(prefill.features, expected[:len(PROMPT)], **TOLERANCE)
    torch.testing.assert_close(features, expected[len(PROMPT):], **TOLERANCE)
    assert_planes_equal(target.prefix.planes, cache.planes)


@pytest.mark.parametrize("stage,k,d1", SCHEDULES)
def test_two_dimensional_schedules_reproduce_the_reference(Runner, model, stage, k, d1):
    """LoopSpec (k = 1), block drafting, promotion and forking through the real executor name the reference's
    tokens, and the prefix they leave is the reference cache of the same tokens: rejected and pruned branches
    leave nothing behind."""
    count = 40
    want, _ = reference_decode(model, count)
    result, target = run_schedule(Runner, model, want, stage, k, d1)
    assert result["output_ids"] == want
    assert target.stats["released_nodes"] > 0 or k == 1 and stage == "stage1"
    depths = tuple(sorted(schedule_depths(stage, d1, 2)))
    cache = Inkling2Cache()
    model.forward((PROMPT + want)[:target.prefix_length], depths=depths, cache=cache, pass_end=False)
    assert_planes_equal(target.prefix.planes, cache.planes)


OTHER_LOOPS = [
    (4, "stage1", 1, 4, 2),
    (4, "stage1", 4, 1, 2),
    (4, "stage1", 8, 3, 2),
    (4, "stage1", 8, 4, 2),
    (4, "stage2b", 4, 1, 3),
    (4, "stage2b", 8, 2, 3),
    (5, "stage1", 1, 2, 2),
    (5, "stage1", 4, 2, 2),
    (5, "stage1", 8, 5, 2),
    (5, "stage2a", 4, 2, 3),
    (5, "stage2b", 4, 1, 4),
    (5, "stage2b", 8, 3, 4),
]


@pytest.mark.parametrize("loops,stage,k,d1,d2", OTHER_LOOPS)
def test_schedules_at_another_number_of_passes_reproduce_the_reference_of_that_count(Runner, model, loops, stage,
                                                                                    k, d1, d2):
    """loops = R' runs the core R' times instead of the checkpoint's three (a variable-loop checkpoint): plain
    decoding, depth speculation, block drafting, promotion and forking name the tokens of the reference forward
    with R' passes, which are not those of three passes, and leave its cache of those tokens."""
    count = 40
    cache, step, want = Inkling2Cache(), PROMPT, []
    while len(want) < count:
        out = model.forward(step, num_passes=loops, cache=cache, pass_end=False)
        want.append(int(out.readouts[loops].argmax[-1]))
        step = [want[-1]]
    assert want != reference_decode(model, count)[0]
    depths = {d1, loops} if stage == "stage1" else {d1, d2, loops}
    target = Runner(model, depths, loops=loops, max_len=128)
    assert target.R == loops
    draft = ScriptedDraft(PROMPT + want, model.config.served_vocab_size) if k > 1 else None
    result = Executor(target, draft, k, stage, d1, d2).generate(PROMPT, count, [])
    assert result["output_ids"] == want
    cache = Inkling2Cache()
    model.forward((PROMPT + want)[:target.prefix_length], num_passes=loops, depths=tuple(sorted(depths)), cache=cache,
                  pass_end=False)
    assert_planes_equal(target.prefix.planes, cache.planes)


def test_a_window_longer_than_the_sequence_shares_nothing(Runner, model):
    count = 24
    want, _ = reference_decode(model, count)
    result, _ = run_schedule(Runner, model, want, "stage2b", 4, 1, window=64)
    assert result["output_ids"] == want


def test_the_static_runner_takes_a_call_wider_than_its_largest_bucket(model):
    """A wide Stage 2 fork can put more rows in a call than the largest row bucket (304 at k = 16): the call
    then runs at its own size. 300 siblings and their 300 children in one call give the batched runner's
    readouts at every depth."""
    width, vocab = 300, model.config.served_vocab_size
    nodes = [(i, None, (7 * i + 3) % vocab) for i in range(width)]
    nodes += [(width + i, i, (11 * i + 5) % vocab) for i in range(width)]
    found = []
    for Runner in (Inkling2BatchRunner, Inkling2GraphRunner):
        target = Runner(model, (1, 2, 3), max_len=64)
        target.prefill(torch.tensor(PROMPT))
        target.ensure_capacity(len(nodes))
        target.register_many(nodes)
        ids, tokens = [node_id for node_id, _, _ in nodes], []
        for loop in range(R):
            rows, argmax, logits = target.advance_nodes(ids, [loop] * len(ids))
            assert rows == list(range(len(ids)))
            tokens.append((argmax.clone(), logits.clone()))
        found.append(tokens)
    for (want, want_logits), (got, got_logits) in zip(*found):
        assert torch.equal(got, want)
        torch.testing.assert_close(got_logits, want_logits, **TOLERANCE)


@pytest.mark.parametrize("k", [1, 4])
def test_the_static_runner_runs_the_passes_of_a_full_depth_schedule_ahead(model, k):
    """Plain decoding and block drafting read out after the last pass only. The static runner then runs a
    row's three passes and its readout with the row's first call: the same tokens, counts and prefix as
    pass by pass, in a third of the host rounds that launch anything."""
    want, _ = reference_decode(model, 24)
    runs = {}
    for fuse in (True, False):
        target = Inkling2GraphRunner(model, {R}, max_len=128, fuse_passes=fuse)
        draft = ScriptedDraft(PROMPT + want, model.config.served_vocab_size) if k > 1 else None
        runs[fuse] = Executor(target, draft, k, "stage1", R, 2).generate(PROMPT, len(want), []), target
    (ahead, fused), (stepped, plain) = runs[True], runs[False]
    assert ahead["output_ids"] == stepped["output_ids"] == want
    for name in ("calls", "cells", "prelude_rows", "readout_rows", "committed_nodes"):
        assert fused.stats[name] == plain.stats[name], name
    assert fused._ahead is None
    assert_planes_equal(fused.prefix.planes, plain.prefix.planes)


@pytest.mark.parametrize("d1", [1, 2])
def test_stage1_that_drops_rejected_suffixes_names_the_same_tokens(Runner, model, d1):
    """A greedy Stage 1 that drops the suffix its review rejects, instead of freezing it for the last loop,
    names the reference's tokens and leaves the reference cache. (Whether it runs fewer rows depends on how
    often the last loop takes a frozen token back.)"""
    want, _ = reference_decode(model, 40)
    result, target = run_schedule(Runner, model, want, "stage1", 4, d1, frozen_suffix=False)
    assert result["output_ids"] == want
    assert result["stats"]["frozen_created"] == 0 and target.stats["released_nodes"] > 0
    cache = Inkling2Cache()
    model.forward((PROMPT + want)[:target.prefix_length], depths=tuple(sorted(schedule_depths("stage1", d1, 2))),
                  cache=cache, pass_end=False)
    assert_planes_equal(target.prefix.planes, cache.planes)
    with pytest.raises(ValueError, match="greedy Stage 1"):
        run_schedule(Runner, model, want, "stage2b", 4, 1, frozen_suffix=False)


@pytest.mark.parametrize("stage,k,d1", [("stage1", 1, 1), ("stage1", 4, 1), ("stage2b", 1, 1), ("stage2b", 4, 1),
                                        ("stage1", 4, 3)])
def test_shared_schedules_reproduce_sequential_shared_decoding(Runner, model, stage, k, d1):
    """With W = 16 on the full-attention layers of the core and the coda, every schedule names the tokens of
    token-by-token decoding under the same rule, which differ from the unshared model's."""
    count, window = 40, 16
    depths = schedule_depths(stage, d1, 2)
    target = Runner(model, depths, window=window)
    assert target.share_layers == {3, 5, 7}
    want = SharedSequence(model, depths, window, target.share_layers).decode(count)
    result, target = run_schedule(Runner, model, want, stage, k, d1, window=window)
    assert result["output_ids"] == want
    assert want == SharedSequence(model, {R}, window, {3, 5}).decode(count)  # coda sharing is not served
    assert want != reference_decode(model, count)[0]


def test_sharing_changes_only_planes_below_the_final_one(Runner, model):
    """After the same tokens, the final planes of a sharing runner hold what its early planes read; the
    prelude and every sliding layer are untouched by the rule."""
    target = Runner(model, (1, 2, 3), window=4, max_len=64)
    plain = Runner(model, (1, 2, 3), max_len=64)
    tokens = reference_decode(model, 12)[0]
    for runner in (target, plain):
        runner.prefill(torch.tensor(PROMPT))
        for step, token in enumerate(tokens):
            runner.register_many([(step, None, token)])
            for depth in range(R):
                runner.advance_nodes([step], [depth])
            runner.commit([step])
    shared, native = target.prefix.planes, plain.prefix.planes
    changed = {name for name, plane in shared.items() if not torch.equal(plane.keys, native[name].keys)}
    assert changed and all(layer >= 2 for layer, _ in changed)
    for (layer, index), plane in shared.items():
        if layer < 2:  # the prelude never reads a shared plane
            assert torch.equal(plane.keys, native[(layer, index)].keys)


def test_sampling_schedules_run_and_repeat_under_a_seed(Runner, model):
    want, _ = reference_decode(model, 24)
    outputs = []
    for _ in range(2):
        target = Runner(model, (1, 2, 3), max_len=64)
        draft = ScriptedDraft(PROMPT + want, model.config.served_vocab_size)
        executor = Executor(target, draft, 4, "stage2b", 1, 2, temperature=0.8)
        outputs.append(executor.generate(PROMPT, 24, [], seed=7)["output_ids"])
    assert outputs[0] == outputs[1] and len(outputs[0]) == 24


@pytest.mark.parametrize("precision", ["exact", "served"])
def test_packed_experts_are_dequantized_exactly_and_decode_the_same_tokens(checkpoint_dir, model, precision):
    """The bank's table lookup of a packed checkpoint (NVFP4 layers 0-5, MXFP8 layers 6-7) gives the loader's
    dense factors bit for bit, for one expert and for a set, and the batched runner on packed experts names
    the tokens of the dense checkpoint."""
    packed = NativeInkling2(weights.load_inkling2_checkpoint(checkpoint_dir, experts="packed"), precision)
    dense = NativeInkling2(weights.load_inkling2_checkpoint(checkpoint_dir), precision)
    bank, dtype = ExpertBank.of(packed), packed.precision.dtype
    for layer in (0, 3, 7):
        experts = packed.checkpoint.experts(layer)
        assert experts.quant_format == ("mxfp8" if layer >= 6 else "nvfp4")
        index = torch.tensor([5, 0, 5, 2])
        for projection in weights.PROJECTIONS:
            want, scale = experts.dequantize(projection, index, dtype)
            got, got_scale = bank.factors(layer, projection, index)
            assert got.dtype == dtype and torch.equal(got, want) and torch.equal(got_scale, scale.to(dtype))
        w13, s13, w2, s2 = bank.single(experts, 2)
        assert torch.equal(w13, experts.dequantize("w13", 2, dtype)[0][0])
        assert torch.equal(w2, experts.dequantize("w2", 2, dtype)[0][0]) and s13.ndim == 0
        assert bank.single(experts, 2)[0] is w13 and bank.hits >= 1  # kept for the reference forward
    outputs = []
    for target in (packed, dense):
        runner = Inkling2BatchRunner(target, (1, 2, R), max_len=64)
        outputs.append(Executor(runner, None, 1, "stage2b", 1, 2).generate(PROMPT, 24, [])["output_ids"])
    assert outputs[0] == outputs[1]
    if precision == "exact":
        assert outputs[0] == reference_decode(model, 24)[0]


def test_the_tensor_core_scope_is_restored_and_nests():
    matmul = torch.backends.cuda.matmul
    before = matmul.allow_tf32
    with tensor_cores():
        assert matmul.allow_tf32
        with tensor_cores(False):
            assert not matmul.allow_tf32
        assert matmul.allow_tf32
    assert matmul.allow_tf32 == before


def test_the_tensor_core_model_is_the_reference_on_a_cpu(checkpoint_dir):
    """Without tensor cores and the grouped expert GEMM, FastInkling2 is the served reference bit for bit:
    states, logits and feature taps of every depth, in one forward and continued from a cache."""
    served, fast = NativeInkling2.from_pretrained(checkpoint_dir), FastInkling2.from_pretrained(checkpoint_dir)
    assert fast.tensor_cores and not fast.fused_experts
    tokens = PROMPT + reference_decode(served, 24)[0]
    options = dict(depths=(1, 2, R), taps=(0, 3, 9, 15), coda_taps=(7,), return_logits=True)
    want = served.forward(tokens, **options)
    cache = Inkling2Cache()
    first = fast.forward(tokens[:10], cache=cache, **options)
    rest = fast.forward(tokens[10:], cache=cache, **options)
    for got in (fast.forward(tokens, **options), None):
        for depth in (1, 2, R):
            w = want.readouts[depth]
            if got is None:  # the two cached calls
                hidden = torch.cat((first.readouts[depth].hidden, rest.readouts[depth].hidden))
                logits = torch.cat((first.readouts[depth].logits, rest.readouts[depth].logits))
            else:
                hidden, logits = got.readouts[depth].hidden, got.readouts[depth].logits
                assert torch.equal(got.readouts[depth].taps[7], w.taps[7])
            assert torch.equal(hidden, w.hidden) and torch.equal(logits, w.logits)
        if got is not None:
            assert got.features.keys() == want.features.keys()
            assert all(torch.equal(got.features[visit], want.features[visit]) for visit in want.features)
            assert torch.equal(got.experts, want.experts)
    with pytest.raises(ValueError, match="served precision"):
        FastInkling2(weights.load_inkling2_checkpoint(checkpoint_dir), "exact")


def test_runners_follow_the_tensor_core_model(Runner, checkpoint_dir):
    """A runner on FastInkling2 gives the tokens and the prefix of the same runner on the served reference
    (on a CPU the two are one arithmetic), through a forking schedule."""
    served = NativeInkling2.from_pretrained(checkpoint_dir)
    script = reference_decode(served, 24)[0]  # what the scripted drafter proposes
    outputs, planes = [], []
    for target in (served, FastInkling2.from_pretrained(checkpoint_dir)):
        result, runner = run_schedule(Runner, target, script, "stage2b", 4, 1)
        outputs.append(result["output_ids"])
        planes.append(runner.prefix.planes)
    assert outputs[0] == outputs[1]
    assert planes[0].keys() == planes[1].keys()
    for key, plane in planes[0].items():
        assert torch.equal(plane.keys, planes[1][key].keys) and torch.equal(plane.values, planes[1][key].values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("stage,k,d1", [("stage1", 1, 3), ("stage1", 4, 3), ("stage1", 4, 1), ("stage2b", 4, 1)])
def test_the_static_runner_s_graphs_replay_its_eager_run(checkpoint_dir, stage, k, d1):
    """With CUDA graphs (its phases and a commit's state move) the static runner names the tokens of its eager
    run and leaves the same prefix, from the first commit of every size on: a capture runs nothing."""
    model = NativeInkling2.from_pretrained(checkpoint_dir, device="cuda", dtype=torch.bfloat16)
    script = reference_decode(model, 32)[0]
    runs = []
    for graph in (False, True):
        target = Inkling2GraphRunner(model, schedule_depths(stage, d1, 2), max_len=128, use_cuda_graph=graph)
        if graph and not target.use_cuda_graph:
            pytest.skip("the runner captures no graphs for this checkpoint")
        draft = ScriptedDraft(PROMPT + script, model.config.served_vocab_size) if k > 1 else None
        result = Executor(target, draft, k, stage, d1, 2).generate(PROMPT, len(script), [])
        runs.append((result["output_ids"], target.prefix.planes, target.stats["graph_replays"]))
    (eager, kept, _), (replayed, planes, replays) = runs
    assert replays > 0 and replayed == eager and planes.keys() == kept.keys()
    for key, plane in kept.items():
        other = planes[key]
        assert plane.first == other.first, key
        assert torch.equal(plane.keys, other.keys) and torch.equal(plane.values, other.values), key
        for stream, history in plane.conv.items():
            assert torch.equal(history, other.conv[stream]), (key, stream)


@pytest.fixture(scope="module")
def kernel_checkpoint_dir(tmp_path_factory):
    """The tiny model at sizes that the static runner's row kernels take (head dimension and controller rank
    16) and with a hidden size of 256: at 32, cuBLAS sums a row of one call and of eight in the same order,
    and the code from before the row kernels became row-independent passes the test below."""
    loop = dict(core_start=2, core_end=6, num_passes=R, gate_rank=16, is_varloop=False)
    config = tiny.tiny_config(num_hidden_layers=8, local_layer_ids=[0, 2, 4, 6], loop_config=loop, head_dim=16,
                              swa_head_dim=16, qk_l2_clip_threshold=2 * 16**0.5, hidden_size=256,
                              intermediate_size=128)
    return tiny.write_tiny_checkpoint(tmp_path_factory.mktemp("inkling2-kernels"), config, mxfp8_layers=(6, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_every_schedule_leaves_the_bits_of_plain_decoding_with_row_kernels(kernel_checkpoint_dir):
    """In the served precision with row kernels, a position's keys, values and conv inputs are the same bits
    whether it ran alone or among the rows of a drafted block, at full depth or after an early readout: a
    schedule accepts the tokens of plain decoding and leaves its prefix."""
    model = FastInkling2.from_pretrained(kernel_checkpoint_dir, device="cuda", precision="served",
                                         dtype=torch.bfloat16)
    schedules = [("stage1", 1, 3), ("stage1", 8, 3), ("stage1", 8, 2), ("stage1", 8, 1), ("stage2a", 8, 1),
                 ("stage2b", 8, 1)]
    script, kept = None, None
    for stage, k, d1 in schedules:
        target = Inkling2GraphRunner(model, schedule_depths(stage, d1, 2), max_len=128, use_cuda_graph=False)
        assert target.row_kernels
        draft = ScriptedDraft(PROMPT + script, model.config.served_vocab_size) if k > 1 else None
        result = Executor(target, draft, k, stage, d1, 2).generate(PROMPT, 48, [])
        if script is None:
            script, kept = result["output_ids"], target.prefix.planes
            continue
        assert result["output_ids"] == script, (stage, d1)
        for key, plane in kept.items():
            other = target.prefix.planes[key]
            assert torch.equal(plane.keys, other.keys) and torch.equal(plane.values, other.values), (stage, d1, key)
            for stream, history in plane.conv.items():
                assert torch.equal(history, other.conv[stream]), (stage, d1, key, stream)


def test_the_attention_split_of_a_position_does_not_depend_on_the_committed_length():
    """A block and plain decoding commit different lengths before the same query, so the split of a full
    layer's keys must group a position's tiles alike under the prefix bucket of either: tile t in program
    t // SEGMENT_TILES, also for a row whose tree positions reach two tiles past the bucket."""
    from vllm.model_executor.models import inkling2_graph as graph

    runner = SimpleNamespace(row_kernels=True, window_size=1024, config=SimpleNamespace(is_local=lambda layer: False))

    def groups(capacity, last):
        split = graph.Inkling2GraphRunner._segments(runner, 0, capacity)
        if split is None:
            return [0] * (last + 1)
        count, per = split
        return [min(tile // per, count - 1) for tile in range(last + 1)]

    for smaller, larger in zip(graph.PREFIX_BUCKETS, graph.PREFIX_BUCKETS[1:]):
        for last in range(smaller // graph.TILE_KEYS + 2):
            assert groups(smaller, last) == groups(larger, last), (smaller, larger, last)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_row_kernel_embedding_gives_a_token_the_same_bits_alone_and_in_a_batch():
    """At the hidden size of the released checkpoints. A torch reduction sums a row of a batch in another
    order than the row alone (a few rows in a thousand end on another BF16 value), which gave a drafted
    block other tokens than plain decoding."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    embedding = torch.randn(1024, 2304, device="cuda", generator=generator).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(2304, device="cuda", generator=generator)).to(torch.bfloat16).float()
    runner = SimpleNamespace(row_kernels=True, dtype=torch.float32, config=SimpleNamespace(rms_norm_eps=1e-6),
                             model=SimpleNamespace(checkpoint=SimpleNamespace(embedding=embedding),
                                                   _w=lambda name: weight),
                             _cached=lambda key, build: build())
    tokens = torch.arange(1024, device="cuda")
    whole = Inkling2GraphRunner._embed(runner, tokens)
    alone = torch.cat([Inkling2GraphRunner._embed(runner, tokens[i:i + 1]) for i in range(1024)])
    assert torch.equal(alone, whole)
    assert torch.equal(Inkling2GraphRunner._embed(runner, tokens[100:333]), whole[100:333])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tf32_does_not_round_bf16_operands():
    """The claim FastInkling2 rests on: a TF32 product of BF16-valued operands is as close to the exact
    product as the FP32 GEMM is. Rounded operands would be off by about 2 ** -11 of the result."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2048, 1024, device="cuda", generator=generator).to(torch.bfloat16).float()
    w = torch.randn(768, 1024, device="cuda", generator=generator).to(torch.bfloat16).float()
    exact = x.double() @ w.double().T
    with tensor_cores(False):
        plain = (x @ w.T).double()
    with tensor_cores():
        fast = (x @ w.T).double()
    scale = exact.abs().max()
    plain_error, fast_error = (plain - exact).abs().max() / scale, (fast - exact).abs().max() / scale
    assert plain_error < 1e-5 and fast_error < 1e-5, (float(plain_error), float(fast_error))


def test_guards(Runner, model):
    with pytest.raises(ValueError, match="final loop"):
        Runner(model, (1, 2))
    with pytest.raises(ValueError, match="full-attention"):
        Runner(model, (R,), window=16, share_layers=[2, 3])
    with pytest.raises(ValueError, match="at least one"):
        Runner(model, (R,), window=0)
    target = Runner(model, (1, R), window=2, max_len=32)
    target.prefill(torch.tensor(PROMPT))
    with pytest.raises(RuntimeError, match="sharing window"):  # node 2 is three positions past the prefix
        target.register_many([(0, None, 1), (1, 0, 2), (2, 1, 3)])
    assert not target.nodes
    target.register_many([(0, None, 1), (1, 0, 2), (3, None, 4)])
    with pytest.raises(RuntimeError, match="registered once"):
        target.register_many([(4, None, 1), (1, 4, 2)])
    assert set(target.nodes) == {0, 1, 3}
    with pytest.raises(RuntimeError, match="differ from the schedule"):
        target.advance_nodes([0, 1], [0, 1])
    with pytest.raises(RuntimeError, match="behind its descendant"):
        target.advance_nodes([1], [0])
    with pytest.raises(RuntimeError, match="once per call"):
        target.advance_nodes([0, 0], [0, 0])
    assert all(node.depth == 0 for node in target.nodes.values())  # refused calls ran nothing
    for depth in range(R):
        target.advance_nodes([1, 0], [depth, depth])  # any row order: parents run first
    with pytest.raises(RuntimeError, match="finished every loop"):
        target.commit([3])
    with pytest.raises(RuntimeError, match="live children"):
        target.release([0])
    with pytest.raises(RuntimeError, match="after the prefix"):
        target.commit([1])
    with pytest.raises(RuntimeError, match="after the prefix"):
        target.commit([1, 0])
    assert target.prefix_length == len(PROMPT)
    target.commit([0, 1])
    with pytest.raises(RuntimeError, match="does not start after the prefix"):
        target.advance_nodes([3], [0])  # a sibling of the committed node is stale
    target.release([3])
    assert not target.nodes and target.prefix_length == len(PROMPT) + 2
