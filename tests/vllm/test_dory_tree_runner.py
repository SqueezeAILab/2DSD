"""Dory tree runner against an independent FP32 reference and sequential decoding.

Every test runs on the dense exact path (``DoryTreeRunner``) and on the
static-shape path (``DoryGraphRunner`` without CUDA graphs or compilation) of a
tiny FP32 checkpoint. The readouts of hand-driven trees match the reference
replay of each node's own path, 2D schedules through the real executor return
the tokens of one-token-at-a-time decoding under the same KV rule, and the
compact prefix storage gives the values of keeping every plane bit for bit.
"""

from types import SimpleNamespace

import dory_reference as reference_module
import pytest
import torch
from dory_tiny import tiny_model, write_model
from vllm.model_executor.models import dory_2d as dory
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor as Executor

PROMPT = [1, 7, 9, 11, 14, 3]
TOKENS = [8, 21, 5, 2, 40, 33, 17, 60, 4, 71, 12, 90, 25, 6]
SHARED = dict(share_layers="full_attention", output_share_layers="full_attention")


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    """FP32 tiny checkpoint; the reference's BF16 rounding is disabled so that the
    comparison isolates the KV read rule."""
    directory = tmp_path_factory.mktemp("dory")
    write_model(directory, tiny_model())
    old, threads = reference_module.bf, torch.get_num_threads()
    reference_module.bf = lambda x: x
    torch.set_num_threads(1)
    weights = dory.DoryWeights.from_checkpoint(directory, "cpu", torch.float32)
    yield weights, reference_module.DoryReference(directory, loops=4, threads=1)
    reference_module.bf = old
    torch.set_num_threads(threads)


RUNNERS = {
    "dense": dory.DoryTreeRunner,
    "static": lambda *a, **k: dory.DoryGraphRunner(
        *a, use_cuda_graph=False, compile_blocks=False, **k
    ),
}
KIND = "dense"


@pytest.fixture(autouse=True, params=list(RUNNERS))
def runner_kind(request):
    """Every test runs on the dense path and on the static-shape path (no graph on CPU)."""
    global KIND
    KIND = request.param


def runner(weights, window=None, depths=(1, 2, 3, 4), **kwargs):
    return RUNNERS[KIND](weights, 4, depths, window=window, max_len=256, **kwargs)


def expected(reference, tokens, window):
    if window is None:
        return reference.replay_readouts(tokens, len(PROMPT))
    return reference.replay_readouts(
        tokens, len(PROMPT), window=window, output_window=window, **SHARED
    )


class Tree:
    """Hand-driven schedule: every call advances all unfinished nodes by one loop."""

    def __init__(self, target):
        self.target, self.hidden, self.depth, self.logits = target, {}, {}, {}
        self.position = {}

    def add(self, node, parent, token):
        self.target.register(node, parent, token)
        self.hidden[node] = self.target.embed(torch.tensor([token]))[0]
        self.depth[node] = 0
        self.position[node] = (
            self.target.prefix_length if parent is None else self.position[parent] + 1
        )

    def call(self):
        active = sorted(
            (n for n, d in self.depth.items() if d < 4),
            key=lambda n: (self.position[n], n),
        )
        spans = [dory.TreeSpan(n, self.depth[n], self.hidden[n]) for n in active]
        results, logits, ids = self.target.advance_readout(spans)
        for node, result in zip(active, results):
            self.hidden[node] = result.hidden[0]
            self.depth[node] += 1
        for row, node in enumerate(ids):
            self.logits[node, self.depth[node]] = logits[row]

    def drop(self, nodes):
        self.target.release(nodes)
        for node in nodes:
            del self.depth[node]

    def commit(self, nodes):
        self.target.commit(nodes)
        for node in nodes:
            del self.depth[node]


@pytest.mark.parametrize("window", [None, 4])
def test_sequential_readouts_match_the_reference_at_every_depth(fixture, window):
    weights, reference = fixture
    target = runner(weights, window)
    prefill = target.prefill(torch.tensor(PROMPT))
    want = expected(reference, PROMPT + TOKENS, window)
    torch.testing.assert_close(
        target.logits(prefill.hidden[-1:])[0], want[4][0], atol=2e-4, rtol=1e-4
    )
    tree = Tree(target)
    for i, token in enumerate(TOKENS[:-1]):
        tree.add(i, None, token)
        for _ in range(4):
            tree.call()
        tree.commit([i])
        for depth in range(1, 5):
            torch.testing.assert_close(
                tree.logits[i, depth], want[depth][i + 1], atol=2e-4, rtol=1e-4
            )
    assert target.prefix_length == len(PROMPT) + len(TOKENS) - 1


@pytest.mark.parametrize("window", [None, 4])
def test_overlapped_tree_equals_each_linear_path(fixture, window):
    """Depth-concurrent calls with in-flight ancestors, a sibling branch, release
    and commit reproduce the reference replay of every node's own path."""
    weights, reference = fixture
    target = runner(weights, window)
    target.prefill(torch.tensor(PROMPT))
    main, branch = TOKENS[:4], [TOKENS[0], 55, 66]
    tree = Tree(target)
    tree.add("a0", None, main[0])
    tree.call()
    tree.add("a1", "a0", main[1])
    tree.add("b1", "a0", branch[1])
    tree.call()
    tree.add("a2", "a1", main[2])
    tree.add("b2", "b1", branch[2])
    tree.call()
    tree.add("a3", "a2", main[3])
    tree.call()
    tree.drop(["b1", "b2"])
    tree.commit(["a0"])
    for node in ("a1", "a2", "a3"):
        tree.call()
        tree.commit([node])
    paths = {
        "a": expected(reference, PROMPT + main + [0], window),
        "b": expected(reference, PROMPT + branch + [0], window),
    }
    checked = 0
    for (node, depth), logits in tree.logits.items():
        torch.testing.assert_close(
            logits, paths[node[0]][depth][int(node[1]) + 1], atol=2e-4, rtol=1e-4
        )
        checked += 1
    assert checked == 4 * 4 + 3 + 2 and target.prefix_length == len(PROMPT) + 4


class ScriptedDraft:
    """Proposes the known continuation with every fifth position corrupted.

    A loop-context request carries the verified prefix length and the in-flight
    span from the cursor to the anchor's parent, so the anchor sits at
    ``prefix_len + len(span)`` and the block starts one position later.
    """

    input_format = "loop-context"

    def __init__(self, sequence):
        self.sequence, self.stats = sequence, {}

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
                slots.append((token + 1) % 96 if position % 5 == 0 else token)
            results.append(SimpleNamespace(token_ids=torch.tensor(slots, dtype=torch.long)))
        return results


def sequential(target, count):
    prefill = target.prefill(torch.tensor(PROMPT))
    output = [int(target.logits(prefill.hidden[-1:]).argmax())]
    tree = Tree(target)
    while len(output) < count:
        node = len(output)
        tree.add(node, None, output[-1])
        for _ in range(4):
            tree.call()
        tree.commit([node])
        output.append(int(tree.logits[node, 4].argmax()))
    return output


@pytest.mark.parametrize("window", [None, 16])
def test_two_dimensional_schedules_reproduce_sequential_decoding(fixture, window):
    """LoopSpec (k=1), block drafting and deeper branching through the real executor
    return the tokens of one-token-at-a-time decoding under the same KV rule."""
    weights, _ = fixture
    count = 40
    want = sequential(runner(weights, window, depths=(4,)), count)
    cases = [
        ("stage1", 1, 1, 2),
        ("stage1", 1, 4, 2),
        ("stage2b", 1, 1, 2),
        ("stage1", 4, 2, 3),
        ("stage2a", 4, 1, 2),
        ("stage2b", 4, 1, 2),
        ("stage2b", 4, 2, 3),
        ("stage1", 4, 4, 2),
    ]
    for stage, k, d1, d2 in cases:
        depths = {d1, 4} if stage == "stage1" else {d1, d2, 4}
        target = runner(weights, window, depths=depths, max_nodes=8)
        draft = ScriptedDraft(PROMPT + want) if k > 1 else None
        result = Executor(target, draft, k, stage, d1, d2).generate(PROMPT, count, [])
        assert result["output_ids"] == want, (stage, k, d1, d2)
        # Cells at a non-readout depth skip the output block.
        assert target.stats["output_rows"] < target.stats["recurrent_rows"]
        assert target.stats["input_rows"] == target.stats["registered_nodes"]


@pytest.mark.parametrize("rings", [None, (4, 4)])
@pytest.mark.parametrize("window", [None, 4])
def test_compact_prefix_matches_every_plane_bit_for_bit(fixture, window, rings):
    """Rings for the sliding layers and the shared planes below R (here also the
    smallest ones, the window and W, which wrap several times) give the logits
    and the schedules' tokens of keeping every plane over the whole context."""
    weights, _ = fixture

    def logits(**kwargs):
        target = runner(weights, window, **kwargs)
        prefill = target.prefill(torch.tensor(PROMPT))
        tree = Tree(target)
        for i, token in enumerate(TOKENS):
            tree.add(i, None, token)
            for _ in range(4):
                tree.call()
            tree.commit([i])
        return target, prefill.hidden, tree.logits

    full, hidden, want = logits(compact_prefix=False)
    compact, got_hidden, got = logits(prefix_rings=rings)
    assert compact.prefix_kv_bytes < full.prefix_kv_bytes
    assert torch.equal(got_hidden, hidden) and want.keys() == got.keys()
    assert all(torch.equal(got[key], want[key]) for key in want)
    # Blocks of k=4 need W > 4: the schedules run with W = 16 (and its smallest ring).
    shared = None if window is None else 16
    rings = None if rings is None else (rings[0], shared or rings[1])
    for stage, k, d1, d2 in (("stage2b", 4, 1, 2), ("stage1", 4, 2, 3), ("stage2a", 1, 1, 3)):
        depths = {d1, 4} if stage == "stage1" else {d1, d2, 4}
        tokens = []
        for kwargs in ({"compact_prefix": False}, {"prefix_rings": rings}):
            target = runner(weights, shared, depths=depths, max_nodes=8, **kwargs)
            draft = (
                ScriptedDraft(PROMPT + [(7 * i) % 96 for i in range(64)]) if k > 1 else None
            )
            result = Executor(target, draft, k, stage, d1, d2).generate(PROMPT, 32, [])
            tokens.append(result["output_ids"])
        assert tokens[0] == tokens[1], (stage, k, d1, d2)


def test_sampling_schedules_run_and_repeat_under_a_seed(fixture):
    """T=0.6 / top-p 0.95 through the executor: in-call readouts feed the sampling
    review, and a fixed seed reproduces the tokens."""
    weights, _ = fixture
    for stage, k in (("stage1", 1), ("stage2b", 4)):
        outputs = []
        for _ in range(2):
            depths = {1, 4} if stage == "stage1" else {1, 2, 4}
            target = runner(weights, 16, depths=depths, max_nodes=8)
            draft = (
                ScriptedDraft(PROMPT + [(7 * i) % 96 for i in range(64)]) if k > 1 else None
            )
            executor = Executor(target, draft, k, stage, 1, 2, temperature=0.6, top_p=0.95)
            outputs.append(executor.generate(PROMPT, 24, [], seed=7)["output_ids"])
        assert outputs[0] == outputs[1] and len(outputs[0]) == 24, (stage, k)


def test_guards(fixture):
    weights, _ = fixture
    with pytest.raises(ValueError, match="include R"):
        runner(weights, depths=(1, 2))
    target = runner(weights, 2, depths=(1, 4))
    target.prefill(torch.tensor(PROMPT))
    target.register(0, None, 5)
    target.register(1, 0, 6)
    with pytest.raises(ValueError, match="in-flight path exceeds"):
        target.register(2, 1, 7)
    with pytest.raises(RuntimeError, match="advance_readout"):
        target.advance([])
    with pytest.raises(ValueError, match="every unfinished node"):
        target.advance_readout([dory.TreeSpan(0, 0, target.embed(torch.tensor([5]))[0])])
    with pytest.raises(ValueError, match="lacks a loop"):
        target.commit([0])


def graph_runner(weights, window, **kwargs):
    return RUNNERS["static"](weights, 4, (1, 2, 4), window=window, max_len=256, **kwargs)


class NodeTree:
    """The Tree schedule through the node interface: the runner keeps hidden states and features."""

    def __init__(self, target):
        self.target, self.depth, self.position, self.logits = target, {}, {}, {}

    def add(self, *nodes):
        self.target.register_many(list(nodes))
        for node, parent, _ in nodes:
            self.depth[node] = 0
            self.position[node] = (
                self.target.prefix_length if parent is None else self.position[parent] + 1
            )

    def call(self):
        active = sorted(
            (n for n, d in self.depth.items() if d < 4),
            key=lambda n: (self.position[n], n),
        )
        rows, argmax, logits = self.target.advance_nodes(active, [self.depth[n] for n in active])
        for node in active:
            self.depth[node] += 1
        assert rows == [i for i, n in enumerate(active) if self.depth[n] in (1, 2, 4)]
        if rows:
            assert argmax.tolist() == logits.argmax(-1).tolist()
        for i, row in enumerate(rows):
            self.logits[active[row], self.depth[active[row]]] = logits[i].clone()

    def drop(self, nodes):
        self.target.release(nodes)
        for node in nodes:
            del self.depth[node]

    def commit(self, nodes, features):
        self.target.commit(nodes, features=features)
        for node in nodes:
            del self.depth[node]


@pytest.mark.parametrize("window", [None, 8])
def test_node_interface_equals_the_span_interface(fixture, window):
    """advance_nodes with the state kept in the runner returns the readouts, feature taps and
    committed prefix of the span interface, across branches, releases, commits and growth."""
    if KIND != "static":
        pytest.skip("the node interface belongs to the static-shape runner")
    weights, _ = fixture
    spans = Tree(graph_runner(weights, window))
    nodes = NodeTree(graph_runner(weights, window, max_nodes=2))
    features, logits = {}, {}

    def call():
        active = sorted(
            (n for n, d in spans.depth.items() if d < 4),
            key=lambda n: (spans.position[n], n),
        )
        loops = [spans.depth[n] for n in active]
        results, rows, ids = spans.target.advance_readout(
            [dory.TreeSpan(n, d, spans.hidden[n]) for n, d in zip(active, loops)]
        )
        for node, loop, result in zip(active, loops, results):
            spans.hidden[node] = result.hidden[0]
            spans.depth[node] += 1
            features[node, loop + 1] = result.features[0]
        for row, node in enumerate(ids):
            logits[node, spans.depth[node]] = rows[row].clone()
        nodes.call()

    for target in (spans.target, nodes.target):
        target.prefill(torch.tensor(PROMPT))
    chain = [
        ("a0", None, 8),
        ("a1", "a0", 21),
        ("b1", "a0", 55),
        ("a2", "a1", 5),
        ("b2", "b1", 66),
        ("a3", "a2", 2),
        ("a4", "a3", 40),
    ]
    steps = [chain[:1], chain[1:3], chain[3:5], chain[5:], [], [], []]
    for step, fresh in enumerate(steps):
        for node, parent, token in fresh:
            spans.add(node, parent, token)
        if fresh:
            nodes.add(*fresh)
        call()
        if step == 3:  # a0 has finished: its other branch goes, then it commits
            spans.drop(["b1", "b2"])
            nodes.drop(["b1", "b2"])
            row = torch.empty((1, features["a0", 4].numel()))
            nodes.commit(["a0"], row)
            spans.commit(["a0"])
            assert torch.equal(row[0], features["a0", 4])
        if step == 5:  # a1 and a2 are at loop R: two tokens in one commit
            rows = torch.empty((2, features["a1", 4].numel()))
            nodes.commit(["a1", "a2"], rows)
            spans.commit(["a1", "a2"])
            assert torch.equal(rows, torch.stack([features["a1", 4], features["a2", 4]]))
    assert nodes.target.max_nodes > 2 and nodes.target.prefix_length == spans.target.prefix_length
    assert nodes.logits.keys() == logits.keys() == {key for key in features if key[1] in (1, 2, 4)}
    for key, value in logits.items():
        assert torch.equal(nodes.logits[key], value), key
    assert all(features[key][-weights.hidden :].isnan().all() for key in features if key[1] == 3)
    live = [n for n in ("a3", "a4") if n in nodes.target._nodes]
    for depth in (1, 2, 4):
        for node in live:
            assert torch.equal(nodes.target.node_features(node, depth), features[node, depth])
        gathered = nodes.target.span_features([(depth, live), (depth, live[:1])])
        assert torch.equal(gathered[0], torch.stack([features[n, depth] for n in live]))
        assert torch.equal(gathered[1], features[live[0], depth][None])
    for store in ("_prefix_store", "_tree_store"):
        assert getattr(nodes.target, store).shape[1] == getattr(spans.target, store).shape[1]
    length = spans.target.prefix_length
    assert torch.equal(
        nodes.target._prefix_store[:, :, :length], spans.target._prefix_store[:, :, :length]
    )


def test_batched_registration_equals_single_registration(fixture):
    """register_many leaves the host tables of one registration per node, through commits."""
    weights, _ = fixture
    single, batched = runner(weights, 16, depths=(1, 4)), runner(weights, 16, depths=(1, 4))
    for target in (single, batched):
        target.prefill(torch.tensor(PROMPT))
    block = [(0, None, 5), (1, 0, 6), (2, 1, 7), (3, 2, 8), (4, 3, 9), (5, 1, 10)]
    for node in block:
        single.register(*node)
    assert batched.register_many(block) == [single._nodes[node] for node, _, _ in block]

    def same():
        for name in ("_table", "_token", "_parent", "_live"):
            assert (getattr(single, name) == getattr(batched, name)).all(), name
        assert single._nodes == batched._nodes and single._free == batched._free

    same()
    for target in (single, batched):
        target.release([5])
        for loop in range(4):
            active = [n for n in range(5) if target._depth[target._nodes[n]] < 4]
            embedded = target.embed(torch.tensor([block[n][2] for n in active]))
            target.advance_readout(
                [dory.TreeSpan(n, loop, embedded[i]) for i, n in enumerate(active)]
            )
        target.commit([0, 1])
        target.register_many([(6, 4, 11), (7, 6, 12)])
    same()
    assert single.prefix_length == len(PROMPT) + 2 and single._span == 5


def test_node_interface_guards(fixture):
    if KIND != "static":
        pytest.skip("the node interface belongs to the static-shape runner")
    weights, _ = fixture
    target = graph_runner(weights, 4)
    target.prefill(torch.tensor(PROMPT))
    target.register_many([(0, None, 5), (1, 0, 6), (2, 0, 7)])
    with pytest.raises(ValueError, match="registered once"):
        target.register_many([(1, 0, 6)])
    with pytest.raises(ValueError, match="every unfinished node"):
        target.advance_nodes([0, 1], [0, 0])
    with pytest.raises(ValueError, match="every unfinished node"):
        target.advance_nodes([0, 1, 1], [0, 0, 0])
    with pytest.raises(ValueError, match="differs from the node depth"):
        target.advance_nodes([0, 1, 2], [0, 1, 0])
    other = target.embed(torch.tensor([6, 6, 7]))  # node 0 was registered with token 5
    with pytest.raises(ValueError, match="embedding of its registered token"):
        target.advance_readout([dory.TreeSpan(n, 0, other[n]) for n in range(3)])
    for loop in range(4):
        target.advance_nodes([0, 1, 2], [loop] * 3)
    with pytest.raises(ValueError, match="live descendant"):
        target.release([0])
    with pytest.raises(ValueError, match="contiguous verified path"):
        target.commit([1])
    with pytest.raises(ValueError, match="rejected sibling survived"):
        target.commit([0, 1])
