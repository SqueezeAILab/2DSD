# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""The Triton attention kernel of Inkling 2 against the reference attention (GPU only)."""
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.inkling2_fast import FastInkling2, tensor_cores
from vllm.model_executor.models.inkling2_reference import PRECISIONS, NativeInkling2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEADS, KV_HEADS, DIM, RANK, EXTENT = 24, 8, 128, 16, 1024  # Inkling-2-Mini's attention geometry


@pytest.fixture(params=[False, True], ids=["rows", "packed"])
def pack(request):
    """The per-row tree kernel, and the one that holds several rows per program."""
    return request.param


def inputs(tokens, keys, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def draw(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    return SimpleNamespace(
        query=draw(tokens, HEADS, DIM, scale=0.5), states=draw(tokens, HEADS, RANK, scale=0.5),
        proj=draw(RANK, EXTENT, scale=0.5), gate=draw(tokens, HEADS, DIM), sink=draw(tokens, HEADS),
        keys=draw(keys, KV_HEADS, DIM, scale=0.5), values=draw(keys, KV_HEADS, DIM))


def kernel(x, *, block_q=16, tuning=(2, 4), **where):
    from vllm.model_executor.models.inkling2_attention import sequence_attention

    rel = (x.states.float() @ x.proj.float()).to(torch.bfloat16)
    return sequence_attention(x.query, rel, x.keys, x.values, x.gate, x.sink, block_q=block_q, tuning=tuning, **where)


@pytest.mark.parametrize("tokens,keys,first_query,first_key,window", [
    (300, 300, 0, 0, None), (300, 300, 0, 0, 64), (2500, 2500, 0, 0, 1024), (700, 1900, 1200, 0, None),
    (700, 1723, 1200, 177, 1024)])
def test_the_kernel_is_the_reference_attention_up_to_rounding(tokens, keys, first_query, first_key, window):
    """The kernel rounds a probability relative to the running maximum and the reference relative to the row
    maximum, so outputs differ by BF16 steps, not more: a wrong key, bias or mask would move them by far more."""
    x = inputs(tokens, keys)
    stub = SimpleNamespace(precision=PRECISIONS["served"], attention_elements=1 << 24, fused_attention=True)
    arguments = [t.float() for t in (x.query, x.keys, x.values, x.states, x.proj, x.gate, x.sink)]
    where = dict(first_query=first_query, first_key=first_key, window=window)
    want = NativeInkling2._attend(stub, *arguments, **where)
    with tensor_cores():
        got = FastInkling2._attend(stub, *arguments, **where)  # the kernel, in the model's chunked launches
    difference = (got - want).abs()
    assert not torch.isnan(got).any()
    assert float(difference.max()) <= 2 ** -7  # two BF16 steps of a value below 1
    assert float(difference.sum() / want.abs().sum()) < 5e-3
    assert float((got == want).float().mean()) > 0.5


def test_grouping_and_launch_ranges_do_not_change_a_bit():
    """A query's output depends on its own visible keys only: not on the queries per program, the warps, or
    which queries and keys a launch covers. 2DSD needs this to name the tokens of plain decoding."""
    x = inputs(333, 1500, seed=5)
    outputs = [kernel(x, first_query=1167, window=1024, block_q=block_q, tuning=(2, warps))
               for block_q in (1, 4, 16, 32) for warps in (1, 4, 8)]
    assert all(torch.equal(outputs[0], other) for other in outputs[1:])
    x = inputs(900, 900, seed=7)
    cut = 517
    for window in (None, 1024, 128):
        whole = kernel(x, window=window)
        low = 0 if window is None else max(0, cut - (window - 1))
        head = SimpleNamespace(**{name: (value if name == "proj" else value[:cut].contiguous())
                                  for name, value in vars(x).items()})
        tail = SimpleNamespace(query=x.query[cut:].contiguous(), states=x.states[cut:].contiguous(), proj=x.proj,
                               gate=x.gate[cut:].contiguous(), sink=x.sink[cut:].contiguous(),
                               keys=x.keys[low:].contiguous(), values=x.values[low:].contiguous())
        joined = torch.cat((kernel(head, window=window), kernel(tail, first_query=cut, first_key=low, window=window)))
        assert torch.equal(joined, whole)


@pytest.mark.parametrize("window,cut,origin", [(None, 0, 0), (None, 700, 0), (1024, 0, 250)])
def test_the_tree_kernel_gives_a_row_the_bits_of_the_sequence_kernel(window, cut, origin, pack):
    """The static runner's kernel reads a row's context from the committed prefix and from its path in the
    tree. Whatever the split, the other rows of the call, the plane of the positions below a sharing cut and
    the first position the prefix holds, the row gets the bits the sequence kernel gives its position."""
    from vllm.model_executor.models.inkling2_attention import sequence_attention
    from vllm.model_executor.models.inkling2_tree_attention import tree_attention

    tokens, planes, slots, width = 1500, 3, 80, 32
    x, other = inputs(tokens, tokens, seed=11), inputs(tokens, tokens, seed=12)
    rel = (x.states.float() @ x.proj.float()).to(torch.bfloat16)
    # Plane 1 holds the row's own sequence, plane 2 the one it reads below the cut, plane 0 neither.
    keys = torch.cat((other.keys[:cut], x.keys[cut:])).contiguous()
    values = torch.cat((other.values[:cut], x.values[cut:])).contiguous()
    want = sequence_attention(x.query, rel, keys, values, x.gate, x.sink, window=window)
    noise = inputs(planes * (tokens + slots), 1, seed=13)

    def planes_of(count, tensor):
        return tensor[:planes * count].reshape(planes, count, KV_HEADS, DIM).clone()

    def index(values):
        return torch.tensor(values, dtype=torch.long, device="cuda")

    for length in (1300, 1343, 1472):
        for positions in ([length], [length + 3], [length, length + 1, length + 17, length + 27],
                          list(range(length, length + 24))):
            held = length - origin
            prefix_k, prefix_v = planes_of(tokens, noise.query.reshape(-1, KV_HEADS, DIM)), \
                planes_of(tokens, noise.gate.reshape(-1, KV_HEADS, DIM))
            prefix_k[1, :held], prefix_v[1, :held] = x.keys[origin:length], x.values[origin:length]
            prefix_k[2, :held], prefix_v[2, :held] = other.keys[origin:length], other.values[origin:length]
            tree_k, tree_v = prefix_k[:, :slots].clone(), prefix_v[:, :slots].clone()
            slot_of = torch.randperm(slots - 1, device="cuda")  # position length + e is in slot slot_of[e]
            last = max(positions) - length + 1
            tree_k[1, slot_of[:last]], tree_v[1, slot_of[:last]] = x.keys[length:length + last], \
                x.values[length:length + last]
            path = torch.full((len(positions), width), slots - 1, dtype=torch.long, device="cuda")
            mask = torch.zeros(len(positions), width, dtype=torch.bool, device="cuda")
            for row, position in enumerate(positions):
                path[row, :position - length + 1], mask[row, :position - length + 1] = \
                    slot_of[:position - length + 1], True
            at = index(positions)
            lowest = torch.zeros_like(at) if window is None else at - (window - 1)
            got = tree_attention(
                x.query[at].contiguous(), rel[at].contiguous(), x.gate[at].contiguous(), x.sink[at].contiguous(),
                prefix_k, prefix_v, tree_k, tree_v, at, lowest, torch.ones_like(at), torch.full_like(at, 2),
                torch.full_like(at, cut), path, mask, index([length]), index([origin]), pack=pack)
            assert torch.equal(got, want[at])



def test_segments_of_the_tree_kernel_merge_to_the_same_attention(pack):
    """Reading a row's tiles in parallel segments and merging them in order is the same attention up to the
    rounding of the output: the bits differ from one program per row, the values by a BF16 step at most."""
    from vllm.model_executor.models.inkling2_tree_attention import tree_attention

    tokens, length, slots, width = 2200, 2100, 40, 16
    x = inputs(tokens, tokens, seed=21)
    rel = (x.states.float() @ x.proj.float()).to(torch.bfloat16)
    at = torch.arange(length, length + 8, device="cuda")
    path = torch.zeros(8, width, dtype=torch.long, device="cuda")
    mask = torch.zeros(8, width, dtype=torch.bool, device="cuda")
    for row in range(8):
        path[row, :row + 1], mask[row, :row + 1] = torch.arange(row + 1, device="cuda"), True
    one = torch.tensor([length], device="cuda")
    zero = torch.zeros_like(at)
    prefix = (x.keys[None, :length].contiguous(), x.values[None, :length].contiguous())
    tree = (x.keys[None, length:length + slots].contiguous(), x.values[None, length:length + slots].contiguous())
    for lowest, segments in ((zero, (5, 8)), (zero, (2, 30)), (at - 1023, (4, 5))):
        arguments = (x.query[at].contiguous(), rel[at].contiguous(), x.gate[at].contiguous(), x.sink[at].contiguous(),
                     *prefix, *tree, at, lowest, zero, zero, zero, path, mask, one, torch.zeros_like(one))
        whole, split = tree_attention(*arguments, pack=pack), tree_attention(*arguments, segments=segments, pack=pack)
        difference = (split.float() - whole.float()).abs()
        assert float(difference.max()) <= 2 ** -7 and float(difference.sum() / whole.float().abs().sum()) < 5e-3


def test_a_split_row_gets_the_same_bits_whatever_is_committed_before_it(pack):
    """The static runner splits a full layer's keys in runs of a fixed number of tiles from position 0. A row
    then gets the same bits whether the positions before it are in the tree (a drafted block) or in the
    prefix (plain decoding committed them), and whatever number of programs the committed length asks for:
    a program past the row's last tile is empty and the merge skips it. A width that changed with the
    committed length gave a block other tokens than plain decoding across a prefix bucket."""
    from vllm.model_executor.models.inkling2_tree_attention import tree_attention

    tokens, slots, width, per = 2400, 48, 32, 8
    x = inputs(tokens, tokens, seed=31)
    rel = (x.states.float() @ x.proj.float()).to(torch.bfloat16)

    def run(length, positions, count):
        at = torch.tensor(positions, dtype=torch.long, device="cuda")
        prefix = (x.keys[None, :length].contiguous(), x.values[None, :length].contiguous())
        tree = (x.keys[None, length:length + slots].contiguous(), x.values[None, length:length + slots].contiguous())
        path = torch.zeros(len(positions), width, dtype=torch.long, device="cuda")
        mask = torch.zeros(len(positions), width, dtype=torch.bool, device="cuda")
        for row, position in enumerate(positions):
            depth = position - length + 1
            path[row, :depth], mask[row, :depth] = torch.arange(depth, device="cuda"), True
        one, zero = torch.tensor([length], device="cuda"), torch.zeros_like(at)
        return tree_attention(x.query[at].contiguous(), rel[at].contiguous(), x.gate[at].contiguous(),
                              x.sink[at].contiguous(), *prefix, *tree, at, zero, zero, zero, zero, path, mask, one,
                              torch.zeros_like(one), segments=(count, per), pack=pack)

    # Positions in the tiles 32 and 33, the last of them 23 positions past the committed length 2090.
    positions = [2090, 2097, 2111, 2112, 2113]
    block = run(2090, positions, count=5)          # five programs: the last one also reads tile 33
    for count in (5, 9, 17, 33):                   # the counts of longer committed prefixes
        assert torch.equal(run(2090, positions, count), block), count
    for row, position in enumerate(positions):     # plain decoding: everything before the row is committed
        for count in (5, 9):
            assert torch.equal(run(position, [position], count)[0], block[row]), (position, count)
    assert not torch.equal(tree_attention(*_whole(x, rel, 2090, positions, slots, width), pack=pack), block)


def _whole(x, rel, length, positions, slots, width):
    """Arguments of one program per row for the rows of the test above (other bits than the split ones)."""
    at = torch.tensor(positions, dtype=torch.long, device="cuda")
    path = torch.zeros(len(positions), width, dtype=torch.long, device="cuda")
    mask = torch.zeros(len(positions), width, dtype=torch.bool, device="cuda")
    for row, position in enumerate(positions):
        depth = position - length + 1
        path[row, :depth], mask[row, :depth] = torch.arange(depth, device="cuda"), True
    one, zero = torch.tensor([length], device="cuda"), torch.zeros_like(at)
    return (x.query[at].contiguous(), rel[at].contiguous(), x.gate[at].contiguous(), x.sink[at].contiguous(),
            x.keys[None, :length].contiguous(), x.values[None, :length].contiguous(),
            x.keys[None, length:length + slots].contiguous(), x.values[None, length:length + slots].contiguous(),
            at, zero, zero, zero, zero, path, mask, one, torch.zeros_like(one))


@pytest.mark.parametrize("segments", [None, (5, 8)])
def test_the_rows_that_share_a_program_do_not_change_a_row(segments, pack):
    """A program of the packed tree kernel holds several rows and loads a committed tile once for the rows
    that read it from one plane. A row gets the same bits alone and in any such company: rows of its plane, of
    other planes, with other sharing cuts, at other positions of the tree. (The per-row kernel: by layout.)"""
    from vllm.model_executor.models.inkling2_tree_attention import tree_attention

    tokens, length, slots, width, planes = 2200, 2090, 48, 32, 3
    x, other = inputs(tokens, tokens, seed=41), inputs(planes * (tokens + slots), 1, seed=42)
    rel = (x.states.float() @ x.proj.float()).to(torch.bfloat16)
    prefix_k = other.query.reshape(-1, KV_HEADS, DIM)[:planes * length].reshape(planes, length, KV_HEADS, DIM).contiguous()
    prefix_v = other.gate.reshape(-1, KV_HEADS, DIM)[:planes * length].reshape(planes, length, KV_HEADS, DIM).contiguous()
    tree_k = other.query.reshape(-1, KV_HEADS, DIM)[-planes * slots:].reshape(planes, slots, KV_HEADS, DIM).contiguous()
    tree_v = other.gate.reshape(-1, KV_HEADS, DIM)[-planes * slots:].reshape(planes, slots, KV_HEADS, DIM).contiguous()
    # (position, plane, plane below the cut, cut): one block of a plane, rows of two other planes, sharing rows.
    rows = [(length + i, 0, 0, 0) for i in range(8)] + [(length + i, 1, 1, 0) for i in (0, 3, 5)] \
        + [(length + i, 2, 0, length + i - 63) for i in (1, 2, 9)] + [(length + 11, 0, 0, 0), (length, 1, 2, 2000)]

    def run(chosen):
        at = torch.tensor([rows[i][0] for i in chosen], dtype=torch.long, device="cuda")
        column = lambda j: torch.tensor([rows[i][j] for i in chosen], dtype=torch.long, device="cuda")
        path = torch.zeros(len(chosen), width, dtype=torch.long, device="cuda")
        mask = torch.zeros(len(chosen), width, dtype=torch.bool, device="cuda")
        for row, i in enumerate(chosen):
            depth = rows[i][0] - length + 1
            path[row, :depth], mask[row, :depth] = torch.arange(depth, device="cuda"), True
        one = torch.tensor([length], device="cuda")
        return tree_attention(x.query[at].contiguous(), rel[at].contiguous(), x.gate[at].contiguous(),
                              x.sink[at].contiguous(), prefix_k, prefix_v, tree_k, tree_v, at, torch.zeros_like(at),
                              column(1), column(2), column(3), path, mask, one, torch.zeros_like(one),
                              segments=segments, pack=pack)

    everyone = list(range(len(rows)))
    together = run(everyone)
    for i in everyone:
        assert torch.equal(run([i])[0], together[i]), i
    shuffled = [15, 3, 9, 0, 12, 7, 1, 14, 5, 10, 2, 13, 6, 11, 4, 8]
    assert torch.equal(run(shuffled), together[shuffled])
