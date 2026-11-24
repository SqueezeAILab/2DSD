# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""The row kernels of the static runner against the reference's torch functions (GPU only).

Rows are random BF16 values. A kernel that rounds its output must give the reference's rounded value in nearly
every element (FP32 sums in another order move a value across a rounding boundary now and then); a wrong tap,
head or source would change most of them.
"""
import pytest
import torch
from vllm.model_executor.models import inkling2_ops as ops
from vllm.model_executor.models import inkling2_reference as ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, ROWS, HALF = 2304, 24, torch.bfloat16


def stored(x):
    return x.to(HALF).float()


def draw(*shape, seed=0, scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed + sum(shape))
    return stored(torch.randn(*shape, device="cuda", generator=generator) * scale)


def same(got, want):
    """Fraction of elements with the same BF16 value."""
    return float((got.to(HALF) == want.to(HALF)).float().mean())


def test_attn_res_mixes_and_norms_as_the_reference():
    blocks, last, addend = [draw(ROWS, H, seed=i) for i in range(5)], draw(ROWS, H, seed=7), draw(ROWS, H, seed=8)
    queries, weight = (draw(3, H, seed=9) * H ** -0.5).contiguous(), draw(3, H, seed=10)
    stack = torch.stack(blocks)
    want = ref.attn_res(blocks, last, queries, 1e-6)
    got = ops.attn_res(stack, last, None, queries, None, 0.0, True, 1e-6)
    assert got.dtype == torch.float32 and torch.allclose(got, want, rtol=0, atol=1e-5)
    normed = ops.attn_res(stack, last, None, queries, weight, 1e-6, True, 1e-6, out_dtype=HALF)
    assert same(normed, stored(ref.rms(want, 1e-6) * weight[:, None, :])) > 0.999
    want = ref.attn_res(blocks, last + addend, queries, 1e-6)
    normed = ops.attn_res(stack, last, addend, queries, weight, 1e-6, True, 1e-6, out_dtype=HALF)
    assert same(normed, stored(ref.rms(want, 1e-6) * weight[:, None, :])) > 0.999
    alone = ops.attn_res(stack[:, :1].contiguous(), last[:1].contiguous(), addend[:1].contiguous(), queries, weight,
                         1e-6, True, 1e-6, out_dtype=HALF)
    assert torch.equal(alone[:, 0], normed[:, 0])  # a row does not depend on the other rows


def test_the_controller_kernels_blend_and_norm_as_the_reference():
    rank = 64
    dest, message = draw(3, ROWS, H, seed=1), draw(3, ROWS, H, seed=2)
    down, gain, forget = draw(3, rank, 2 * H, seed=3, scale=0.02), draw(3, H, rank, seed=4, scale=0.1), \
        draw(3, H, rank, seed=5, scale=0.1)
    weight = draw(3, H, seed=6)
    pair = ops.pair_rms_norm(dest, message)
    want_pair = stored(torch.cat((ref.rms(dest, 1e-6), ref.rms(message, 1e-6)), -1))
    assert same(pair, want_pair) > 0.999
    hidden = stored(want_pair @ down.transpose(1, 2))
    gates = (hidden @ torch.cat((gain, forget), 1).transpose(1, 2)).contiguous()
    blended = (1.0 - torch.tanh(hidden @ forget.transpose(1, 2))) * dest + torch.tanh(hidden @ gain.transpose(1, 2)) * message
    got = ops.blend_rms_norm(dest, message, gates, weight, 1e-6)
    assert same(got, stored(ref.rms(blended, 1e-6) * weight[:, None, :])) > 0.99
    # Zero gates leave the destination: how the runner skips the blend for a row without a message.
    kept = ops.blend_rms_norm(dest, message, torch.zeros_like(gates), weight, 1e-6)
    assert same(kept, stored(ref.rms(dest, 1e-6) * weight[:, None, :])) > 0.999


@pytest.mark.parametrize("width,threshold", [(128, 5.0), (16, 2.0)])
def test_clip_heads_is_the_reference_clip(width, threshold):
    heads = 24
    x, scale = draw(ROWS, heads * width, seed=1), draw(heads, width, seed=2).abs() + 0.5
    tau = (torch.rand(ROWS, device="cuda") + 1.0).contiguous()
    want = stored(ref.l2_clip(x.view(ROWS, heads, width) * scale, threshold, 1e-6))
    assert torch.equal(ops.clip_heads(x.to(HALF), scale, heads, threshold, 1e-6).float(), want)
    assert torch.equal(ops.clip_heads(x.to(HALF), scale, heads, threshold, 1e-6, tau).float(),
                       stored(want * tau[:, None, None]))


@pytest.mark.parametrize("dynamic", [False, True])
def test_conv_rows_reads_its_history_from_the_tree_or_the_prefix(dynamic):
    channels, kernel, heads, planes, slots = 1024, 4, 8, 3, 9
    x = draw(ROWS, channels, seed=1)
    tree, prefix = draw(planes, slots, channels, seed=2), draw(planes, kernel - 1, channels, seed=3)
    generator = torch.Generator(device="cuda").manual_seed(4)
    plane = torch.randint(0, planes, (ROWS,), device="cuda", generator=generator)
    hist_tree = torch.rand(ROWS, kernel - 1, device="cuda", generator=generator) < 0.5
    hist_slot = torch.randint(0, slots, (ROWS, kernel - 1), device="cuda", generator=generator)
    hist_prefix = torch.randint(0, kernel - 1, (ROWS, kernel - 1), device="cuda", generator=generator)
    history = torch.where(hist_tree[..., None], tree[plane[:, None], hist_slot], prefix[plane[:, None], hist_prefix])
    taps = draw(ROWS, heads, kernel, seed=5) if dynamic else None
    weight = draw(kernel, channels, seed=6) if dynamic else draw(channels, kernel, seed=6)
    want = torch.stack([ref.short_conv(x[i:i + 1], weight, history[i], None if taps is None else taps[i:i + 1])[0][0]
                        for i in range(ROWS)])
    got = ops.conv_rows(x, tree, prefix, plane, hist_tree, hist_slot, hist_prefix, weight, taps)
    assert got.dtype == HALF and torch.equal(got.float(), stored(want))


@pytest.mark.parametrize("dynamic", [False, True])
def test_conv_rows_stores_its_inputs_and_reads_the_call_s_own_rows_from_them(dynamic):
    """The launch that stores the inputs gives the bits of storing them first: a chain of rows in one plane
    reads the rows before it, the tree where they are other calls' and the prefix history before the tree."""
    channels, kernel, heads, planes, slots = 1024, 4, 8, 2, ROWS + 5
    x = draw(ROWS, channels, seed=1).to(HALF)
    tree, prefix = draw(planes, slots, channels, seed=2).to(HALF), draw(planes, kernel - 1, channels, seed=3).to(HALF)
    plane = torch.ones(ROWS, dtype=torch.long, device="cuda")
    slot = torch.arange(3, 3 + ROWS, device="cuda")  # row t continues slots 1, 2 and then the rows before it
    hist_tree = torch.ones(ROWS, kernel - 1, dtype=torch.bool, device="cuda")
    hist_slot = (slot[:, None] - torch.arange(kernel - 1, 0, -1, device="cuda")).contiguous()
    hist_tree[hist_slot < 1] = False
    hist_prefix = (hist_slot + kernel - 2).clamp(0, kernel - 2).contiguous()
    hist_row = torch.where(hist_slot >= 3, hist_slot - 3, -1).contiguous()
    taps = draw(ROWS, heads, kernel, seed=5).to(HALF) if dynamic else None
    weight = draw(kernel, channels, seed=6) if dynamic else draw(channels, kernel, seed=6)
    first = tree.clone()
    first[plane, slot] = x
    want = ops.conv_rows(x, first, prefix, plane, hist_tree, hist_slot.clamp(min=0), hist_prefix, weight, taps)
    stale = tree.clone()
    got = ops.conv_rows(x, stale, prefix, plane, hist_tree, hist_slot.clamp(min=0), hist_prefix, weight, taps,
                        store=(slot, hist_row))
    assert torch.equal(got, want) and torch.equal(stale, first)


def test_routing_activation_and_sums_are_the_reference_s():
    logits, bias = torch.randn(ROWS, 128, device="cuda") * 2, torch.randn(128, device="cuda") * 0.1
    logits[0, 5] = logits[0, 9]  # a tie goes to the lower index
    ids, weights = ops.route_rows(logits, bias, 8, 1.3)
    want_ids, want_weights = ref.route_experts(logits, bias, 8, 1.3)
    assert torch.equal(ids, want_ids) and torch.allclose(weights, want_weights, rtol=0, atol=1e-6)
    gate_up, weight = draw(ROWS * 8, 2 * 1152, seed=1), torch.rand(ROWS * 8, device="cuda")
    want = stored(ref.situ(gate_up[:, 0::2], gate_up[:, 1::2], weight[:, None], 30.0, 30.0))
    assert same(ops.situ_rows(gate_up.to(HALF), weight, 30.0, 30.0), want) > 0.999
    down = draw(ROWS, 8, H, seed=2)
    total = down[:, 0]
    for slot in range(1, 8):
        total = total + down[:, slot]
    assert torch.equal(ops.sum_slots(down.to(HALF)).float(), stored(total))
    a, b, c = draw(ROWS, H, seed=3), draw(ROWS, H, seed=4), draw(ROWS, H, seed=5)
    assert torch.equal(ops.sum_rows(a, b.to(HALF), c.to(HALF), out_dtype=torch.float32), stored((a + b) + c))
    assert torch.equal(ops.sum_rows(a.to(HALF), b.to(HALF)).float(), stored(a + b))


@pytest.mark.parametrize("rows,inputs,outputs", [(1, 4608, 64), (7, 64, 4608), (20, 4608, 64)])
def test_gemv_streams_is_each_stream_s_product(rows, inputs, outputs):
    """The controllers' products: as close to the exact product as an FP32 product of the same operands, and a
    row's result is the same alone."""
    x, weight = draw(3, rows, inputs, seed=1).to(HALF), draw(3, outputs, inputs, seed=2, scale=0.05).to(HALF)
    exact = torch.bmm(x.double(), weight.double().transpose(1, 2))
    got = ops.gemv_streams(x, weight)
    assert got.dtype == torch.float32 and got.shape == (3, rows, outputs)
    scale = exact.abs().max()
    assert (got.double() - exact).abs().max() <= 2e-5 * scale  # FP32 sums of thousands of exact products
    assert torch.equal(ops.gemv_streams(x[:, :1].contiguous(), weight), got[:, :1])
    assert torch.equal(ops.gemv_streams(x, weight, out_dtype=HALF), got.to(HALF))
    partials = ops.gemv_streams(x, weight, chunked=True)  # the inputs' chunks of 1024 in parallel
    assert partials.shape == (-(-inputs // 1024), 3, rows, outputs)
    assert (partials.double().sum(0) - exact).abs().max() <= 2e-5 * scale
    if outputs == 64:  # a product that reads such partials: their ordered sum, rounded
        gates = draw(3, 128, outputs, seed=3, scale=0.05).to(HALF)
        total = partials[0]
        for part in partials[1:]:
            total = total + part
        assert torch.equal(ops.gemv_streams(partials, gates), ops.gemv_streams(total.to(HALF), gates))


@pytest.mark.parametrize("inputs,outputs", [(H, 3 * H), (16, 1024), (H, 128)])
def test_gemv_any_gives_a_row_the_same_bits_in_any_call(inputs, outputs):
    """A product of the static runner: a row's outputs are its own ordered sums, so they are the same bits
    alone, among 7, 20 or 70 rows (other rows per program), and equal to gemv_rows' up to its 16 rows. This is
    what makes a schedule's target rows independent of how many rows a call carries."""
    x, weight = draw(70, inputs, seed=1).to(HALF), draw(outputs, inputs, seed=2, scale=0.05).to(HALF)
    whole = ops.gemv_any(x, weight)
    exact = x.double() @ weight.double().T
    assert (whole.double() - exact).abs().max() <= 2e-5 * exact.abs().max()
    for rows in (1, 7, 16, 20, 33):
        assert torch.equal(ops.gemv_any(x[:rows], weight), whole[:rows]), rows
    assert torch.equal(ops.gemv_any(x[40:41], weight), whole[40:41])
    assert torch.equal(ops.gemv_rows(x[:16], weight), whole[:16])
    assert torch.equal(ops.gemv_any(x, weight, out_dtype=HALF), whole.to(HALF))


def test_blend_rms_norm_without_messages_is_a_masked_row_s_arithmetic():
    """A call of first-pass rows only norms its mixes with the bits a first-pass row gets among later-pass rows."""
    dest, message, gates = draw(3, ROWS, H, seed=1), draw(3, ROWS, H, seed=2), draw(3, ROWS, 2 * H, seed=3)
    norm = draw(3, H, seed=4)
    live = torch.arange(ROWS, device="cuda") % 3 != 0
    mixed = ops.blend_rms_norm(dest, message, gates, norm, 1e-6, live)
    plain = ops.blend_rms_norm(dest, None, None, norm, 1e-6)
    assert torch.equal(mixed[:, ~live], plain[:, ~live]) and not torch.equal(mixed[:, live], plain[:, live])
    shared = ops.blend_rms_norm(dest[:1].expand(3, -1, -1), None, None, norm[:1].expand(3, -1), 1e-6)
    assert torch.equal(shared[1], plain[0]) and torch.equal(shared[2], plain[0])


def test_router_rows_routes_the_exact_logits_whatever_the_other_rows():
    gate, bias = draw(128, H, seed=1, scale=0.05).to(HALF), torch.randn(128, device="cuda") * 0.1
    x = draw(ROWS, H, seed=2).to(HALF)
    ids, weights = ops.router_rows(x, gate, bias, 8, 1.3)
    want_ids, want_weights = ref.route_experts((x.double() @ gate.double().T).float(), bias, 8, 1.3)
    assert torch.equal(ids, want_ids) and torch.allclose(weights, want_weights, rtol=0, atol=1e-5)
    alone = [ops.router_rows(x[i:i + 1], gate, bias, 8, 1.3) for i in range(ROWS)]
    assert torch.equal(torch.cat([a[0] for a in alone]), ids) and torch.equal(torch.cat([a[1] for a in alone]), weights)
    padding = torch.arange(ROWS, device="cuda") >= ROWS - 3  # padding rows read row 0's experts
    padded_ids, padded_weights = ops.router_rows(x, gate, bias, 8, 1.3, padding)
    assert torch.equal(padded_ids[:-3], ids[:-3]) and torch.equal(padded_weights[:-3], weights[:-3])
    assert torch.equal(padded_ids[-3:], ids[:1].expand(3, -1)) and torch.equal(padded_weights[-3:], weights[:1].expand(3, -1))


def test_blend_rms_norm_leaves_a_row_without_a_message_its_own_mix():
    dest, message, gates = draw(2, ROWS, H, seed=1), draw(2, ROWS, H, seed=2), draw(2, ROWS, 2 * H, seed=3)
    norm = draw(2, H, seed=4)
    live = torch.arange(ROWS, device="cuda") % 3 != 0
    want = ops.blend_rms_norm(dest, message, gates * live[None, :, None], norm, 1e-6)
    assert torch.equal(ops.blend_rms_norm(dest, message, gates, norm, 1e-6, live), want)


def test_store_key_value_clips_norms_and_writes_the_rows_slots():
    kv, dim, planes, slots = 8, 128, 3, 9
    keys, values = draw(ROWS, kv * dim, seed=1), draw(ROWS, kv * dim, seed=2)
    tree_k, tree_v = draw(planes, ROWS + slots, kv, dim, seed=3).to(HALF), draw(planes, ROWS + slots, kv, dim, seed=4).to(HALF)
    before = tree_k.clone()
    generator = torch.Generator(device="cuda").manual_seed(5)
    plane = torch.randint(0, planes, (ROWS,), device="cuda", generator=generator)
    slot = torch.randperm(ROWS + slots, device="cuda", generator=generator)[:ROWS].contiguous()
    wide = torch.cat((keys, values), 1).to(HALF)  # the inputs as columns of one product
    ops.store_key_value(wide[:, :kv * dim], wide[:, kv * dim:], tree_k, tree_v, plane, slot, 3.0, 1e-6, 1e-6)
    assert torch.equal(tree_k[plane, slot].float(), stored(ref.l2_clip(keys.view(ROWS, kv, dim), 3.0, 1e-6)))
    assert same(tree_v[plane, slot], stored(ref.rms(values.view(ROWS, kv, dim), 1e-6))) > 0.999
    untouched = torch.ones(planes, ROWS + slots, dtype=torch.bool, device="cuda")
    untouched[plane, slot] = False
    assert torch.equal(tree_k[untouched], before[untouched])


@pytest.mark.parametrize("rows", [1, 7, 16])
def test_gemv_rows_accumulates_in_fp32(rows):
    """A product of a few BF16 rows with a BF16 weight that is not rounded to BF16: as close to the exact
    product as an FP32 product of the same operands."""
    x, weight = draw(rows, H, seed=1).to(HALF), draw(5000, H, seed=2, scale=0.05).to(HALF)
    exact = x.double() @ weight.double().T
    got = ops.gemv_rows(x, weight)
    assert got.dtype == torch.float32
    assert float((got.double() - exact).abs().max() / exact.abs().max()) < 1e-5
