# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Row-wise Triton kernels of Inkling 2's small steps, for the static tree runner.

The first part is taken from the served Hopper path, vLLM PR #69028
(vllm/models/inkling2/nvidia/ops/norm.py, attn_res.py and loop.py, Apache-2.0),
as shipped in vllm/vllm-openai:inkling-2-mini: RMSNorm, the block attention
residual with the following RMSNorm in the same kernel, and the two kernels of
a loop message controller. One program computes one row, so a row's result
does not depend on the other rows of a call. All math is FP32 and each output
is rounded once to its storage dtype. Changes from the source: ``attn_res``
takes the output dtype and always uses the small-batch kernel (the source
switches kernels with the batch size), and programmatic dependent launch is
not requested.

The second part is this project's: the per-head L2 clip of the query and
relative states, the short conv of a tree row, whose earlier inputs are read
where the runner keeps them (the tree by slot, or the prefix history), the
router's choice of experts, the experts' activation, the sum of a row's
experts and the residual sum that ends a layer.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

@triton.jit
def rsqrt(x):
    # Non-FTZ rsqrt.approx, as emitted by CuTe's cute.math.rsqrt.
    return tl.inline_asm_elementwise(
        "rsqrt.approx.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rms_norm_kernel(
    X,
    W,
    Y,
    stride_x,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), N) + eps)
    y = x * rstd
    y = y * tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """RMSNorm over the last dim of a row-major (possibly row-strided) tensor."""
    n = x.shape[-1]
    assert x.stride(-1) == 1
    x2d = x if x.dim() == 2 else x.reshape(-1, n)
    out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    if x2d.shape[0]:
        block = triton.next_power_of_2(n)
        _rms_norm_kernel[(x2d.shape[0],)](
            x2d,
            weight,
            out,
            x2d.stride(0),
            n,
            eps,
            BLOCK=block,
            num_warps=min(max(block // 512, 1), 16),
        )
    return out


@triton.jit
def _load_source(
    Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
):
    if n < N_STACK:
        x = tl.load(Stack + n * stride_s + row_off + offs, mask, other=0)
        x = x.to(tl.float32)
    else:
        x = tl.load(Last + row_off + offs, mask, other=0).to(tl.float32)
        if HAS_ADDEND:
            x = x + tl.load(Addend + row_off + offs, mask, other=0).to(tl.float32)
    return x


@triton.jit
def _mix(
    Stack,
    stride_s,
    Last,
    Addend,
    row_off,
    offs,
    mask,
    probs,
    n_idx,
    N_STACK,
    HAS_ADDEND,
    BLOCK_D,
):
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        x = _load_source(
            Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
        )
        acc = tl.fma(tl.sum(tl.where(n_idx == n, probs, 0.0)), x, acc)
    return acc


@triton.jit
def _attn_res_row_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # Small batches: one program per (token, query) holds the row in registers
    # as a power-of-2 head [0, BLOCK_A) and a masked tail [BLOCK_A, D), so the
    # sources are read twice instead of three times.
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    da = tl.arange(0, BLOCK_A)
    db = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = da < D
    mask_b = db < D
    n_idx = tl.arange(0, BLOCK_N)
    qa = tl.load(Queries + q_off + da, mask_a, other=0)
    qb = tl.load(Queries + q_off + db, mask_b, other=0)
    src = (Stack, stride_s, Last, Addend, row_off)

    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        rstd = rsqrt(tl.fma(tl.sum(xa * xa) + tl.sum(xb * xb), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        dot = tl.sum(qa * xa) + tl.sum(qb * xb)
        logits = tl.where(n_idx == n, dot * rstd, logits)

    probs = tldevice.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    acc_a = tl.zeros((BLOCK_A,), tl.float32)
    acc_b = tl.zeros((BLOCK_B,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        p = tl.sum(tl.where(n_idx == n, probs, 0.0))
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        acc_a = tl.fma(p, xa, acc_a)
        acc_b = tl.fma(p, xb, acc_b)
    acc_a = acc_a * inv_sum
    acc_b = acc_b * inv_sum
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    if APPLY_NORM:
        sum_sq = tl.sum(acc_a * acc_a) + tl.sum(acc_b * acc_b)
        out_rstd = rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
        wa = tl.load(NormW + q_off + da, mask_a, other=0).to(tl.float32)
        wb = tl.load(NormW + q_off + db, mask_b, other=0).to(tl.float32)
        acc_a = acc_a * out_rstd * wa
        acc_b = acc_b * out_rstd * wb
    tl.store(out + da, acc_a.to(Out.dtype.element_ty), mask_a)
    tl.store(out + db, acc_b.to(Out.dtype.element_ty), mask_b)


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Mix `[*stack, last (+ addend)]` with each pseudo-query.

    `queries` are fp32 `(Q, D)`, already scaled by `D**-0.5`; each fp32
    mix is RMS-normalized with its row of `norm_weight` `(Q, D)`. Returns
    `(Q, T, D)` in `out_dtype` (default `last.dtype`), or the fp32 mixes
    without `norm_weight`.
    """
    num_tokens, d = last.shape
    num_stack = stack.shape[0]
    num_q = queries.shape[0]
    assert last.is_contiguous() and queries.is_contiguous()
    assert queries.dtype == torch.float32
    assert num_stack == 0 or (stack.stride(1) == d and stack.stride(2) == 1)
    if addend is not None:
        assert addend.is_contiguous() and addend.shape == last.shape
    apply_norm = norm_weight is not None
    if norm_weight is None:
        norm_weight = queries  # unread
    assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    out_dtype = (out_dtype or last.dtype) if apply_norm else torch.float32
    out = torch.empty((num_q, num_tokens, d), dtype=out_dtype, device=last.device)
    block_a = triton.next_power_of_2(d + 1) // 2
    _attn_res_row_kernel[(num_q, num_tokens)](
        stack,
        last,
        addend,
        queries,
        norm_weight,
        out,
        stack.stride(0) if num_stack else 0,
        num_tokens,
        eps,
        norm_eps,
        D=d,
        N_STACK=num_stack,
        HAS_ADDEND=addend is not None,
        NORMALIZE=normalize_values,
        APPLY_NORM=apply_norm,
        BLOCK_N=triton.next_power_of_2(num_stack + 1),
        BLOCK_A=block_a,
        BLOCK_B=max(triton.next_power_of_2(d - block_a), 16),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + (s * T + t) * D + cols, mask, other=0)
    dest_rstd = rsqrt(tl.div_rn(tl.sum(dest * dest), D) + eps)
    message_rstd = rsqrt(tl.div_rn(tl.sum(message * message), D) + eps)
    out = Out + (s * T + t) * (2 * D)
    tl.store(out + cols, (dest * dest_rstd).to(Out.dtype.element_ty), mask)
    tl.store(out + D + cols, (message * message_rstd).to(Out.dtype.element_ty), mask)


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """bf16 `(S, T, 2D)` `[rms(dest), rms(message)]` of fp32 `(S, T, D)`
    inputs, weightless; `dest` may broadcast over streams (stride 0)."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    out = message.new_empty((num_streams, num_tokens, 2 * d), dtype=torch.bfloat16)
    if num_tokens:
        _pair_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            out,
            dest.stride(0),
            num_tokens,
            1e-6,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
        )
    return out


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormW,
    Out,
    Live,
    stride_dest_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
    PLAIN: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    row = s * T + t
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    if PLAIN:  # no row has a message: the norm of the mixes, the arithmetic of a row without one below
        x = dest
    else:
        message = tl.load(Message + row * D + cols, mask, other=0)
        gain = tldevice.tanh(tl.load(GainForget + row * (2 * D) + cols, mask, other=0))
        forget = tldevice.tanh(
            tl.load(GainForget + row * (2 * D) + D + cols, mask, other=0)
        )
        x = (1.0 - forget) * dest + gain * message
        if MASKED:  # a row without a message keeps its own mix, bit for bit
            x = tl.where(tl.load(Live + t) != 0, x, dest)
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), D) + norm_eps)
    y = x * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor | None,
    gain_forget: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    live: torch.Tensor | None = None,
) -> torch.Tensor:
    """bf16 `(S, T, D)` RMSNorm (row `s` of `norm_weight` `(S, D)`) of
    `(1 - tanh(forget)) * dest + tanh(gain) * message`, with fp32 `(S, T, D)`
    `dest` (may broadcast over streams) and `message`, and fp32 `(S, T, 2D)`
    `gain_forget` = `[gain, forget]`. `norm_weight` may broadcast too. A row
    t with `live[t]` false (bool `(T,)`) has no message: the norm of its `dest`.
    With `message` None no row has one. A row without a message gets the same
    bits either way, so its result does not depend on the other rows of a call."""
    plain = message is None
    num_streams, num_tokens, d = dest.shape
    assert dest.stride(1) == d and dest.stride(2) == 1 and dest.dtype == torch.float32
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    if not plain:
        assert dest.shape == message.shape and message.is_contiguous() and message.dtype == torch.float32
        assert gain_forget.shape == (num_streams, num_tokens, 2 * d)
        assert gain_forget.is_contiguous() and gain_forget.dtype == torch.float32
        assert live is None or (live.dtype == torch.bool and live.shape == (num_tokens,) and live.is_contiguous())
    out = dest.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    if num_tokens:
        _blend_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            dest if plain else message,
            dest if plain else gain_forget,
            norm_weight,
            out,
            live if live is not None and not plain else out,
            dest.stride(0),
            norm_weight.stride(0),
            num_tokens,
            norm_eps,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            MASKED=live is not None and not plain,
            PLAIN=plain,
            num_warps=8,
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
        )
    return out


# From ops/gemv.py of the same source: a product of a few BF16 rows with a BF16 weight, accumulated in FP32.
# The source uses it on SM12x only; it is a plain Triton kernel.

GEMV_ROWS = 16


@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def gemv_rows(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """`x @ weight.T` of up to 16 BF16 rows `(M, K)` with a BF16 weight `(N, K)`, accumulated in fp32 and
    returned in `out_dtype`: a product that reads the weight as stored and does not round to BF16."""
    m, k = x.shape
    n = weight.shape[0]
    assert m <= GEMV_ROWS and x.dtype == weight.dtype == torch.bfloat16 and weight.shape[1] == k
    assert x.stride(1) == weight.stride(1) == 1
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if m:
        _gemv_kernel[(triton.cdiv(n, 32),)](
            x, weight, None, out, m, n, k, x.stride(0), weight.stride(0), out.stride(0),
            HAS_BIAS=False, BN=32, BK=256, MP=GEMV_ROWS, num_warps=4,
        )
    return out


# This project's kernels.


@triton.jit
def _gemv_streams_kernel(
    X,
    W,
    Y,
    M,
    N,
    K,
    SXP,
    SXS,
    SX,
    SWS,
    SYP,
    SYS,
    SY,
    ROW_BLOCKS,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    CHUNK: tl.constexpr,
    PARTS: tl.constexpr,
):
    """`_gemv_kernel` for one stream of a batch: program (column block, stream and block of MP rows,
    K chunk). A row's result is its own inputs' ordered sum, whatever MP and the other rows are.

    CHUNK > 0: the program sums the inputs `[chunk * CHUNK, (chunk + 1) * CHUNK)` only and writes the
    fp32 partial `Y[chunk]`. PARTS > 0: the input row is the ordered fp32 sum of the PARTS partials
    `X[part]`, rounded to bf16 (a product whose input is such a chunked product)."""
    pid = tl.program_id(0)
    cell = tl.program_id(1).to(tl.int64)
    stream = cell // ROW_BLOCKS
    chunk = tl.program_id(2).to(tl.int64)
    cols = pid * BN + tl.arange(0, BN)
    rows = (cell % ROW_BLOCKS) * MP + tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    first, last = 0, K
    if CHUNK > 0:
        first = chunk * CHUNK
        last = tl.minimum(first + CHUNK, K)
    for k0 in range(first, last, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + stream * SWS + cols[None, :].to(tl.int64) * K + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < last),
            other=0,
        )
        at = stream * SXS + rows[:, None] * SX + ks[None, :]
        live = (rows[:, None] < M) & (ks[None, :] < last)
        if PARTS > 0:
            total = tl.zeros((MP, BK), tl.float32)
            for part in tl.static_range(PARTS):
                total += tl.load(X + part * SXP + at, live, other=0)
            x = total.to(tl.bfloat16)
        else:
            x = tl.load(X + at, live, other=0)
        acc += tl.dot(x, w)
    tl.store(
        Y + chunk * SYP + stream * SYS + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


GEMV_CHUNK = 1024  # inputs per partial of a chunked product (the served linear's K group)


def _row_lanes(rows: int) -> int:
    """Rows per program of a GEMV: one program reads a weight block once for up to 64 rows."""
    return 16 if rows <= 16 else 32 if rows <= 32 else 64


def gemv_streams(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32,
                 chunked: bool = False) -> torch.Tensor:
    """`x[s] @ weight[s].T` of bf16 rows `(S, M, K)` with bf16 weights `(S, N, K)` in one launch:
    accumulated in fp32 over ordered blocks of the inputs and returned `(S, M, N)` in `out_dtype`. A row's
    result does not depend on the other rows nor on their number.

    `chunked` returns instead the fp32 partials `(P, S, M, N)` of the inputs' chunks of 1024, computed in
    parallel programs; their ordered sum is the product. `x` may itself be such partials `(P, S, M, K)`
    (fp32): the input is then their ordered sum rounded to bf16."""
    parts = x.shape[0] if x.dim() == 4 else 0
    streams, m, k = x.shape[-3:]
    n = weight.shape[1]
    assert weight.dtype == torch.bfloat16 and weight.shape == (streams, n, k)
    assert x.dtype == (torch.float32 if parts else torch.bfloat16) and x.is_contiguous() and weight.is_contiguous()
    chunks = triton.cdiv(k, GEMV_CHUNK) if chunked else 1
    shape = (chunks, streams, m, n) if chunked else (streams, m, n)
    out = torch.empty(shape, dtype=torch.float32 if chunked else out_dtype, device=x.device)
    if m:
        lanes = _row_lanes(m)
        blocks = triton.cdiv(m, lanes)
        stride = out.stride()[-3:]
        _gemv_streams_kernel[(triton.cdiv(n, 32), streams * blocks, chunks)](
            x, weight, out, m, n, k,
            x.stride(0) if parts else 0, x.stride(-3), x.stride(-2), weight.stride(0),
            out.stride(0) if chunked else 0, stride[0], stride[1], blocks,
            BN=32, BK=max(16, min(256, triton.next_power_of_2(k))), MP=lanes, CHUNK=GEMV_CHUNK if chunked else 0,
            PARTS=parts, num_warps=4,
        )
    return out


def gemv_any(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """`x @ weight.T` of any number of bf16 rows `(M, K)` with a bf16 weight `(N, K)` in one launch,
    accumulated in fp32 over ordered blocks of 256 inputs (the sums of `gemv_rows`). A row's result does
    not depend on the other rows of the call nor on their number: the product of a row is the same in a
    call of one row and in a call of many."""
    assert x.dim() == 2 and weight.dim() == 2 and weight.is_contiguous()
    return gemv_streams(x.contiguous()[None], weight[None], out_dtype)[0]


@triton.jit
def _clip_heads_kernel(
    X,
    Scale,
    Tau,
    Out,
    stride_x,
    threshold,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One row: per head, `x * scale` clipped to L2 norm `threshold`, rounded,
    then times the row's `tau` and rounded again (`inkling2_reference.l2_clip`)."""
    row = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, BLOCK_H)[:, None]
    cols = tl.arange(0, BLOCK_D)[None, :]
    mask = (heads < H) & (cols < D)
    x = tl.load(X + row * stride_x + heads * D + cols, mask, other=0).to(tl.float32)
    x = x * tl.load(Scale + heads * D + cols, mask, other=0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=1))
    y = x * tl.minimum(tl.div_rn(threshold, norm + eps), 1.0)[:, None]
    if HAS_TAU:
        y = y.to(Out.dtype.element_ty).to(tl.float32) * tl.load(Tau + row)
    tl.store(Out + row * (H * D) + heads * D + cols, y.to(Out.dtype.element_ty), mask)


def clip_heads(
    x: torch.Tensor,
    scale: torch.Tensor,
    heads: int,
    threshold: float,
    eps: float,
    tau: torch.Tensor | None = None,
) -> torch.Tensor:
    """bf16 `(T, heads, D)`: each head of `x` `(T, heads * D)` times `scale`
    (`(heads, D)`, any float dtype), clipped to L2 norm `threshold`, then
    times the fp32 per-row `tau` `(T,)` if given. Rounded after the clip and
    after `tau`."""
    tokens, width = x.shape
    d = width // heads
    assert x.stride(1) == 1 and scale.shape == (heads, d) and scale.is_contiguous()
    assert tau is None or (tau.shape == (tokens,) and tau.dtype == torch.float32 and tau.is_contiguous())
    out = torch.empty((tokens, heads, d), dtype=torch.bfloat16, device=x.device)
    if tokens:
        _clip_heads_kernel[(tokens,)](
            x,
            scale,
            tau if tau is not None else scale,
            out,
            x.stride(0),
            threshold,
            eps,
            H=heads,
            D=d,
            HAS_TAU=tau is not None,
            BLOCK_H=triton.next_power_of_2(heads),
            BLOCK_D=triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


CONV_BLOCK, CONV_WARPS = 512, 4  # channels per program of the short conv and its warps (elementwise: no effect on bits)


@triton.jit
def _conv_rows_kernel(
    X,
    Tree,
    Prefix,
    Weight,
    Dynamic,
    Out,
    Planes,
    HistTree,
    HistSlot,
    HistPrefix,
    HistRow,
    Slots,
    stride_x,
    tree_slots,
    scale,
    C: tl.constexpr,
    K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DYNAMIC: tl.constexpr,
    STORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One row of `inkling2_reference.short_conv`: the K - 1 inputs before the
    row's own come from the tree slot `HistSlot` where `HistTree`, else from
    entry `HistPrefix` of the prefix history, both in the row's plane. With
    STORE the row writes its own input to its tree slot `Slots`, and reads an
    earlier input that is another row's of this launch (`HistRow` >= 0) from
    `X`, since that row's write is not ordered with this read."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)  # a program per block of channels
    mask = cols < C
    plane = tl.load(Planes + row)
    own = tl.load(X + row * stride_x + cols, mask, other=0)
    x = own.to(tl.float32)
    if STORE:
        tl.store(Tree + (plane * tree_slots + tl.load(Slots + row)) * C + cols, own.to(Tree.dtype.element_ty), mask)
    acc = tl.zeros((BLOCK,), tl.float32)
    for step in tl.static_range(K):
        # Static taps run from the oldest input to the row's own, dynamic ones from the row's own back.
        entry = K - 1 - step if DYNAMIC else step
        if entry == K - 1:
            value = x
        else:
            at = row * (K - 1) + entry
            in_tree = tl.load(HistTree + at) != 0
            tree_at = (plane * tree_slots + tl.load(HistSlot + at)) * C + cols
            prefix_at = (plane * (K - 1) + tl.load(HistPrefix + at)) * C + cols
            if STORE:
                source = tl.load(HistRow + at)
                here = source >= 0
                value = (
                    tl.load(X + source * stride_x + cols, mask & here, other=0).to(tl.float32)
                    + tl.load(Tree + tree_at, mask & in_tree & (not here), other=0).to(tl.float32)
                    + tl.load(Prefix + prefix_at, mask & (not in_tree) & (not here), other=0).to(tl.float32)
                )
            else:
                value = tl.load(Tree + tree_at, mask & in_tree, other=0).to(tl.float32) + tl.load(
                    Prefix + prefix_at, mask & (not in_tree), other=0
                ).to(tl.float32)
        if DYNAMIC:
            taps = tl.load(Weight + step * C + cols, mask, other=0).to(tl.float32) + tl.load(
                Dynamic + (row * (C // HEAD_DIM) + cols // HEAD_DIM) * K + step, mask, other=0
            ).to(tl.float32)
        else:
            taps = tl.load(Weight + cols * K + step, mask, other=0).to(tl.float32)
        acc = acc + value * taps
    tl.store(Out + row * C + cols, ((acc + x) * scale).to(Out.dtype.element_ty), mask)


def conv_rows(
    x: torch.Tensor,
    tree: torch.Tensor,
    prefix: torch.Tensor,
    planes: torch.Tensor,
    hist_tree: torch.Tensor,
    hist_slot: torch.Tensor,
    hist_prefix: torch.Tensor,
    weight: torch.Tensor,
    dynamic: torch.Tensor | None = None,
    store: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """bf16 `(T, C)` causal short conv with skip of the rows `x` `(T, C)`.

    `tree` `(planes, slots, C)` holds the inputs of in-flight rows (the rows'
    own already written) and `prefix` `(planes, K - 1, C)` the last inputs of
    the committed sequence. Row t in plane `planes[t]` takes its earlier input
    j (oldest first) from `tree[plane, hist_slot[t, j]]` where `hist_tree[t, j]`,
    else from `prefix[plane, hist_prefix[t, j]]`. Static: `weight` `(C, K)`,
    oldest tap first. Dynamic: `weight` `(K, C)` and `dynamic` `(T, heads, K)`,
    newest tap first, the per-row deltas shared within a head.

    `store=(slots, hist_row)` writes each row's own input to
    `tree[planes[t], slots[t]]` in the same launch (`tree` then need not hold
    it): an earlier input j of row t that is the input of row `hist_row[t, j]`
    of this call (>= 0, a row of the same plane) is read from `x`.
    """
    tokens, channels = x.shape
    kernel = weight.shape[1] if dynamic is None else weight.shape[0]
    rows = (planes, hist_tree, hist_slot, hist_prefix)
    assert x.stride(1) == 1 and all(t.is_contiguous() for t in (tree, prefix, weight, *rows))
    assert tree.shape[2] == prefix.shape[2] == channels and prefix.shape[1] == kernel - 1
    assert hist_tree.dtype == torch.bool and hist_tree.shape == hist_slot.shape == hist_prefix.shape == (tokens, kernel - 1)
    if dynamic is not None:
        assert dynamic.is_contiguous() and dynamic.shape[0] == tokens and dynamic.shape[2] == kernel
        assert weight.shape == (kernel, channels) and channels % dynamic.shape[1] == 0
    else:
        assert weight.shape == (channels, kernel)
    slots, hist_row = store if store is not None else (planes, hist_slot)
    if store is not None:
        assert slots.is_contiguous() and hist_row.is_contiguous() and x.dtype == tree.dtype
        assert slots.shape == (tokens,) and hist_row.shape == hist_slot.shape
    out = torch.empty((tokens, channels), dtype=torch.bfloat16, device=x.device)
    if tokens:
        _conv_rows_kernel[(tokens, triton.cdiv(channels, CONV_BLOCK))](
            x,
            tree,
            prefix,
            weight,
            dynamic if dynamic is not None else weight,
            out,
            planes,
            hist_tree,
            hist_slot,
            hist_prefix,
            hist_row,
            slots,
            x.stride(0),
            tree.shape[1],
            (kernel + 1) ** -0.5 if dynamic is None else (2 * kernel + 1) ** -0.5,
            C=channels,
            K=kernel,
            HEAD_DIM=channels // dynamic.shape[1] if dynamic is not None else 1,
            DYNAMIC=dynamic is not None,
            STORE=store is not None,
            BLOCK=CONV_BLOCK,
            num_warps=CONV_WARPS,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _route(logits, bias, lanes, mask, route_scale, E_POW2: tl.constexpr, TOPK: tl.constexpr, TOPK_POW2: tl.constexpr):
    """`inkling2_reference.route_experts` of one row's fp32 logits `(E_POW2,)`: ids and weights `(TOPK_POW2,)`."""
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias
    scores = tl.where(mask, scores, float("-inf"))
    slots = tl.arange(0, TOPK_POW2)
    chosen = tl.zeros((TOPK_POW2,), tl.int64)
    selected = tl.zeros((TOPK_POW2,), tl.float32)
    for k in tl.static_range(TOPK):  # the largest score, the lower index on a tie
        best = tl.min(tl.where(scores == tl.max(scores), lanes, E_POW2))
        chosen = tl.where(slots == k, best.to(tl.int64), chosen)
        selected = tl.where(slots == k, tl.sum(tl.where(lanes == best, logits, 0.0)), selected)
        scores = tl.where(lanes == best, float("-inf"), scores)
    live = slots < TOPK
    log_probs = tl.minimum(selected, 0.0) - tldevice.log1p(tl.exp(-tl.abs(selected)))
    top = tl.max(tl.where(live, log_probs, float("-inf")))
    shifted = tl.where(live, tl.exp(log_probs - top), 0.0)
    if TOPK == 8:
        s0 = tl.sum(tl.where(slots == 0, shifted, 0.0))
        s1 = tl.sum(tl.where(slots == 1, shifted, 0.0))
        s2 = tl.sum(tl.where(slots == 2, shifted, 0.0))
        s3 = tl.sum(tl.where(slots == 3, shifted, 0.0))
        s4 = tl.sum(tl.where(slots == 4, shifted, 0.0))
        s5 = tl.sum(tl.where(slots == 5, shifted, 0.0))
        s6 = tl.sum(tl.where(slots == 6, shifted, 0.0))
        s7 = tl.sum(tl.where(slots == 7, shifted, 0.0))
        total = ((s0 + s2) + (s1 + s3)) + ((s4 + s6) + (s5 + s7))
    else:
        total = tl.sum(shifted)
    weights = tl.exp(log_probs - (top + tl.log(total))) * route_scale
    return chosen, weights


@triton.jit
def _route_rows_kernel(
    Logits,
    Bias,
    Ids,
    Weights,
    Padding,
    route_scale,
    stride_part,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    PADDED: tl.constexpr,
    PARTS: tl.constexpr,
):
    """One row of `inkling2_reference.route_experts`. A padding row takes row
    0's routing, so it reads no expert of its own. With PARTS the logits are the
    ordered sum of that many fp32 partials (a chunked router product)."""
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, E_POW2)
    mask = lanes < E
    source = row
    if PADDED:
        source = tl.where(tl.load(Padding + row) != 0, 0, row)
    logits = tl.load(Logits + source * E + lanes, mask, other=0.0)
    for part in tl.static_range(1, PARTS):
        logits += tl.load(Logits + part * stride_part + source * E + lanes, mask, other=0.0)
    bias = tl.load(Bias + lanes, mask, other=0.0)
    chosen, weights = _route(logits, bias, lanes, mask, route_scale, E_POW2, TOPK, TOPK_POW2)
    slots = tl.arange(0, TOPK_POW2)
    tl.store(Ids + row * TOPK + slots, chosen, slots < TOPK)
    tl.store(Weights + row * TOPK + slots, weights, slots < TOPK)


def route_rows(logits: torch.Tensor, bias: torch.Tensor, top_k: int, route_scale: float,
               padding: torch.Tensor | None = None):
    """`inkling2_reference.route_experts` row by row: ids `(T, top_k)` (int64,
    best first, the lower index on a tie) and routing weights `(T, top_k)`
    (fp32) of the fp32 router logits `(T, E)`. A row t with `padding[t]`
    (bool `(T,)`) gets row 0's ids and weights: its result is discarded, and
    it then adds no expert to the ones the call reads. `logits` may be the
    fp32 partials `(P, T, E)` of a chunked product: a logit is their ordered sum."""
    parts = logits.shape[0] if logits.dim() == 3 else 1
    tokens, experts = logits.shape[-2:]
    assert logits.is_contiguous() and bias.is_contiguous() and logits.dtype == bias.dtype == torch.float32
    assert padding is None or (padding.dtype == torch.bool and padding.shape == (tokens,) and padding.is_contiguous())
    ids = torch.empty((tokens, top_k), dtype=torch.int64, device=logits.device)
    weights = torch.empty((tokens, top_k), dtype=torch.float32, device=logits.device)
    if tokens:
        _route_rows_kernel[(tokens,)](
            logits, bias, ids, weights, padding if padding is not None else ids, route_scale,
            logits.stride(0) if parts > 1 else 0,
            E=experts, E_POW2=triton.next_power_of_2(experts), TOPK=top_k,
            TOPK_POW2=triton.next_power_of_2(top_k), PADDED=padding is not None, PARTS=parts, num_warps=1,
            enable_fp_fusion=False,
        )
    return ids, weights


def router_rows(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, top_k: int, route_scale: float,
                padding: torch.Tensor | None = None):
    """The router of bf16 rows `x` `(T, H)`: `route_rows` of the logits
    `x @ gate.T` with the bf16 gate `(E, H)`. A logit is accumulated in fp32
    over the inputs' chunks of 1024 in parallel and the chunks are summed in
    order, as the served router does. Two launches up to 16 rows; a row's
    result does not depend on the other rows of the call."""
    tokens, hidden = x.shape
    assert x.dtype == gate.dtype == torch.bfloat16 and gate.shape[1] == hidden and gate.is_contiguous()
    partials = gemv_streams(x.contiguous()[None], gate[None], chunked=True)  # (P, 1, T, E)
    return route_rows(partials.view(partials.shape[0], tokens, -1), bias, top_k, route_scale, padding)


@triton.jit
def _situ_rows_kernel(
    GateUp,
    Weights,
    Out,
    gate_cap,
    up_cap,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP_GATE: tl.constexpr,
    CAP_UP: tl.constexpr,
):
    """`inkling2_reference.situ` of one (row, expert) pair: the gate and up
    values are the even and odd columns."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(GateUp + row * (2 * N) + 2 * cols, mask, other=0).to(tl.float32)
    up = tl.load(GateUp + row * (2 * N) + 2 * cols + 1, mask, other=0).to(tl.float32)
    capped = gate
    if CAP_GATE:
        capped = gate_cap * tldevice.tanh(gate / gate_cap)
    if CAP_UP:
        up = up_cap * tldevice.tanh(up / up_cap)
    y = ((capped * (1.0 / (1.0 + tl.exp(-gate)))) * up) * tl.load(Weights + row)
    tl.store(Out + row * N + cols, y.to(Out.dtype.element_ty), mask)


def situ_rows(gate_up: torch.Tensor, weights: torch.Tensor, gate_cap, up_cap) -> torch.Tensor:
    """bf16 `(R, N)`: the softcapped SwiGLU of the interleaved (gate, up) columns
    of `gate_up` `(R, 2N)` times the fp32 routing weight `(R,)` of each row."""
    rows, n = gate_up.shape[0], gate_up.shape[1] // 2
    assert gate_up.is_contiguous() and weights.is_contiguous() and weights.shape == (rows,)
    assert weights.dtype == torch.float32
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    if rows:
        block = min(triton.next_power_of_2(n), 512)
        _situ_rows_kernel[(rows, triton.cdiv(n, block))](
            gate_up, weights, out, float(gate_cap or 0.0), float(up_cap or 0.0),
            N=n, BLOCK=block, CAP_GATE=bool(gate_cap and gate_cap > 0), CAP_UP=bool(up_cap and up_cap > 0),
            num_warps=2, enable_fp_fusion=False,
        )
    return out


@triton.jit
def _sum_rows_kernel(
    A,
    B,
    C3,
    Out,
    stride_a,
    N: tl.constexpr,
    TERMS: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    if SLOTS > 0:  # the row's slots of A, summed in order
        acc = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
        for slot in tl.static_range(1, SLOTS):
            acc = acc + tl.load(A + row * stride_a + slot * N + cols, mask, other=0).to(tl.float32)
    else:  # (A + B) + C
        acc = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
        acc = acc + tl.load(B + row * N + cols, mask, other=0).to(tl.float32)
        if TERMS == 3:
            acc = acc + tl.load(C3 + row * N + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * N + cols, acc.to(tl.bfloat16).to(Out.dtype.element_ty), mask)


def sum_slots(x: torch.Tensor) -> torch.Tensor:
    """bf16 `(T, N)`: the slots of `x` `(T, slots, N)` summed in order in fp32, rounded once."""
    tokens, slots, n = x.shape
    assert x.is_contiguous()
    out = torch.empty((tokens, n), dtype=torch.bfloat16, device=x.device)
    if tokens:
        block = min(triton.next_power_of_2(n), 1024)
        _sum_rows_kernel[(tokens, triton.cdiv(n, block))](
            x, x, x, out, slots * n, N=n, TERMS=0, SLOTS=slots, BLOCK=block, num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def sum_rows(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None,
             out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """`(a + b) + c` of `(T, N)` rows in fp32, rounded once to BF16 and held in `out_dtype`."""
    tokens, n = a.shape
    assert a.stride(1) == 1 and b.is_contiguous() and b.shape == a.shape
    assert c is None or (c.is_contiguous() and c.shape == a.shape)
    out = torch.empty((tokens, n), dtype=out_dtype, device=a.device)
    if tokens:
        block = min(triton.next_power_of_2(n), 1024)
        _sum_rows_kernel[(tokens, triton.cdiv(n, block))](
            a, b, c if c is not None else b, out, a.stride(0), N=n, TERMS=3 if c is not None else 2,
            SLOTS=0, BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
    return out


@triton.jit
def _store_key_value_kernel(
    Keys,
    Values,
    TreeK,
    TreeV,
    Planes,
    Slots,
    stride_k,
    stride_v,
    tree_slots,
    threshold,
    clip_eps,
    norm_eps,
    D: tl.constexpr,
):
    """One (row, KV head): the key clipped to L2 norm `threshold` and the value
    RMS-normalized (`inkling2_reference.l2_clip`, `rms`), written to the row's
    slot of its plane."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, D)
    at = ((tl.load(Planes + row) * tree_slots + tl.load(Slots + row)) * tl.num_programs(1) + head) * D + cols
    k = tl.load(Keys + row * stride_k + head * D + cols).to(tl.float32)
    k = k * tl.minimum(tl.div_rn(threshold, tl.sqrt(tl.sum(k * k)) + clip_eps), 1.0)
    tl.store(TreeK + at, k.to(TreeK.dtype.element_ty))
    v = tl.load(Values + row * stride_v + head * D + cols).to(tl.float32)
    v = v * rsqrt(tl.div_rn(tl.sum(v * v), D) + norm_eps)
    tl.store(TreeV + at, v.to(TreeV.dtype.element_ty))


def store_key_value(
    keys: torch.Tensor,
    values: torch.Tensor,
    tree_k: torch.Tensor,
    tree_v: torch.Tensor,
    planes: torch.Tensor,
    slots: torch.Tensor,
    threshold: float,
    clip_eps: float,
    norm_eps: float,
) -> None:
    """`tree_k[planes[t], slots[t]]` = the keys `(T, kv * D)` clipped per head,
    `tree_v[...]` = the values RMS-normalized per head. `tree_k`, `tree_v`
    `(P, S, kv, D)`; a row writes only its own slot."""
    tokens = keys.shape[0]
    kv, d = tree_k.shape[2], tree_k.shape[3]
    assert keys.stride(1) == values.stride(1) == 1 and keys.shape == values.shape == (tokens, kv * d)
    assert tree_k.is_contiguous() and tree_v.is_contiguous() and tree_k.shape == tree_v.shape
    assert planes.is_contiguous() and slots.is_contiguous() and d & (d - 1) == 0
    if tokens:
        _store_key_value_kernel[(tokens, kv)](
            keys, values, tree_k, tree_v, planes, slots, keys.stride(0), values.stride(0),
            tree_k.shape[1], threshold, clip_eps, norm_eps, D=d, num_warps=1, enable_fp_fusion=False,
        )
