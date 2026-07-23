# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import contextvars
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal


import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune

from .compat import *  # noqa: F401,F403
from .recipes import *  # noqa: F401,F403
from .swizzle import *  # noqa: F401,F403
from .tile_layout import *  # noqa: F401,F403
from .quant import *  # noqa: F401,F403
from .scales import *  # noqa: F401,F403



@triton.jit
def mx_dot_scaled(acc, a, a_scale, w, w_scale):
    """MX 'dot_scaled' path: scaled MMA folding the UE8M0 group scales into the tensor core —
    each operand's format is its loaded tile's dtype (``uint8`` = packed E2M1, else E4M3).
    fp4 on BOTH operands lowers to the native ``kind::mxf4`` MMA (2x the fp8 rate; probed
    bit-exact on sm_100, native iff the M operand is 128 — same gate as mxf8f6f4). Caller
    pre-shapes ``w``/``w_scale`` (e.g. ``tl.trans(gu)``)."""
    lhs_format: tl.constexpr = "e2m1" if a.dtype == tl.uint8 else "e4m3"
    rhs_format: tl.constexpr = "e2m1" if w.dtype == tl.uint8 else "e4m3"
    return tl.dot_scaled(a, a_scale, lhs_format, w, w_scale, rhs_format, acc)



@triton.jit
def mx_dot_rescale(acc, a, w, a_scale, w_scale):
    """MX 'dot' path (BK == group): unpack MXFP4 weights to E4M3, fp8 ``tl.dot`` + per-group
    software rescale (decoding both UE8M0 scales internally), accumulating into ``acc`` (returned
    updated). The batched gate_up kernel passes the stacked
    gate|up tile (2*BN columns) — per-column independence keeps that bit-exact."""
    aq = e2m1_cols_to_e4m3(a) if a.dtype == tl.uint8 else a
    wq = e2m1_to_e4m3(w) if w.dtype == tl.uint8 else w
    return acc + tl.dot(aq, wq) * decode_group_scale(a_scale) * tl.trans(
        decode_group_scale(w_scale)
    )



@triton.jit
def mx_scalar_reduce(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    BLOCK_SIZE_M: tl.constexpr,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """MX 'scalar' path: CUDA-core FMA GEMV, unpacking MXFP4 weights to E4M3 then dequantizing
    activation + weight per-element by their expanded group scales, reducing and accumulating into
    ``acc`` (returned updated). No tensor core (so no M→16 MMA pad) — wins for the memory-bound
    decode GEMV (M=1). The batched gate_up kernel passes the stacked gate|up tile (ROWS_W = 2*BN).

    The UE8M0 scale is constant within each group of ``SCALE_GROUP_K``, so it factors out of the
    inner sum: instead of expanding it to every K element and doing ``BLOCK_SIZE_K`` scale-muls,
    reduce the raw products within each group, then apply ONE combined (act × weight) scale per
    group — ``SCALE_GROUP_K``× fewer scale-muls. Measured ~18% faster on the decode reduce
    (the per-element expand was pure overhead), bit-identical to the expanded form (rel 1e-7)."""
    aq = (
        e2m1_cols_to_e4m3(a).to(tl.float32) if a.dtype == tl.uint8 else a.to(tl.float32)
    )
    wq = e2m1_to_e4m3(w) if w.dtype == tl.uint8 else w
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    prod = tl.trans(aq) * wq.to(tl.float32)  # [BK, ROWS_W]
    grp = tl.sum(
        tl.reshape(prod, (NG, SCALE_GROUP_K, ROWS_W)), axis=1
    )  # per-group partial
    scale = tl.trans(decode_group_scale(a_scale)) * tl.trans(
        decode_group_scale(w_scale)
    )
    return acc + tl.sum(grp * scale, axis=0)[None, :]



@triton.jit
def mx_compute(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Single-projection MMA step. Under ``SWAP_AB`` the swapped decode path runs (weight output rows
    in the MMA M dim — different acc shape/finalize; see ``mx_swap_compute``); otherwise dispatch on
    ``COMPUTE_MODE``: scaled-MMA on the raw weight (``w``), or fp8 ``tl.dot`` + per-group rescale /
    scalar reduce on the E4M3-decoded weight. Single return — only the taken branch compiles.
    A ``uint8`` ``a`` tile is packed-E2M1 activations (W4A4, the dtype is the format):
    dot_scaled consumes it natively; the dot/scalar/swap arms column-unpack it to E4M3
    (lossless) first."""
    if SWAP_AB:
        acc = mx_swap_compute(
            acc,
            a,
            a_scale,
            w,
            w_scale,
            COMPUTE_MODE,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
        )
    elif COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled(acc, a, a_scale, w, w_scale)
    elif COMPUTE_MODE == "dot":
        acc = mx_dot_rescale(acc, a, w, a_scale, w_scale)
    else:  # scalar
        acc = mx_scalar_reduce(
            acc,
            a,
            a_scale,
            w,
            w_scale,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
        )
    return acc



# ── swap-AB decode compute: M=1 batched GEMV with output rows in the MMA M dim ───────
#
# The batched (decode) kernels are structurally M=1, where the sm_100 scaled MMA pads M→128.
# Putting the WEIGHT's output rows in M (fully used) and the single token in N cuts that pad to
# the N-atom (16) — a ~1.5× decode win for fp4 (dot_scaled), neutral-to-worse for fp8 (scalar
# stays memory-bound and wins, so the tuner keeps it). Weight is loaded output-rows-major
# ``[BN, BK]`` for every mode, so the kernel does ONE load; each helper returns ``[1, BN]``.


# The sm_100 MMA's minimum N tile (16). In the swap path the single decode token sits in the MMA's
# N dim, so it must be padded up to this width (col 0 = the token, cols 1..15 = zero). It is NOT a
# block size — BLOCK_SIZE_M stays 1 under swap; this is the token's *padded N extent*, fixed by the
# hardware. Assigned via tl.constexpr(...), the only module-global form a @triton.jit fn can read.
MMA_N_ATOM = tl.constexpr(16)



@triton.jit
def mx_dot_scaled_swapped(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Swapped ``dot_scaled`` decode step: weight ``w`` [BN, BK] (E2M1 packed if fp4 else E4M3)
    is the MMA lhs (output rows in M); the activation is the N=16 rhs (col 0 real). ``acc`` is the
    persistent ``[BN, MMA_N_ATOM]`` MMA accumulator (accumulated across the K-loop, then the caller
    takes column 0) — NOT a fresh per-step init, which trips the sm_100 accumulator-init pass.
    Each side's format is its dtype (a packed ``a`` stays packed — the E4M3-scaled mxf4nvf4
    kind is fp4 x fp4 only); the token's group scale broadcasts to the rhs columns."""
    fmt: tl.constexpr = "e2m1" if w.dtype == tl.uint8 else "e4m3"
    rhs_fmt: tl.constexpr = "e2m1" if a.dtype == tl.uint8 else "e4m3"
    # the token becomes a [bytes, MMA_N_ATOM] rhs with only column 0 real (16 is
    # Triton's tcgen05-selection gate, not the hardware floor: N=8 drops to the
    # bf16-upcast fallback, bare-1 was 1.83x)
    rhs = swap_pad_rhs(a, BLOCK_SIZE_K // 2 if a.dtype == tl.uint8 else BLOCK_SIZE_K)
    if a_scale.dtype == tl.uint8:  # UE8M0 broadcast via the zero-add idiom
        asc = tl.trans(
            a_scale[:, None]
            + tl.zeros((BLOCK_SIZE_K // SCALE_GROUP_K, MMA_N_ATOM), tl.uint8)
        )
    else:  # E4M3 (NVFP4) — no fp8 arithmetic; materialize the broadcast directly
        asc = tl.trans(
            tl.broadcast_to(
                a_scale[:, None], (BLOCK_SIZE_K // SCALE_GROUP_K, MMA_N_ATOM)
            )
        )
    return tl.dot_scaled(w, w_scale, fmt, rhs, asc, rhs_fmt, acc)



@triton.jit
def mx_dot_rescale_swapped(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Swapped MX 'dot' step (BK == one scale group): weight output rows in the MMA M dim
    (``[ROWS, BK]`` after the column-unpack for MXFP4), the [BK] token padded to the N=16
    atom — the well-shaped fp8 MMA at M=1 (M quantizes to 64/128, N only to 8, so weight
    rows fill the big atom). Both UE8M0 scales factor out of the single-group step: the
    weight's per-output-row scale broadcasts down the acc columns, the token's group scale
    is a scalar. ``acc`` is the persistent ``[ROWS, MMA_N_ATOM]`` accumulator (col 0 taken
    by the caller's ``acc_finalize``)."""
    if w.dtype == tl.uint8:  # column-unpack E2M1 -> E4M3 (K order: low nibble first)
        wq = e2m1_cols_to_e4m3(w)
    else:
        wq = w
    aq = e2m1_cols_to_e4m3(a) if a.dtype == tl.uint8 else a
    rhs = swap_pad_rhs(aq, BLOCK_SIZE_K)
    a_s = decode_group_scale(a_scale)  # [1] — the single group's token scale
    w_s = decode_group_scale(w_scale)  # [ROWS, 1] — per output row
    return acc + tl.dot(wq, rhs) * w_s * a_s



@triton.jit
def mx_scalar_reduce_swapped(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Swapped scalar reduce: weight ``w`` output-rows-major ``[ROWS_W, BK]``, ``a`` the [BK]
    activation. No transpose (vs ``mx_scalar_reduce``); MXFP4 unpacks along columns (K). Per-group
    scale factored out of the reduce (grpscale). Reduces over K; returns ``acc + [1, ROWS_W]``."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    if w.dtype == tl.uint8:  # column-unpack E2M1 -> f32, K-order via interleave
        wq = e2m1_cols_to_e4m3(w).to(tl.float32)
    else:
        wq = w.to(tl.float32)
    aq = (
        e2m1_cols_to_e4m3(a).to(tl.float32) if a.dtype == tl.uint8 else a.to(tl.float32)
    )
    prod = aq[None, :] * wq  # [ROWS_W, BK]
    grp = tl.sum(tl.reshape(prod, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)  # [ROWS_W, NG]
    scale = decode_group_scale(a_scale)[None, :] * decode_group_scale(w_scale)
    return acc + tl.reshape(tl.sum(grp * scale, axis=1), (1, ROWS_W))



@triton.jit
def mx_swap_compute(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Swapped-AB counterpart to ``mx_compute``: weight output-rows in the MMA M dim, the single
    decode token flattened to the [BK] rhs. Dispatches the three swapped modes — ``dot_scaled``
    and ``dot`` (persistent ``[BLOCK_SIZE_N, MMA_N_ATOM]`` MMA acc, col 0 taken by the caller)
    and ``scalar`` (``[1, BLOCK_SIZE_N]`` reduce). The acc shapes diverge, but only the taken
    constexpr branch compiles so the single return never has to unify them. ``BLOCK_SIZE_N`` is the weight tile's row count — the gate_up kernel passes ``2*BN``
    with its STACKED gate|up tile (gate rows first, split back via ``split_gate_up``): one
    load and one MMA for both projections keeps the native microscaled-MMA M=128 operand at BN=64, doubling
    the CTAs on the parallelism-starved decode grid (dsv4 gate_up 1.34x, bit-exact)."""
    # packed-E2M1 activations flatten to their BYTE length; the dot/scalar leaves
    # column-unpack (lossless), dot_scaled consumes the packed rhs natively (the E4M3-scaled
    # mxf4nvf4 kind exists only for fp4 x fp4 — unpacking would forfeit it)
    if a.dtype == tl.uint8:
        a1 = tl.reshape(a, (BLOCK_SIZE_K // 2,))
    else:
        a1 = tl.reshape(a, (BLOCK_SIZE_K,))
    as1 = tl.reshape(a_scale, (BLOCK_SIZE_K // SCALE_GROUP_K,))
    if COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled_swapped(
            acc, a1, as1, w, w_scale, BLOCK_SIZE_K, SCALE_GROUP_K
        )
    elif COMPUTE_MODE == "dot":
        acc = mx_dot_rescale_swapped(acc, a1, as1, w, w_scale, BLOCK_SIZE_K)
    elif COMPUTE_MODE == "scalar":
        acc = mx_scalar_reduce_swapped(
            acc,
            a1,
            as1,
            w,
            w_scale,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
        )
    else:
        tl.static_assert(False, "unknown COMPUTE_MODE under SWAP_AB")
    return acc



@triton.jit
def swap_pad_rhs(a, BLOCK_SIZE_K: tl.constexpr):
    """Pad the ``[BLOCK_SIZE_K]`` M=1 token to the ``[BLOCK_SIZE_K, MMA_N_ATOM]`` swap-AB MMA rhs —
    only column 0 is the real token (16 is measured, not a hardware floor — see acc_init). Used
    by the M=1 batched / fused-MoE fp8 ``tl.dot`` swap paths (weight output rows in the MMA M dim);
    the caller takes column 0 of the ``[BN, MMA_N_ATOM]`` result after the K-loop."""
    return tl.where(
        tl.arange(0, MMA_N_ATOM)[None, :] == 0,
        a[:, None],
        tl.zeros((BLOCK_SIZE_K, MMA_N_ATOM), a.dtype),
    )



@triton.jit
def fp8_dot(a, b, SWAP_AB: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """Swap-aware plain ``tl.dot`` for the fp8 (block/tensor-dynamic) paths — no scaling; the caller
    applies its own per-block/per-tensor scales. Swap: weight ``b`` ``[N, BK]`` × the single token
    padded to the N=16 atom → ``[N, MMA_N_ATOM]`` (col 0 real). No-swap: token ``a`` ``[M, BK]`` ×
    weight ``b`` ``[BK, N]`` → ``[M, N]``. ``BLOCK_SIZE_K`` is the contraction tile (the down
    projection passes its intermediate tile). Single return: only the taken branch compiles."""
    if SWAP_AB:
        out = tl.dot(b, swap_pad_rhs(tl.reshape(a, (BLOCK_SIZE_K,)), BLOCK_SIZE_K))
    else:
        out = tl.dot(a, b)
    return out



@triton.jit
def block_dynamic_dot(
    acc, a, a_s, b, b_s,
    block_k: tl.constexpr, SWAP_AB: tl.constexpr, USE_DOT_SCALED: tl.constexpr,
    FAKE_BATCH: tl.constexpr = False,
):
    """Accumulate one block-dynamic (1x128/128x128) fp8 K-tile into ``acc``, oriented by
    ``SWAP_AB``. ``USE_DOT_SCALED`` (UE8M0 scales on a native-M tile): fold the group scales
    into the tcgen05 MMA — the tile's single 128-group scale broadcasts in-register to the
    ``block_k // 32`` group-32 columns ``dot_scaled`` consumes, identical to a 128-group
    rescale but with no 4x scale memory and no software multiply. Else: plain fp8 ``tl.dot``
    + per-group software rescale (``decode_group_scale`` is a no-op on fp32 scales, decodes
    UE8M0). ``FAKE_BATCH`` (single-token decode): ``fp8_dot`` pads the lone token to the MMA N
    atom and both scales broadcast down the weight-row (M) dim. Single-exit if/else so only the
    taken arm type-checks (a trailing fall-through arm would be checked even when an earlier
    branch is taken)."""
    if USE_DOT_SCALED:
        reps: tl.constexpr = block_k // 32
        a_sg = a_s[:, None].broadcast_to(a_s.shape[0], reps)
        b_sg = b_s[:, None].broadcast_to(b_s.shape[0], reps)
        if SWAP_AB:
            acc = tl.dot_scaled(b, b_sg, "e4m3", a, a_sg, "e4m3", acc)
        else:
            acc = tl.dot_scaled(a, a_sg, "e4m3", b, b_sg, "e4m3", acc)
    else:
        # plain fp8 tl.dot + per-group decoded scales, oriented by SWAP_AB (weight rows in the MMA
        # M dim under swap). decode_group_scale: fp32 passthrough, UE8M0 -> 2^(e-127).
        a_sd = decode_group_scale(a_s)
        b_sd = decode_group_scale(b_s)
        if FAKE_BATCH:
            acc = acc + fp8_dot(a, b, SWAP_AB, block_k) * a_sd[:, None] * b_sd[:, None]
        elif SWAP_AB:
            acc = acc + tl.dot(b, a) * b_sd[:, None] * a_sd[None, :]
        else:
            acc = acc + tl.dot(a, b) * a_sd[:, None] * b_sd[None, :]
    return acc



@triton.jit
def accumulate(
    acc,
    a,
    a_s,
    b,
    b_s,
    RECIPE: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr,
    USE_DOT_SCALED: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr = 32,
    FAKE_BATCH: tl.constexpr = False,
):
    """Unified K-step accumulate — the single "do math" of every matmul/grouped/batched kernel,
    dispatched by ``RECIPE`` so the kernel loops are identical:

    - ``"mx"``: microscaled MMA / dot+rescale / scalar (``mx_compute``), swap-aware.
    - ``"block_dynamic"``: UE8M0 ``dot_scaled`` broadcast or fp8 ``tl.dot`` + software rescale
      (``block_dynamic_dot``).
    - ``"static"``: plain (swap-aware) dot + per-K-block weight rescale (the per-tensor act scale
      is applied post-loop).
    - ``"tensor"`` / ``"full_precision"``: plain (swap-aware) dot; per-row/per-tensor scale (if any)
      is applied post-loop in the epilogue.

    ``FAKE_BATCH`` (single-token decode) routes the block_dynamic/static rescale down the weight-row
    (M) dim — the per-weight-row block scale sits there under the swap — and pads the lone token via
    ``fp8_dot``; the prefill tiles broadcast the weight scale across the N columns instead.

    Single return (if/elif/else) — only the taken recipe arm compiles, so the dead arms are
    never type-checked (e.g. the ``fp8_dot`` arms would reject packed-E2M1 activations, whose
    reduction dim is halved vs an unpacked weight). ``a_s``/``b_s`` are dead on the recipes that
    scale post-loop."""
    if RECIPE == "mx":
        acc = mx_compute(
            acc, a, a_s, b, b_s,
            COMPUTE_MODE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, SWAP_AB,
        )
    elif RECIPE == "block_dynamic":
        acc = block_dynamic_dot(acc, a, a_s, b, b_s, BLOCK_SIZE_K, SWAP_AB, USE_DOT_SCALED, FAKE_BATCH)
    elif RECIPE == "static":
        b_sd = decode_group_scale(b_s)
        if FAKE_BATCH:
            acc = acc + fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K) * b_sd[:, None]
        else:
            acc = acc + fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K) * b_sd[None, :]
    else:  # tensor / full_precision
        acc = acc + fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K)
    return acc
