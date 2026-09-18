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

import triton
import triton.language as tl

from .quant import e2m1_cols_to_e4m3, e2m1_cols_to_f32, e2m1_to_bf16, e2m1_to_e4m3, e2m1_to_f32
from .loading.scales import decode_group_scale


@triton.jit
def mx_dot_scaled(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    SWAP_AB: tl.constexpr = False,
    BLOCK_SIZE_K: tl.constexpr = 0,
    SCALE_GROUP_K: tl.constexpr = 0,
):
    """MX 'dot_scaled' step: the tcgen05 scaled MMA folds the group scales into the tensor core;
    each operand's format is its tile's dtype (``uint8`` = packed E2M1, E4M3, or bf16 for the
    weight-only unscaled lhs). fp4 on BOTH operands lowers to the native ``kind::mxf4`` MMA (2x the
    fp8 rate; native iff the M operand is 128 — same gate as mxf8f6f4). Plain: ``a`` [BM, BK] x
    ``w`` [BK, BN] (the caller pre-shapes ``w``/``w_scale``, e.g. ``tl.trans(gu)``). ``SWAP_AB``
    (decode): the weight ``w`` [BN, BK] is the lhs (output rows in M) and the single [BK] token is
    padded to the N=16 rhs (col 0 real — 16 is Triton's tcgen05-selection gate, N=8 drops to the
    bf16-upcast fallback, bare-1 was 1.83x); ``acc`` is the persistent ``[BN, MMA_N_ATOM]``
    accumulator the caller takes column 0 of (never a fresh per-step init, which trips the sm_100
    accumulator-init pass). A packed ``a`` stays packed (the E4M3-scaled mxf4nvf4 kind is fp4 x fp4
    only) and the token's group scale broadcasts to the rhs columns."""
    a_fmt: tl.constexpr = (
        "e2m1" if a.dtype == tl.uint8
        else ("e4m3" if a.dtype == tl.float8e4nv else "bf16")
    )
    w_fmt: tl.constexpr = "e2m1" if w.dtype == tl.uint8 else "e4m3"
    if SWAP_AB:
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
        acc = tl.dot_scaled(w, w_scale, w_fmt, rhs, asc, a_fmt, acc)
    else:
        acc = tl.dot_scaled(a, a_scale, a_fmt, w, w_scale, w_fmt, acc)
    return acc


@triton.jit
def mx_dot_rescale(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    SWAP_AB: tl.constexpr = False,
    BLOCK_SIZE_K: tl.constexpr = 0,
):
    """MX 'dot' step (BK == one scale group): fp8 ``tl.dot`` on E4M3-decoded operands + per-group
    software rescale (both UE8M0 scales decoded here), accumulating into ``acc``. Plain: ``a``
    [BM, BK] x ``w`` [BK, BN] with the weight's per-column scale transposed onto the product (the
    batched gate_up kernel passes the INTERLEAVED gate|up 2*BN tile — per-column independence keeps
    that bit-exact). ``SWAP_AB`` (decode): the weight ``w`` [ROWS, BK] is the lhs (E2M1 column-unpacked,
    K order low nibble first) and the [BK] token padded to the N=16 atom is the rhs — the
    well-shaped fp8 MMA at M=1 (M quantizes to 64/128, N only to 8, so weight rows fill the big
    atom); the weight's per-output-row scale broadcasts down the acc columns and the token's single
    group scale is a scalar; ``acc`` is the persistent ``[ROWS, MMA_N_ATOM]`` accumulator."""
    aq = e2m1_cols_to_e4m3(a) if a.dtype == tl.uint8 else a
    a_s = decode_group_scale(a_scale)
    w_s = decode_group_scale(w_scale)
    if SWAP_AB:
        wq = e2m1_cols_to_e4m3(w) if w.dtype == tl.uint8 else w
        acc = acc + tl.dot(wq, swap_pad_rhs(aq, BLOCK_SIZE_K)) * w_s * a_s
    else:
        wq = e2m1_to_e4m3(w) if w.dtype == tl.uint8 else w
        acc = acc + tl.dot(aq, wq) * a_s * tl.trans(w_s)
    return acc


@triton.jit
def mx_weight_upcast(w, w_scale, BLOCK_SIZE_K: tl.constexpr, N: tl.constexpr, SCALE_GROUP_K: tl.constexpr,
                     OUT_DTYPE: tl.constexpr = tl.bfloat16):
    """Upcast one MXFP4/MXFP8 weight K-tile ``[BK, N]`` to ``OUT_DTYPE`` for the weight-only path: unpack
    E2M1 -> bf16 directly (fp8 passes through a cast) and apply each ``SCALE_GROUP_K``-row K-group's
    UE8M0 group scale. ``w_scale`` is ``[N, BK // SCALE_GROUP_K]`` (per-N-row, per-group); transposed
    and broadcast across each group's rows to ``[BK, N]``. Unlike the ``dot`` arm (BK == group, scale
    folded onto the [M,N] product) this dequantizes IN the tile, so BK spans any number of groups — a
    full-BK bf16 ``tl.dot`` (the matmul_ogs format: fp4 weight, bf16 acts).

    The scale is applied as a plain bf16 multiply, NOT a hand-rolled exponent-add: the UE8M0 scale is
    a power of two so a ``bf16 * 2^k`` is already an exact exponent shift (a cheap FMA), and folding it
    into the bits as an int add + zero-mask ``tl.where`` measured ~13% SLOWER (gate_up 4.68->5.29ms) —
    the mask + uint16<->int32 bitcast chain costs more than the multiply it replaces."""
    ng: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    wq = e2m1_to_bf16(w) if w.dtype == tl.uint8 else w.to(tl.bfloat16)  # [BK, N] bf16 (direct, no E4M3)
    # Reshape the (real, contiguous) weight to [ng, g, N] and broadcast the group scale [ng, 1, N]
    # across each group's g rows in the multiply, then reshape back to [BK, N]. (Reshaping a
    # broadcasted stride-0 tensor doesn't lower, so broadcast happens IN the op, not before it.)
    ws = tl.trans(decode_group_scale(w_scale)).to(tl.bfloat16)[:, None, :]  # [ng, 1, N]
    w3 = wq.reshape(ng, SCALE_GROUP_K, N) * ws
    # dequant + scale happen in bf16 (exact: E2M1/E4M3 codes and power-of-two UE8M0 scales all fit),
    # then widen if the activation is wider — bf16->fp32 is lossless and keeps the multiply cheap.
    return w3.reshape(BLOCK_SIZE_K, N).to(OUT_DTYPE)


@triton.jit
def mx_weight_only_compute(
    acc,
    a,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_WIDTH: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Weight-only (W4A16/W8A16) MMA step: raw bf16/fp16 activations against an MXFP4/MXFP8 weight
    tile, dispatched on ``COMPUTE_MODE``. ``"dot_scaled"`` hands the packed weight and its UE8M0
    group scales straight to the tcgen05 scaled MMA with an UNSCALED activation operand (the
    ``lhs_scale=None`` / bf16-format form), so the tensor core does the fp4 decode and the group
    rescale; ``"dot"`` upcasts the tile to the activation dtype in-loop and runs a plain
    ``tl.dot`` (the matmul_ogs Hopper format). Both are correct everywhere and the tuner picks per
    workload — the two differ by arm: measured on gpt-oss K=2880, ``dot_scaled`` wins the down
    projection by 18.5% while the stacked gate|up tile prefers ``dot`` by 7.3``%``. ``SWAP_AB``
    is the decode form: weight output rows lead the tile and the FMA reduce replaces the MMA
    (both dot arms keep the token in M). Single return — only the taken branch compiles."""
    if SWAP_AB:
        acc = mx_weight_only_scalar_swapped(
            acc, tl.reshape(a, (BLOCK_SIZE_K,)), w, w_scale,
            N_WIDTH, BLOCK_SIZE_K, SCALE_GROUP_K,
        )
    elif COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled(acc, a, None, w, w_scale)
    else:
        acc = acc + tl.dot(
            a, mx_weight_upcast(w, w_scale, BLOCK_SIZE_K, N_WIDTH, SCALE_GROUP_K, a.dtype)
        )
    return acc


@triton.jit
def mx_weight_only_scalar_swapped(
    acc,
    a,
    w,
    w_scale,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Swapped weight-only reduce: CUDA-core FMA GEVM against RAW bf16 activations (no act
    scale — the weight-only counterpart of ``mx_scalar_reduce_swapped``). Weight
    ``w`` is output-rows-major ``[ROWS_W, BK]``, ``a`` the ``[BK]`` token; the UE8M0 group
    scale factors out of the inner sum, so it costs one multiply per group instead of per
    element. Returns ``acc + [1, ROWS_W]``.

    This is the decode structure for W4A16: with no MMA there is no M->16 pad to waste, and
    the loop runs at the dequantized weight stream's own rate (gpt-oss gate_up 32.0 -> 13.7us,
    down 19.8 -> 9.1us, against a 12.3us/6us dequant-inclusive load floor)."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    wq = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)  # [ROWS_W, BK]
    prod = wq * a.to(tl.float32)[None, :]
    grp = tl.sum(tl.reshape(prod, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)  # [ROWS_W, NG]
    return acc + tl.reshape(
        tl.sum(grp * decode_group_scale(w_scale), axis=1), (1, ROWS_W)
    )


@triton.jit
def mx_scalar_reduce(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """MX 'scalar' step: CUDA-core FMA GEMV, unpacking MXFP4 to fp32 and dequantizing activation +
    weight by their group scales, reducing over K into ``acc``. No tensor core (so no M->16 MMA
    pad) — wins the memory-bound decode GEMV (M=1). The gate_up kernels pass the INTERLEAVED
    gate|up tile (gate even, up odd; ROWS_W = 2*BN). The UE8M0 scale is constant within each
    ``SCALE_GROUP_K`` group, so it
    factors out of the inner sum: reduce the raw products per group, then apply ONE combined
    (act x weight) scale per group — ``SCALE_GROUP_K``x fewer scale-muls (~18% faster, bit-identical
    to the expanded form). Plain: ``a`` [BM, BK] against ``w`` [BK, ROWS_W]. ``SWAP_AB`` (decode):
    ``w`` output-rows-major [ROWS_W, BK] against the [BK] token, MXFP4 unpacked along columns (K),
    no transposes. Both return ``acc + [1, ROWS_W]``."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    aq = e2m1_cols_to_f32(a) if a.dtype == tl.uint8 else a.to(tl.float32)
    if SWAP_AB:
        wq = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)  # [ROWS_W, BK]
        grp = tl.sum(tl.reshape(aq[None, :] * wq, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)
        scale = decode_group_scale(a_scale)[None, :] * decode_group_scale(w_scale)  # [ROWS_W, NG]
        acc = acc + tl.reshape(tl.sum(grp * scale, axis=1), (1, ROWS_W))
    else:
        wq = e2m1_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)  # [BK, ROWS_W]
        grp = tl.sum(tl.reshape(tl.trans(aq) * wq, (NG, SCALE_GROUP_K, ROWS_W)), axis=1)
        scale = tl.trans(decode_group_scale(a_scale)) * tl.trans(decode_group_scale(w_scale))
        acc = acc + tl.sum(grp * scale, axis=0)[None, :]
    return acc


@triton.jit
def mx_compute(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Single-projection MMA step, dispatched on ``COMPUTE_MODE`` (scaled-MMA on the raw weight,
    fp8 ``tl.dot`` + per-group rescale, or the scalar reduce); ``SWAP_AB`` selects each leaf's
    decode form — weight output rows in the MMA M dim, the single token flattened to the [BK] rhs
    (a packed-E2M1 token flattens to its BYTE length; dot_scaled consumes it packed, dot/scalar
    column-unpack it losslessly). The acc shapes diverge across modes and orientations, but only
    the taken constexpr branch compiles, so the single return never has to unify them.
    ``BLOCK_SIZE_N`` is the weight tile's row count — the gate_up kernels pass ``2*BN`` with the
    INTERLEAVED gate|up tile (gate on even rows, up on odd; split back via ``split_gate_up``): one load and one MMA for both
    projections keeps the native microscaled-MMA M=128 operand at BN=64, doubling the CTAs on the
    parallelism-starved decode grid (dsv4 gate_up 1.34x, bit-exact)."""
    if SWAP_AB:
        if a.dtype == tl.uint8:
            a1 = tl.reshape(a, (BLOCK_SIZE_K // 2,))
        else:
            a1 = tl.reshape(a, (BLOCK_SIZE_K,))
        as1 = tl.reshape(a_scale, (BLOCK_SIZE_K // SCALE_GROUP_K,))
    else:
        a1 = a
        as1 = a_scale
    if COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled(acc, a1, as1, w, w_scale, SWAP_AB, BLOCK_SIZE_K, SCALE_GROUP_K)
    elif COMPUTE_MODE == "dot":
        acc = mx_dot_rescale(acc, a1, as1, w, w_scale, SWAP_AB, BLOCK_SIZE_K)
    else:  # scalar
        acc = mx_scalar_reduce(acc, a1, as1, w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, SWAP_AB)
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
# hardware. The plain int is the source of truth: host-side consumers (the autotune pruners, which
# filter configs in ordinary Python) compare against it directly, while kernels read the
# tl.constexpr view derived from it — the only module-global form a @triton.jit fn can read.
MMA_N_ATOM_WIDTH = 16
MMA_N_ATOM = tl.constexpr(MMA_N_ATOM_WIDTH)


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
            d = fp8_dot(a, b, SWAP_AB, block_k)
            if SWAP_AB:  # [BN, N-atom]: weight-row scale down the M=BN dim, act scalar broadcasts
                acc = acc + d * a_sd[:, None] * b_sd[:, None]
            else:  # no-swap [BM, BN] GEVM: act scale down M, weight-N scale across N (upstream form)
                acc = acc + d * a_sd[:, None] * b_sd[None, :]
        elif SWAP_AB:
            acc = acc + tl.dot(b, a) * b_sd[:, None] * a_sd[None, :]
        else:
            acc = acc + tl.dot(a, b) * a_sd[:, None] * b_sd[None, :]
    return acc


@triton.jit
def static_dot(acc, a, b, b_s, SWAP_AB: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, FAKE_BATCH: tl.constexpr):
    """static format K-step: plain (swap-aware) fp8 dot + per-K-block weight rescale. FAKE_BATCH
    (single-token decode) routes the rescale down the weight-row (M) dim; else it broadcasts across
    the N columns. The per-tensor activation scale is applied post-loop."""
    b_sd = decode_group_scale(b_s)
    d = fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K)
    if FAKE_BATCH:
        acc = acc + d * b_sd[:, None]
    else:
        acc = acc + d * b_sd[None, :]
    return acc


