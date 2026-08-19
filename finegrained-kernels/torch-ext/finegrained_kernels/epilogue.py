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


import torch
import triton
import triton.language as tl


from .compat import *  # noqa: F401,F403
from .recipes import *  # noqa: F401,F403
from .swizzle import *  # noqa: F401,F403
from .tile_layout import *  # noqa: F401,F403
from .quant import *  # noqa: F401,F403
from .scales import *  # noqa: F401,F403
from .mma import *  # noqa: F401,F403
from .scheduling import *  # noqa: F401,F403
from .tiles import *  # noqa: F401,F403



@triton.jit
def store_masked(
    C,
    accumulator,
    pid_m,
    pid_n,
    M,
    N,
    stride_c_m,
    stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Shared output epilogue of the kernels below: cast the fp32 accumulator to
    ``C``'s dtype and store the ``(BLOCK_SIZE_M, BLOCK_SIZE_N)`` tile at the swizzled
    ``(pid_m, pid_n)``, masked to the ``(M, N)`` bounds. (A descriptor-store arm was
    measured an EXACT tie at the store-heavy gap shape and dropped, 2026-07-16 —
    stores are fire-and-forget, TMA has nothing to hide there. B200-only verdict;
    re-measure on H100 or the target device.)"""
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_c_m * offs_cm[:, None] + stride_c_n * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



@triton.jit
def store_masked_oriented(
    C,
    accumulator,
    pid_m,
    pid_n,
    M,
    N,
    stride_c_m,
    stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """``store_masked`` with the (m, n) roles swapped under ``SWAP_AB``, where the
    accumulator is ``(BLOCK_SIZE_N, BLOCK_SIZE_M)``. Only the taken branch compiles."""
    if SWAP_AB:
        store_masked(
            C,
            accumulator,
            pid_n,
            pid_m,
            N,
            M,
            stride_c_n,
            stride_c_m,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
        )
    else:
        store_masked(
            C,
            accumulator,
            pid_m,
            pid_n,
            M,
            N,
            stride_c_m,
            stride_c_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )



@triton.jit
def acc_init(
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """Zero accumulator shaped for the layout: swapped scalar reduces into ``[1, N]``; any other
    swapped mode keeps the ``[N, M]`` MMA acc (weight rows in M, act tile in N — padded up to
    the ``MMA_N_ATOM`` when the caller is the single-token decode GEVM, col 0 taken after the
    K-loop); no-swap uses ``[M, N]``. ``COMPUTE_MODE`` matters only under swap — kernels with no
    mode axis (fp8 ``tl.dot``) pass ``"dot"``. ``N`` is the weight-output tile (``BLOCK_SIZE_H``
    for the fp8 down projection). Single return: only the taken branch compiles."""
    if SWAP_AB and COMPUTE_MODE == "scalar":
        acc = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)
    elif SWAP_AB and BLOCK_SIZE_M < MMA_N_ATOM:
        acc = tl.zeros((BLOCK_SIZE_N, MMA_N_ATOM), dtype=tl.float32)
    elif SWAP_AB:
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    return acc



@triton.jit
def acc_finalize(
    acc, COMPUTE_MODE: tl.constexpr, ROWS: tl.constexpr, SWAP_AB: tl.constexpr
):
    """Bookend to ``acc_init``: when the acc was built as the persistent ``[ROWS, MMA_N_ATOM]`` MMA
    tile (any swapped non-scalar mode), collapse the padded token dim to column 0 → ``[1, ROWS]``.
    Swapped scalar (already ``[1, ROWS]``) and no-swap pass through unchanged. ``COMPUTE_MODE``
    matches ``acc_init``'s (fp8 ``tl.dot`` kernels, which have no mode axis, pass ``"dot"``)."""
    if SWAP_AB and COMPUTE_MODE != "scalar":
        # take column 0: the padded token dim collapses back to the single real token
        acc = tl.reshape(
            tl.sum(acc * (tl.arange(0, MMA_N_ATOM)[None, :] == 0), axis=1), (1, ROWS)
        )
    return acc



@triton.jit
def split_gate_up(
    acc,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """Bookend to the gate|up accumulator: finalize it (swap MMA col-0 collapse or pass-through,
    via ``acc_finalize``) and de-interleave the doubled N extent into the ``(gate, up)`` pair, each
    ``[rows, BN]`` (rows = 1 under swap, else BM). The accumulator's N axis alternates gate/up per
    column, so this is a trailing-axis split."""
    rows: tl.constexpr = 1 if SWAP_AB else BLOCK_SIZE_M
    flat = acc_finalize(acc, COMPUTE_MODE, 2 * BLOCK_SIZE_N, SWAP_AB)
    g, u = tl.split(tl.reshape(flat, (rows, BLOCK_SIZE_N, 2)))
    return g, u



@triton.jit
def glu(
    gate,
    up,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.float32,
):
    """Gated linear unit on the gate/up matmul accumulators. ``SWIGLU_LIMIT`` clamps gate above and up
    to ``[-LIMIT, LIMIT]``; ``SWIGLU_ALPHA`` gives the clamped/scaled SwiGLU ``(up + 1) * gate * sigmoid(ALPHA *
    gate)`` (GPT-OSS / MiniMax), else ``ACT_FN(gate) * up`` (``ACT_FN`` in {silu, gelu, relu}, gelu exact
    via erf). ``SIMULATE_UNFUSED`` rounds each materialized value through ``INTERMEDIATE_DTYPE`` (the dtype the unfused path lands intermediates in) to match the
    unfused (separate-kernel) path, where every intermediate lands in that dtype."""
    g = gate
    u = up

    if SIMULATE_UNFUSED:
        g = g.to(INTERMEDIATE_DTYPE).to(tl.float32)
        u = u.to(INTERMEDIATE_DTYPE).to(tl.float32)

    if SWIGLU_LIMIT is not None:
        g = tl.minimum(g, SWIGLU_LIMIT)
        u = tl.minimum(tl.maximum(u, -SWIGLU_LIMIT), SWIGLU_LIMIT)

    if SWIGLU_ALPHA is not None:
        gate_scaled = g * SWIGLU_ALPHA
        if SIMULATE_UNFUSED:
            gate_scaled = gate_scaled.to(INTERMEDIATE_DTYPE).to(tl.float32)
        sig = tl.sigmoid(gate_scaled)
        if SIMULATE_UNFUSED:
            sig = sig.to(INTERMEDIATE_DTYPE).to(tl.float32)
        act = g * sig
        u = u + 1.0
    elif ACT_FN == "silu":
        sig = tl.sigmoid(g)
        # SIMULATE_UNFUSED must be bit-exact vs the unfused ``apply_glu`` (``g * torch.sigmoid(g)``),
        # where torch.sigmoid returns bf16 — i.e. the sigmoid is rounded before the multiply. Round
        # it here to match (the fp32 sigmoid otherwise flips e4m3 requant bits, ~35% on MXFP4 down).
        if SIMULATE_UNFUSED:
            sig = sig.to(INTERMEDIATE_DTYPE).to(tl.float32)
        act = g * sig
    elif ACT_FN == "gelu":
        if SIMULATE_UNFUSED:
            # Bit-match the unfused ``apply_glu`` gelu ``0.5 * g * (1 + erf(g * c))``, which rounds
            # to bf16 at every torch op (input to erf, erf, 1+erf, 0.5*g, final mul). Rounding only
            # a subset diverges (~0.7 rel) on the MX requant — round each op, like torch does.
            gc = (g * 0.7071067811865476).to(INTERMEDIATE_DTYPE).to(tl.float32)
            e = tl.erf(gc).to(INTERMEDIATE_DTYPE).to(tl.float32)
            one_plus = (1.0 + e).to(INTERMEDIATE_DTYPE).to(tl.float32)
            half_g = (0.5 * g).to(INTERMEDIATE_DTYPE).to(tl.float32)
            act = half_g * one_plus
        else:
            act = 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))
    elif ACT_FN == "relu":
        act = tl.maximum(g, 0.0)
    else:
        tl.static_assert(
            False, "unsupported ACT_FN; expected 'silu', 'gelu', or 'relu'"
        )

    if SIMULATE_UNFUSED:
        act = act.to(INTERMEDIATE_DTYPE).to(tl.float32)
        u = u.to(INTERMEDIATE_DTYPE).to(tl.float32)

    gated = act * u

    if SIMULATE_UNFUSED:
        gated = gated.to(INTERMEDIATE_DTYPE).to(tl.float32)

    return gated



@triton.jit
def _glu_kernel(
    GateUp,  # (S, 2I) interleaved [g0,u0,g1,u1,...] GEMM output; fp32 = exact accumulators
    Out,  # (S, I) activation-dtype GLU result; the E4M3 tensor under the requant arm
    QuantScales,  # block-FP8 requant arm (``fused_glu(quant_group=...)``): per-``QUANT_GROUP``
    #             scales (fp32, or UE8M0 exponent bytes); ``None`` folds the arm out — the GLU
    #             result then rounds through bf16 (the reference two-kernel order) and quantizes
    n_out,
    I: tl.constexpr,
    ACT_FN: tl.constexpr,
    SWIGLU_ALPHA: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
    QUANT_GROUP: tl.constexpr = 128,
    UE8M0: tl.constexpr = False,
):
    """Fused GLU over a contiguous (S, 2I) interleaved gate|up tensor -> (S, I), covering every
    ``apply_glu`` variant (silu/gelu/relu, ``SWIGLU_ALPHA``/``SWIGLU_LIMIT`` clamped-SwiGLU
    — ``None`` folds the arm out, the in-kernel ``glu``'s convention). Rounds through the
    output dtype after each op exactly where the torch chain does (each torch tensor op
    materializes), so the two are bit-identical."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    # gate|up columns alternate (gate 2j, up 2j+1) — the interleaved row order the GEMM emits
    base = (offs // I) * 2 * I + 2 * (offs % I)
    # fp32 input = exact GEMM accumulators (the unstacked fused-order path): compute the
    # GLU in fp32 and round ONCE at the store — the gated epilogue's math. Narrow inputs
    # keep the torch-chain per-op rounding (bit-identical to apply_glu).
    FUSED_ORDER: tl.constexpr = GateUp.dtype.element_ty == tl.float32
    dt = tl.float32 if FUSED_ORDER else (
        tl.bfloat16 if QuantScales is not None else Out.dtype.element_ty)
    g = tl.load(GateUp + base, mask=mask).to(tl.float32)
    u = tl.load(GateUp + base + 1, mask=mask).to(tl.float32)
    if SWIGLU_LIMIT is not None:
        g = tl.minimum(g, SWIGLU_LIMIT)
        u = tl.clamp(u, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    if SWIGLU_ALPHA is not None:  # (up + 1) * gate * sigmoid(alpha * gate)
        sig = tl.sigmoid((g * SWIGLU_ALPHA).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        act = (g * sig).to(dt).to(tl.float32)
        out = ((u + 1.0).to(dt).to(tl.float32) * act).to(dt)
    elif ACT_FN == "silu":
        sig = tl.sigmoid(g).to(dt).to(tl.float32)
        act = (g * sig).to(dt).to(tl.float32)
        out = (act * u).to(dt)
    elif ACT_FN == "gelu":  # exact via erf, torch's evaluation order
        e = tl.erf((g * 0.7071067811865476).to(dt).to(tl.float32)).to(dt).to(tl.float32)
        d = (1.0 + e).to(dt).to(tl.float32)
        a = (0.5 * g).to(dt).to(tl.float32)
        out = ((a * d).to(dt).to(tl.float32) * u).to(dt)
    else:  # relu
        out = (tl.maximum(g, 0.0) * u).to(dt)
    if QuantScales is None:
        tl.store(Out + offs, out.to(Out.dtype.element_ty), mask=mask)
    else:
        grp = tl.reshape(out.to(tl.float32), (BLOCK // QUANT_GROUP, QUANT_GROUP))
        amax = tl.max(tl.abs(grp), axis=1)
        if UE8M0:
            sc = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1e-10) / 448.0)))
        else:
            sc = tl.maximum(amax, 1e-10) / 448.0
        q = tl.clamp(grp / sc[:, None], -448.0, 448.0)
        tl.store(Out + offs, tl.reshape(q, (BLOCK,)).to(tl.float8e4nv), mask=mask)
        soffs = tl.program_id(0) * (BLOCK // QUANT_GROUP) + tl.arange(0, BLOCK // QUANT_GROUP)
        smask = soffs * QUANT_GROUP < n_out
        if UE8M0:
            tl.store(QuantScales + soffs, (tl.log2(sc) + 127.0).to(tl.uint8), mask=smask)
        else:
            tl.store(QuantScales + soffs, sc, mask=smask)


def fused_glu(
    gate_up: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    quant_group: int | None = None,
    use_ue8m0: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """GLU over the interleaved (S, 2I) gate|up GEMM output — the one-kernel ``_glu_kernel``
    on a contiguous tensor, the torch ``apply_glu`` on any fallback layout; bit-identical
    either way. ``quant_group`` additionally block-FP8-requantizes the result per group in
    the same kernel (the unstacked decode band's handoff to the down projection) and
    returns ``(q, scales)`` — scales fp32, or UE8M0 exponent bytes under ``use_ue8m0``;
    non-SiLU/clamped variants take the two-step order (same bits — the fused arm rounds
    through bf16 first for exactly this equivalence)."""
    I = gate_up.shape[-1] // 2
    if quant_group is not None:
        if act_fn == "silu" and swiglu_alpha is None and swiglu_limit is None and gate_up.is_contiguous():
            S = gate_up.shape[0]
            q = torch.empty(S, I, dtype=torch.float8_e4m3fn, device=gate_up.device)
            sc = torch.empty(S, I // quant_group,
                             dtype=torch.uint8 if use_ue8m0 else torch.float32,
                             device=gate_up.device)
            n = q.numel()
            with device_context(gate_up.device):
                compile_time_only_triton_wrap(_glu_kernel)[(triton.cdiv(n, 1024),)](
                    gate_up, q, sc, n, I, "silu", None, None, BLOCK=1024,
                    QUANT_GROUP=quant_group, UE8M0=use_ue8m0,
                )
            return q, sc
        from .quant import fp8_act_quant_block_dynamic  # deferred: module import order

        inter = fused_glu(gate_up, act_fn, swiglu_alpha, swiglu_limit)
        return fp8_act_quant_block_dynamic(inter, quant_group, use_ue8m0=use_ue8m0)
    if gate_up.is_contiguous() and act_fn in ("silu", "gelu", "relu"):
        out = torch.empty(*gate_up.shape[:-1], I,
                          dtype=out_dtype or gate_up.dtype, device=gate_up.device)
        n = out.numel()
        with device_context(gate_up.device):
            compile_time_only_triton_wrap(_glu_kernel)[(triton.cdiv(n, 1024),)](
                gate_up, out, None, n, I, act_fn, swiglu_alpha, swiglu_limit, BLOCK=1024
            )
        return out
    gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
    return apply_glu(gate, up, act_fn, swiglu_alpha, swiglu_limit)


def apply_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Host-side (torch) gated linear unit — the unfused path's activation, mirroring the triton
    ``glu``. ``swiglu_limit`` clamps gate above / up to ``[-limit, limit]``; ``swiglu_alpha`` gives
    the clamped/scaled SwiGLU ``(up + 1) * gate * sigmoid(alpha * gate)`` (GPT-OSS / MiniMax), else
    ``act_fn(gate) * up`` (``act_fn`` in {silu, gelu, relu}, gelu exact via erf)."""
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    if swiglu_alpha is not None:
        return (up + 1.0) * (gate * torch.sigmoid(gate * swiglu_alpha))
    if act_fn == "silu":
        act = gate * torch.sigmoid(gate)
    elif act_fn == "gelu":
        act = 0.5 * gate * (1.0 + torch.erf(gate * 0.7071067811865476))
    elif act_fn == "relu":
        act = gate.clamp(min=0.0)
    else:
        raise ValueError(
            f"unsupported act_fn {act_fn!r}; expected 'silu', 'gelu', or 'relu'"
        )
    return act * up



@triton.jit
def split_gate_up_glu(
    acc,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.float32,
):
    """Gate|up epilogue in one step: split the stacked accumulator into its (gate, up) pair
    (``split_gate_up``) and apply the ``ACT_FN``/SwiGLU gated linear unit (``glu``), returning
    the combined intermediate. See those two for the orientation and activation details."""
    gate, up = split_gate_up(acc, COMPUTE_MODE, BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    return glu(
        gate,
        up,
        ACT_FN,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        SIMULATE_UNFUSED,
        INTERMEDIATE_DTYPE,
    )



@triton.jit
def _store_out(
    C, acc, out_row, pid_n, row_mask, stride_c_m, stride_c_n,
    BLOCK_SIZE_M: tl.constexpr, WIDTH: tl.constexpr, FAKE_BATCH: tl.constexpr,
    N_COLS: tl.constexpr = 0,
):
    """Cast + store one output tile of N-width ``WIDTH`` (halved when the recipe packs nibble
    pairs). ``FAKE_BATCH`` (batched decode): the BM lanes alias one C row (``C`` pre-advanced), so
    a plain store would duplicate-write the same bytes (hardware-undefined on Intel XPU) — mask to
    lane 0 (the replicated rows are identical). Else a real scatter to global rows ``out_row`` under
    ``row_mask``. ``N_COLS`` > 0 also masks the column tail (the 2D dense output isn't ``BN``-aligned
    like the ``N % BN == 0`` grouped/batched MoE outputs); 0 skips it. Single return."""
    c = acc.to(C.dtype.element_ty)
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * WIDTH + tl.arange(0, WIDTH)
    col_ok = (offs_cn < N_COLS) if N_COLS > 0 else (offs_cn >= 0)
    if FAKE_BATCH:
        c_ptrs = C + offs_cm[:, None] * 0 + stride_c_n * offs_cn[None, :]
        tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None] & col_ok[None, :])
    else:
        c_ptrs = C + stride_c_m * out_row[:, None] + stride_c_n * offs_cn[None, :]
        tl.store(c_ptrs, c, mask=row_mask[:, None] & col_ok[None, :])


@triton.jit
def _epilogue_requant_fp8(
    C, Cs, out, out_row, pid_n, row_mask, stride_c_m, stride_c_n, stride_cs_m, stride_cs_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, FAKE_BATCH: tl.constexpr,
    N_COLS: tl.constexpr,
):
    """Requantize the SwiGLU intermediate to fp8 with a per-(row, N-tile) scalar scale. UE8M0
    intermediate scales under a UE8M0 model (inferred from Cs's dtype) keep the down proj's activation
    scales power-of-two so its dot_scaled arm fires. FAKE_BATCH collapses the replicated rows to one
    scalar per (row, N-tile)."""
    q, q_s = fp8_act_quant_inline(out, UE8M0=Cs.dtype.element_ty == tl.uint8)
    _store_out(C, q, out_row, pid_n, row_mask, stride_c_m, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N, FAKE_BATCH, N_COLS)
    cs_ptr = Cs + out_row * stride_cs_m + pid_n * stride_cs_n
    if FAKE_BATCH:  # replicated rows -> one scalar per (row, N-tile)
        tl.store(cs_ptr, tl.max(q_s))
    else:
        tl.store(cs_ptr, q_s, mask=row_mask)


@triton.jit
def _epilogue_requant_mx(
    C, Cs, out, out_row, pid_n, pid_m, row_mask, stride_c_m, stride_c_n, stride_cs_m, stride_cs_n,
    CSDescriptor, CsGlobal,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, SCALE_GROUP_K: tl.constexpr,
    OUTPUT_RECIPE: tl.constexpr, SWIZZLED_OUT: tl.constexpr, FAKE_BATCH: tl.constexpr,
    N_COLS: tl.constexpr,
):
    """Requantize the SwiGLU intermediate to MX group-``SCALE_GROUP_K`` (mxfp8 UE8M0 / mxfp4 / nvfp4
    E4M3 — the fp4 recipes pack nibble pairs so ``C`` halves). ``SWIZZLED_OUT`` writes ``Cs`` straight
    into the down proj's SWIZZLE_32_4_4 descriptor at block ``(pid_m, pid_n)`` (BM/BN pinned 128), else
    a row-major affine store. NVFP4 two-level: normalize by the next proj's provided input_scale
    (``CsGlobal``) before the block quant. FAKE_BATCH collapses replicated rows via the row-max."""
    if CsGlobal is not None:
        # NVFP4 two-level requant: normalize the fp32 GLU intermediate by the NEXT proj's provided
        # (calibrated) input_scale before the block quant — the canonical two-step. The down folds it
        # back via its As pair ([Cs, g_out]); nothing is computed at runtime.
        out = out / tl.load(CsGlobal).to(tl.float32)
    q, q_s = mx_act_quant_inline(out, BLOCK_SIZE_M, BLOCK_SIZE_N, SCALE_GROUP_K, OUTPUT_RECIPE)
    width: tl.constexpr = BLOCK_SIZE_N if OUTPUT_RECIPE == "mxfp8" else BLOCK_SIZE_N // 2
    _store_out(C, q, out_row, pid_n, row_mask, stride_c_m, stride_c_n, BLOCK_SIZE_M, width, FAKE_BATCH,
               N_COLS if OUTPUT_RECIPE == "mxfp8" else N_COLS // 2)
    if SWIZZLED_OUT:
        # group scales straight into the down proj's SWIZZLE_32_4_4 layout (inverse of
        # load_swizzled_scale) at block (pid_m, pid_n) — BM/BN pinned 128.
        REP_K_CS: tl.constexpr = (BLOCK_SIZE_N // SCALE_GROUP_K) // 4
        sw = (
            q_s.reshape(1, 4, 32, REP_K_CS, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(1, 1, REP_K_CS, 2, 256)
        )
        CSDescriptor.store([0, pid_m, pid_n * REP_K_CS, 0, 0], sw)
    else:
        offs_sc = pid_n * (BLOCK_SIZE_N // SCALE_GROUP_K) + tl.arange(0, BLOCK_SIZE_N // SCALE_GROUP_K)
        if FAKE_BATCH:
            # replicated rows -> the row-max IS the row's scale (f32-exact for UE8M0 exponent bytes
            # and E4M3 values alike)
            tl.store(
                Cs + out_row * stride_cs_m + offs_sc[None, :] * stride_cs_n,
                tl.reshape(tl.max(q_s.to(tl.float32), axis=0), (1, BLOCK_SIZE_N // SCALE_GROUP_K)),
            )
        else:
            tl.store(
                Cs + out_row[:, None] * stride_cs_m + offs_sc[None, :] * stride_cs_n,
                q_s,
                mask=row_mask[:, None],
            )


@triton.jit
def gemm_epilogue(
    C,
    Cs,  # row-major requant-scale pointer (None under SWIZZLED_OUT — CSDescriptor writes instead)
    acc,
    out_row,
    pid_n,
    pid_m,
    row_mask,
    stride_c_m,
    stride_c_n,
    stride_cs_m,
    stride_cs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GATE: tl.constexpr,
    OUTPUT_RECIPE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    ACT_FN: tl.constexpr,
    SWIGLU_ALPHA: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
    INTERMEDIATE_DTYPE: tl.constexpr,
    COMPUTE_MODE: tl.constexpr = "dot",
    SWAP_AB: tl.constexpr = False,
    SWIZZLED_OUT: tl.constexpr = False,
    FAKE_BATCH: tl.constexpr = False,
    N_COLS: tl.constexpr = 0,  # >0 masks the column tail (2D dense N isn't BN-aligned); 0 = no mask
    CSDescriptor=0,  # SWIZZLE_32_4_4 requant-scale descriptor; read only under SWIZZLED_OUT (else dummy)
    CsGlobal=None,  # (1,) fp32 NVFP4 output global (the NEXT proj's provided input_scale); normalizes the requant, None folds out
    GlobalScale=None,  # (E,)|(1,) fp32 NVFP4 accumulator global (g_a·g_b or the weight-only g_b); applied PRE-GLU, None folds out
    global_row=0,  # index into GlobalScale (expert id; 0 for the per-tensor 2D case)
    PreAct=None,  # (M, 2N) pre-activation buffer; written iff not None — the GLU backward's Z
    stride_pa_m=0,
    stride_pa_n=0,
):
    """Unified output epilogue for grouped (a real scatter tile) and batched (fake-batch decode:
    one token replicated across the BM lanes) GEMMs. Plain: cast + store the accumulator. ``GATE``:
    split the stacked gate|up accumulator + SwiGLU (``split_gate_up_glu``); ``OUTPUT_RECIPE`` — the
    ``Quantization`` vocabulary — then requantizes into ``C`` + ``Cs``: ``"fp8"`` (per-(row, N-tile)
    scalar), or MX group-``SCALE_GROUP_K`` (UE8M0/E4M3 — the fp4 recipes pack nibble pairs so ``C``
    halves), with ``SWIZZLED_OUT`` writing the MX ``Cs`` straight into the down proj's SWIZZLE_32_4_4
    descriptor at block ``(pid_m, pid_n)`` (the tcgen05 fast path, BM/BN pinned 128). ``FAKE_BATCH``
    shims the store: value masks to lane 0 (``C`` pre-advanced), the scale collapses the replicated
    rows with ``tl.max``; else a real BM-row scatter (``out_row`` + ``row_mask``).
    ``COMPUTE_MODE``/``SWAP_AB`` orient the decode GLU/finalize (grouped passes ``"dot"``/no-swap,
    both no-ops there). Every arm is constexpr-pruned."""
    acc = apply_global_scale(acc, GlobalScale, global_row)
    if GATE:
        # SAVE_PREACT: the GLU backward needs Z, the 2*BN-wide pre-activation. Interleaved, that
        # tile IS an ungated output tile at doubled width (pid_n * 2BN == 2 * pid_n * BN), so the
        # ordinary store serves it — no separate path, no gated store arm.
        if PreAct is not None:
            _store_out(
                PreAct,
                acc_finalize(acc, COMPUTE_MODE, 2 * BLOCK_SIZE_N, SWAP_AB),
                out_row, pid_n, row_mask, stride_pa_m, stride_pa_n,
                BLOCK_SIZE_M, 2 * BLOCK_SIZE_N, FAKE_BATCH,
                2 * N_COLS if N_COLS > 0 else 0,
            )
        out = split_gate_up_glu(
            acc, COMPUTE_MODE, BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB,
            ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        )
        if OUTPUT_RECIPE == "fp8":
            _epilogue_requant_fp8(
                C, Cs, out, out_row, pid_n, row_mask, stride_c_m, stride_c_n, stride_cs_m, stride_cs_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, FAKE_BATCH, N_COLS,
            )
        elif OUTPUT_RECIPE is not None:  # "mxfp8" | "mxfp4" | "nvfp4"
            _epilogue_requant_mx(
                C, Cs, out, out_row, pid_n, pid_m, row_mask, stride_c_m, stride_c_n, stride_cs_m,
                stride_cs_n, CSDescriptor, CsGlobal,
                BLOCK_SIZE_M, BLOCK_SIZE_N, SCALE_GROUP_K, OUTPUT_RECIPE, SWIZZLED_OUT, FAKE_BATCH, N_COLS,
            )
        else:  # bf16 (unquantized) SwiGLU output
            _store_out(C, out, out_row, pid_n, row_mask, stride_c_m, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N, FAKE_BATCH, N_COLS)
    else:  # plain GEMM: cast + store the accumulator (no fused requant on the non-gate path)
        acc = acc_finalize(acc, COMPUTE_MODE, BLOCK_SIZE_N, SWAP_AB)
        _store_out(C, acc, out_row, pid_n, row_mask, stride_c_m, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N, FAKE_BATCH, N_COLS)
