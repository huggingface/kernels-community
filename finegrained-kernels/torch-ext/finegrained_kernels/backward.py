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
"""Backward for the quantized ops: dgrad only, registered on the forward ops themselves.

``torch.library.register_autograd`` on the ops themselves rather than an ``autograd.Function``
wrapper: the ops take the RAW activation (they quantize it in one pass internally), so the op
boundary is exactly where a usable gradient lives, and hanging autograd there means an ordinary
``matmul_2d`` call differentiates — through compile, export and AOTAutograd, with no second
entry point to keep in sync. ``compile_time_only_triton_op``'s dispatch routes to the op
whenever an input wants a gradient, so this fires in eager as well.

dgrad ONLY: ``dA = dY @ B``, contracting N. The quantized weight never receives a gradient, and
not merely by omission — autograd accumulates into a leaf, and an E4M3 leaf has no ``add``
("ufunc_add_CUDA not implemented for Float8_e4m3fn"), so an fp8/fp4 tensor cannot be an autograd
parameter at all. A frozen quantized base with trainable high-precision adapters in front of it
(QLoRA) is therefore the shape this serves, and the only gradient it needs is the one that flows
THROUGH the weight to whatever precedes it.

Block-FP8 and per-tensor formats reach that product on the forward kernel via transposed views
(their scale grids subdivide both axes). MX/NVFP4 cannot be reoriented by a view — group scales
bind to K and packed E2M1 packs along K — so they contract N against the weight's natural (N, K)
tile, dequantized per-tile in-register; see ``backward.py``. Either way the FORWARD-oriented
weight is the only copy that exists.

── the gradient products ──

Forward contracts K (``C = A @ B.T``, both operands K-contiguous). dgrad contracts N::

    dX[m, k] = sum_n dY[m, n] * W[n, k]

The weight is STORED ``(N, K)`` K-contiguous, so the ``[BN, BK]`` tile this reduction wants is
already a natural contiguous tile — n leads, k is contiguous. Nothing is transposed and no second
quantized copy is kept: the tile is dequantized in-register and fed to a plain ``tl.dot``, exactly
as the weight-only forward's ``dot`` arm does. That is what makes a 4-bit frozen weight
differentiable at no memory cost, which is the whole QLoRA case.

The group axis stays K, which here is the OUTPUT axis rather than the contraction axis. That is
why ``dot_scaled`` cannot serve this pass — the tensor core's scaled MMA requires the scaled axis
to be the one it reduces over — so dgrad is plain-dot only. The tuner already prefers ``dot`` over
``dot_scaled`` on the gate|up tile, so little is given up.

Format genericity lives in ``_dgrad_weight_tile`` alone. A format is fully described to this
kernel by three numbers plus the value dtype:

    SCALE_GROUP_K    scale extent along K   32 (MX)   128 (block-FP8)
    SCALE_ROW_DIV    rows sharing a scale    1 (MX)   128 (block-FP8)
    WEIGHT_VALUES_PER_BYTE  2 for packed E2M1, else 1

so MX (per-row K-groups) and block-FP8 (128x128 blocks) are the same code path at different
divisors, and no new arm is needed to add a format whose scales tile the same way.
"""

import inspect

import torch
import triton
import triton.language as tl

from .bayesian_autotuner import bayesian_autotune
from ._ops import add_op_namespace_prefix
from .compat import (
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    device_context,
    get_accelerator_autotuning_configs,
    sm_count,
)
from .descriptors import maybe_descriptor, rebind_dgrad_descriptors, rebind_dgrad_grouped_descriptors, weight_descriptor
from . import batched, grouped, matmul
from .matmul import matmul_2d
from .pruners import (
    PATH_ANCHOR_AXES,
    block_within_dim_pruner,
    config_filter,
    descriptor_support_pruner,
    dgrad_grouped_warps_pruner,
    compose_pruners,
    smem_pruner,
    warp_spec_compile_guard_pruner,
)
from .formats import global_scale_stride, normalize_global_scale, ue8m0_as_uint8
from .swizzle import unswizzle_mx_scales
from .quant import e2m1_cols_to_bf16, e2m1_cols_to_f32
from .loading.scales import decode_group_scale
from .mma import int_dequant_tile
from .loading.tiles import load_grouped_act_tile, operand_tile_ptrs
from .scheduling import build_tile_layout, resolve_grouped_tile


# Internal: these are the gradient PRODUCTS, not API. Backward hangs off the forward ops via
# ``register_autograd`` (the table at the bottom of this file), so an ordinary forward call
# differentiates and there is
# no second entry point to keep in sync. Nothing here is re-exported from the package root.


@triton.jit
def _dgrad_load_weight(
    b_ptrs, b_descriptor, row0, n_off, kb_off,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr, GROUPED: tl.constexpr = False,
):
    """One ``[BN, BK_bytes]`` weight tile: the explicit pointer tile, or the host-TMA box at the
    same absolute offset. The box is a natural contiguous window — unlike the forward's gated
    tile there is no interleaved span to express — so the descriptor arm needs no doubling.
    Single return; only the taken arm compiles."""
    if B_MEMORY_MODE == "pointer":
        w = tl.load(b_ptrs)
    elif GROUPED:  # (E, N, K_bytes) box — the expert leads
        w = tl.reshape(b_descriptor.load([row0, n_off, kb_off]), BLOCK_ROWS, BLOCK_COLS)
    else:
        w = b_descriptor.load([n_off, kb_off])
    return w


@triton.jit
def _dgrad_weight_tile(
    w,
    w_scale,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One natural ``[BLOCK_SIZE_N, BLOCK_SIZE_K]`` weight tile, dequantized. The K-oriented
    counterpart of ``mx_weight_upcast`` — the group axis is already last here, so unlike the
    forward no transpose is needed.

    The scale broadcast happens INSIDE the multiply: the real (contiguous) weight reshapes to
    ``[BN, ng, g]`` and the ``[BN, ng, 1]`` scale broadcasts across each group's columns.
    Materializing the broadcast first and reshaping it does not lower — the same trap
    ``mx_weight_upcast`` documents.

    ``OUT_DTYPE`` picks the arm, and both are this codebase's measured verdicts:
    bf16 for the MMA consumers (exact — E2M1/E4M3 codes and power-of-two UE8M0 scales all fit in
    bf16, and the multiply is a cheap exponent shift), fp32 for the scalar reduce, which widens
    anyway and for which the bf16/E4M3 hop is pure cost. Single return; only the taken arm
    compiles."""
    TILE_GROUP: tl.constexpr = SCALE_GROUP_K if SCALE_GROUP_K < BLOCK_SIZE_K else BLOCK_SIZE_K  # a group wider than the tile
    NG: tl.constexpr = BLOCK_SIZE_K // TILE_GROUP
    if w_scale is None:  # full-precision weight
        out = w.to(OUT_DTYPE)
    elif OUT_DTYPE == tl.float32:  # scalar-reduce arm: unpack straight to fp32, no narrow hop
        vals = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)
        sc = decode_group_scale(w_scale).to(tl.float32)[:, :, None]
    else:  # MMA arm: dequantize in bf16, exactly as the forward's weight upcast does
        vals = e2m1_cols_to_bf16(w) if w.dtype == tl.uint8 else w.to(tl.bfloat16)
        sc = decode_group_scale(w_scale).to(tl.bfloat16)[:, :, None]
    if w_scale is not None:
        out = (vals.reshape(BLOCK_SIZE_N, NG, TILE_GROUP) * sc).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).to(OUT_DTYPE)
    return out


@triton.jit
def _dgrad_scale_cols(pid_k, K, BLOCK_SIZE_K: tl.constexpr, SCALE_GROUP_K: tl.constexpr):
    """The scale columns of K tile ``pid_k``, one per ``min(SCALE_GROUP_K, BLOCK_SIZE_K)`` sub-group: the tile's own
    groups, or the one group it lies inside (repeated) when groups are wider than the tile."""
    TILE_GROUP: tl.constexpr = SCALE_GROUP_K if SCALE_GROUP_K < BLOCK_SIZE_K else BLOCK_SIZE_K
    starts = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // TILE_GROUP) * TILE_GROUP
    return tl.minimum(starts // SCALE_GROUP_K, K // SCALE_GROUP_K - 1)


@triton.jit
def _dgrad_int_tile(w, w_scale, w_zero, divisor, BLOCK_SIZE_K: tl.constexpr, SCALE_GROUP_K: tl.constexpr,
                    INT_BITS: tl.constexpr, OUT_DTYPE: tl.constexpr):
    """One forward-oriented ``[BN, BK]`` integer weight tile from its ``[BN, BK // VPW]`` interleaved words, the value
    the forward multiplies by: ``int_dequant_tile`` rounded in the scale dtype into ``OUT_DTYPE`` (the activation
    dtype), or, under ModelOpt W4A8 (``divisor``: the second-level scale), into E4M3 against it."""
    TILE_GROUP: tl.constexpr = SCALE_GROUP_K if SCALE_GROUP_K < BLOCK_SIZE_K else BLOCK_SIZE_K
    if divisor is None:
        out = tl.trans(int_dequant_tile(w, w_scale, w_zero, True, TILE_GROUP, INT_BITS, OUT_DTYPE))
    else:
        out = tl.trans(int_dequant_tile(w, w_scale, w_zero, True, TILE_GROUP, INT_BITS, tl.float8e4nv, divisor=divisor))
        out = out.to(OUT_DTYPE)
    return out


@bayesian_autotune(
    # Plain-dot loop over the N reduction — same physics as the full-precision 2D kernel (tile +
    # WS), no memory-mode axis: the dequantize is in-tile so both operands read through pointers,
    # and no COMPUTE_MODE axis because the scaled MMA cannot reduce over an unscaled axis.
    get_accelerator_autotuning_configs(
        tune_block_nk=True,
        tune_block_m=True,
        warp_spec=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=rebind_dgrad_descriptors,
    ),
    ["N", "K", "m_bit_length"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    finite_check_args=("C",),
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("N"),  # exact N-loop: no tail mask, constant-stride advance
            config_filter(  # the scale row advances by BN // SCALE_ROW_DIV — must be whole
                lambda c, a: c.kwargs["BLOCK_SIZE_N"] % max(a.get("SCALE_ROW_DIV", 1), 1) == 0
            ),
            config_filter(  # a K tile holds whole scale groups, or lies inside one (per-channel INT, wide groups)
                lambda c, a: c.kwargs["BLOCK_SIZE_K"] % a.get("SCALE_GROUP_K", 1) == 0
                or a.get("SCALE_GROUP_K", 1) % c.kwargs["BLOCK_SIZE_K"] == 0
            ),
            warp_spec_compile_guard_pruner(),
            descriptor_support_pruner(),
            smem_pruner("BLOCK_SIZE_K"),
        )
    },
)
@triton.jit
def dgrad_matmul_2d_kernel(
    A,  # (M, N) upstream gradient, raw bf16/fp16
    ADescriptor,  # host TMA descriptor over A, box (BM, BN); read iff A_MEMORY_MODE != "pointer"
    B,  # (N, K) quantized weight, K-contiguous — read in its FORWARD orientation
    BDescriptor,  # host TMA descriptor over B, box (BN, BK_bytes); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (N // SCALE_ROW_DIV, K // SCALE_GROUP_K) weight scales
    Bz,  # (N, K // SCALE_GROUP_K) integer zero points; None for a symmetric or non-integer weight
    BsGlobal,  # fp32 NVFP4 / W4A8 second-level weight scale; folded on the accumulator, None folds out
    C,  # (M, K) output gradient
    Perm,  # (K,) act-order K order: column j of the product is stored at Perm[j]; None = natural
    # Shape
    M,
    N,
    K,
    m_bit_length,  # autotune key only (log2 M bucket); unused in body
    # Strides
    stride_a_m,
    stride_a_n,
    stride_b_n,
    stride_b_k,
    stride_bs_n,
    stride_bs_k,
    stride_bz_n,
    stride_bz_k,
    stride_c_m,
    stride_c_k,
    # Format
    SCALE_GROUP_K: tl.constexpr,
    SCALE_ROW_DIV: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
    # Meta-parameters
    A_MEMORY_MODE: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WARP_SPEC: tl.constexpr,
    INT_BITS: tl.constexpr = 0,  # integer weight width; 0: E2M1 / E4M3 / full-precision weight
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.minimum(
        pid_k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
        + tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE),
        K // WEIGHT_VALUES_PER_BYTE - 1,
    )
    offs_kg = _dgrad_scale_cols(pid_k, K, BLOCK_SIZE_K, SCALE_GROUP_K)
    m_mask = offs_m < M
    k_mask = offs_k < K

    # BLOCK_SIZE_N divides N (block_within_dim_pruner), so the reduction loop is EXACT: no tail
    # mask on the activation, no clamp on the weight rows, and the addresses advance by a constant
    # stride instead of being rebuilt from offsets each step — the forward's loop shape.
    offs_n0 = tl.arange(0, BLOCK_SIZE_N)
    # SWAP_AB is the WEIGHT's viewpoint, inverted for activations: True = rows-major [BM, BN]
    dy_ptrs = operand_tile_ptrs(A, offs_m, offs_n0, stride_a_m, stride_a_n, A_MEMORY_MODE, True)
    b_ptrs = operand_tile_ptrs(B, offs_n0, offs_kb, stride_b_n, stride_b_k, B_MEMORY_MODE, True)
    if Bs is not None:
        bs_ptrs = Bs + (offs_n0[:, None] // SCALE_ROW_DIV) * stride_bs_n + offs_kg[None, :] * stride_bs_k
    if Bz is not None:
        bz_ptrs = Bz + offs_n0[:, None] * stride_bz_n + offs_kg[None, :] * stride_bz_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in tl.range(0, tl.cdiv(N, BLOCK_SIZE_N), warp_specialize=WARP_SPEC):
        n_off = n * BLOCK_SIZE_N
        dy = load_grouped_act_tile(
            dy_ptrs, ADescriptor, pid_m * BLOCK_SIZE_M, n_off, m_mask, 0, A_MEMORY_MODE
        )
        # the tile the reduction wants is the weight's NATURAL layout: n leads, k contiguous
        w = _dgrad_load_weight(
            b_ptrs, BDescriptor, 0, n_off,
            pid_k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE),
            BLOCK_SIZE_N, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE, B_MEMORY_MODE,
        )
        w_scale = tl.load(bs_ptrs) if Bs is not None else None
        if INT_BITS:
            w_zero = tl.load(bz_ptrs) if Bz is not None else None
            divisor = tl.load(BsGlobal) if BsGlobal is not None else None
            accumulator += tl.dot(dy, _dgrad_int_tile(w, w_scale, w_zero, divisor, BLOCK_SIZE_K, SCALE_GROUP_K, INT_BITS, dy.dtype))
        else:
            accumulator += tl.dot(
                dy.to(tl.bfloat16),
                _dgrad_weight_tile(w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, tl.bfloat16),
            )
        dy_ptrs += BLOCK_SIZE_N * stride_a_n
        b_ptrs += BLOCK_SIZE_N * stride_b_n
        if Bs is not None:
            bs_ptrs += (BLOCK_SIZE_N // SCALE_ROW_DIV) * stride_bs_n
        if Bz is not None:
            bz_ptrs += BLOCK_SIZE_N * stride_bz_n

    if BsGlobal is not None:  # the second-level scale: one matrix, so one value
        accumulator = accumulator * tl.load(BsGlobal)

    cols = tl.load(Perm + offs_k, mask=k_mask, other=0) if Perm is not None else offs_k
    tl.store(
        C + offs_m.to(tl.int64)[:, None] * stride_c_m + cols[None, :] * stride_c_k,
        accumulator.to(C.dtype.element_ty),
        mask=m_mask[:, None] & k_mask[None, :],
    )


@compile_time_only_triton_op(add_op_namespace_prefix("dgrad_matmul_2d"), mutates_args=())
def dgrad_matmul_2d(
    dY: torch.Tensor,
    W: torch.Tensor,
    Ws: torch.Tensor,
    scale_group_k: int,
    scale_row_div: int,
    output_dtype: torch.dtype | None = None,
    w_global_scale: torch.Tensor | None = None,
    w_zero: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    int_bits: int = 0,
) -> torch.Tensor:
    """``dX = dY @ W`` against the FORWARD-oriented quantized weight — the gradient a frozen
    quantized base weight passes to whatever trains in front of it (LoRA adapters, earlier
    layers).

    ``W`` is ``(N, K)`` exactly as the forward holds it; nothing is transposed or re-quantized.
    ``scale_group_k``/``scale_row_div`` describe the scale grid (MX: ``32, 1``; block-FP8:
    ``128, 128``), which is the only thing that differs between formats. An integer weight (``int_bits``) is its
    interleaved uint32 words, with ``w_zero`` its zero points and ``perm`` its act-order K order."""
    assert dY.ndim == 2 and W.ndim == 2, f"expected 2D operands, got {dY.ndim}D and {W.ndim}D"
    M, N = dY.shape
    values_per_byte = 32 // int_bits if int_bits else 2 if W.dtype == torch.uint8 else 1
    K = W.shape[1] * values_per_byte
    assert W.shape[0] == N, f"weight rows {W.shape[0]} != dY's N {N}"
    assert K % scale_group_k == 0, (
        f"K={K} must be a whole number of scale groups ({scale_group_k})"
    )

    dX = torch.empty(M, K, device=dY.device, dtype=output_dtype or dY.dtype)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )
    with device_context(dY.device):
        # built INSIDE the device context: the tensormap is created against the CURRENT context,
        # so constructing it before the switch fails as "invalid device context" whenever the
        # caller's device is not the active one. Boxes are placeholders; the pre_hook rebinds
        # them to the tuned tile per config.
        a_descriptor = maybe_descriptor(dY, [1, 128])
        b_descriptor = weight_descriptor(W, [1, 128])
        compile_time_only_triton_wrap(dgrad_matmul_2d_kernel)[grid](
            dY,
            a_descriptor,
            W,
            b_descriptor,
            Ws,
            w_zero,
            w_global_scale,
            dX,
            perm,
            M,
            N,
            K,
            max(int(M).bit_length(), 1),
            dY.stride(0),
            dY.stride(1),
            W.stride(0),
            W.stride(1),
            *_scale_strides(Ws, 2),
            *_scale_strides(w_zero, 2),
            dX.stride(0),
            dX.stride(1),
            SCALE_GROUP_K=scale_group_k,
            SCALE_ROW_DIV=scale_row_div,
            WEIGHT_VALUES_PER_BYTE=values_per_byte,
            INT_BITS=int_bits,
        )
    return dX


@bayesian_autotune(
    # Persistent grouped loop, plain dot over the N reduction. No memory-mode axis (the
    # dequantize is in-tile, so both operands read through pointers) and no COMPUTE_MODE axis
    # (the scaled MMA cannot reduce over an unscaled axis — see the module docstring).
    get_accelerator_autotuning_configs(
        tune_block_nk=True,
        tune_block_m=True,
        warp_spec=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=rebind_dgrad_grouped_descriptors,
    ),
    ["N", "K", "tokens_per_expert_bit_length"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    finite_check_args=("C",),
    prune_configs_by={
        "early_config_prune": compose_pruners(
            dgrad_grouped_warps_pruner(),
            block_within_dim_pruner("N"),  # exact N-loop: no tail mask, constant-stride advance
            config_filter(  # the scale row advances by BN // SCALE_ROW_DIV — must be whole
                lambda c, a: c.kwargs["BLOCK_SIZE_N"] % max(a.get("SCALE_ROW_DIV", 1), 1) == 0
            ),
            config_filter(  # a K tile holds whole scale groups, or lies inside one (per-channel INT, wide groups)
                lambda c, a: c.kwargs["BLOCK_SIZE_K"] % a.get("SCALE_GROUP_K", 1) == 0
                or a.get("SCALE_GROUP_K", 1) % c.kwargs["BLOCK_SIZE_K"] == 0
            ),
            warp_spec_compile_guard_pruner(),
            descriptor_support_pruner(),
            smem_pruner("BLOCK_SIZE_K"),
        )
    },
)
@triton.jit
def dgrad_matmul_grouped_kernel(
    A,  # (S, N) upstream gradient in the forward's OUTPUT row order
    ADescriptor,  # host TMA descriptor over A; box (1|BM, BN) — 1 row when gathering
    B,  # (E, N, K) quantized weights, K-contiguous — the forward orientation
    BDescriptor,  # host TMA descriptor over B, box (1, BN, BK_bytes)
    Bs,  # (E, N // SCALE_ROW_DIV, K // SCALE_GROUP_K) weight scales
    Bz,  # (E, N, K // SCALE_GROUP_K) integer zero points; None for a symmetric or non-integer weight
    BsGlobal,  # fp32 NVFP4 / W4A8 second-level weight scale, per tensor or per expert; folded on the accumulator
    stride_bs_global,  # expert stride of BsGlobal (0 broadcasts a scalar)
    C,  # (S, K) output gradient, written in the forward's INPUT row order
    Perm,  # (E, K) act-order K orders: column j of an expert's product is stored at Perm[e, j]; None = natural
    GatherIdx,  # (S,) sorted position -> source row of the forward's A
    ScatterIdx,  # (S,) sorted position -> destination row of the forward's C
    RoutedIdx,  # (S,) sorted position -> its token-major routed row, for a gather-only launch; else None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) cumulative row starts, S sentinel
    # Shape
    S,
    N,
    K,
    tokens_per_expert_bit_length,  # autotune key only; unused in body
    # Strides
    stride_a_m,
    stride_a_n,
    stride_b_e,
    stride_b_n,
    stride_b_k,
    stride_bs_e,
    stride_bs_n,
    stride_bs_k,
    stride_bz_e,
    stride_bz_n,
    stride_bz_k,
    stride_perm_e,
    stride_c_m,
    stride_c_k,
    # Format
    SCALE_GROUP_K: tl.constexpr,
    SCALE_ROW_DIV: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
    # Meta-parameters
    A_MEMORY_MODE: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WARP_SPEC: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr,
    STORE: tl.constexpr = "position",
    INT_BITS: tl.constexpr = 0,  # integer weight width; 0: E2M1 / E4M3 / full-precision weight
):
    """Grouped dgrad. The tile resolver is reused verbatim with K-tiles in its N-tile slot —
    it is generic in that second axis, so ``offs_bn`` comes back as this kernel's K offsets and
    ``pid_n`` as its K-tile index (GATE=False: the gradient of a gate|up GEMM is an UNGATED
    product at the doubled N extent, so no gated arm exists here).

    The two row maps SWAP roles against the forward: the forward gathered A at ``in_row`` and
    scattered C to ``out_row``, so its gradient reads ``A`` at ``out_row`` and writes ``C`` at
    ``in_row``. Both are already real parameters, which is why routing needs no new machinery."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_k_tiles, NUM_SMS):
        pid_k, _, expert_id64, in_row, out_row, row_mask, offs_k, row0, _, m_start = (
            resolve_grouped_tile(
                tile_id,
                num_k_tiles,
                exp_start,
                freqs,
                tile_start_excl,
                e_offs,
                GatherIdx,
                ScatterIdx,
                BLOCK_SIZE_K,
                BLOCK_SIZE_M,
                False,
            )
        )
        offs_kb = tl.minimum(
            pid_k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
            + tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE),
            K // WEIGHT_VALUES_PER_BYTE - 1,
        )
        offs_kg = _dgrad_scale_cols(pid_k, K, BLOCK_SIZE_K, SCALE_GROUP_K)
        b_base = B + expert_id64 * stride_b_e

        # exact N-loop (BLOCK_SIZE_N divides N): maskless weight, constant-stride advance
        offs_n0 = tl.arange(0, BLOCK_SIZE_N)
        # SWAP_AB is the WEIGHT's viewpoint and activation callers pass it INVERTED: False builds
        # [K, rows], so both of these want True to get the rows-major tiles the dot expects
        # ([BM, BN] and [BN, BK]). Only the pointer arm is affected — the descriptor arm returns
        # the scalar base and reads its box from the descriptor.
        dy_ptrs = operand_tile_ptrs(
            A, out_row, offs_n0, stride_a_m, stride_a_n, A_MEMORY_MODE, True
        )
        b_ptrs = operand_tile_ptrs(
            b_base, offs_n0, offs_kb, stride_b_n, stride_b_k, B_MEMORY_MODE, True
        )
        if Bs is not None:
            bs_ptrs = (
                Bs + expert_id64 * stride_bs_e
                + (offs_n0[:, None] // SCALE_ROW_DIV) * stride_bs_n
                + offs_kg[None, :] * stride_bs_k
            )
        if Bz is not None:
            bz_ptrs = Bz + expert_id64 * stride_bz_e + offs_n0[:, None] * stride_bz_n + offs_kg[None, :] * stride_bz_k
        divisor = tl.load(BsGlobal + expert_id64 * stride_bs_global) if INT_BITS and BsGlobal is not None else None

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for n in tl.range(0, tl.cdiv(N, BLOCK_SIZE_N), warp_specialize=WARP_SPEC):
            n_off = n * BLOCK_SIZE_N
            # the forward gathered A at in_row and scattered C to out_row, so its gradient READS
            # at out_row — ScatterIdx is this pass's gather map. The box anchor must be out_row's
            # tile start too: the resolver's `m_start` is min(in_row) — the FORWARD's read side —
            # and with a gather-only launch (ScatterIdx None, in_row = the accumulation targets)
            # it anchors the box at a TOKEN id, reading the wrong dY rows.
            dy = load_grouped_act_tile(
                dy_ptrs, ADescriptor, tl.min(out_row).to(tl.int32), n_off, row_mask, out_row,
                A_MEMORY_MODE, ScatterIdx is not None,
            )
            w = _dgrad_load_weight(
                b_ptrs, BDescriptor, row0, n_off,
                pid_k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE),
                BLOCK_SIZE_N, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE, B_MEMORY_MODE, True,
            )
            w_scale = tl.load(bs_ptrs) if Bs is not None else None
            if INT_BITS:
                w_zero = tl.load(bz_ptrs) if Bz is not None else None
                accumulator += tl.dot(dy, _dgrad_int_tile(w, w_scale, w_zero, divisor, BLOCK_SIZE_K, SCALE_GROUP_K, INT_BITS, dy.dtype))
            else:
                accumulator += tl.dot(
                    dy.to(tl.bfloat16),
                    _dgrad_weight_tile(w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, tl.bfloat16),
                )
            dy_ptrs += BLOCK_SIZE_N * stride_a_n
            b_ptrs += BLOCK_SIZE_N * stride_b_n
            if Bs is not None:
                bs_ptrs += (BLOCK_SIZE_N // SCALE_ROW_DIV) * stride_bs_n
            if Bz is not None:
                bz_ptrs += BLOCK_SIZE_N * stride_bz_n

        if BsGlobal is not None:  # the second-level scale, this tile's expert
            accumulator = accumulator * tl.load(BsGlobal + expert_id64 * stride_bs_global)

        k_mask = offs_k < K
        cols = tl.load(Perm + expert_id64 * stride_perm_e + offs_k, mask=k_mask, other=0) if Perm is not None else offs_k
        _store_dgrad_rows(C, accumulator, in_row, out_row, row_mask, cols, k_mask, stride_c_m, stride_c_k, RoutedIdx, STORE)


@triton.jit
def _store_dgrad_rows(C, accumulator, in_row, out_row, row_mask, cols, col_mask, stride_c_m, stride_c_k,
                      RoutedIdx, STORE: tl.constexpr):
    """Store a dgrad tile where its rows' gradients belong, each row exactly once, no atomics:
    - ``"routed"``: each position's TOKEN-MAJOR routed row (``out_row`` with a scatter map, ``RoutedIdx`` of a
      gather-only launch), so the caller sums each token's top_k rows contiguously: the forward's own shape
      (``weighted_reduce``), 1.40x faster than atomics at DeepSeek-V3 top_k=8 (3003 vs 4215 us).
    - ``"position"``: ``in_row``, the sorted position of a scatter-only launch or the source row of a gather
      that is a permutation (one position per input row)."""
    if STORE == "position":
        row = in_row
    elif RoutedIdx is not None:
        row = tl.load(RoutedIdx + out_row, mask=row_mask, other=0)
    else:
        row = out_row
    tl.store(
        C + row.to(tl.int64)[:, None] * stride_c_m + cols[None, :] * stride_c_k,
        accumulator,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@compile_time_only_triton_op(add_op_namespace_prefix("dgrad_matmul_grouped"), mutates_args=())
def dgrad_matmul_grouped(
    dY: torch.Tensor,
    W: torch.Tensor,
    Ws: torch.Tensor,
    expert_start: torch.Tensor,
    scale_group_k: int,
    scale_row_div: int,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    num_input_rows: int | None = None,
    w_global_scale: torch.Tensor | None = None,
    routed_idx: torch.Tensor | None = None,
    w_zero: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    int_bits: int = 0,
) -> torch.Tensor:
    """Routed ``dX = dY @ W[expert]`` over expert-sorted positions — the MoE counterpart of
    ``dgrad_matmul_2d``.

    ``gather_idx`` / ``scatter_idx`` are the FORWARD's maps, passed unchanged: the kernel swaps
    their roles itself (reads ``dY`` at the forward's scatter destination, writes ``dX`` at its
    gather source), so callers hand over the same two tensors the forward got. A gathered launch sums each
    token's top_k rows, so it needs them token-major: its scatter map, or ``routed_idx`` (each position's
    token-major routed row, the chain's scatter map) when it has none. An integer weight (``int_bits``) is its
    interleaved uint32 words, with ``w_zero`` its zero points and ``perm`` its per-expert act-order K orders."""
    assert dY.ndim == 2 and W.ndim == 3, f"expected (S,N) and (E,N,K), got {dY.shape}, {W.shape}"
    S, N = dY.shape
    values_per_byte = 32 // int_bits if int_bits else 2 if W.dtype == torch.uint8 else 1
    K = W.shape[2] * values_per_byte
    assert W.shape[1] == N, f"weight N {W.shape[1]} != dY's N {N}"
    assert K % scale_group_k == 0, (
        f"K={K} must be a whole number of scale groups ({scale_group_k})"
    )

    num_experts = W.shape[0]
    num_sms = sm_count(dY.device.index)
    rows = S if num_input_rows is None else num_input_rows
    # a gather that is a permutation (one position per input row) has nothing to sum: each position stores at its
    # gather source; otherwise every position stores token-major and the top_k rows are summed below
    store = "position" if gather_idx is None or rows == S else "routed"
    assert store == "position" or scatter_idx is not None or routed_idx is not None, (
        "a gathered launch needs a token-major map: its scatter map or routed_idx"
    )
    assert store == "position" or S % rows == 0, f"{S} positions are not top_k rows of {rows} tokens"
    # zero-initialised: the EP sentinel positions are never stored
    dX = torch.zeros(S, K, device=dY.device, dtype=torch.float32)
    with device_context(dY.device):
        # inside the context — see dgrad_matmul_2d; placeholder boxes, rebound by the pre_hook
        a_descriptor = maybe_descriptor(dY, [1, 128])
        b_descriptor = weight_descriptor(W, [1, 1, 128])
        compile_time_only_triton_wrap(dgrad_matmul_grouped_kernel)[(num_sms,)](
            dY,
            a_descriptor,
            W,
            b_descriptor,
            Ws,
            w_zero,
            w_global_scale,
            global_scale_stride(w_global_scale),
            dX,
            perm,
            gather_idx,
            scatter_idx,
            routed_idx if store == "routed" and scatter_idx is None else None,
            expert_start,
            S,
            N,
            K,
            max(int(max(S // max(num_experts, 1), 1)).bit_length(), 1),
            dY.stride(0),
            dY.stride(1),
            W.stride(0),
            W.stride(1),
            W.stride(2),
            *_scale_strides(Ws, 3),
            *_scale_strides(w_zero, 3),
            perm.stride(0) if perm is not None else 0,
            dX.stride(0),
            dX.stride(1),
            SCALE_GROUP_K=scale_group_k,
            SCALE_ROW_DIV=scale_row_div,
            WEIGHT_VALUES_PER_BYTE=values_per_byte,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            STORE=store,
            INT_BITS=int_bits,
        )
    if store == "routed":
        dX = dX.view(rows, S // rows, K).sum(1)
    return dX.to(output_dtype or dY.dtype)


@triton.jit
def _glu_grad(g, u, dh, ACT_FN: tl.constexpr, SWIGLU_ALPHA: tl.constexpr,
              SWIGLU_LIMIT: tl.constexpr):
    """(dgate, dup) for one tile, mirroring ``epilogue.glu`` arm for arm.

    The alpha arm is ``(u + 1) * g * sigmoid(ALPHA * g)`` — the ``+1`` is load-bearing and easy to
    drop: without it the gate gradient is wrong on exactly the GPT-OSS / MiniMax path and nowhere
    else. ``SWIGLU_LIMIT`` clamps gate ABOVE and up to ``[-LIMIT, LIMIT]``, so the gradient is zero
    wherever the forward saturated. Single return; only the taken arms compile."""
    gc = tl.minimum(g, SWIGLU_LIMIT) if SWIGLU_LIMIT is not None else g
    uc = (
        tl.minimum(tl.maximum(u, -SWIGLU_LIMIT), SWIGLU_LIMIT)
        if SWIGLU_LIMIT is not None
        else u
    )
    if SWIGLU_ALPHA is not None:
        sig = tl.sigmoid(gc * SWIGLU_ALPHA)
        act = gc * sig
        dact = sig + gc * SWIGLU_ALPHA * sig * (1.0 - sig)
        dg = dh * (uc + 1.0) * dact
    elif ACT_FN == "silu":
        sig = tl.sigmoid(gc)
        act = gc * sig
        dg = dh * uc * (sig + gc * sig * (1.0 - sig))
    elif ACT_FN == "relu":
        act = tl.maximum(gc, 0.0)
        dg = dh * uc * (gc > 0.0).to(tl.float32)
    else:  # gelu, exact via erf — the forward's form
        cdf = 0.5 * (1.0 + tl.erf(gc * 0.7071067811865476))
        act = gc * cdf
        dg = dh * uc * (cdf + gc * tl.exp(-0.5 * gc * gc) * 0.3989422804014327)
    du = dh * act
    if SWIGLU_LIMIT is not None:  # saturated inputs pass no gradient
        dg = tl.where(g > SWIGLU_LIMIT, 0.0, dg)
        du = tl.where(tl.abs(u) > SWIGLU_LIMIT, 0.0, du)
    return dg, du


@triton.jit
def _glu_backward_kernel(
    Z,  # (M, 2*INTER) pre-activation, gate at even columns / up at odd
    DH,  # (M, INTER) gradient w.r.t. the GLU output
    DZ,  # (M, 2*INTER) output, interleaved to match Z
    M, INTER,
    stride_z_m, stride_z_n, stride_dh_m, stride_dh_n, stride_dz_m, stride_dz_n,
    ACT_FN: tl.constexpr, SWIGLU_ALPHA: tl.constexpr, SWIGLU_LIMIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    """Elementwise GLU backward over the interleaved layout. Gate ``j`` sits at column ``2j`` and
    up at ``2j+1``, so one tile reads both halves with a stride-2 pair of loads and writes them
    back the same way — no split, no concatenate, no materialized halves."""
    offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_i = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < INTER)

    z_gate = Z + offs_m[:, None] * stride_z_m + (2 * offs_i[None, :]) * stride_z_n
    g = tl.load(z_gate, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(z_gate + stride_z_n, mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(
        DH + offs_m[:, None] * stride_dh_m + offs_i[None, :] * stride_dh_n,
        mask=mask, other=0.0,
    ).to(tl.float32)

    dg, du = _glu_grad(g, u, dh, ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT)

    dz_gate = DZ + offs_m[:, None] * stride_dz_m + (2 * offs_i[None, :]) * stride_dz_n
    tl.store(dz_gate, dg.to(DZ.dtype.element_ty), mask=mask)
    tl.store(dz_gate + stride_dz_n, du.to(DZ.dtype.element_ty), mask=mask)


def glu_backward(
    dH: torch.Tensor,
    Z: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Gradient of the fused GLU epilogue: ``dH`` (grad w.r.t. ``H = f(Z_gate) * Z_up``) and the
    2I-wide pre-activation ``Z`` -> ``dZ``, interleaved.

    One pass over Z, no materialized halves. The GEMM that consumes ``dZ`` is then an UNGATED
    product at the doubled N extent — the forward kernel with no gate flag, because
    ``pid_n * 2BN == 2 * pid_n * BN`` — so no gated backward kernel exists or is needed."""
    assert dH.ndim == 2 and Z.ndim == 2, f"expected 2D, got {dH.shape}, {Z.shape}"
    M, INTER = dH.shape
    assert Z.shape == (M, 2 * INTER), f"Z must be (M, 2I) = {(M, 2 * INTER)}, got {tuple(Z.shape)}"
    dZ = torch.empty_like(Z)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(INTER, META["BLOCK_SIZE_I"]),
    )
    with device_context(Z.device):
        compile_time_only_triton_wrap(_glu_backward_kernel)[grid](
            Z, dH, dZ, M, INTER,
            Z.stride(0), Z.stride(1), dH.stride(0), dH.stride(1), dZ.stride(0), dZ.stride(1),
            ACT_FN=act_fn, SWIGLU_ALPHA=swiglu_alpha, SWIGLU_LIMIT=swiglu_limit,
            BLOCK_SIZE_M=32, BLOCK_SIZE_I=64, num_warps=4,
        )
    return dZ


@bayesian_autotune(
    # Neither a BLOCK_SIZE_M nor a WARP_SPEC axis. One row per program (the rows are not
    # expert-sorted, so a tile cannot span them) and the reduce is FMA rather than MMA, so there is
    # no M tile to choose — and WS, which needs ``(BN if swapped else BM) >= 64`` of M work, has no
    # M operand to satisfy it. Emitting either axis would search configs that cannot compile.
    get_accelerator_autotuning_configs(tune_block_nk=True),
    ["N", "K", "S"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    finite_check_args=("C",),
    # No pruners. The smem model is fit to MMA loops — ``num_stages`` buffers of an A tile and a B
    # tile, keyed on BLOCK_SIZE_M — and this loop has neither: the activation is a ``[BN]`` vector
    # and the reduce is FMA, so there is no M dimension for it to reason about. A config that
    # genuinely will not fit raises OutOfResources, which the tuner already inf-forgives.
)
@triton.jit
def dgrad_matmul_batched_kernel(
    A,  # (S, N) upstream gradient
    B,  # (E, N, K) quantized weights, K-contiguous — the forward orientation
    Bs,  # (E, N // SCALE_ROW_DIV, K // SCALE_GROUP_K) weight scales
    Bz,  # (E, N, K // SCALE_GROUP_K) integer zero points; None for a symmetric or non-integer weight
    BsGlobal,  # fp32 NVFP4 / W4A8 second-level weight scale, per tensor or per expert; folded on the accumulator
    stride_bs_global,  # expert stride of BsGlobal (0 broadcasts a scalar)
    C,  # (S, K) output gradient
    Perm,  # (E, K) act-order K orders: column j of an expert's product is stored at Perm[e, j]; None = natural
    ExpertIds,  # (S,) which expert each row routed to
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_n,
    stride_b_e,
    stride_b_n,
    stride_b_k,
    stride_bs_e,
    stride_bs_n,
    stride_bs_k,
    stride_bz_e,
    stride_bz_n,
    stride_bz_k,
    stride_perm_e,
    stride_c_m,
    stride_c_k,
    # Format
    SCALE_GROUP_K: tl.constexpr,
    SCALE_ROW_DIV: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    INT_BITS: tl.constexpr = 0,  # integer weight width; 0: E2M1 / E4M3 / full-precision weight
):
    """Batched dgrad: ``dX[s] = dY[s] @ B[ExpertIds[s]]``.

    Unlike the grouped form the rows are NOT expert-sorted — each row names its own expert — so a
    tile must cover one expert. ``BLOCK_SIZE_M`` is pinned to 1 by the wrapper for that reason,
    which is also the shape the batched forward runs at."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_ok = pid_m < S
    expert_id = tl.load(ExpertIds + pid_m, mask=row_ok, other=0).to(tl.int64)

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.minimum(
        pid_k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
        + tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE),
        K // WEIGHT_VALUES_PER_BYTE - 1,
    )
    offs_kg = _dgrad_scale_cols(pid_k, K, BLOCK_SIZE_K, SCALE_GROUP_K)
    b_base = B + expert_id * stride_b_e

    # CUDA-core FMA reduce, NOT tl.dot: one row per program is an M=1 GEVM, and feeding the MMA a
    # broadcast [1, BN] operand runs it emulated at a fraction of peak. The forward reaches the same
    # conclusion for its decode path (``mx_weight_only_scalar_swapped``) — the scalar arm wins at
    # M=1 and the tuner picks it there. Same reduction, no MMA, no padded M dimension.
    accumulator = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    for n in tl.range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dy = tl.load(
            A + pid_m * stride_a_m + offs_n * stride_a_n,
            mask=row_ok & (offs_n < N),
            other=0.0,
        ).to(tl.float32)
        offs_nc = tl.minimum(offs_n, N - 1)  # clamp: dy is already zeroed past N
        w = tl.load(b_base + offs_nc[:, None] * stride_b_n + offs_kb[None, :] * stride_b_k)
        w_scale = tl.load(
            Bs + expert_id * stride_bs_e + (offs_nc[:, None] // SCALE_ROW_DIV) * stride_bs_n + offs_kg[None, :] * stride_bs_k
        ) if Bs is not None else None
        if INT_BITS:
            w_zero = tl.load(
                Bz + expert_id * stride_bz_e + offs_nc[:, None] * stride_bz_n + offs_kg[None, :] * stride_bz_k
            ) if Bz is not None else None
            divisor = tl.load(BsGlobal + expert_id * stride_bs_global) if BsGlobal is not None else None
            wdeq = _dgrad_int_tile(
                w, w_scale, w_zero, divisor, BLOCK_SIZE_K, SCALE_GROUP_K, INT_BITS, A.dtype.element_ty
            ).to(tl.float32)
        else:
            wdeq = _dgrad_weight_tile(w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, tl.float32)
        accumulator += tl.sum(dy[:, None] * wdeq, axis=0)
    if BsGlobal is not None:  # the second-level scale, this row's expert
        accumulator = accumulator * tl.load(BsGlobal + expert_id * stride_bs_global)
    k_mask = row_ok & (offs_k < K)
    cols = tl.load(Perm + expert_id * stride_perm_e + offs_k, mask=k_mask, other=0) if Perm is not None else offs_k
    tl.store(C + pid_m * stride_c_m + cols * stride_c_k, accumulator.to(C.dtype.element_ty), mask=k_mask)


@compile_time_only_triton_op(add_op_namespace_prefix("dgrad_matmul_batched"), mutates_args=())
def dgrad_matmul_batched(
    dY: torch.Tensor,
    W: torch.Tensor,
    Ws: torch.Tensor,
    expert_ids: torch.Tensor,
    scale_group_k: int,
    scale_row_div: int,
    output_dtype: torch.dtype | None = None,
    w_global_scale: torch.Tensor | None = None,
    w_zero: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    int_bits: int = 0,
) -> torch.Tensor:
    """``dX[s] = dY[s] @ W[expert_ids[s]]`` — the batched (per-row expert) counterpart of
    ``dgrad_matmul_grouped``. Registered because the batched experts forward is a SELECTABLE
    dispatch in transformers, not a decode-only path: without this, finetuning against it
    silently produces no gradient."""
    assert dY.ndim == 2 and W.ndim == 3, f"expected (S,N) and (E,N,K), got {dY.shape}, {W.shape}"
    S, N = dY.shape
    values_per_byte = 32 // int_bits if int_bits else 2 if W.dtype == torch.uint8 else 1
    K = W.shape[2] * values_per_byte
    assert W.shape[1] == N, f"weight N {W.shape[1]} != dY's N {N}"

    dX = torch.empty(S, K, device=dY.device, dtype=output_dtype or dY.dtype)
    grid = lambda META: (S, triton.cdiv(K, META["BLOCK_SIZE_K"]))  # noqa: E731
    with device_context(dY.device):
        compile_time_only_triton_wrap(dgrad_matmul_batched_kernel)[grid](
            dY, W, Ws, w_zero, w_global_scale, global_scale_stride(w_global_scale), dX, perm, expert_ids, S, N, K,
            dY.stride(0), dY.stride(1),
            W.stride(0), W.stride(1), W.stride(2),
            *_scale_strides(Ws, 3),
            *_scale_strides(w_zero, 3),
            perm.stride(0) if perm is not None else 0,
            dX.stride(0), dX.stride(1),
            SCALE_GROUP_K=scale_group_k,
            SCALE_ROW_DIV=scale_row_div,
            WEIGHT_VALUES_PER_BYTE=values_per_byte,
            INT_BITS=int_bits,
        )
    return dX



# ── gradient products, per format family ─────────────────────────────────────────
# What each format's dgrad actually computes. ``backward.py`` holds only the table saying which
# op gets which of these — compute lives here, attachment lives there.

def _scale_strides(Ws, ndim):
    """The scale grid's strides, zeros for a full-precision weight (no grid)."""
    return Ws.stride() if Ws is not None else (0,) * ndim


def _dgrad_operands(W, Ws):
    """``(W, Ws, scale_group_k, scale_row_div)`` as the dgrad kernels read them:
    - packed E2M1 and UE8M0 as ``uint8`` bytes (UE8M0 decoded as floats is off by orders of magnitude);
    - a SWIZZLE_32_4_4 artifact back on the affine grid (group 16 for NVFP4's E4M3 scales, else 32);
    - one scale per tensor, expert or output row as a stride-0 view of the group-32 grid it equals;
    - the divisors read off the grid's shape, so MX group-32 and NVFP4 group-16 need no naming here;
    - a full-precision weight with no grid, its divisors only satisfying the tile rules."""
    if Ws is None:
        return W, None, 32, 1
    values_per_byte = 2 if W.dtype in (torch.int8, torch.uint8) else 1
    grid = (*W.shape[:-1], W.shape[-1] * values_per_byte // 32)
    if Ws.numel() == 1:
        Ws = Ws.reshape([1] * len(grid)).expand(grid)
    elif W.ndim == 3 and Ws.numel() == W.shape[0]:
        Ws = Ws.reshape(-1, 1, 1).expand(grid)
    elif Ws.shape[-1] == 1:  # row-wise
        Ws = Ws.reshape(*W.shape[:-1], 1).expand(grid)
    elif Ws.ndim == 5:
        group = 16 if Ws.dtype == torch.float8_e4m3fn else 32
        Ws = unswizzle_mx_scales(Ws, W.shape[-2], W.shape[-1] * values_per_byte // group,
                                 num_experts=W.shape[0] if W.ndim == 3 else None)
    if W.dtype == torch.int8:
        W = W.view(torch.uint8)
    Ws = ue8m0_as_uint8(Ws)
    return W, Ws, W.shape[-1] * values_per_byte // Ws.shape[-1], max(W.shape[-2] // Ws.shape[-2], 1)


def _token_sum(routed_rows, ctx):
    """Each token's gradient from its token-major routed rows ``(T * top_k, K)``: a contiguous fp32 top_k sum, no
    atomics."""
    return routed_rows.view(ctx.a_rows, -1, routed_rows.shape[1]).sum(1, dtype=torch.float32).to(ctx.a_dtype)


def _int_dgrad_2d(ctx, dY, B, Bs, b_zero=None, b_perm=None, b_scale2=None):
    """Integer dA on the dgrad kernel: the interleaved words decoded in registers to the value the forward multiplies
    by (zero points, group or per-channel scales, ModelOpt W4A8's E4M3 rounding against ``b_scale2``), the act-order
    columns stored back through ``b_perm``."""
    return dgrad_matmul_2d(
        dY.contiguous(), B, Bs, ctx.a_cols // Bs.shape[-1], 1, output_dtype=ctx.a_dtype,
        **_int_operands(ctx, B, Bs, b_zero, b_perm, b_scale2, 1),
    )


def _int_operands(ctx, B, Bs, b_zero, b_perm, b_scale2, num_experts):
    """The dgrad kernels' integer kwargs: zero points, act-order, the W4A8 second level and the bit width (read off
    the word count against the input's K)."""
    return dict(
        w_zero=b_zero, perm=b_perm, w_global_scale=normalize_global_scale(b_scale2, num_experts),
        int_bits=32 // (ctx.a_cols // B.shape[-1]),
    )


def _int_grouped_dgrad(ctx, dY, B, expert_start, Bs, gather_idx=None, scatter_idx=None, routed_idx=None,
                       b_zero=None, b_perm=None, b_scale2=None):
    """Integer routed dA on the grouped dgrad kernel (``_int_dgrad_2d``'s decode, per expert)."""
    return dgrad_matmul_grouped(
        dY.contiguous(), B, Bs, expert_start, ctx.a_cols // Bs.shape[-1], 1, gather_idx=gather_idx,
        scatter_idx=scatter_idx, output_dtype=ctx.a_dtype, num_input_rows=ctx.a_rows, routed_idx=routed_idx,
        **_int_operands(ctx, B, Bs, b_zero, b_perm, b_scale2, B.shape[0]),
    )


def _int_batched_dgrad(ctx, dY, B, expert_ids, Bs, scatter_idx=None, b_zero=None, b_perm=None, b_scale2=None):
    """Integer per-row-expert dA on the batched dgrad kernel, a gate|up launch's top_k rows summed per token."""
    summed = expert_ids.shape[0] != ctx.a_rows
    if scatter_idx is not None:
        dY = dY.index_select(0, scatter_idx)
    rows = dgrad_matmul_batched(
        dY.contiguous(), B, Bs, expert_ids, ctx.a_cols // Bs.shape[-1], 1,
        output_dtype=torch.float32 if summed else ctx.a_dtype,
        **_int_operands(ctx, B, Bs, b_zero, b_perm, b_scale2, B.shape[0]),
    )
    return _token_sum(rows, ctx) if summed else rows


def _dgrad_2d(ctx, dY, B, Bs, b_global_scale=None):
    """dA on the dgrad kernel (``dgrad_matmul_2d``): contract N against the forward-oriented tile decoded in registers,
    the NVFP4 second-level global folded on the accumulator."""
    W, Ws, group, row_div = _dgrad_operands(B, Bs)
    return dgrad_matmul_2d(
        dY.contiguous(), W, Ws, group, row_div, output_dtype=ctx.a_dtype,
        w_global_scale=normalize_global_scale(b_global_scale, 1),
    )


def _dgrad_transposed_weight(ctx, dY, B, Bs):
    """dA on the quantized weight reoriented by transposition: a per-tensor scalar is orientation-free and a block
    grid subdivides both axes, so transposing the grid is the tile's own reindexing. The grid is made contiguous
    (strided, every tile's re-read cost 79 us of a 4096x8192x6144 dgrad for a ~3 KB copy); the weight stays a view."""
    return matmul_2d(dY.contiguous(), B.t(), None, Bs if Bs.numel() == 1 else Bs.t().contiguous(), output_dtype=ctx.a_dtype)


def _dgrad_full_precision_2d(ctx, dY, B):
    """dA for an unquantized weight: ``dY @ B``."""
    return (dY.to(B.dtype) @ B).to(ctx.a_dtype)


def _grouped_dgrad(ctx, dY, B, expert_start, Bs=None, gather_idx=None, scatter_idx=None, routed_idx=None,
                   b_global_scale=None):
    """Routed dA for every format: the grouped dgrad kernel contracts N against the forward-oriented weight tile,
    decoded in registers, and turns the forward's maps around itself."""
    W, Ws, group, row_div = _dgrad_operands(B, Bs)
    return dgrad_matmul_grouped(
        dY.contiguous(), W, Ws, expert_start, group, row_div, gather_idx=gather_idx, scatter_idx=scatter_idx,
        output_dtype=ctx.a_dtype, num_input_rows=ctx.a_rows,
        w_global_scale=normalize_global_scale(b_global_scale, W.shape[0]), routed_idx=routed_idx,
    )


def _batched_dgrad(ctx, dY, B, expert_ids, Bs=None, scatter_idx=None, b_global_scale=None):
    """Per-row-expert dA, one GEVM per routed row (``dgrad_matmul_batched``) on the forward-oriented weight tile:
    routed row ``r`` reads ``dY`` at its forward destination, and a launch reading one row per token (the gate|up on
    the unexpanded hidden) sums each token's top_k rows."""
    summed = expert_ids.shape[0] != ctx.a_rows
    W, Ws, group, row_div = _dgrad_operands(B, Bs)
    if scatter_idx is not None:
        dY = dY.index_select(0, scatter_idx)
    rows = dgrad_matmul_batched(
        dY.contiguous(), W, Ws, expert_ids, group, row_div, output_dtype=torch.float32 if summed else ctx.a_dtype,
        w_global_scale=normalize_global_scale(b_global_scale, W.shape[0]),
    )
    return _token_sum(rows, ctx) if summed else rows


def _with_input_transform(dgrad):
    """``dgrad`` for an op that may transform its input before the GEMM:
    - ``a_pre_quant_scale`` (AWQ / ModelOpt smoothing, ``A * s`` before quantizing): ``dA = dgrad ⊙ s``; a per-expert
      scale would apply per routed row before the token sum, which is not implemented;
    - SVDQuant's low-rank correction: an inference-only recipe, refused."""

    def wrapped(ctx, dY, a_pre_quant_scale=None, svdquant_lora_a=None, **kwargs):
        if svdquant_lora_a is not None:
            raise NotImplementedError("no backward for SVDQuant: it is an inference-only recipe")
        dA = dgrad(ctx, dY, **kwargs)
        if a_pre_quant_scale is None:
            return dA
        if a_pre_quant_scale.numel() != dA.shape[-1]:
            raise NotImplementedError("no backward for a per-expert pre-quant scale")
        return (dA.float() * a_pre_quant_scale.float().reshape(-1)).to(dA.dtype)

    return wrapped


__all__: list[str] = []


def _register(op, reads, dgrad):
    """Attach ``dgrad`` to ``op``. It receives ``ctx``, the output gradient and the ``reads`` parameters by name, and
    returns the activation's gradient; weights and scales take none (a quantized tensor cannot be an autograd leaf).
    A fused GLU (``gate`` with an ``act_fn``) is refused: its gradient needs the pre-activation."""
    names = list(inspect.signature(op).parameters)

    def setup_context(ctx, inputs, output):
        bound = dict(zip(names, inputs))
        ctx.fused_glu = bool(bound.get("gate")) and bound.get("act_fn") is not None
        values = {name: bound.get(name) for name in reads}
        ctx.tensor_names = [name for name, value in values.items() if isinstance(value, torch.Tensor)]
        ctx.save_for_backward(*(values[name] for name in ctx.tensor_names))
        ctx.config = {name: value for name, value in values.items() if name not in ctx.tensor_names}
        # a gathered launch's rows sum back to its source rows, a different count from the routed rows it consumed
        ctx.a_dtype, ctx.a_rows, ctx.a_cols = inputs[0].dtype, inputs[0].shape[0], inputs[0].shape[-1]

    def backward(ctx, *grads):
        # `needs_input_grad` is sized to the args as invoked plus torch's trailing metadata slot
        n_grads = len(ctx.needs_input_grad)
        if not ctx.needs_input_grad[0]:
            return (None,) * n_grads
        if ctx.fused_glu:
            raise NotImplementedError(f"{op.__name__}: no backward through a fused GLU; call with act_fn=None")
        dY = grads[0]
        while isinstance(dY, (list, tuple)):  # the ops return list[Tensor]
            dY = dY[0]
        saved = dict(zip(ctx.tensor_names, ctx.saved_tensors))
        return (dgrad(ctx, dY, **saved, **ctx.config),) + (None,) * (n_grads - 1)

    torch.library.register_autograd(add_op_namespace_prefix(op.__name__), backward, setup_context=setup_context)


_GROUPED = ("B", "Bs", "expert_start", "gather_idx", "scatter_idx", "routed_idx")
# 2D. Block FP8: the transposed weight view on the forward GEMM (a block grid subdivides both axes). Per-tensor /
# row-wise FP8, MX / NVFP4 and weight-only (the QLoRA base): the dgrad kernel contracts N against the forward-oriented
# tile, so a scale bound to the output rows or to K-groups decodes as stored.
for _op in (matmul.w8a8_block_dynamic_fp8_matmul, matmul.w8a8_block_static_fp8_matmul):
    _register(_op, ("B", "Bs"), _dgrad_transposed_weight)
_register(matmul.w8a8_tensor_dynamic_fp8_matmul, ("B", "Bs"), _dgrad_2d)
_register(matmul.mx_dynamic_matmul, ("B", "Bs", "b_global_scale", "a_pre_quant_scale", "svdquant_lora_a"),
          _with_input_transform(_dgrad_2d))
_register(matmul.mx_weight_only_matmul_2d, ("B", "Bs", "b_global_scale"), _dgrad_2d)
_register(matmul.block_fp8_weight_only_matmul_2d, ("B", "Bs"), _dgrad_2d)
_register(matmul.int_weight_only_matmul_2d, ("B", "Bs", "b_zero", "b_perm", "b_scale2", "a_pre_quant_scale"),
          _with_input_transform(_int_dgrad_2d))
_register(matmul.full_precision_matmul_2d, ("B",), _dgrad_full_precision_2d)
# grouped: every format on the grouped dgrad kernel
for _op in (grouped.w8a8_block_dynamic_fp8_matmul_grouped, grouped.w8a8_block_static_fp8_matmul_grouped,
            grouped.w8a8_tensor_dynamic_fp8_matmul_grouped, grouped.full_precision_matmul_grouped):
    _register(_op, _GROUPED, _grouped_dgrad)
_register(grouped.mx_dynamic_matmul_grouped, (*_GROUPED, "b_global_scale", "a_pre_quant_scale"),
          _with_input_transform(_grouped_dgrad))
_register(grouped.mx_weight_only_matmul_grouped, (*_GROUPED, "b_global_scale"), _grouped_dgrad)
_register(grouped.block_fp8_weight_only_matmul_grouped, _GROUPED, _grouped_dgrad)
_INT = ("b_zero", "b_perm", "b_scale2", "a_pre_quant_scale")
_register(grouped.int_weight_only_matmul_grouped, (*_GROUPED, *_INT), _with_input_transform(_int_grouped_dgrad))
# batched: the batched experts forward is a selectable dispatch in transformers, not a decode-only path
_BATCHED = ("B", "Bs", "expert_ids", "scatter_idx", "b_global_scale")
for _op in (batched.w8a8_block_dynamic_fp8_matmul_batched, batched.w8a8_block_static_fp8_matmul_batched,
            batched.w8a8_tensor_dynamic_fp8_matmul_batched, batched.mx_weight_only_matmul_batched,
            batched.block_fp8_weight_only_matmul_batched, batched.full_precision_matmul_batched):
    _register(_op, _BATCHED, _batched_dgrad)
_register(batched.mx_dynamic_matmul_batched, (*_BATCHED, "a_pre_quant_scale"), _with_input_transform(_batched_dgrad))
_register(batched.int_weight_only_matmul_batched, ("B", "Bs", "expert_ids", "scatter_idx", *_INT),
          _with_input_transform(_int_batched_dgrad))
