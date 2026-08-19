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

Block-FP8 and per-tensor recipes reach that product on the forward kernel via transposed views
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

Recipe genericity lives in ``_dgrad_weight_tile`` alone. A recipe is fully described to this
kernel by three numbers plus the value dtype:

    SCALE_GROUP_K    scale extent along K   32 (MX)   128 (block-FP8)
    SCALE_ROW_DIV    rows sharing a scale    1 (MX)   128 (block-FP8)
    WEIGHT_VALUES_PER_BYTE  2 for packed E2M1, else 1

so MX (per-row K-groups) and block-FP8 (128x128 blocks) are the same code path at different
divisors, and no new arm is needed to add a recipe whose scales tile the same way.
"""

import torch
import triton
import triton.language as tl

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune
from .compat import (
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    device_context,
    get_accelerator_autotuning_configs,
    sm_count,
)
from .matmul import _maybe_descriptor, _rebind_operand_box, matmul_2d
from .pruners import (
    PATH_ANCHOR_AXES,
    block_within_dim_pruner,
    config_filter,
    descriptor_box_pruner,
    compose_pruners,
    smem_pruner,
    warp_spec_compile_guard_pruner,
)
from .quant import e2m1_cols_to_bf16, e2m1_cols_to_f32
from .scales import decode_group_scale
from .tiles import load_grouped_act_tile, operand_tile_ptrs
from .scheduling import resolve_grouped_tile
from .tile_layout import build_tile_layout


# Internal: these are the gradient PRODUCTS, not API. Backward hangs off the forward ops via
# ``register_autograd`` (the table at the bottom of this file), so an ordinary forward call
# differentiates and there is
# no second entry point to keep in sync. Nothing here is re-exported from the package root.


def _rebind_dgrad_descriptors(nargs):
    """Per-config pre_hook: set the dgrad boxes to the tuned tile — ``[BM, BN]`` over the (M, N)
    gradient and ``[BN, BK // WEIGHT_VALUES_PER_BYTE]`` over the (N, K_bytes) weight. Both are
    natural contiguous windows in the FORWARD-oriented operands, which is why dgrad can take a
    descriptor at all: a transposed view has no unit-stride innermost dim (see
    ``_maybe_descriptor``). Scales stay affine on the pointer arm, as weight-only does."""
    wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
    _rebind_operand_box(
        nargs, "A_MEMORY_MODE", "ADescriptor", nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"]
    )
    _rebind_operand_box(
        nargs, "B_MEMORY_MODE", "BDescriptor",
        nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"] // wvpb,
    )


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
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    if OUT_DTYPE == tl.float32:  # scalar-reduce arm: unpack straight to fp32, no narrow hop
        vals = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)
        sc = decode_group_scale(w_scale).to(tl.float32)[:, :, None]
    else:  # MMA arm: dequantize in bf16, exactly as the forward's weight upcast does
        vals = e2m1_cols_to_bf16(w) if w.dtype == tl.uint8 else w.to(tl.bfloat16)
        sc = decode_group_scale(w_scale).to(tl.bfloat16)[:, :, None]
    tile = vals.reshape(BLOCK_SIZE_N, NG, SCALE_GROUP_K) * sc
    return tile.reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).to(OUT_DTYPE)


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
        pre_hook=_rebind_dgrad_descriptors,
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
            warp_spec_compile_guard_pruner(),
            descriptor_box_pruner("BLOCK_SIZE_K"),
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
    C,  # (M, K) output gradient
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
    stride_c_m,
    stride_c_k,
    # Recipe
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
    offs_kg = tl.minimum(
        (pid_k * BLOCK_SIZE_K) // SCALE_GROUP_K + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K),
        K // SCALE_GROUP_K - 1,
    )
    m_mask = offs_m < M
    k_mask = offs_k < K

    # BLOCK_SIZE_N divides N (block_within_dim_pruner), so the reduction loop is EXACT: no tail
    # mask on the activation, no clamp on the weight rows, and the addresses advance by a constant
    # stride instead of being rebuilt from offsets each step — the forward's loop shape.
    offs_n0 = tl.arange(0, BLOCK_SIZE_N)
    dy_ptrs = operand_tile_ptrs(A, offs_m, offs_n0, stride_a_m, stride_a_n, A_MEMORY_MODE, False)
    b_ptrs = operand_tile_ptrs(B, offs_n0, offs_kb, stride_b_n, stride_b_k, B_MEMORY_MODE, False)
    bs_ptrs = (
        Bs + (offs_n0[:, None] // SCALE_ROW_DIV) * stride_bs_n + offs_kg[None, :] * stride_bs_k
    )

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
        w_scale = tl.load(bs_ptrs)
        accumulator += tl.dot(
            dy.to(tl.bfloat16),
            _dgrad_weight_tile(
                w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, tl.bfloat16
            ),
        )
        dy_ptrs += BLOCK_SIZE_N * stride_a_n
        b_ptrs += BLOCK_SIZE_N * stride_b_n
        bs_ptrs += (BLOCK_SIZE_N // SCALE_ROW_DIV) * stride_bs_n

    tl.store(
        C + offs_m[:, None] * stride_c_m + offs_k[None, :] * stride_c_k,
        accumulator.to(C.dtype.element_ty),
        mask=m_mask[:, None] & k_mask[None, :],
    )


@compile_time_only_triton_op("finegrained::dgrad_matmul_2d", mutates_args=())
def dgrad_matmul_2d(
    dY: torch.Tensor,
    W: torch.Tensor,
    Ws: torch.Tensor,
    scale_group_k: int,
    scale_row_div: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``dX = dY @ W`` against the FORWARD-oriented quantized weight — the gradient a frozen
    quantized base weight passes to whatever trains in front of it (LoRA adapters, earlier
    layers).

    ``W`` is ``(N, K)`` exactly as the forward holds it; nothing is transposed or re-quantized.
    ``scale_group_k``/``scale_row_div`` describe the scale grid (MX: ``32, 1``; block-FP8:
    ``128, 128``), which is the only thing that differs between recipes."""
    assert dY.ndim == 2 and W.ndim == 2, f"expected 2D operands, got {dY.ndim}D and {W.ndim}D"
    M, N = dY.shape
    values_per_byte = 2 if W.dtype == torch.uint8 else 1
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
        a_descriptor = _maybe_descriptor(dY, [1, 128])
        b_descriptor = _maybe_descriptor(W, [1, 128])
        compile_time_only_triton_wrap(dgrad_matmul_2d_kernel)[grid](
            dY,
            a_descriptor,
            W,
            b_descriptor,
            Ws,
            dX,
            M,
            N,
            K,
            max(int(M).bit_length(), 1),
            dY.stride(0),
            dY.stride(1),
            W.stride(0),
            W.stride(1),
            Ws.stride(0),
            Ws.stride(1),
            dX.stride(0),
            dX.stride(1),
            SCALE_GROUP_K=scale_group_k,
            SCALE_ROW_DIV=scale_row_div,
            WEIGHT_VALUES_PER_BYTE=values_per_byte,
        )
    return dX


def _rebind_dgrad_grouped_descriptors(nargs):
    """Grouped dgrad boxes. The activation box is 1 row when the pass gathers (tma gather4 needs a
    1-row box) and ``[BM, BN]`` otherwise — ScatterIdx is the gather map here, since dgrad reads
    dY at the forward's SCATTER destination. The weight box is ``[1, BN, BK_bytes]`` over
    (E, N, K_bytes): the expert leads, so this descriptor takes THREE offsets."""
    wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
    gathering = nargs.get("ScatterIdx") is not None
    _rebind_operand_box(
        nargs, "A_MEMORY_MODE", "ADescriptor",
        1 if gathering else nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"],
    )
    desc = nargs["BDescriptor"]
    if nargs.get("B_MEMORY_MODE", "pointer") != "pointer" and not isinstance(desc, int) and desc is not None:
        desc.block_shape = [1, nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"] // wvpb]


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
        pre_hook=_rebind_dgrad_grouped_descriptors,
    ),
    ["N", "K", "tokens_per_expert_bit_length"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    finite_check_args=("C",),
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("N"),  # exact N-loop: no tail mask, constant-stride advance
            config_filter(  # the scale row advances by BN // SCALE_ROW_DIV — must be whole
                lambda c, a: c.kwargs["BLOCK_SIZE_N"] % max(a.get("SCALE_ROW_DIV", 1), 1) == 0
            ),
            warp_spec_compile_guard_pruner(),
            descriptor_box_pruner("BLOCK_SIZE_K"),
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
    C,  # (S, K) output gradient, written in the forward's INPUT row order
    GatherIdx,  # (S,) sorted position -> source row of the forward's A
    ScatterIdx,  # (S,) sorted position -> destination row of the forward's C
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
    stride_c_m,
    stride_c_k,
    # Recipe
    SCALE_GROUP_K: tl.constexpr,
    SCALE_ROW_DIV: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
    # Meta-parameters
    A_MEMORY_MODE: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WARP_SPEC: tl.constexpr,
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
        offs_kg = tl.minimum(
            (pid_k * BLOCK_SIZE_K) // SCALE_GROUP_K + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K),
            K // SCALE_GROUP_K - 1,
        )
        b_base = B + expert_id64 * stride_b_e
        bs_base = Bs + expert_id64 * stride_bs_e

        # exact N-loop (BLOCK_SIZE_N divides N): maskless weight, constant-stride advance
        offs_n0 = tl.arange(0, BLOCK_SIZE_N)
        dy_ptrs = operand_tile_ptrs(
            A, out_row, offs_n0, stride_a_m, stride_a_n, A_MEMORY_MODE, False
        )
        b_ptrs = operand_tile_ptrs(
            b_base, offs_n0, offs_kb, stride_b_n, stride_b_k, B_MEMORY_MODE, False
        )
        bs_ptrs = (
            bs_base
            + (offs_n0[:, None] // SCALE_ROW_DIV) * stride_bs_n
            + offs_kg[None, :] * stride_bs_k
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for n in tl.range(0, tl.cdiv(N, BLOCK_SIZE_N), warp_specialize=WARP_SPEC):
            n_off = n * BLOCK_SIZE_N
            # the forward gathered A at in_row and scattered C to out_row, so its gradient READS
            # at out_row — ScatterIdx is this pass's gather map
            dy = load_grouped_act_tile(
                dy_ptrs, ADescriptor, m_start, n_off, row_mask, out_row,
                A_MEMORY_MODE, ScatterIdx is not None,
            )
            w = _dgrad_load_weight(
                b_ptrs, BDescriptor, row0, n_off,
                pid_k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE),
                BLOCK_SIZE_N, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE, B_MEMORY_MODE, True,
            )
            w_scale = tl.load(bs_ptrs)
            accumulator += tl.dot(
                dy.to(tl.bfloat16),
                _dgrad_weight_tile(
                    w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, tl.bfloat16
                ),
            )
            dy_ptrs += BLOCK_SIZE_N * stride_a_n
            b_ptrs += BLOCK_SIZE_N * stride_b_n
            bs_ptrs += (BLOCK_SIZE_N // SCALE_ROW_DIV) * stride_bs_n

        tl.store(
            C + in_row[:, None] * stride_c_m + offs_k[None, :] * stride_c_k,
            accumulator.to(C.dtype.element_ty),
            mask=row_mask[:, None] & (offs_k[None, :] < K),
        )


@compile_time_only_triton_op("finegrained::dgrad_matmul_grouped", mutates_args=())
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
) -> torch.Tensor:
    """Routed ``dX = dY @ W[expert]`` over expert-sorted positions — the MoE counterpart of
    ``dgrad_matmul_2d``.

    ``gather_idx`` / ``scatter_idx`` are the FORWARD's maps, passed unchanged: the kernel swaps
    their roles itself (reads ``dY`` at the forward's scatter destination, writes ``dX`` at its
    gather source), so callers hand over the same two tensors the forward got."""
    assert dY.ndim == 2 and W.ndim == 3, f"expected (S,N) and (E,N,K), got {dY.shape}, {W.shape}"
    S, N = dY.shape
    values_per_byte = 2 if W.dtype == torch.uint8 else 1
    K = W.shape[2] * values_per_byte
    assert W.shape[1] == N, f"weight N {W.shape[1]} != dY's N {N}"
    assert K % scale_group_k == 0, (
        f"K={K} must be a whole number of scale groups ({scale_group_k})"
    )

    num_experts = W.shape[0]
    num_sms = sm_count(dY.device.index)
    # every sorted position belongs to exactly one expert and is written by its tile,
    # so there is nothing for a memset to cover
    dX = torch.empty(S, K, device=dY.device, dtype=output_dtype or dY.dtype)
    with device_context(dY.device):
        # inside the context — see dgrad_matmul_2d; placeholder boxes, rebound by the pre_hook
        a_descriptor = _maybe_descriptor(dY, [1, 128])
        b_descriptor = _maybe_descriptor(W, [1, 1, 128])
        compile_time_only_triton_wrap(dgrad_matmul_grouped_kernel)[(num_sms,)](
            dY,
            a_descriptor,
            W,
            b_descriptor,
            Ws,
            dX,
            gather_idx,
            scatter_idx,
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
            Ws.stride(0),
            Ws.stride(1),
            Ws.stride(2),
            dX.stride(0),
            dX.stride(1),
            SCALE_GROUP_K=scale_group_k,
            SCALE_ROW_DIV=scale_row_div,
            WEIGHT_VALUES_PER_BYTE=values_per_byte,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
        )
    return dX


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
    Z,  # (M, 2I) pre-activation, gate at even columns / up at odd
    DH,  # (M, I) gradient w.r.t. the GLU output
    DZ,  # (M, 2I) output, interleaved to match Z
    M, I,
    stride_z_m, stride_z_n, stride_dh_m, stride_dh_n, stride_dz_m, stride_dz_n,
    ACT_FN: tl.constexpr, SWIGLU_ALPHA: tl.constexpr, SWIGLU_LIMIT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    """Elementwise GLU backward over the interleaved layout. Gate ``j`` sits at column ``2j`` and
    up at ``2j+1``, so one tile reads both halves with a stride-2 pair of loads and writes them
    back the same way — no split, no concatenate, no materialized halves."""
    offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_i = tl.program_id(1) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I)

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
    M, I = dH.shape
    assert Z.shape == (M, 2 * I), f"Z must be (M, 2I) = {(M, 2 * I)}, got {tuple(Z.shape)}"
    dZ = torch.empty_like(Z)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(I, META["BLOCK_SIZE_I"]),
    )
    with device_context(Z.device):
        compile_time_only_triton_wrap(_glu_backward_kernel)[grid](
            Z, dH, dZ, M, I,
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
    C,  # (S, K) output gradient
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
    stride_c_m,
    stride_c_k,
    # Recipe
    SCALE_GROUP_K: tl.constexpr,
    SCALE_ROW_DIV: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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
    offs_kg = tl.minimum(
        (pid_k * BLOCK_SIZE_K) // SCALE_GROUP_K + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K),
        K // SCALE_GROUP_K - 1,
    )
    b_base = B + expert_id * stride_b_e
    bs_base = Bs + expert_id * stride_bs_e

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
            bs_base
            + (offs_nc[:, None] // SCALE_ROW_DIV) * stride_bs_n
            + offs_kg[None, :] * stride_bs_k
        )
        wdeq = _dgrad_weight_tile(
            w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, tl.float32
        )
        accumulator += tl.sum(dy[:, None] * wdeq, axis=0)

    tl.store(
        C + pid_m * stride_c_m + offs_k * stride_c_k,
        accumulator.to(C.dtype.element_ty),
        mask=row_ok & (offs_k < K),
    )


@compile_time_only_triton_op("finegrained::dgrad_matmul_batched", mutates_args=())
def dgrad_matmul_batched(
    dY: torch.Tensor,
    W: torch.Tensor,
    Ws: torch.Tensor,
    expert_ids: torch.Tensor,
    scale_group_k: int,
    scale_row_div: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``dX[s] = dY[s] @ W[expert_ids[s]]`` — the batched (per-row expert) counterpart of
    ``dgrad_matmul_grouped``. Registered because the batched experts forward is a SELECTABLE
    dispatch in transformers, not a decode-only path: without this, finetuning against it
    silently produces no gradient."""
    assert dY.ndim == 2 and W.ndim == 3, f"expected (S,N) and (E,N,K), got {dY.shape}, {W.shape}"
    S, N = dY.shape
    values_per_byte = 2 if W.dtype == torch.uint8 else 1
    K = W.shape[2] * values_per_byte
    assert W.shape[1] == N, f"weight N {W.shape[1]} != dY's N {N}"

    dX = torch.empty(S, K, device=dY.device, dtype=output_dtype or dY.dtype)
    grid = lambda META: (S, triton.cdiv(K, META["BLOCK_SIZE_K"]))  # noqa: E731
    with device_context(dY.device):
        compile_time_only_triton_wrap(dgrad_matmul_batched_kernel)[grid](
            dY, W, Ws, dX, expert_ids, S, N, K,
            dY.stride(0), dY.stride(1),
            W.stride(0), W.stride(1), W.stride(2),
            Ws.stride(0), Ws.stride(1), Ws.stride(2),
            dX.stride(0), dX.stride(1),
            SCALE_GROUP_K=scale_group_k,
            SCALE_ROW_DIV=scale_row_div,
            WEIGHT_VALUES_PER_BYTE=values_per_byte,
        )
    return dX



# ── gradient products, per recipe family ─────────────────────────────────────────
# What each recipe's dgrad actually computes. ``autograd.py`` holds only the table saying which
# op gets which of these — compute lives here, attachment lives there.

def _dgrad_mx(dY, B, Bs, b_global, out_dtype):
    """dA for an MX/NVFP4 weight: contract N on the forward-oriented weight (see backward.py).

    The scale grid is DERIVED FROM SHAPES, not from a recipe constant — ``Bs`` is
    ``(N // row_div, K // group)``, so the divisors read straight off it. That covers group-32
    UE8M0 (MX) and group-16 E4M3 (NVFP4) without either being named here, and cannot drift when a
    recipe's block changes.

    Containers are normalized, not reinterpreted: packed E2M1 reaches the kernel as ``int8`` or
    ``uint8`` depending on the producer, and UE8M0 as ``float8_e8m0fnu`` or ``uint8``. The kernel
    keys its unpack/decode off ``uint8``, and decoding UE8M0 bytes as floats is off by orders of
    magnitude — the failure mode is a non-finite dX, which the tuner's numerics veto turns into an
    all-inf tune rather than a wrong answer.

    The NVFP4 second-level global is a scalar on the product, so it folds afterwards."""
    if B.dtype == torch.int8:
        B = B.view(torch.uint8)
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = Bs.view(torch.uint8)
    values_per_byte = 2 if B.dtype == torch.uint8 else 1
    K = B.shape[-1] * values_per_byte
    group = K // Bs.shape[-1]
    row_div = max(B.shape[-2] // Bs.shape[-2], 1)
    dX = dgrad_matmul_2d(
        dY.contiguous(), B, Bs, group, row_div, output_dtype=torch.float32
    )
    if b_global is not None:
        dX = dX * b_global.float().reshape(-1)[0]
    return dX.to(out_dtype)

def _dgrad_transposed_weight(dY, B, Bs, out_dtype):
    """dA on the QUANTIZED weight, reoriented by transposition. Valid when the scale does not
    bind a group to K: a per-tensor scalar is orientation-free, and a BLOCK grid subdivides both
    axes, so transposing it is the same reindexing as transposing the tile.

    The scale grid is made CONTIGUOUS, not just transposed: reading it column-major cost 79us on
    a 4096x8192x6144 dgrad (405.7 -> 326.5us) while the copy is ~3KB, because every tile re-reads
    it strided. The weight itself stays a view — materializing it saves a further 47us but costs
    211us to produce, so that only pays when the caller already keeps both orientations."""
    Bs_t = Bs if Bs.numel() == 1 else Bs.t().contiguous()
    return matmul_2d(dY.contiguous(), B.t(), None, Bs_t, output_dtype=out_dtype)

def _dgrad_grouped(ctx, dY, B, Bs, expert_start, gather_idx, scatter_idx, b_global):
    """Routed dgrad. GATE is not a case to handle: the gradient of a gate|up GEMM is an UNGATED
    product at the doubled N extent, which is what a 2N-row weight already is. The forward's two
    row maps are passed through unchanged — the kernel swaps their roles itself."""
    if B.dtype == torch.int8:
        B = B.view(torch.uint8)
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = Bs.view(torch.uint8)
    values_per_byte = 2 if B.dtype == torch.uint8 else 1
    K = B.shape[-1] * values_per_byte
    dX = dgrad_matmul_grouped(
        dY.contiguous(), B, Bs, expert_start,
        K // Bs.shape[-1], max(B.shape[-2] // Bs.shape[-2], 1),
        gather_idx=gather_idx, scatter_idx=scatter_idx, output_dtype=torch.float32,
    )
    if b_global is not None:  # per-expert NVFP4 global folds on the product
        dX = dX * b_global.float().reshape(-1)[0]
    return dX.to(ctx.a_dtype)


__all__: list[str] = []

def _register(name: str, saved: tuple[int, ...], dgrad):
    """Attach a backward to one op. ``saved`` names the positional inputs backward needs — they
    differ per op, e.g. the MX signature is ``(A, B, As, Bs, ...)`` so its weight scale is index
    3, not 2.

    Everything but the activation gets None — weights and scales take no gradient, the rest is
    non-differentiable config."""

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*(inputs[i] for i in saved))
        ctx.a_dtype = inputs[0].dtype


    def backward(ctx, *grads):
        # torch matches backward's return against `tree_flatten(args, is_leaf=not_list_of_tensor)`
        # of the args AS INVOKED — which setup_context cannot see (it gets the bound args, with
        # defaults filled). `needs_input_grad` is sized to that same flattened list plus the
        # trailing metadata arg, so it is the one place the right count is observable. torch
        # appends the metadata's own None itself, so this returns exactly that many.
        n_grads = len(ctx.needs_input_grad)
        if not ctx.needs_input_grad[0]:
            return (None,) * n_grads
        # the ops return list[Tensor] (a second entry under output_recipe), so autograd hands the
        # gradient back in that shape — unwrap to the gradient of the primary output
        dY = grads[0]
        while isinstance(dY, (list, tuple)):
            dY = dY[0]
        return (dgrad(ctx, dY, *ctx.saved_tensors),) + (None,) * (n_grads - 1)

    torch.library.register_autograd(
        add_op_namespace_prefix(name), backward, setup_context=setup_context
    )


# block-FP8 (128x128 / 1x128, fp32 or UE8M0 scales): the grid subdivides both axes, so dgrad
# reuses the quantized weight through a transposed view.
_register(
    "w8a8_block_dynamic_fp8_matmul", saved=(1, 2),
    dgrad=lambda ctx, dY, B, Bs: _dgrad_transposed_weight(dY, B, Bs, ctx.a_dtype),
)

# per-tensor FP8: a scalar scale is orientation-free, so the same route needs no reorientation.
_register(
    "w8a8_tensor_dynamic_fp8_matmul", saved=(1, 2),
    dgrad=lambda ctx, dY, B, Bs: _dgrad_transposed_weight(dY, B, Bs, ctx.a_dtype),
)

# MX / NVFP4: per-row group scales bind to K and packed E2M1 packs along K, so the weight cannot
# be reoriented by a VIEW — but it does not need to be. The reduction runs over N against the
# weight's natural (N, K) tile, dequantized per-tile in-register, so the FORWARD-oriented weight
# serves dgrad as-is: no transposed copy, no second quantized orientation, no fp32 materialization.
_register(
    "mx_dynamic_matmul", saved=(1, 3, 13),
    dgrad=lambda ctx, dY, B, Bs, b_global: _dgrad_mx(dY, B, Bs, b_global, ctx.a_dtype),
)

# Weight-only (W4A16 / W8A16) — the QLoRA recipe: a frozen 4-bit base weight with raw bf16
# activations. Same route; the weight takes no gradient, so nothing here needs a master copy.
_register(
    "mx_weight_only_matmul_2d", saved=(1, 2, 9),
    dgrad=lambda ctx, dY, B, Bs, b_global: _dgrad_mx(dY, B, Bs, b_global, ctx.a_dtype),
)


def _dgrad_batched(ctx, dY, B, Bs, expert_ids, b_global):
    """Per-row-expert dgrad. Same shape-derived scale grid as the others; the only difference from
    ``_dgrad_grouped`` is that the expert comes from a per-row index rather than sorted ranges."""
    if B.dtype == torch.int8:
        B = B.view(torch.uint8)
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = Bs.view(torch.uint8)
    values_per_byte = 2 if B.dtype == torch.uint8 else 1
    K = B.shape[-1] * values_per_byte
    dX = dgrad_matmul_batched(
        dY.contiguous(), B, Bs, expert_ids,
        K // Bs.shape[-1], max(B.shape[-2] // Bs.shape[-2], 1),
        output_dtype=torch.float32,
    )
    if b_global is not None:
        dX = dX * b_global.float().reshape(-1)[0]
    return dX.to(ctx.a_dtype)


# Routed MoE, weight-only — QLoRA on an MoE model. Positional slots: B=1, Bs=2, expert_start=3,
# gather_idx=10, scatter_idx=11, b_global_scale=12.
_register(
    "mx_weight_only_matmul_grouped", saved=(1, 2, 3, 10, 11, 12),
    dgrad=_dgrad_grouped,
)

# The batched experts forward is a SELECTABLE dispatch in transformers, not a decode-only path, so
# it needs a formula too — without one, finetuning against `batched_mm` silently trains nothing.
# Rows are not expert-sorted here: slot 3 is per-row `expert_ids`, not `expert_start`.
_register(
    "mx_weight_only_matmul_batched", saved=(1, 2, 3, 12),
    dgrad=_dgrad_batched,
)
