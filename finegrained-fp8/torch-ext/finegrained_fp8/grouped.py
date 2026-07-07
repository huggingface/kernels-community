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
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix, ops
from .bayesian_autotuner import bayesian_autotune
from .utils import (
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    UE8M0_SCALE_DTYPES,
    adaptive_block_size_m,
    device_context,
    mxfp_act_quant_inline,
    fp8_act_quant,
    fp8_act_quant_inline,
    get_accelerator_autotuning_configs,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_tensor_wide,
    e2m1_as_uint8,
    ue8m0_as_uint8,
    decode_ue8m0_scale,
    mx_dot_rescale,
    mx_dot_scaled,
)


def _grouped_tile_layout(
    tokens_per_expert: torch.Tensor,
    block_size_m: int,
    S: int,
    num_experts: int,
) -> tuple[torch.Tensor, int]:
    """Compute the M-tile layout for the grouped kernels.

    Returns ``(tile_offsets, max_m_tiles)``:
    - ``tile_offsets``: int32 (num_experts,) cumulative tile-end per expert, used by
      ``_grouped_tile_setup`` to locate an M-tile's owning expert.
    - ``max_m_tiles``: upper bound on total M-tiles, used as the grid axis-0
      size. Real tile count <= this; surplus programs early-return inside the
      kernel. Keeps the grid data-independent (cuda-graph / torch.compile safe).
    """
    tiles_per_expert = (tokens_per_expert + block_size_m - 1) // block_size_m
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    max_m_tiles = triton.cdiv(S, block_size_m) + num_experts
    return tile_offsets, max_m_tiles


@triton.jit
def _grouped_tile_setup(
    pid_m,
    pid_n,
    Offsets,
    TileOffsets,
    stride_offs,
    stride_tile,
    num_experts,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Map a grouped M-tile to its expert and build the per-tile offset vectors —
    the prologue shared by every grouped kernel.

    Returns ``(expert_id, offs_global_m, row_mask, offs_bn, offs_k)``:
    - ``expert_id``: int64 owning expert
    - ``offs_global_m``: ``(BLOCK_SIZE_M,)`` global row indices into A
    - ``row_mask``: ``(BLOCK_SIZE_M,)`` validity mask within the expert's M
    - ``offs_bn``: ``(BLOCK_SIZE_N,)`` output column offsets
    - ``offs_k``: ``(BLOCK_SIZE_K,)`` K range

    Caller must have early-returned if ``pid_m`` exceeds total_tiles
    (``TileOffsets[(num_experts - 1) * stride_tile]``) — the ``Offsets`` load below
    is out of bounds for an out-of-range tile otherwise.
    """
    # Binary search: upper_bound(TileOffsets, pid_m). NUM_EXPERTS_BIT_LENGTH is
    # ceil(log2(num_experts))+1, giving one harmless extra iteration; constexpr so the
    # loop unrolls.
    lo = 0
    hi = num_experts
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid * stride_tile)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    # Cast to int64 so ``expert_id * stride_be`` doesn't overflow for large num_experts
    # × large weight matrices (e.g. 255 * 9_437_184 > 2^31).
    expert_id = lo.to(tl.int64)

    prev_eid = tl.maximum(expert_id - 1, 0)
    expert_start = tl.where(
        expert_id == 0, 0, tl.load(Offsets + prev_eid * stride_offs)
    )
    expert_end = tl.load(Offsets + expert_id * stride_offs)
    M_expert = expert_end - expert_start

    expert_tile_start = tl.where(
        expert_id == 0, 0, tl.load(TileOffsets + prev_eid * stride_tile)
    )
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_SIZE_M

    offs_am = m_off + tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_am < M_expert
    offs_global_m = expert_start + offs_am
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    return expert_id, offs_global_m, row_mask, offs_bn, offs_k


@triton.jit
def store_tile(C, accumulator, offs_global_m, offs_bn, row_mask, stride_cm, stride_cn):
    """Output epilogue shared by the grouped kernels: cast the fp32 accumulator to
    ``C``'s dtype and store the tile at expert-sorted global rows ``offs_global_m`` ×
    columns ``offs_bn``, masked to the expert's valid rows (``row_mask``)."""
    c = accumulator.to(C.dtype.element_ty)
    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=row_mask[:, None])


@triton.autotune(
    configs=get_accelerator_autotuning_configs(),
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_grouped_kernel(
    A,  # (S, K) raw BF16/FP16 activations, sorted/grouped by expert id
    B,  # (num_experts, N, K) FP8 weight matrices
    C,  # (S, N) output
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales
    Offsets,  # (num_experts,) int32 — cumulative row-end per expert
    TileOffsets,  # (num_experts,) int32 — cumulative tile-end per expert
    # Shape
    S,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_offs,
    stride_tile,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
):
    """Block-scale grouped FP8 expert matmul kernel.

    Tokens are assumed sorted by expert. The kernel maps each M-tile to its
    owning expert via ``TileOffsets`` and applies fused activation quantization.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + (num_experts - 1) * stride_tile)
    if pid_m >= total_tiles:
        return

    expert_id, offs_global_m, row_mask, offs_bn, offs_k = _grouped_tile_setup(
        pid_m,
        pid_n,
        Offsets,
        TileOffsets,
        stride_offs,
        stride_tile,
        num_experts,
        NUM_EXPERTS_BIT_LENGTH,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    bs_ptrs = Bs + expert_id * stride_bs_e + pid_n * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a, a_s = fp8_act_quant_inline(a_raw)
        b = tl.load(b_ptrs)
        b_s = decode_ue8m0_scale(tl.load(bs_ptrs))
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    store_tile(C, accumulator, offs_global_m, offs_bn, row_mask, stride_cm, stride_cn)


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_nk=True),
    ["N", "K", "BLOCK_SIZE_M"],
    n_trials=60,
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_grouped_kernel(
    A,  # (S, K) pre-quantized FP8 activations, sorted/grouped by expert id
    B,  # (num_experts, N, K) FP8 weight matrices
    C,  # (S, N) output
    As,  # (S,) per-token activation scales
    Bs,  # (num_experts, 1, 1) per-tensor weight scales
    Offsets,  # (num_experts,) int32 — cumulative row-end per expert
    TileOffsets,  # (num_experts,) int32 — cumulative tile-end per expert
    # Shape
    S,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_m,
    stride_bs_e,
    stride_offs,
    stride_tile,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
):
    """Tensor-scale grouped FP8 expert matmul kernel.

    Uses grouped expert scheduling with pre-quantized activations plus
    per-token activation scales and per-expert tensor weight scales.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + (num_experts - 1) * stride_tile)
    if pid_m >= total_tiles:
        return

    expert_id, offs_global_m, row_mask, offs_bn, offs_k = _grouped_tile_setup(
        pid_m,
        pid_n,
        Offsets,
        TileOffsets,
        stride_offs,
        stride_tile,
        num_experts,
        NUM_EXPERTS_BIT_LENGTH,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    a_s = tl.load(As + offs_global_m * stride_as_m, mask=row_mask, other=0.0)
    b_s = tl.load(Bs + expert_id * stride_bs_e)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    accumulator = accumulator * a_s[:, None] * b_s

    store_tile(C, accumulator, offs_global_m, offs_bn, row_mask, stride_cm, stride_cn)


@bayesian_autotune(
    get_mxfp_autotuning_configs(
        compute_modes=("dot_scaled", "dot")
    ),  # prefill: no scalar branch
    ["N", "K", "BLOCK_SIZE_M"],
    n_trials=60,
)
@triton.jit
def mxfp_dynamic_matmul_grouped_kernel(
    A,  # (S, K) raw BF16/FP16 activations, sorted by expert id
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4) expert weights
    C,  # (S, N) output
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales
    Offsets,  # (num_experts,) int32 — cumulative row-end per expert
    TileOffsets,  # (num_experts,) int32 — cumulative tile-end per expert
    # Shape
    S,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_offs,
    stride_tile,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
):
    """Unified grouped MXFP4/MXFP8 (W4A8/W8A8) expert matmul with fused act quant.

    Tokens sorted by expert; each M-tile maps to its expert via ``TileOffsets``. ``A`` is
    quantized to E4M3 per K-group inline (UE8M0 scale). ``VALUES_PER_BYTE`` picks the
    weight format (2 = packed E2M1 / MXFP4, 1 = unpacked E4M3 / MXFP8); ``COMPUTE_MODE``
    picks ``tl.dot_scaled`` vs fp8 ``tl.dot`` + per-group rescale (decode; FP4 unpacks
    E2M1->E4M3 first, lossless).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + (num_experts - 1) * stride_tile)
    if pid_m >= total_tiles:
        return

    expert_id, offs_global_m, row_mask, offs_bn, offs_k = _grouped_tile_setup(
        pid_m,
        pid_n,
        Offsets,
        TileOffsets,
        stride_offs,
        stride_tile,
        num_experts,
        NUM_EXPERTS_BIT_LENGTH,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_be
        + offs_kb[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    bs_ptrs = (
        Bs
        + expert_id * stride_bs_e
        + offs_bn[:, None] * stride_bs_n
        + offs_sf[None, :] * stride_bs_k
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a, a_scale = mxfp_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs).to(tl.uint8)
        if COMPUTE_MODE == "dot_scaled":
            accumulator = mx_dot_scaled(
                accumulator, a, a_scale, b, b_s, VALUES_PER_BYTE
            )
        else:  # dot
            accumulator = mx_dot_rescale(accumulator, a, b, a_scale, b_s, VALUES_PER_BYTE)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    store_tile(C, accumulator, offs_global_m, offs_bn, row_mask, stride_cm, stride_cn)


@triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_grouped"), mutates_args=()
)
def _w8a8_block_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale grouped FP8 matmul: C = A @ B.T per expert, with fused act quant.

    A:  (S, K) raw bf16/fp16 activations, sorted by expert
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    S, K = A.shape
    num_experts, N, _ = B.shape

    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]
    # MoE expert dimensions must be block-aligned; non-aligned N/K is not supported.
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
    assert Bs.ndim == 3, (
        f"Bs must be 3D (num_experts, N//block_n, K//block_k), got ndim={Bs.ndim}"
    )
    assert Bs.shape == (num_experts, N // block_n, K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, {N // block_n}, {K // block_k})"
    )

    Bs = ue8m0_as_uint8(Bs)
    C = A.new_empty(S, N, dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m((S + num_experts - 1) // num_experts)
    tile_offsets, max_m_tiles = _grouped_tile_layout(
        tokens_per_expert, BLOCK_SIZE_M, S, num_experts
    )
    grid = (max_m_tiles, triton.cdiv(N, block_n))

    with device_context(A.device):
        wrap_triton(w8a8_block_dynamic_fp8_matmul_grouped_kernel)[grid](
            A,
            B,
            C,
            Bs,
            offsets,
            tile_offsets,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            Bs.stride(0),
            Bs.stride(2),
            Bs.stride(1),
            offsets.stride(0),
            tile_offsets.stride(0),
            # Meta-parameters
            num_experts=num_experts,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
        )

    return C


@triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_grouped"), mutates_args=()
)
def _w8a8_tensor_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tensor-scale grouped FP8 matmul: C = A @ B.T per expert, with fused act quant.

    A:  (S, K) raw bf16/fp16 activations, sorted by expert
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    S, K = A.shape
    num_experts, N, _ = B.shape

    # Normalize Bs to (num_experts, 1, 1)
    if Bs.ndim == 1:
        assert Bs.shape[0] == num_experts, (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts},)"
        )
        Bs = Bs.reshape(num_experts, 1, 1)
    else:
        assert Bs.shape == (num_experts, 1, 1), (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, 1, 1)"
        )

    qA, As = fp8_act_quant(A, K)
    C = A.new_empty(S, N, dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m((S + num_experts - 1) // num_experts)
    tile_offsets, max_m_tiles = _grouped_tile_layout(
        tokens_per_expert, BLOCK_SIZE_M, S, num_experts
    )

    def grid(META):
        return (max_m_tiles, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(w8a8_tensor_dynamic_fp8_matmul_grouped_kernel)[grid](
            qA,
            B,
            C,
            As,
            Bs,
            offsets,
            tile_offsets,
            S,
            N,
            K,
            qA.stride(0),
            qA.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            As.stride(0),
            Bs.stride(0),
            offsets.stride(0),
            tile_offsets.stride(0),
            # Meta-parameters (BLOCK_SIZE_N, BLOCK_SIZE_K come from autotune Config)
            num_experts=num_experts,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
        )

    return C


def w8a8_block_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale grouped FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations sorted by expert, bf16/fp16/fp32
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales
    output_dtype: defaults to ``A.dtype``
    """
    return ops.w8a8_block_dynamic_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size, output_dtype
    )


def w8a8_tensor_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tensor-scale grouped FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations sorted by expert, bf16/fp16/fp32
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    output_dtype: defaults to ``A.dtype``
    """
    return ops.w8a8_tensor_dynamic_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, output_dtype
    )


@triton_op(add_op_namespace_prefix("mxfp_dynamic_matmul_grouped"), mutates_args=())
def _mxfp_dynamic_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Grouped MX matmul with fused act quant — per-expert ``C[s] = A[s] @ B[e].T`` over
    contiguous, expert-sorted rows. Weight format detected from ``B.dtype``: ``int8`` →
    packed E2M1 (MXFP4, ``B`` is ``(num_experts, N, K//2)``); ``float8_e4m3fn`` → unpacked E4M3
    (MXFP8, ``(num_experts, N, K)``). UE8M0 group-32 scales ``(num_experts, N, K//32)``; tile + dot autotuned.

    A:  (S, K) raw activations, bf16/fp16/fp32, expert-sorted (quantized inline to E4M3)
    offsets: (num_experts,) — exclusive prefix of expert token counts (cumsum)
    tokens_per_expert: (num_experts,) — per-expert row count
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert offsets.ndim == 1 and tokens_per_expert.ndim == 1
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be float8_e8m0fnu or uint8 (UE8M0), got {Bs.dtype}"
    )
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    S, K = A.shape
    num_experts, N, K_b = B.shape
    assert offsets.shape[0] == num_experts and tokens_per_expert.shape[0] == num_experts
    assert K == VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {VALUES_PER_BYTE} * B.shape[2] (={K_b})"
    )
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (num_experts, N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {N}, {K // MX_SCALE_GROUP_K})"
    )

    B = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    C = A.new_empty((S, N), dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m((S + num_experts - 1) // num_experts)
    tile_offsets, max_m_tiles = _grouped_tile_layout(
        tokens_per_expert, BLOCK_SIZE_M, S, num_experts
    )

    def grid(META):
        return (max_m_tiles, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(mxfp_dynamic_matmul_grouped_kernel)[grid](
            A,
            B,
            C,
            bs_u8,
            offsets,
            tile_offsets,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            offsets.stride(0),
            tile_offsets.stride(0),
            # Meta-parameters (BLOCK_SIZE_N, BLOCK_SIZE_K come from autotune Config)
            num_experts=num_experts,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
        )
    return C


def mxfp_dynamic_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Grouped MX (MXFP4/MXFP8) matmul with fused act quant — per-expert
    ``C[s] = A[s] @ B[e].T`` over expert-sorted rows; weight format detected from
    ``B.dtype``. Tile + dot path autotuned."""
    return ops.mxfp_dynamic_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, output_dtype
    )


def matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Grouped quantized matmul dispatcher (W8A8 FP8 or W4A8 FP4). Tokens must
    be sorted by expert; M-tiles are mapped to experts via ``offsets``.

    ``output_dtype`` defaults to ``A.dtype``.

    Routes by weight dtype and ``block_size``:
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` (shape ``[num_experts, N, K//32]``) → ``mxfp_dynamic_matmul_grouped``
      (``block_size`` ignored; tile + dot path autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul_grouped``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul_grouped``.
    """
    if is_mxfp(B, Bs):
        return mxfp_dynamic_matmul_grouped(
            A, B, Bs, offsets, tokens_per_expert, output_dtype
        )

    if is_tensor_wide(block_size, B):
        return w8a8_tensor_dynamic_fp8_matmul_grouped(
            A, B, Bs, offsets, tokens_per_expert, output_dtype
        )

    return w8a8_block_dynamic_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size, output_dtype
    )
