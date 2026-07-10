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

from ._ops import add_op_namespace_prefix

from .bayesian_autotuner import bayesian_autotune
from .utils import (
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    UE8M0_SCALE_DTYPES,
    decode_ue8m0_scale,
    device_context,
    mx_compute,
    oriented_weight_ptrs,
    acc_init,
    fp8_dot,
    batched_mx_pruner,
    block_k_within_k_pruner,
    acc_finalize,
    mxfp_act_quant_inline,
    fp8_act_quant,
    fp8_act_quant_2d,
    maybe_act_quant,
    fp8_act_quant_inline,
    get_accelerator_autotuning_configs,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_tensor_wide,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)

# maybe_act_quant crossover for the block-dynamic batched matmul: offline always —
# inherited from the fused gate_up sweep (same per-row decode structure, fewer streams):
# offline won at T=1/4/16 and lost only ~2% (noise-tier) at T=64, outside this kernel's
# decode regime.
BLOCK_DYNAMIC_BATCHED_ACT_PREQUANT_MIN_M = 0


@triton.jit
def expert_setup(
    A,
    B,
    C,
    Bs,
    ExpertIds,
    stride_am,
    stride_be,
    stride_cm,
    stride_bs_e,
    stride_eid,
):
    """Per-(row, expert) prologue shared by the batched kernels: read the program
    ids, look up the routed expert, and advance the A/B/C/Bs base pointers to this
    row's slice. Returns ``(batch_id, pid_n, expert_id, A, B, C, Bs)``.

    The caller must early-return on the EP sentinel (``expert_id >= num_experts``)
    before any load — the pointer arithmetic itself is harmless, only the loads on a
    non-local expert would be out of bounds."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # Cast to int64 to prevent overflow on expert_id * stride_be.
    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    A = A + batch_id * stride_am
    B = B + expert_id * stride_be
    C = C + batch_id * stride_cm
    Bs = Bs + expert_id * stride_bs_e
    return batch_id, pid_n, expert_id, A, B, C, Bs


@triton.jit
def store_row(
    C,
    accumulator,
    pid_n,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Output epilogue shared by the batched kernels (``C`` already advanced to the
    row). The fake-batch trick aliases all ``BLOCK_SIZE_M`` lanes to the same C row,
    so a plain store would issue ``BLOCK_SIZE_M`` duplicate-address writes — benign on
    NVIDIA WGMMA (last-write-wins of identical bytes) but hardware-undefined on Intel
    XPU, where it corrupts the output. Mask so only lane 0 stores; the accumulator
    rows are mathematically identical (same A row × same B), so lane 0 is correct."""
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None])


@triton.autotune(
    configs=get_accelerator_autotuning_configs(swap_ab=True, for_decode=True),
    key=["N", "K", "S"],
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) E4M3 activations (pre-quantized once by the wrapper)
    AScale,  # (S, K // BLOCK_SIZE_K) fp32 per-row, per-K-block activation scales
    B,  # (num_experts, N, K) FP8 weight matrices
    C,  # (S, N) output
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales
    ExpertIds,  # (S,) — which expert each batch element routes to
    # Shape
    S,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_as_m,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Block-scale batched FP8 expert matmul kernel.

    Each program handles one routed token row and one N-tile, looking up the
    owning expert from ``ExpertIds``. Activations arrive pre-quantized (one wrapper
    pass — the inline quant re-ran per N-tile and paid a per-tile amax reduction).

    ``SWAP_AB`` (tuner axis, M=1 decode): load the weight output-rows-major ``[BN, BK]`` and put
    those rows in the MMA M dim, padding the single token to the N=16 atom; column 0 of the
    ``[BN, 16]`` accumulator is the result. No-swap keeps the token in M (padded to 16)."""
    batch_id, pid_n, expert_id, A, B, C, Bs = expert_setup(
        A, B, C, Bs, ExpertIds, stride_am, stride_be, stride_cm, stride_bs_e, stride_eid
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    as_ptrs = AScale + batch_id * stride_as_m + tl.arange(0, BLOCK_SIZE_M) * 0
    b_ptrs = oriented_weight_ptrs(B, offs_bn, offs_k, stride_bn, stride_bk, SWAP_AB)
    bs_ptrs = Bs + pid_n * stride_bs_n

    accumulator = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if A.dtype.element_ty == tl.float8e4nv:  # pre-quantized offline
            a = tl.load(a_ptrs)
            a_s = tl.load(as_ptrs)
        else:  # raw bf16/fp16 — quantize inline
            a, a_s = fp8_act_quant_inline(tl.load(a_ptrs).to(tl.float32))
        b = tl.load(b_ptrs)
        b_s = decode_ue8m0_scale(tl.load(bs_ptrs))
        # a_s is [BM], b_s a per-block scalar; a_s[:, None] broadcasts onto the acc either way (under
        # swap BM=1, so it is the single token's scale), so no swap branch — as in the down projection.
        dot = fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K)
        accumulator += dot * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K * stride_ak
        as_ptrs += 1
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    accumulator = acc_finalize(accumulator, "dot", BLOCK_SIZE_N, SWAP_AB)
    store_row(C, accumulator, pid_n, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)


@bayesian_autotune(
    # S (routed rows) keyed like the block-dynamic/mxfp batched siblings — decode re-tunes per batch.
    get_accelerator_autotuning_configs(
        tune_block_nk=True, swap_ab=True, for_decode=True
    ),
    ["N", "K", "S"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs.
    prune_configs_by={"early_config_prune": block_k_within_k_pruner("K")},
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) pre-quantized FP8 activations
    B,  # (num_experts, N, K) FP8 weight matrices
    C,  # (S, N) output
    As,  # (S,) per-token activation scales
    Bs,  # (num_experts, 1, 1) per-tensor weight scales
    ExpertIds,  # (S,) — which expert each batch element routes to
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
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Tensor-scale batched FP8 expert matmul kernel.

    Activations are already quantized; the kernel applies per-token activation
    scales and per-expert tensor weight scales.

    ``SWAP_AB`` (tuner axis, M=1 decode): weight output rows in the MMA M dim (``B`` as ``[BN, BK]``,
    single token padded to N=16); column 0 of the ``[BN, 16]`` accumulator is the result. Both
    scales are per-token/per-tensor scalars, applied once after the loop, orientation-agnostic."""
    batch_id, pid_n, expert_id, A, B, C, Bs = expert_setup(
        A, B, C, Bs, ExpertIds, stride_am, stride_be, stride_cm, stride_bs_e, stride_eid
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = oriented_weight_ptrs(B, offs_bn, offs_k, stride_bn, stride_bk, SWAP_AB)
    b_s = tl.load(Bs)
    a_s = tl.load(As + batch_id * stride_as_m)

    accumulator = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = acc_finalize(accumulator, "dot", BLOCK_SIZE_N, SWAP_AB) * a_s * b_s
    store_row(C, accumulator, pid_n, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)


# VALUES_PER_BYTE keys the MXFP4/MXFP8 split so a cached winner is only reused for its packing.
# BLOCK_SIZE_M is always 1 here (per-token decode), so — like the fused MXFP batched kernels —
# dot is excluded (dead at M=1) and SWAP_AB is the operand-swap axis over {dot_scaled, scalar}.
@bayesian_autotune(
    get_mxfp_autotuning_configs(
        compute_modes=("dot_scaled", "scalar"), swap_ab=True, for_decode=True
    ),
    ["N", "K", "S", "VALUES_PER_BYTE"],
    n_trials=100,
    # BK-within-K + the sm_10x MMA-shape guards (swapped dot_scaled needs BN >= 128 for the
    # native scaled-MMA; smaller-BN swap configs never win and mislead the TPE).
    prune_configs_by={"early_config_prune": batched_mx_pruner("K")},
)
@triton.jit
def mxfp_dynamic_matmul_batched_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4) expert weights
    C,  # (S, N) output
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales
    ExpertIds,  # (S,) — which expert each routed row uses
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
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Unified batched MXFP4/MXFP8 (W4A8/W8A8) expert matmul with fused act quant.

    One routed row + one N-tile per program; expert looked up from ``ExpertIds``. ``A`` is
    quantized to E4M3 per K-group inline (UE8M0 scale). ``VALUES_PER_BYTE`` picks the
    weight format (2 = packed E2M1 / MXFP4, 1 = unpacked E4M3 / MXFP8); ``COMPUTE_MODE``
    picks ``tl.dot_scaled`` (native M=128) vs the scalar CUDA-core reduce (wins at decode).

    ``SWAP_AB`` (tuner axis, M=1 decode): weight output rows in the MMA M dim (``B`` as ``[BN, BK]``,
    single token padded to N=16); column 0 of the ``[BN, 16]`` accumulator is the result. dot_scaled
    uses the swapped scaled-MMA; scalar reduces over K with the weight output-rows-major.
    """
    _, pid_n, expert_id, A, B, C, Bs = expert_setup(
        A, B, C, Bs, ExpertIds, stride_am, stride_be, stride_cm, stride_bs_e, stride_eid
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    # Weight [BN, BK_packed] output-rows-major when swapped, else [BK_packed, BN]; scales stay
    # [BN, NG] (output-rows-major) either way.
    b_ptrs = oriented_weight_ptrs(B, offs_bn, offs_kb, stride_bn, stride_bk, SWAP_AB)
    bs_ptrs = Bs + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k

    accumulator = acc_init(COMPUTE_MODE, BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a, a_scale = mxfp_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs).to(tl.uint8)
        accumulator = mx_compute(
            accumulator,
            a,
            a_scale,
            b,
            b_s,
            COMPUTE_MODE,
            VALUES_PER_BYTE,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
            SWAP_AB,
        )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    accumulator = acc_finalize(accumulator, COMPUTE_MODE, BLOCK_SIZE_N, SWAP_AB)
    store_row(C, accumulator, pid_n, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_batched"), mutates_args=()
)
def w8a8_block_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale batched FP8 matmul: C[s] = A[s] @ B[expert_ids[s]].T; activations
    quantized offline in one pass.

    A:  (S, K) raw bf16/fp16 activations
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
    grid = (S, triton.cdiv(N, block_n))
    C = A.new_empty(S, N, dtype=output_dtype)
    A_q, A_s = maybe_act_quant(
        A,
        lambda x: fp8_act_quant_2d(x, block_k),
        BLOCK_DYNAMIC_BATCHED_ACT_PREQUANT_MIN_M,
    )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            A_q,
            A_s,
            B,
            C,
            Bs,
            expert_ids,
            S,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            Bs.stride(0),
            Bs.stride(2),
            Bs.stride(1),
            expert_ids.stride(0),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            num_experts=num_experts,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_batched"), mutates_args=()
)
def w8a8_tensor_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tensor-scale batched FP8 matmul: C[s] = A[s] @ B[expert_ids[s]].T; activations
    quantized offline per row in the wrapper.

    A:  (S, K) raw bf16/fp16 activations
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

    Bs = ue8m0_as_uint8(Bs)
    qA, As = fp8_act_quant(A, K)
    C = A.new_empty(S, N, dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            qA,
            B,
            C,
            As,
            Bs,
            expert_ids,
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
            expert_ids.stride(0),
            num_experts=num_experts,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mxfp_dynamic_matmul_batched"), mutates_args=()
)
def mxfp_dynamic_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched MX matmul ``C[s] = A[s] @ B[expert_ids[s]].T``; activations quantized
    inline in the kernel (decode: one act row per program, inline is free).
    Weight format is detected from ``B.dtype``: ``int8`` → packed E2M1 (MXFP4, ``B`` is
    ``(num_experts, N, K//2)``); ``float8_e4m3fn`` → unpacked E4M3 (MXFP8, ``(num_experts, N, K)``). Both use
    UE8M0 group-32 scales ``(num_experts, N, K//32)``; tile + dot path are autotuned.

    A:  (S, K) raw activations, bf16/fp16/fp32 (quantized inline to E4M3)
    expert_ids: (S,) which expert each routed row uses
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert expert_ids.ndim == 1
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be float8_e8m0fnu or uint8 (UE8M0), got {Bs.dtype}"
    )
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    S, K = A.shape
    num_experts, N, K_b = B.shape
    assert expert_ids.shape[0] == S
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

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        compile_time_only_triton_wrap(mxfp_dynamic_matmul_batched_kernel)[grid](
            A,
            B,
            C,
            bs_u8,
            expert_ids,
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
            expert_ids.stride(0),
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            num_experts=num_experts,
        )
    return C


def matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched quantized matmul dispatcher (W8A8 FP8 or W4A8 FP4). Routes one
    routed row per program to ``B[expert_ids[s]]``.

    ``output_dtype`` defaults to ``A.dtype``.

    Routes by weight dtype and ``block_size``:
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` (shape ``[num_experts, N, K//32]``) → ``mxfp_dynamic_matmul_batched``
      (``block_size`` ignored; tile + dot path autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul_batched``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul_batched``.
    """
    if is_mxfp(B, Bs):
        return mxfp_dynamic_matmul_batched(A, B, Bs, expert_ids, output_dtype)

    if is_tensor_wide(block_size, B):
        return w8a8_tensor_dynamic_fp8_matmul_batched(
            A, B, Bs, expert_ids, output_dtype
        )

    return w8a8_block_dynamic_fp8_matmul_batched(
        A, B, Bs, expert_ids, block_size, output_dtype
    )
