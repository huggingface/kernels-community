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
    block_dynamic_grouped_matmul_pruner,
    batched_mx_pruner,
    build_tile_layout,
    resolve_grouped_tile,
    block_k_within_k_pruner,
    compose_pruners,
    device_context,
    sm_count,
    mxfp_act_quant,
    load_mx_act_tile,
    store_tile,
    fp8_act_quant,
    fp8_act_quant_2d,
    get_accelerator_autotuning_configs,
    warp_spec_compile_guard_pruner,
    GroupedScheduling,
    is_mxfp,
    is_tensor_wide,
    e2m1_as_uint8,
    ue8m0_as_uint8,
    decode_ue8m0_scale,
    mx_compute,
)


@bayesian_autotune(
    # SWAP_AB/MEMORY_MODE (TMA) arms were ported here, measured, and REMOVED: at dsv4-like
    # prefill (S=8192, E=256, N=4096, K=7168) WS-pointer BM=64 w4 s4 = 1796us beats
    # descriptor+swap (1944us, its best) and pointer+swap (2088us); descriptor+swap was
    # also numerically WRONG at BM=16 at large K. The dormant reference implementation
    # lives on the 2D kernel (w8a8_block_dynamic_fp8_matmul_kernel); see OPTIMIZATION_LOG.
    get_accelerator_autotuning_configs(warp_spec=True, tune_block_m=True),
    ["N", "K", "tokens_per_expert_bit_length"],
    n_trials=100,
    # Pipeliner-race guard: per launch-BM, WS-only at BM >= 64 and non-WS below (see the pruner).
    prune_configs_by={"early_config_prune": block_dynamic_grouped_matmul_pruner()},
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    As,  # (S, K // BLOCK_SIZE_K) fp32 per-row, per-K-block activation scales
    B,  # (num_experts, N, K) FP8 weight matrices
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales
    C,  # (S, N) output
    InputPerm,  # (S,) int32 — sorted position -> source row of A; read iff HAS_INPUT_PERM
    OutputPerm,  # (S,) int32 — sorted position -> destination row of C; read iff HAS_OUTPUT_PERM
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_INPUT_PERM: tl.constexpr,
    HAS_OUTPUT_PERM: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
):
    """Block-scale grouped FP8 expert matmul kernel — persistent grid-stride over tiles.

    Each M-tile maps to its owning expert via ``ExpertStart`` and gathers its rows
    through ``InputPerm`` — the expert sort is virtual, ``A`` arrives in any row order.
    Activations arrive pre-quantized (one pass in the wrapper — the
    inline per-N-tile quant would repeat N//BN times per element; see the fused kernels' log).
    """
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = resolve_grouped_tile(
            tile_id,
            num_n_tiles,
            exp_start,
            freqs,
            tile_start_excl,
            e_offs,
            InputPerm,
            OutputPerm,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
            HAS_INPUT_PERM,
            HAS_OUTPUT_PERM,
        )
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        as_ptrs = As + in_row * stride_as_m
        b_ptrs = (
            B
            + expert_id64 * stride_b_e
            + offs_k[:, None] * stride_b_k
            + offs_bn[None, :] * stride_b_n
        )
        bs_ptrs = Bs + expert_id64 * stride_bs_e + pid_n * stride_bs_n

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            a_s = tl.load(as_ptrs, mask=row_mask, other=0.0)
            b = tl.load(b_ptrs)
            b_s = decode_ue8m0_scale(tl.load(bs_ptrs))
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            as_ptrs += 1
            b_ptrs += BLOCK_SIZE_K * stride_b_k
            bs_ptrs += stride_bs_k

        store_tile(C, accumulator, out_row, offs_bn, row_mask, stride_c_m, stride_c_n)


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        tune_block_nk=True, warp_spec=True, tune_block_m=True
    ),
    ["N", "K", "tokens_per_expert_bit_length"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs;
    # WS is a pure perf axis here (non-WS is the validated state), compile-guarded.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_k_within_k_pruner("K"), warp_spec_compile_guard_pruner()
        )
    },
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) pre-quantized FP8 activations, any row order
    As,  # (S,) per-token activation scales
    B,  # (num_experts, N, K) FP8 weight matrices
    Bs,  # (num_experts, 1, 1) per-tensor weight scales
    C,  # (S, N) output
    InputPerm,  # (S,) int32 — sorted position -> source row of A; read iff HAS_INPUT_PERM
    OutputPerm,  # (S,) int32 — sorted position -> destination row of C; read iff HAS_OUTPUT_PERM
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_c_m,
    stride_c_n,
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_INPUT_PERM: tl.constexpr,
    HAS_OUTPUT_PERM: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
):
    """Tensor-scale grouped FP8 expert matmul kernel — persistent grid-stride over tiles.

    Uses grouped expert scheduling with pre-quantized activations plus
    per-token activation scales and per-expert tensor weight scales.
    """
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = resolve_grouped_tile(
            tile_id,
            num_n_tiles,
            exp_start,
            freqs,
            tile_start_excl,
            e_offs,
            InputPerm,
            OutputPerm,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
            HAS_INPUT_PERM,
            HAS_OUTPUT_PERM,
        )
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        b_ptrs = (
            B
            + expert_id64 * stride_b_e
            + offs_k[:, None] * stride_b_k
            + offs_bn[None, :] * stride_b_n
        )
        a_s = tl.load(As + in_row * stride_as_m, mask=row_mask, other=0.0)
        b_s = tl.load(Bs + expert_id64 * stride_bs_e)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k
        accumulator = accumulator * a_s[:, None] * b_s

        store_tile(C, accumulator, out_row, offs_bn, row_mask, stride_c_m, stride_c_n)


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        mx=True,
        tune_block_nk=True,
        tune_block_m=True,
        compute_modes=("dot_scaled", "dot"),
    ),  # prefill: no scalar branch
    # VALUES_PER_BYTE keys the MXFP4/MXFP8 split so a cached winner is only reused for its packing.
    ["N", "K", "tokens_per_expert_bit_length", "VALUES_PER_BYTE"],
    n_trials=100,
    # BK-within-K veto + the sm_10x dot_scaled shape/trap gates (this kernel had no
    # pruner while its BK span was {128,256} — the union span's BK=64 rows made the
    # gates load-bearing).
    prune_configs_by={"early_config_prune": batched_mx_pruner("K")},
)
@triton.jit
def mxfp_dynamic_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    As,  # (S, K // 32) UE8M0 group-32 activation scales
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4) expert weights
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales
    C,  # (S, N) output
    InputPerm,  # (S,) int32 — sorted position -> source row of A; read iff HAS_INPUT_PERM
    OutputPerm,  # (S,) int32 — sorted position -> destination row of C; read iff HAS_OUTPUT_PERM
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_INPUT_PERM: tl.constexpr,
    HAS_OUTPUT_PERM: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
):
    """Unified grouped MXFP4/MXFP8 (W4A8/W8A8) expert matmul — persistent grid-stride.

    Each M-tile maps to its expert via ``ExpertStart`` and gathers its rows through
    ``PermToken`` (virtual sort — ``A`` in any row order). ``A``
    arrives pre-quantized (E4M3 + UE8M0 group-32 scales, one pass in the wrapper — the
    inline per-N-tile quant would repeat N//BN times per element). ``VALUES_PER_BYTE`` picks the
    weight format (2 = packed E2M1 / MXFP4, 1 = unpacked E4M3 / MXFP8); ``COMPUTE_MODE``
    picks ``tl.dot_scaled`` vs fp8 ``tl.dot`` + per-group rescale (decode; FP4 unpacks
    E2M1->E4M3 first, lossless).
    """
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = resolve_grouped_tile(
            tile_id,
            num_n_tiles,
            exp_start,
            freqs,
            tile_start_excl,
            e_offs,
            InputPerm,
            OutputPerm,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
            HAS_INPUT_PERM,
            HAS_OUTPUT_PERM,
        )
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        as_ptrs = As + in_row[:, None] * stride_as_m + offs_sf[None, :]
        b_ptrs = (
            B
            + expert_id64 * stride_b_e
            + offs_kb[:, None] * stride_b_k
            + offs_bn[None, :] * stride_b_n
        )
        bs_ptrs = (
            Bs
            + expert_id64 * stride_bs_e
            + offs_bn[:, None] * stride_bs_n
            + offs_sf[None, :] * stride_bs_k
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a, a_scale = load_mx_act_tile(
                a_ptrs, as_ptrs, row_mask, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
            )
            as_ptrs += BLOCK_SIZE_K // SCALE_GROUP_K
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
            )
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_b_k
            bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

        store_tile(C, accumulator, out_row, offs_bn, row_mask, stride_c_m, stride_c_n)


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_grouped"), mutates_args=()
)
def w8a8_block_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
    input_perm: torch.Tensor | None = None,
    output_perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-scale grouped FP8 matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the
    maps); activations quantized offline in one pass.

    A:  raw bf16/fp16 activations — rows addressed via ``input_perm`` (expert-sorted as-is when None)
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    input_perm: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    output_perm: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    _, K = A.shape
    # S = routed rows (num_tokens * top_k), carried by the (S,) perms — A's rows are
    # gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden).
    # Only with no perms at all is A itself the expert-sorted (S, K) matrix.
    if input_perm is not None:
        S = input_perm.numel()
    elif output_perm is not None:
        S = output_perm.numel()
    else:
        S = A.shape[0]

    for perm_map in (input_perm, output_perm):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())

    num_experts, N, _ = B.shape
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"

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
    A_q, A_s = fp8_act_quant_2d(A, block_k)
    C = A.new_empty(S, N, dtype=output_dtype)
    num_sms = sm_count(A.device.index)

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            A_q,
            A_s,
            B,
            Bs,
            C,
            input_perm if input_perm is not None else expert_start,  # dummy ptr
            output_perm if output_perm is not None else expert_start,  # dummy ptr
            expert_start,
            S,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            Bs.stride(2),
            Bs.stride(1),
            C.stride(0),
            C.stride(1),
            # Meta-parameters
            num_experts=num_experts,
            tokens_per_expert_bit_length=int(
                (S + num_experts - 1) // num_experts
            ).bit_length(),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            HAS_INPUT_PERM=input_perm is not None,
            HAS_OUTPUT_PERM=output_perm is not None,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_grouped"), mutates_args=()
)
def w8a8_tensor_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    input_perm: torch.Tensor | None = None,
    output_perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensor-scale grouped FP8 matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the
    maps); activations quantized offline per row in the wrapper.

    A:  raw bf16/fp16 activations — rows addressed via ``input_perm`` (expert-sorted as-is when None)
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    input_perm: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    output_perm: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    _, K = A.shape
    # S = routed rows (num_tokens * top_k), carried by the (S,) perms — A's rows are
    # gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden).
    # Only with no perms at all is A itself the expert-sorted (S, K) matrix.
    if input_perm is not None:
        S = input_perm.numel()
    elif output_perm is not None:
        S = output_perm.numel()
    else:
        S = A.shape[0]
    for perm_map in (input_perm, output_perm):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())
    num_experts, N, _ = B.shape
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"

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
    num_sms = sm_count(A.device.index)

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            qA,
            As,
            B,
            Bs,
            C,
            input_perm if input_perm is not None else expert_start,  # dummy ptr
            output_perm if output_perm is not None else expert_start,  # dummy ptr
            expert_start,
            S,
            N,
            K,
            qA.stride(0),
            qA.stride(1),
            As.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            C.stride(0),
            C.stride(1),
            num_experts=num_experts,
            tokens_per_expert_bit_length=int(
                (S + num_experts - 1) // num_experts
            ).bit_length(),
            HAS_INPUT_PERM=input_perm is not None,
            HAS_OUTPUT_PERM=output_perm is not None,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mxfp_dynamic_matmul_grouped"), mutates_args=()
)
def mxfp_dynamic_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    input_perm: torch.Tensor | None = None,
    output_perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped MX matmul over expert-sorted positions (per-tile gather/scatter, the
    sort is virtual — see ``compute_grouped_scheduling`` for the maps); activations
    quantized offline in one pass (always: inline only wins at M=1 decode, which the
    grouped prefill path never serves).
    Weight format detected from ``B.dtype``: ``int8`` →
    packed E2M1 (MXFP4, ``B`` is ``(num_experts, N, K//2)``); ``float8_e4m3fn`` → unpacked E4M3
    (MXFP8, ``(num_experts, N, K)``). UE8M0 group-32 scales ``(num_experts, N, K//32)``; tile + dot autotuned.

    A:  raw activations, bf16/fp16/fp32 — rows addressed via ``input_perm`` (expert-sorted as-is when None)
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    input_perm: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    output_perm: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be float8_e8m0fnu or uint8 (UE8M0), got {Bs.dtype}"
    )
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    _, K = A.shape
    # S = routed rows (num_tokens * top_k), carried by the (S,) perms — A's rows are
    # gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden).
    # Only with no perms at all is A itself the expert-sorted (S, K) matrix.
    if input_perm is not None:
        S = input_perm.numel()
    elif output_perm is not None:
        S = output_perm.numel()
    else:
        S = A.shape[0]
    for perm_map in (input_perm, output_perm):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())
    num_experts, N, K_b = B.shape
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"
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
    # One-pass MX pre-quant (bit-exact with an inline quant: group-32 boundaries align).
    A_q, A_s = mxfp_act_quant(A)
    C = A.new_empty((S, N), dtype=output_dtype)
    num_sms = sm_count(A.device.index)

    with device_context(A.device):
        compile_time_only_triton_wrap(mxfp_dynamic_matmul_grouped_kernel)[(num_sms,)](
            A_q,
            A_s,
            B,
            bs_u8,
            C,
            input_perm if input_perm is not None else expert_start,  # dummy ptr
            output_perm if output_perm is not None else expert_start,  # dummy ptr
            expert_start,
            S,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            C.stride(0),
            C.stride(1),
            num_experts=num_experts,
            tokens_per_expert_bit_length=int(
                (S + num_experts - 1) // num_experts
            ).bit_length(),
            HAS_INPUT_PERM=input_perm is not None,
            HAS_OUTPUT_PERM=output_perm is not None,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
        )
    return C


def matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    scheduling: GroupedScheduling,
    block_size: list[int] | None = None,
    output_dtype: torch.dtype | None = None,
    input_gather: str = "unordered",
    output_scatter: str = "unordered",
) -> torch.Tensor:
    """Grouped quantized matmul dispatcher (W8A8 FP8 or W4A8 FP4). ``scheduling`` is one
    ``compute_grouped_scheduling`` pass, shared by every grouped GEMM of the layer; the
    maps are applied per tile (the expert sort is virtual — nothing is physically
    permuted). ``input_gather`` / ``output_scatter`` give the row order of ``A`` / ``C``:

    - ``"unordered"`` (default): token-major — the input side gathers via ``perm_token``
      (for gate_up that reads straight from ``hidden``, no replication); the output side
      scatters to routed rows ``(t*K + j)`` via ``perm_routed``.
    - ``"ordered"``: expert-ordered, the frame grouped GEMMs hand to each other — no map
      on that side.

    The unfused MoE chain is one scheduling + one gather + one scatter: gate_up with
    ``output_scatter="ordered"`` (the intermediate stays expert-ordered), then down with
    ``input_gather="ordered"``.
    EP-sentinel routes fall past ``expert_start[-1]`` and are never touched (their
    output rows are uninitialized). ``output_dtype`` defaults to ``A.dtype``.

    Routes by weight dtype and ``block_size``:
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` (shape ``[num_experts, N, K//32]``) → ``mxfp_dynamic_matmul_grouped``
      (``block_size`` ignored; tile + dot path autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul_grouped``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul_grouped``.
    """
    assert input_gather in ("ordered", "unordered"), input_gather
    assert output_scatter in ("ordered", "unordered"), output_scatter
    input_perm = None if input_gather == "ordered" else scheduling.perm_token
    output_perm = None if output_scatter == "ordered" else scheduling.perm_routed
    expert_start = scheduling.expert_start

    if is_mxfp(B, Bs):
        return mxfp_dynamic_matmul_grouped(
            A, B, Bs, expert_start, output_dtype, input_perm, output_perm
        )

    if is_tensor_wide(block_size, B):
        return w8a8_tensor_dynamic_fp8_matmul_grouped(
            A, B, Bs, expert_start, output_dtype, input_perm, output_perm
        )

    return w8a8_block_dynamic_fp8_matmul_grouped(
        A, B, Bs, expert_start, block_size, output_dtype, input_perm, output_perm
    )
