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
"""Decode and temporary-workspace routes for coarse-scale FP8 W8A16 GEMMs."""

import torch
import triton
import triton.language as tl

from . import compat
from .bayesian_autotuner import bayesian_autotune
from .epilogue import bias_strides, gemm_epilogue
from .loading.scales import block_fp8_scale_ptrs, decode_group_scale
from .scheduling import resolve_row


@bayesian_autotune(
    [
        triton.Config({"BLOCK": b}, num_warps=w, num_stages=s)
        for b in (256, 512, 1024, 2048, 4096, 8192)
        for w in (4, 8, 16)
        for s in (1, 2, 3)
    ],
    key=["E", "N", "K", "SCALE_N", "SCALE_K", "GATE_SCALES"],
    n_trials=100,
    do_bench=compat.replay_benchmark,
)
@triton.jit
def _dequantize(
    B,
    Bs,
    Out,
    E,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sk: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_K: tl.constexpr,
    GATE_SCALES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n, k = i // K, i % K
    value = tl.load(
        B + expert * stride_be + n * stride_bn + k * stride_bk, i < N * K, other=0.0
    )
    scale = tl.load(
        block_fp8_scale_ptrs(
            Bs, expert, n, k, stride_se, stride_sn, stride_sk,
            SCALE_N, SCALE_K, GATE_SCALES,
        ),
        i < N * K,
        other=0.0,
    )
    tl.store(
        Out + expert * N * K + i,
        value.to(tl.float32) * decode_group_scale(scale),
        i < N * K,
    )


def dequantize_weight(B, Bs, dtype, scale_n, scale_k, gate_scales=False):
    """Dequantize into a temporary contiguous BF16/FP16 workspace, never a persistent copy."""
    rows, k = B.shape[-2:]
    experts = B.shape[0] if B.ndim == 3 else 1
    out = torch.empty(B.shape, device=B.device, dtype=dtype)
    compat.compile_time_only_triton_wrap(_dequantize)[
        lambda meta: (experts, triton.cdiv(rows * k, meta["BLOCK"]))
    ](
        B,
        Bs,
        out,
        experts,
        rows,
        k,
        B.stride(0) if B.ndim == 3 else 0,
        B.stride(-2),
        B.stride(-1),
        Bs.stride(0) if Bs.ndim == 3 else 0,
        Bs.stride(-2),
        Bs.stride(-1),
        SCALE_N=scale_n,
        SCALE_K=scale_k,
        GATE_SCALES=gate_scales,
    )
    return out


@bayesian_autotune(
    [
        triton.Config({"BLOCK": b}, num_warps=w, num_stages=1)
        for b in (128, 256, 512, 1024)
        for w in (4, 8)
    ],
    key=["M", "N", "E", "SPLIT_K", "GATE"],
    n_trials=100,
    do_bench=compat.replay_benchmark,
)
@triton.jit
def _finish_decode(
    P,
    C,
    Bias,
    ExpertIds,
    ExpertStart,
    ScatterIdx,
    M,
    N,
    E: tl.constexpr,
    stride_bias,
    stride_bias_e,
    stride_pm,
    SPLIT_K: tl.constexpr,
    GATE: tl.constexpr,
    ACT_FN: tl.constexpr,
    SWIGLU_ALPHA: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    expert, _, destination, valid = resolve_row(
        row, ExpertIds, ExpertStart, None, ScatterIdx, E
    )
    if not valid:
        return
    width: tl.constexpr = 2 * BLOCK if GATE else BLOCK
    n = tile * width + tl.arange(0, width)
    split = tl.arange(0, SPLIT_K)
    values = tl.load(
        P + row * stride_pm + split[:, None] * N + n[None, :],
        n[None, :] < N, other=0.0,
    )
    acc = tl.sum(values, 0).reshape(1, width)
    out_row = tl.full((1,), destination, tl.int64)
    gemm_epilogue(
        C, None, acc, out_row, tile, 0, tl.full((1,), True, tl.int1),
        N // 2 if GATE else N, 1, 0, 0, 1, BLOCK, GATE, None, 1,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, OUTPUT_DTYPE,
        N_COLS=N // 2 if GATE else N,
        Bias=Bias, stride_bias_e=stride_bias_e, stride_bias_n=stride_bias,
        global_row=expert,
    )


def _finish_after_bench(args, exception):
    if exception is None and args["SPLIT_K"] > 1:
        _launch_finish(args)


def _launch_finish(args):
    n = args["N"] // 2 if args["GATE"] else args["N"]
    _finish_decode[lambda meta: (args["M"], triton.cdiv(n, meta["BLOCK"]))](
        args["P"],
        args["C"],
        args["Bias"],
        args["ExpertIds"],
        args["ExpertStart"],
        args["ScatterIdx"],
        args["M"],
        args["N"],
        args["E"],
        args["stride_bias"],
        args["stride_bias_e"],
        args["stride_pm"],
        args["SPLIT_K"],
        args["GATE"],
        args["ACT_FN"],
        args["SWIGLU_ALPHA"],
        args["SWIGLU_LIMIT"],
        args["SIMULATE_UNFUSED"],
        args["OUTPUT_DTYPE"],
    )


def _prune_decode(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    # More splits than 128-value strips only adds empty CTAs/reduction work.
    kept = [
        c for c in configs if c.kwargs["SPLIT_K"] <= max(1, triton.cdiv(args["K"], 128))
    ]
    return kept or configs


@bayesian_autotune(
    [
        triton.Config(
            {"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": split}, num_warps=w, num_stages=s
        )
        for bn in (16, 32, 64)
        for bk in (128, 256, 512, 1024)
        for split in (1, 2, 4, 8, 16)
        for w in (2, 4, 8, 16)
        for s in (1, 2, 3, 4, 5, 6)
    ],
    key=["M", "N", "K", "E", "SCALE_N", "SCALE_K", "GATE", "GATE_SCALES"],
    n_trials=100,
    do_bench=compat.replay_benchmark,
    path_anchor_axes=("SPLIT_K",),
    # Include the reduction and fused epilogue when choosing the split count.
    post_hook=_finish_after_bench,
    prune_configs_by={"early_config_prune": _prune_decode},
)
@triton.jit
def _decode_partial(
    A,
    B,
    Bs,
    P,
    C,
    Bias,
    ExpertIds,
    ExpertStart,
    GatherIdx,
    ScatterIdx,
    M,
    N,
    K: tl.constexpr,
    E: tl.constexpr,
    stride_be,
    stride_se,
    stride_bias_e,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_bias,
    stride_pm,
    SCALE_N: tl.constexpr,
    SCALE_K: tl.constexpr,
    GATE: tl.constexpr,
    GATE_SCALES: tl.constexpr,
    ACT_FN: tl.constexpr,
    SWIGLU_ALPHA: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    row, tile, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    expert, source, destination, valid = resolve_row(
        row, ExpertIds, ExpertStart, GatherIdx, ScatterIdx, E
    )
    B += expert * stride_be
    Bs += expert * stride_se
    n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    if SCALE_K >= K:
        chunk = tl.cdiv(tl.cdiv(K, 128), SPLIT_K) * 128
    else:
        chunk = tl.cdiv(tl.cdiv(K, SCALE_K), SPLIT_K) * SCALE_K
    start = split * chunk
    end = tl.minimum(K, start + chunk)
    acc = tl.full((BLOCK_N,), 0.0, tl.float32)
    for off in range(tl.cdiv(chunk, BLOCK_K)):
        k = start + off * BLOCK_K + kk
        a = tl.load(
            A + source * stride_am + k * stride_ak, (k < end) & valid, other=0.0
        ).to(tl.float32)
        w = tl.load(
            B + n[:, None] * stride_bn + k[None, :] * stride_bk,
            (n[:, None] < N) & (k[None, :] < end) & valid,
            other=0.0,
        ).to(tl.float32)
        if SCALE_K < K and BLOCK_K >= SCALE_K and BLOCK_K % SCALE_K == 0:
            groups: tl.constexpr = BLOCK_K // SCALE_K
            kg = start // SCALE_K + off * groups + tl.arange(0, groups)
            scale = tl.load(
                block_fp8_scale_ptrs(
                    Bs, 0, n[:, None], kg[None, :] * SCALE_K, 0, stride_sn, stride_sk,
                    SCALE_N, SCALE_K, GATE_SCALES,
                ),
                (n[:, None] < N) & (kg[None, :] * SCALE_K < end) & valid,
                other=0.0,
            )
            w = (
                w.reshape(BLOCK_N, groups, SCALE_K)
                * decode_group_scale(scale)[:, :, None]
            ).reshape(BLOCK_N, BLOCK_K)
        else:
            scale = tl.load(
                block_fp8_scale_ptrs(
                    Bs, 0, n[:, None], k[None, :], 0, stride_sn, stride_sk,
                    SCALE_N, SCALE_K, GATE_SCALES,
                ),
                (n[:, None] < N) & (k[None, :] < end) & valid,
                other=0.0,
            )
            w *= decode_group_scale(scale)
        # Preserve the original W8A16 contract: round the scaled weight to A's dtype.
        w = w.to(A.dtype.element_ty).to(tl.float32)
        acc += tl.sum(w * a[None, :], 1)
    if SPLIT_K == 1:
        if valid:
            logical_width: tl.constexpr = BLOCK_N // 2 if GATE else BLOCK_N
            out_row = tl.full((1,), destination, tl.int64)
            gemm_epilogue(
                C, None, acc.reshape(1, BLOCK_N), out_row, tile, 0,
                tl.full((1,), True, tl.int1), N // 2 if GATE else N, 1, 0, 0,
                1, logical_width, GATE, None, 1,
                ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, OUTPUT_DTYPE,
                N_COLS=N // 2 if GATE else N,
                Bias=Bias, stride_bias_e=stride_bias_e, stride_bias_n=stride_bias,
                global_row=expert,
            )
    else:
        tl.store(P + row * stride_pm + split * N + n, acc, n < N)


def decode_matmul(
    A,
    B,
    Bs,
    scale_n,
    scale_k,
    output_dtype,
    *,
    gate=False,
    gate_scales=False,
    bias=None,
    act_fn="silu",
    swiglu_alpha=None,
    swiglu_limit=None,
    simulate_unfused=False,
    expert_ids=None,
    expert_start=None,
    gather_idx=None,
    scatter_idx=None,
    num_rows=None,
):
    """One-token W8A16 GEMM with fused bias/GLU and a tuned optional split-K reduction."""
    m = A.numel() // A.shape[-1] if num_rows is None else num_rows
    k, n = A.shape[-1], B.shape[-2]
    experts = B.shape[0] if B.ndim == 3 else 1
    output_shape = A.shape[:-1] if num_rows is None else (m,)
    out = torch.empty(
        output_shape + (n // 2 if gate else n,), dtype=output_dtype, device=A.device
    )
    if compat.launches_skipped() or m == 0:
        return out
    configs = _prune_decode(_decode_partial.configs, {"K": k})
    max_splits = max(config.kwargs["SPLIT_K"] for config in configs)
    partial = torch.empty((m, max_splits, n), device=A.device, dtype=torch.float32)
    stride_bias_e, stride_bias = bias_strides(bias)
    args = dict(
        A=A, B=B, Bs=Bs, P=partial, C=out, Bias=bias,
        ExpertIds=expert_ids, ExpertStart=expert_start,
        GatherIdx=gather_idx, ScatterIdx=scatter_idx,
        M=m, N=n, K=k, E=experts,
        stride_be=B.stride(0) if B.ndim == 3 else 0,
        stride_se=Bs.stride(0) if Bs.ndim == 3 else 0,
        stride_bias_e=stride_bias_e,
        stride_am=A.stride(-2), stride_ak=A.stride(-1),
        stride_bn=B.stride(-2), stride_bk=B.stride(-1),
        stride_sn=Bs.stride(-2), stride_sk=Bs.stride(-1),
        stride_bias=stride_bias, stride_pm=partial.stride(0),
        SCALE_N=scale_n, SCALE_K=scale_k, GATE=gate, GATE_SCALES=gate_scales,
        ACT_FN=act_fn, SWIGLU_ALPHA=swiglu_alpha, SWIGLU_LIMIT=swiglu_limit,
        SIMULATE_UNFUSED=simulate_unfused, OUTPUT_DTYPE=compat.tl_dtype(output_dtype),
    )
    _decode_partial[
        lambda config: (m, triton.cdiv(n, config["BLOCK_N"]), config["SPLIT_K"])
    ](**args)
    args["SPLIT_K"] = _decode_partial.best_config.kwargs["SPLIT_K"]
    if args["SPLIT_K"] > 1:
        _launch_finish(args)
    return out
