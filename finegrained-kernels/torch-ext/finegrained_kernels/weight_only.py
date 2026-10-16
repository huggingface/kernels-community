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
from .epilogue import glu
from .loading.scales import decode_group_scale


@triton.jit
def _decode_route(row, ExpertIds, ExpertStart, GatherIdx, ScatterIdx, E: tl.constexpr):
    valid = True
    expert = 0
    if ExpertIds is not None:
        expert = tl.load(ExpertIds + row)
        valid = (expert >= 0) & (expert < E)
    elif ExpertStart is not None:
        offsets = tl.arange(0, triton.next_power_of_2(E))
        starts = tl.load(ExpertStart + offsets, offsets < E, other=2147483647)
        expert = tl.sum((row >= starts).to(tl.int32), 0) - 1
        valid = (expert >= 0) & (row < tl.load(ExpertStart + E))
    source = row
    destination = row
    if GatherIdx is not None:
        source = tl.load(GatherIdx + row)
    if ScatterIdx is not None:
        destination = tl.load(ScatterIdx + row)
    return expert, source, destination, valid


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
    expert = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n, k = i // K, i % K
    if GATE_SCALES:
        sn = (n // 2 // SCALE_N) * 2 + n % 2
    else:
        sn = n // SCALE_N
    value = tl.load(
        B + expert * stride_be + n * stride_bn + k * stride_bk, i < N * K, other=0.0
    )
    scale = tl.load(
        Bs + expert * stride_se + sn * stride_sn + (k // SCALE_K) * stride_sk,
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
    GatherIdx,
    ScatterIdx,
    M,
    N,
    E: tl.constexpr,
    stride_bias,
    stride_bias_e,
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
    expert, _, destination, valid = _decode_route(
        row, ExpertIds, ExpertStart, GatherIdx, ScatterIdx, E
    )
    if Bias is not None:
        Bias += expert * stride_bias_e
    out_n = N // 2 if GATE else N
    n = tile * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT_K)
    raw_n = 2 * n if GATE else n
    offset = row * 16 * N + split[:, None] * N + raw_n[None, :]
    values = tl.load(P + offset, n[None, :] < out_n, other=0.0)
    value = tl.sum(values, 0)
    if Bias is not None:
        value += tl.load(Bias + raw_n * stride_bias, (n < out_n) & valid, other=0.0).to(
            tl.float32
        )
    if GATE:
        up = tl.sum(tl.load(P + offset + 1, n[None, :] < out_n, other=0.0), 0)
        if Bias is not None:
            up += tl.load(
                Bias + (raw_n + 1) * stride_bias, (n < out_n) & valid, other=0.0
            ).to(tl.float32)
        value = glu(
            value,
            up,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            OUTPUT_DTYPE,
        )
    tl.store(C + destination * out_n + n, value, (n < out_n) & valid)


def _finish_after_bench(args, exception):
    if exception is None and (args["SPLIT_K"] > 1 or args["GATE"]):
        _launch_finish(args)


def _launch_finish(args):
    n = args["N"] // 2 if args["GATE"] else args["N"]
    _finish_decode[lambda meta: (args["M"], triton.cdiv(n, meta["BLOCK"]))](
        args["P"],
        args["C"],
        args["Bias"],
        args["ExpertIds"],
        args["ExpertStart"],
        args["GatherIdx"],
        args["ScatterIdx"],
        args["M"],
        args["N"],
        args["E"],
        args["stride_bias"],
        args["stride_bias_e"],
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
    expert, source, destination, valid = _decode_route(
        row, ExpertIds, ExpertStart, GatherIdx, ScatterIdx, E
    )
    B += expert * stride_be
    Bs += expert * stride_se
    if Bias is not None:
        Bias += expert * stride_bias_e
    n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    if SCALE_K >= K:
        chunk = tl.cdiv(tl.cdiv(K, 128), SPLIT_K) * 128
    else:
        chunk = tl.cdiv(tl.cdiv(K, SCALE_K), SPLIT_K) * SCALE_K
    start = split * chunk
    end = tl.minimum(K, start + chunk)
    if GATE_SCALES:
        sn = (n // 2 // SCALE_N) * 2 + n % 2
    else:
        sn = n // SCALE_N
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
                Bs + sn[:, None] * stride_sn + kg[None, :] * stride_sk,
                (n[:, None] < N) & (kg[None, :] * SCALE_K < end) & valid,
                other=0.0,
            )
            w = (
                w.reshape(BLOCK_N, groups, SCALE_K)
                * decode_group_scale(scale)[:, :, None]
            ).reshape(BLOCK_N, BLOCK_K)
        else:
            scale = tl.load(
                Bs + sn[:, None] * stride_sn + (k[None, :] // SCALE_K) * stride_sk,
                (n[:, None] < N) & (k[None, :] < end) & valid,
                other=0.0,
            )
            w *= decode_group_scale(scale)
        # Preserve the original W8A16 contract: round the scaled weight to A's dtype.
        w = w.to(A.dtype.element_ty).to(tl.float32)
        acc += tl.sum(w * a[None, :], 1)
    if SPLIT_K == 1 and not GATE:
        if Bias is not None:
            acc += tl.load(Bias + n * stride_bias, (n < N) & valid, other=0.0).to(
                tl.float32
            )
        tl.store(C + destination * N + n, acc, (n < N) & valid)
    else:
        tl.store(P + row * 16 * N + split * N + n, acc, n < N)


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
    if compat._SKIP_LAUNCHES_MIRROR or m == 0:
        return out
    partial = torch.empty((m, 16, n), device=A.device, dtype=torch.float32)
    meta = dict(
        SCALE_N=scale_n,
        SCALE_K=scale_k,
        GATE=gate,
        GATE_SCALES=gate_scales,
        ACT_FN=act_fn,
        SWIGLU_ALPHA=swiglu_alpha,
        SWIGLU_LIMIT=swiglu_limit,
        SIMULATE_UNFUSED=simulate_unfused,
        OUTPUT_DTYPE=compat.tl_dtype(output_dtype),
    )
    stride_bias = bias.stride(-1) if bias is not None else 0
    stride_bias_e = bias.stride(0) if bias is not None and bias.ndim == 2 else 0
    routing = dict(
        ExpertIds=expert_ids,
        ExpertStart=expert_start,
        GatherIdx=gather_idx,
        ScatterIdx=scatter_idx,
        E=experts,
        stride_bias_e=stride_bias_e,
    )
    _decode_partial[
        lambda config: (m, triton.cdiv(n, config["BLOCK_N"]), config["SPLIT_K"])
    ](
        A,
        B,
        Bs,
        partial,
        out,
        bias,
        expert_ids,
        expert_start,
        gather_idx,
        scatter_idx,
        m,
        n,
        k,
        experts,
        B.stride(0) if B.ndim == 3 else 0,
        Bs.stride(0) if Bs.ndim == 3 else 0,
        stride_bias_e,
        A.stride(-2),
        A.stride(-1),
        B.stride(-2),
        B.stride(-1),
        Bs.stride(-2),
        Bs.stride(-1),
        stride_bias,
        **meta,
    )
    splits = _decode_partial.best_config.kwargs["SPLIT_K"]
    if splits > 1 or gate:
        _launch_finish(
            dict(
                P=partial,
                C=out,
                Bias=bias,
                M=m,
                N=n,
                stride_bias=stride_bias,
                **routing,
                SPLIT_K=splits,
                **meta,
            )
        )
    return out
