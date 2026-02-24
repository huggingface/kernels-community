"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import List, Optional, Tuple

from ._ops import ops, add_op_namespace_prefix
from torch.nn.functional import scaled_dot_product_attention as sdpa


# ---------------------------------------------------------------------------
# Low-level ops with torch.compile support (custom_op + register_fake)
# ---------------------------------------------------------------------------

@torch.library.custom_op(
    add_op_namespace_prefix("mha_fwd"), mutates_args=(), device_types="cuda"
)
def mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sfq: torch.Tensor,
    sfk: torch.Tensor,
    sfv: torch.Tensor,
    delta_s: torch.Tensor,
    unpadded_k: int,
    out: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    per_block_mean: bool,
    is_bf16: bool,
) -> List[torch.Tensor]:
    return ops.mha_fwd(
        q, k, v, sfq, sfk, sfv, delta_s,
        unpadded_k, out, softmax_scale, is_causal,
        per_block_mean, is_bf16,
    )


@torch.library.register_fake(add_op_namespace_prefix("mha_fwd"))
def mha_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sfq: torch.Tensor,
    sfk: torch.Tensor,
    sfv: torch.Tensor,
    delta_s: torch.Tensor,
    unpadded_k: int,
    out: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    per_block_mean: bool,
    is_bf16: bool,
) -> List[torch.Tensor]:
    batch_size = q.size(0)
    num_heads = q.size(1)
    seqlen_q = q.size(2)
    head_size_packed = q.size(3)
    unpacked_head_size = head_size_packed * 2
    dtype = torch.bfloat16 if is_bf16 else torch.float16
    fake_out = torch.empty(
        (batch_size, num_heads, seqlen_q, unpacked_head_size),
        dtype=dtype, device=q.device,
    )
    fake_lse = torch.empty(
        (batch_size, num_heads, seqlen_q),
        dtype=torch.float32, device=q.device,
    )
    return [fake_out, fake_lse]


@torch.library.custom_op(
    add_op_namespace_prefix("scaled_fp4_quant"),
    mutates_args=("output", "output_sf"),
    device_types="cuda",
)
def scaled_fp4_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    tensor_layout: int,
) -> None:
    ops.scaled_fp4_quant(input, output, output_sf, tensor_layout)


@torch.library.register_fake(add_op_namespace_prefix("scaled_fp4_quant"))
def scaled_fp4_quant_fake(
    input: torch.Tensor,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    tensor_layout: int,
) -> None:
    pass


@torch.library.custom_op(
    add_op_namespace_prefix("scaled_fp4_quant_permute"),
    mutates_args=("output", "output_sf"),
    device_types="cuda",
)
def scaled_fp4_quant_permute(
    input: torch.Tensor,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    tensor_layout: int,
) -> None:
    ops.scaled_fp4_quant_permute(input, output, output_sf, tensor_layout)


@torch.library.register_fake(add_op_namespace_prefix("scaled_fp4_quant_permute"))
def scaled_fp4_quant_permute_fake(
    input: torch.Tensor,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    tensor_layout: int,
) -> None:
    pass


@torch.library.custom_op(
    add_op_namespace_prefix("scaled_fp4_quant_trans"),
    mutates_args=("output", "output_sf"),
    device_types="cuda",
)
def scaled_fp4_quant_trans(
    input: torch.Tensor,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    tensor_layout: int,
) -> None:
    ops.scaled_fp4_quant_trans(input, output, output_sf, tensor_layout)


@torch.library.register_fake(add_op_namespace_prefix("scaled_fp4_quant_trans"))
def scaled_fp4_quant_trans_fake(
    input: torch.Tensor,
    output: torch.Tensor,
    output_sf: torch.Tensor,
    tensor_layout: int,
) -> None:
    pass


# ---------------------------------------------------------------------------
# Triton kernel for grouped mean subtraction
# ---------------------------------------------------------------------------

@triton.jit
def _group_mean_kernel(
    q_ptr,
    q_out_ptr,
    qm_out_ptr,
    B, H, L, D: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qmb, stride_qmh, stride_qml, stride_qmd,
    GROUP_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)

    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)

    q_offsets = (
        pid_b * stride_qb
        + pid_h * stride_qh
        + offsets[:, None] * stride_ql
        + tl.arange(0, D)[None, :] * stride_qd
    )
    q_group = tl.load(q_ptr + q_offsets)

    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE

    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = (
        pid_b * stride_qmb
        + pid_h * stride_qmh
        + pid_group * stride_qml
        + tl.arange(0, D) * stride_qmd
    )
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)

    grid = (B, H, num_groups)
    _group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE,
    )
    return q_out, qm


# ---------------------------------------------------------------------------
# High-level Python API (ported from sageattn3/api.py)
# ---------------------------------------------------------------------------

def preprocess_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_block_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = map(pad_128, [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s


def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty(
        (B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty(
        (B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty(
        (B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def blockscaled_fp4_attn(
    qlist: Tuple[torch.Tensor, torch.Tensor],
    klist: Tuple[torch.Tensor, torch.Tensor],
    vlist: Tuple[torch.Tensor, torch.Tensor],
    delta_s: torch.Tensor,
    KL: int,
    is_causal: bool = False,
    per_block_mean: bool = True,
    is_bf16: bool = True,
) -> List[torch.Tensor]:
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return mha_fwd(
        qlist[0], klist[0], vlist[0],
        qlist[1], klist[1], vlist[1],
        delta_s, KL, None,
        softmax_scale, is_causal, per_block_mean, is_bf16,
    )


def sageattn3_blackwell(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    per_block_mean: bool = True,
    **kwargs,
) -> torch.Tensor:
    if q.size(-1) >= 256:
        return sdpa(q, k, v, is_causal=is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    qlist = scale_and_quant_fp4(q)
    klist = scale_and_quant_fp4_permute(k)
    vlist = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(
        qlist, klist, vlist, delta_s,
        KL, is_causal, per_block_mean, is_bf16,
    )[0][:, :, :QL, :].contiguous()
    return o_fp4
