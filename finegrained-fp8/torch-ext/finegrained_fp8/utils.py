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

import functools
from contextlib import contextmanager
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix, ops

# ── Format constants ──────────────────────────────────────────────────────────

# FP8 (E4M3) is the main format for weights and activations;
FP8_DTYPE = torch.float8_e4m3fn
# FP4 (E2M1) packs two 4-bit nibbles per byte. MX formats (MXFP4 weights, MXFP8
# E4M3 weights/activations) share one UE8M0 scale per 32-element K-group — the OCP
# MX block size, consumed by ``tl.dot_scaled``. Format constants, not tunables.
NIBBLES_PER_BYTE = 2
MX_SCALE_GROUP_K = 32


# ── Host-side helpers ─────────────────────────────────────────────────────────


@contextmanager
def device_context(device: torch.device):
    """Context manager that sets the active device for any backend (cuda, xpu, etc.)."""
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "device"):
        with backend.device(device):
            yield
    else:
        yield


def ue8m0_as_uint8(scale: torch.Tensor) -> torch.Tensor:
    """View UE8M0 (``float8_e8m0fnu``) weight scales as ``uint8`` for the Triton
    binder, which doesn't recognize the dtype; kernels decode ``2^(exp-127)``
    inline. fp32 (non-UE8M0) scales pass through unchanged."""
    return scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale


def adaptive_block_size_m(target_m: int) -> int:
    """Smallest power-of-2 >= ``target_m``, floored at 16 and capped at 128.

    Used by all matmul wrappers (single / batched / grouped) to size the M tile
    to the workload — small per-expert M wants smaller tiles, large M caps out
    at 128 to keep register pressure bounded. Pass ``M`` for single matmul, or
    ``(S + E - 1) // E`` (avg tokens per expert) for batched/grouped.
    """
    return min(max(triton.next_power_of_2(target_m), 16), 128)


def grouped_tile_layout(
    tokens_per_expert: torch.Tensor,
    block_size_m: int,
    S: int,
    E: int,
) -> Tuple[torch.Tensor, int]:
    """Compute the M-tile layout for grouped kernels.

    Returns ``(tile_offsets, max_m_tiles)``:
    - ``tile_offsets``: int32 (E,) cumulative tile-end per expert, used by
      ``grouped_expert_lookup`` to locate an M-tile's owning expert.
    - ``max_m_tiles``: upper bound on total M-tiles, used as the grid axis-0
      size. Real tile count <= this; surplus programs early-return inside the
      kernel. Keeps the grid data-independent (cuda-graph / torch.compile safe).
    """
    tiles_per_expert = (tokens_per_expert + block_size_m - 1) // block_size_m
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    max_m_tiles = triton.cdiv(S, block_size_m) + E
    return tile_offsets, max_m_tiles


@functools.cache
def get_active_device_type() -> str:
    """Active torch device type for the current Triton backend (``"cuda"``, ``"xpu"``, ...).

    Falls back to ``"cuda"`` when no driver is loaded — Triton raises
    ``RuntimeError: 0 active drivers ([])`` on driverless build boxes, and the
    autotune-config builder is evaluated at module-import time under the
    ``@triton.autotune`` decorator (no kernel launches there, so the default is
    only used to shape the config list).
    """
    try:
        return triton.runtime.driver.active.get_active_torch_device().type
    except RuntimeError:
        return "cuda"


def get_accelerator_autotuning_configs(*, with_block_sizes: bool = False):
    """Autotune search grid for the current accelerator.

    ``num_warps``, ``num_stages`` and ``blocks`` (the ``(BLOCK_SIZE_N, BLOCK_SIZE_K)``
    tile shapes) are fixed up front from ``(is_xpu, with_block_sizes)``, then crossed
    into the config list.

    ``with_block_sizes=True`` sweeps the tile: used by kernels that have no caller
    ``block_size`` to fix it — the MX ``tl.dot_scaled`` paths AND the tensor-dynamic
    FP8 paths. ``with_block_sizes=False`` emits a single empty meta-dict (block-scaled
    kernels take the tile from the caller's ``block_size``).

    The CUDA tile set is a data-driven prune of a B200 sweep across single (BM=128),
    grouped MoE (BM=16/64) and decode (BM=1): winners only ever used these 4 tiles,
    num_warps in {4,8,16}, num_stages in {2,3} (warps=2, stages=4 and 256x256 never
    won) — 108 → 24. (Tuned on dot_scaled; tensor-dynamic tl.dot reuses it.)
    """
    is_xpu = get_active_device_type() == "xpu"
    if with_block_sizes:
        num_stages = [2, 3]
        num_warps = [8, 16] if is_xpu else [4, 8, 16]
        tiles = (
            [(128, 128)] if is_xpu else [(128, 128), (256, 128), (128, 64), (64, 256)]
        )
        blocks = [{"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk} for bn, bk in tiles]
    else:
        num_stages = [2, 3, 4]
        num_warps = [8, 16] if is_xpu else [2, 4, 8, 16]
        blocks = [{}]

    return [
        triton.Config(b, num_warps=w, num_stages=s)
        for b in blocks
        for w in num_warps
        for s in num_stages
    ]


# ── Triton-side helpers (inlined by ``@triton.jit`` callers) ──────────────────


@triton.jit
def fp8_act_quant_inline(a_raw):
    """Inline FP8 (E4M3) activation quant for the W8A8 block-scale path.

    Per-row amax → fp32 scale ``amax/448`` (floored at 1e-12 against zero rows)
    → cast values to FP8. Returns ``(a_fp8, a_s)`` with shapes ``(M, K)`` and
    ``(M,)``.
    """
    a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
    a_fp8 = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)
    return a_fp8, a_s


@triton.jit
def mx_act_quant_inline(
    a_raw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Inline E4M3 activation quant for the MX paths (W4A8 MXFP4 / W8A8 MXFP8).

    Per-row, per-K-group amax → UE8M0 scale (ceil to next power-of-2 via the
    mantissa-nonzero bump trick) → cast values to FP8. Returns ``(a_fp8,
    a_scale_u8)`` with shapes ``(M, K)`` and ``(M, K // SCALE_GROUP_K)``.
    """
    a_groups = tl.reshape(
        a_raw, (BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_K, SCALE_GROUP_K)
    )
    a_s_fp32 = tl.max(tl.abs(a_groups), axis=2) / 448.0
    bits = a_s_fp32.to(tl.int32, bitcast=True)
    # ceil_to_ue8m0: bump exponent by 1 when mantissa is non-zero.
    exp_ceil = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exp_ceil = tl.minimum(tl.maximum(exp_ceil, 1), 254)
    a_scale_u8 = exp_ceil.to(tl.uint8)
    a_s_pow2 = (exp_ceil << 23).to(tl.float32, bitcast=True)
    a_fp8 = tl.reshape(
        a_groups / tl.maximum(a_s_pow2[:, :, None], 1e-12),
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
    ).to(tl.float8e4nv)
    return a_fp8, a_scale_u8


@triton.jit
def grouped_expert_lookup(
    pid_m,
    Offsets,
    TileOffsets,
    stride_offs,
    stride_tile,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Locate the expert owning a grouped-kernel M-tile and compute row offsets.

    Returns ``(expert_id, offs_global_m, row_mask)``:
    - ``expert_id``: int64
    - ``offs_global_m``: ``(BLOCK_SIZE_M,)`` global row indices into A
    - ``row_mask``: ``(BLOCK_SIZE_M,)`` validity mask within the expert's M

    Caller is expected to have already early-returned if ``pid_m`` exceeds
    ``total_tiles`` (``TileOffsets[(NUM_EXPERTS - 1) * stride_tile]``).
    """
    # Binary search: upper_bound(TileOffsets, pid_m). NUM_EXPERTS_BIT_LENGTH is
    # ceil(log2(E))+1, giving one harmless extra iteration; constexpr so the
    # loop unrolls.
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid * stride_tile)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    # Cast to int64 so ``expert_id * stride_be`` doesn't overflow for large E
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
    return expert_id, offs_global_m, row_mask


# ── fp8_act_quant kernel (used by tensor-mode FP8 wrappers) ───────────────────


@triton.jit
def _fp8_act_quant_kernel(
    x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, PADDED_BLOCK: tl.constexpr
):
    # ``tl.arange`` needs a power-of-2 length, so iterate over PADDED_BLOCK (the next
    # power of 2) and mask the tail — lets block_size be non-power-of-2 (e.g. a full
    # row K=14336 in tensor-mode). Masked lanes load 0, which can't affect ``amax``.
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, PADDED_BLOCK)
    mask = cols < BLOCK_SIZE
    offs = pid * BLOCK_SIZE + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0  # float8_e4m3fn max
    y = (x / tl.maximum(s, 1e-12)).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid, s)


@triton_op(add_op_namespace_prefix("fp8_act_quant"), mutates_args=())
def _fp8_act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    grid = (triton.cdiv(x.numel(), block_size),)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    with device_context(x.device):
        wrap_triton(_fp8_act_quant_kernel)[grid](
            x, y, s, BLOCK_SIZE=block_size, PADDED_BLOCK=triton.next_power_of_2(block_size)
        )

    return y, s


def fp8_act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 with per-block dynamic scaling.

    Splits the last dimension of ``x`` into blocks of ``block_size`` elements,
    computes ``scale = max(|x_block|) / 448`` per block, and quantizes to
    ``float8_e4m3fn``.

    Args:
        x: Input tensor in bf16/fp16/fp32. Last dimension must be divisible by
            ``block_size`` and the tensor must be contiguous.
        block_size: Number of elements per quantization block (default: 128).

    Returns:
        A tuple ``(quantized, scales)`` where ``quantized`` has dtype
        ``float8_e4m3fn`` with the same shape as ``x``, and ``scales`` has
        shape ``(*x.shape[:-1], x.shape[-1] // block_size)`` in float32.
    """
    return ops.fp8_act_quant(x, block_size)
