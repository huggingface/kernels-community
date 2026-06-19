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


def is_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """MXFP8 weight/scale pair: E4M3 weights with UE8M0 group-32 scales — last dim
    ``scale.shape[-1] == weight.shape[-1] // MX_SCALE_GROUP_K``, matching leading dims.
    Works for 2D ``(N, K)`` and 3D ``(E, N, K)`` weights. The group-32 layout is what
    separates MXFP8 from 128-block FP8 (which may also carry UE8M0 scales)."""
    return (
        weight.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.float8_e8m0fnu
        and scale.ndim == weight.ndim
        and scale.shape[:-1] == weight.shape[:-1]
        and scale.shape[-1] == weight.shape[-1] // MX_SCALE_GROUP_K
    )


def is_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """MXFP4 weight/scale pair: packed E2M1 weights (``int8``, two codes/byte) with
    UE8M0 group-32 scales — ``scale.shape[-1] == weight.shape[-1] * NIBBLES_PER_BYTE //
    MX_SCALE_GROUP_K`` (unpacked K = ``2 * K_half``), matching leading dims. 2D or 3D."""
    return (
        weight.dtype == torch.int8
        and scale.dtype == torch.float8_e8m0fnu
        and scale.ndim == weight.ndim
        and scale.shape[:-1] == weight.shape[:-1]
        and scale.shape[-1] == weight.shape[-1] * NIBBLES_PER_BYTE // MX_SCALE_GROUP_K
    )


def is_mxfp(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """Any MX weight/scale pair — MXFP8 (``float8_e4m3fn``, one value/byte) or MXFP4
    (``int8``, two E2M1 codes/byte), both with UE8M0 group-32 scales. ``values_per_byte``
    folds the two cases: the scale's last dim covers the unpacked K
    (``weight.shape[-1] * values_per_byte``) in groups of ``MX_SCALE_GROUP_K``. The
    dispatchers route on this; the op picks the format from ``weight.dtype``."""
    if weight.dtype == torch.float8_e4m3fn:
        values_per_byte = 1
    elif weight.dtype == torch.int8:
        values_per_byte = NIBBLES_PER_BYTE
    else:
        return False
    return (
        scale.dtype == torch.float8_e8m0fnu
        and scale.ndim == weight.ndim
        and scale.shape[:-1] == weight.shape[:-1]
        and scale.shape[-1] == weight.shape[-1] * values_per_byte // MX_SCALE_GROUP_K
    )


def is_tensor_wide(block_size, weight: torch.Tensor) -> bool:
    """True when ``block_size`` selects per-tensor (tensor-dynamic) scaling: ``None`` or
    equal to the weight's full ``(N, K)`` — one scale block spanning the whole matrix.
    Handles 2D ``(N, K)`` and 3D ``(E, N, K)`` weights via the last two dims."""
    return block_size is None or (
        block_size[0] == weight.shape[-2] and block_size[1] == weight.shape[-1]
    )


# The per-token batched/fused kernels are decode-shaped — one routed row per program, so
# BLOCK_SIZE_M is 1 (a larger tile would just recompute the same row). The 2D matmul sizes
# its M tile to the workload instead, via ``adaptive_block_size_m``.
DECODE_BLOCK_SIZE_M = 1


def adaptive_block_size_m(target_m: int) -> int:
    """Smallest power-of-2 >= ``target_m``, floored at 16 and capped at 128.

    Used by all matmul wrappers (single / batched / grouped) to size the M tile
    to the workload — small per-expert M wants smaller tiles, large M caps out
    at 128 to keep register pressure bounded. Pass ``M`` for single matmul, or
    ``(S + E - 1) // E`` (avg tokens per expert) for batched/grouped.
    """
    return min(max(triton.next_power_of_2(target_m), 16), 128)


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


def get_mxfp_autotuning_configs(pre_hook=None):
    """Autotune grid for the MXFP8 ``USE_DOT_SCALED`` kernels (batched matmul + fused MoE).

    ``USE_DOT_SCALED`` picks the MMA: True → ``tl.dot_scaled`` (native M=128 scaled MMA,
    wide K — wins once the grid saturates, ~S≥32); False → fp8 ``tl.dot`` + per-group
    software rescale, which needs exactly one scale group per K-step (``BLOCK_SIZE_K == 32``)
    and wins at small S where the scaled MMA's M→128 pad is pure waste. The ``if`` keeps
    only that valid pairing; ``num_warps=16`` and ``BLOCK_SIZE_K=64`` are omitted (always
    dead at M=1 per a config sweep), and the Bayesian tuner prunes the rest per workload.
    """
    return [
        triton.Config(
            {"USE_DOT_SCALED": use_dot_scaled, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk},
            num_warps=w,
            num_stages=s,
            pre_hook=pre_hook,
        )
        for use_dot_scaled in [False, True]
        for bn in [32, 64, 128, 256]
        for bk in [32, 128, 256]
        for s in [2, 3, 4, 5, 6]
        for w in [2, 4, 8]
        if (bk == 32) != use_dot_scaled
    ]


@functools.lru_cache(maxsize=None)
def sm_shared_memory_limit(device_index: int) -> int:
    """Max dynamic shared memory per block (bytes) for a CUDA device — the cap Triton
    reports as the 'Hardware limit' on an ``out of resource: shared memory`` failure
    (~232 KB on B200, ~227 KB on H100, much less on older/consumer parts). Queried from
    the driver so the prune adapts to the hardware instead of hardcoding one GPU."""
    try:
        return triton.runtime.driver.active.utils.get_device_properties(device_index)[
            "max_shared_mem"
        ]
    except Exception:
        return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


def smem_config_pruner(act_bytes: int, n_weight_tiles: int, weight_bytes: int = 1):
    """Build an ``early_config_prune`` that drops configs whose pipelined operand shared
    memory would overflow the SM — the source of ``out of resource: shared memory`` autotune
    failures (and the wasted compiles they cause).

    Per-stage estimate (bytes) = ``BK · (act_bytes·BM + weight_bytes·n_weight_tiles·BN)``:
    one activation tile ``[BM, BK]`` plus ``n_weight_tiles`` weight tiles ``[BN, BK]``
    (gate_up fuses 2; down has 1), times ``num_stages``. ``BM`` is the routing-derived tile,
    read from the launch args. MXFP4 packs 2 weights/byte, so ``weight_bytes=1`` (MXFP8) is a
    safe upper bound. The limit is read from the active device. Never returns empty — keeps
    the smallest-footprint config as a fallback."""
    def prune(configs, named_args, **kwargs):
        # The estimate needs all three tile dims. Each is either an autotuned config meta
        # (MX) or a launch arg (block-dynamic), so look in both; raise clearly if missing.
        all_args = {**named_args, **kwargs}
        limit = sm_shared_memory_limit(torch.cuda.current_device())

        def dim(c, name):
            v = c.kwargs.get(name, all_args.get(name))
            if v is None:
                raise ValueError(
                    f"smem_config_pruner needs {name} (autotune config meta or launch arg) "
                    "to estimate shared memory; none found."
                )
            return v

        def smem(c):
            BM, BN, BK = (dim(c, n) for n in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
            return c.num_stages * BK * (act_bytes * BM + weight_bytes * n_weight_tiles * BN)

        kept = [c for c in configs if smem(c) <= limit]
        return kept or [min(configs, key=smem)]

    return prune


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
def mxfp_act_quant_inline(
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
def decode_ue8m0_scale(scale):
    """Decode a UE8M0 weight scale to fp32: when it was loaded as ``uint8`` exponent
    bits, ``value = 2^(exp - 127)``, built directly as the fp32 bit pattern. fp32
    scales (block-dynamic with float scales) pass through. The dtype branch is a
    compile-time constant, so only the taken path is emitted (single return — Triton
    requires all ``return`` statements to share a type)."""
    if scale.dtype == tl.uint8:
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    return scale


@triton.jit
def _e2m1_code_to_f32(code):
    """One E2M1 4-bit code -> fp32. Layout ``[sign | exp(2) | mant(1)]``; the 8
    magnitudes are ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` (exp==0 is the 0/0.5 subnormal)."""
    code = code.to(tl.int32)
    s = (code >> 3) & 1
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    pow2 = tl.exp2((e - 1).to(tl.float32))  # e in 0..3 -> 0.5, 1, 2, 4
    mag = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * pow2)
    return (1.0 - 2.0 * s.to(tl.float32)) * mag


@triton.jit
def mxfp4_e2m1_to_e4m3(b_packed):
    """Unpack packed MXFP4 (E2M1, two nibbles/byte along K) to E4M3, doubling the K
    (row) dim: ``(R, C) uint8 -> (2R, C) E4M3``. E2M1's 8 magnitudes are all exact in
    E4M3, so this is lossless — it lets the FP8 ``tl.dot`` path stand in for
    ``tl.dot_scaled`` at decode (avoiding its M->128 pad). K order is the low nibble
    first: ``[byte0_lo, byte0_hi, byte1_lo, ...]``."""
    lo = _e2m1_code_to_f32(b_packed & 0xF)
    hi = _e2m1_code_to_f32(b_packed >> 4)
    # interleave along the K (row) dim via trans -> interleave-last-dim -> trans back
    unpacked = tl.trans(tl.interleave(tl.trans(lo), tl.trans(hi)))
    return unpacked.to(tl.float8e4nv)


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
            x,
            y,
            s,
            BLOCK_SIZE=block_size,
            PADDED_BLOCK=triton.next_power_of_2(block_size),
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
