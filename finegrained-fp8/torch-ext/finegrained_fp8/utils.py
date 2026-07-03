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


@functools.lru_cache(maxsize=8)
def sm_count(device_index: int) -> int:
    """Processor (CUDA SM / XPU Xe-core) count for a device (a constant) — cached so callers
    don't re-query it each launch. Read from Triton's active driver so it works on any backend,
    with a CUDA fallback."""
    try:
        return triton.runtime.driver.active.utils.get_device_properties(device_index)[
            "multiprocessor_count"
        ]
    except Exception:
        active_device = get_active_device_type()
        if active_device == "cuda":
            return torch.cuda.get_device_properties(device_index).multi_processor_count
        elif active_device == "xpu":
            return torch.xpu.get_device_properties(device_index).multi_processor_count
        else:
            raise RuntimeError(
                f"Unsupported device type {active_device} for sm_count; only cuda/xpu are supported."
            )


@functools.lru_cache(maxsize=8)
def warp_size(device_index: int = 0) -> int:
    """SIMD sub-group width (threads per warp) of the active target — 32 on Intel PVC/Max and on
    CUDA. Read from Triton's driver so the register estimate adapts to the device instead of
    hardcoding one width. Note the Intel backend may override ``threads_per_warp`` per-kernel at
    compile time; an ``early_config_prune`` runs pre-compile, so this target default is the value
    to estimate against. Falls back to the largest supported sub-group size, then 32."""
    try:
        return triton.runtime.driver.active.get_current_target().warp_size
    except Exception:
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(
                device_index
            )
            return max(props["sub_group_sizes"])
        except Exception:
            return 32


# H-tile each topk_reduce program reduces (one (token, tile) program per BLOCK_H span). The
# reduce is bandwidth-bound and saturates at BLOCK_H>=512: a sweep over {128..2048} x prefill
# shapes put 512/1024/2048 within ~7% (mostly noise) while 128/256 lagged — not worth a tuning
# axis for this small slice of runtime. 512 = smallest that saturates (best at low token counts,
# where bigger tiles waste masked lanes). Power of 2 (tl.arange).
TOPK_REDUCE_BLOCK_H = 512


@triton.jit
def topk_reduce_kernel(
    ProjOut,
    Out,
    ExpertIds,
    H,
    stride_pm,
    stride_ph,
    stride_om,
    stride_oh,
    stride_ei_m,
    stride_ei_k,
    NUM_TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Sum the ``NUM_TOP_K`` flat ``(token, slot)`` rows of ``ProjOut`` into ``Out[token]`` —
    a tight replacement for ``proj_out.view(T, K, H).sum(1)`` (~2.8x faster than torch's
    generic reduce; fp32 accumulate, so bit-identical). Slots whose expert is non-local (EP
    sentinel id ``>= NUM_EXPERTS``) are skipped — the experts kernels never write those rows, so
    they must contribute 0. Caller allocs Out and launches it."""
    t = tl.program_id(0)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for k in tl.static_range(NUM_TOP_K):
        local = tl.load(ExpertIds + t * stride_ei_m + k * stride_ei_k) < NUM_EXPERTS
        acc += tl.load(
            ProjOut + (t * NUM_TOP_K + k) * stride_pm + offs_h * stride_ph,
            mask=mask & local,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        Out + t * stride_om + offs_h * stride_oh,
        acc.to(Out.dtype.element_ty),
        mask=mask,
    )


def ue8m0_as_uint8(scale: torch.Tensor) -> torch.Tensor:
    """View UE8M0 (``float8_e8m0fnu``) weight scales as ``uint8`` for the Triton
    binder, which doesn't recognize the dtype; kernels decode ``2^(exp-127)``
    inline. fp32 (non-UE8M0) scales pass through unchanged."""
    return scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale


def e2m1_as_uint8(weight: torch.Tensor) -> torch.Tensor:
    """View an ``int8``-stored MXFP4 (packed E2M1) weight as ``uint8`` — a zero-cost
    reinterpret. ``tl.dot_scaled`` requires the packed rhs as ``uint8``, so do it once here
    instead of casting in-kernel at every load. E4M3 (MXFP8) weights pass through unchanged."""
    return weight.view(torch.uint8) if weight.dtype == torch.int8 else weight


# UE8M0 group-32 scales arrive either as ``float8_e8m0fnu`` or as raw ``uint8`` — the same 8
# exponent bits, and a common on-disk encoding (e.g. group-32 "mxfp8" checkpoints store the
# scale tensor as uint8). Both are valid MX scales: ``ue8m0_as_uint8`` reinterprets to uint8
# and the kernels decode ``2^(exp-127)`` inline, so the detectors accept either dtype.
UE8M0_SCALE_DTYPES = (torch.float8_e8m0fnu, torch.uint8)


def is_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """MXFP8 weight/scale pair: E4M3 weights with UE8M0 group-32 scales — last dim
    ``scale.shape[-1] == weight.shape[-1] // MX_SCALE_GROUP_K``, matching leading dims.
    Works for 2D ``(N, K)`` and 3D ``(E, N, K)`` weights. The group-32 layout is what
    separates MXFP8 from 128-block FP8 (which may also carry UE8M0 scales)."""
    return (
        weight.dtype == torch.float8_e4m3fn
        and scale.dtype in UE8M0_SCALE_DTYPES
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
        and scale.dtype in UE8M0_SCALE_DTYPES
        and scale.ndim == weight.ndim
        and scale.shape[:-1] == weight.shape[:-1]
        and scale.shape[-1] == (weight.shape[-1] * NIBBLES_PER_BYTE) // MX_SCALE_GROUP_K
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
        scale.dtype in UE8M0_SCALE_DTYPES
        and scale.ndim == weight.ndim
        and scale.shape[:-1] == weight.shape[:-1]
        and scale.shape[-1] == (weight.shape[-1] * values_per_byte) // MX_SCALE_GROUP_K
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


def get_accelerator_autotuning_configs(
    *,
    tune_block_m: bool = False,
    tune_block_nk: bool = False,
):
    """Autotune search grid for the current accelerator.

    ``num_warps``, ``num_stages`` and ``blocks`` (the ``(BLOCK_SIZE_N, BLOCK_SIZE_K)``
    tile shapes) are fixed up front from the accelerator and requested tile axes, then crossed
    into the config list.

    ``tune_block_nk=True`` sweeps the tile: used by kernels that have no caller
    ``block_size`` to fix it — the MX ``tl.dot_scaled`` paths AND the tensor-dynamic
    FP8 paths. ``tune_block_nk=False`` emits a single empty meta-dict (block-scaled
    kernels take the tile from the caller's ``block_size``). ``tune_block_m=True`` also crosses
    in BLOCK_SIZE_M ∈ ``(16, 32, 64, 128)`` (fused grouped, which lets the tuner pick M instead
    of computing it via ``adaptive_block_size_m``).

    The CUDA tile set is a data-driven prune of a B200 sweep across single (BM=128),
    grouped MoE (BM=16/64) and decode (BM=1): winners only ever used these 4 tiles,
    num_warps in {4,8,16}, num_stages in {2,3} (warps=2, stages=4 and 256x256 never
    won) — 108 → 24. (Tuned on dot_scaled; tensor-dynamic tl.dot reuses it.)
    """
    is_xpu = get_active_device_type() == "xpu"
    num_warps = [8, 16] if is_xpu else [2, 4, 8, 16]
    num_stages = [2, 3, 4]

    if tune_block_nk:
        tiles = (
            [(128, 128), (128, 64), (64, 128), (64, 64)]
            if is_xpu
            else [(128, 128), (256, 128), (128, 64), (64, 256)]
        )
        blocks = [{"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk} for bn, bk in tiles]
    else:
        blocks = [{}]

    if tune_block_m:
        blocks = [{**b, "BLOCK_SIZE_M": bm} for b in blocks for bm in (16, 32, 64, 128)]

    return [
        triton.Config(b, num_warps=w, num_stages=s)
        for b in blocks
        for w in num_warps
        for s in num_stages
    ]


def get_mxfp_autotuning_configs(
    pre_hook=None,
    compute_modes=("dot_scaled", "dot", "scalar"),
    memory_modes=(None,),
    tune_block_m=False,
):
    """Autotune grid for the MXFP8 MoE/matmul kernels. ``COMPUTE_MODE`` (a constexpr string) picks
    the multiply-accumulate, and the Bayesian tuner selects it per workload (token count is
    in the key):

    - ``"dot_scaled"``: native group-32 scaled MMA (wide K, ``BLOCK_SIZE_K`` ∈ {128, 256}) —
      wins once the grid saturates (prefill, ~S≥32).
    - ``"dot"``: fp8 ``tl.dot`` + per-group software rescale, one scale group per K-step
      (``BLOCK_SIZE_K == 32``).
    - ``"scalar"``: scalar CUDA-core FMA reduction, no tensor core — same per-group structure
      as ``"dot"`` (hence bit-exact with it) but avoids the scaled-MMA's M→16 pad, so it wins
      for memory-bound **decode** (M=1; ≈2× over ``dot_scaled`` at S≥32 in a sweep). Only emit
      for kernels that implement the branch.

    ``num_warps=16`` and ``BLOCK_SIZE_K=64`` are omitted (dead at M=1 per a sweep).

    ``memory_modes`` (fused_grouped only) is the ``MEMORY_MODE`` axis for the weight load: pass
    ``("descriptor", "pointer")`` — ``"descriptor"`` resolves here to the device's tensor-descriptor
    flavor (host-built / NVIDIA-TMA on CUDA, device-built in-kernel tensormap on XPU), ``"pointer"``
    is explicit pointers. ``(None,)`` (default) emits no axis. A given pre_hook is attached to every
    config and self-guards on ``MEMORY_MODE``."""
    # "descriptor" -> the device's descriptor flavor: host-built (NVIDIA TMA) on CUDA, device-built
    # (in-kernel tensormap) on XPU. "pointer" / "None" pass through.
    descriptor = (
        "device_descriptor" if get_active_device_type() == "xpu" else "host_descriptor"
    )
    memory_modes = tuple(descriptor if m == "descriptor" else m for m in memory_modes)
    if "device_descriptor" in memory_modes:
        # device-side descriptors build in-kernel tensormaps that need a scratch allocator (the
        # XPU path; CUDA keeps host-built descriptors and never reaches this).
        triton.set_allocator(
            lambda size, alignment, stream: torch.empty(
                size, dtype=torch.int8, device="xpu"
            )
        )
    bms = (16, 32, 64, 128) if tune_block_m else (None,)
    return [
        triton.Config(
            {
                "COMPUTE_MODE": mode,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                **({"MEMORY_MODE": mm} if mm is not None else {}),
                **({"BLOCK_SIZE_M": bm} if bm is not None else {}),
            },
            num_warps=w,
            num_stages=s,
            pre_hook=pre_hook,
        )
        for mm in memory_modes
        for mode in compute_modes
        for bn in [32, 64, 128, 256]
        # only "dot" pins BK to the group (one scale group per fp8 tl.dot step); "dot_scaled"
        # and "scalar" handle multiple groups per K-chunk, so they use a wide BK.
        for bk in ([32] if mode == "dot" else [128, 256])
        for s in [2, 3, 4, 5, 6]
        for w in [2, 4, 8]
        for bm in bms
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
        if get_active_device_type() == "cuda":
            return torch.cuda.get_device_properties(
                device_index
            ).shared_memory_per_block_optin
        elif get_active_device_type() == "xpu":
            return torch.xpu.get_device_properties(
                device_index
            ).shared_memory_per_block_optin
        else:
            raise RuntimeError(
                f"Unsupported device type {get_active_device_type()} for sm_shared_memory_limit; only cuda/xpu are supported."
            )


def smem_config_pruner(
    act_bytes: int,
    n_weight_tiles: int,
    weight_bytes: int = 1,
    reduction_dim: str | None = None,
):
    """Build an ``early_config_prune`` that drops configs whose pipelined operand shared
    memory would overflow the SM — the source of ``out of resource: shared memory`` autotune
    failures (and the wasted compiles they cause).

    Per-stage estimate (bytes) = ``BK · (act_bytes·BM + weight_bytes·n_weight_tiles·BN)``:
    one activation tile ``[BM, BK]`` plus ``n_weight_tiles`` weight tiles ``[BN, BK]``
    (gate_up fuses 2; down has 1), times ``num_stages``. ``BM`` is the routing-derived tile,
    read from the launch args. MXFP4 packs 2 weights/byte, so ``weight_bytes=1`` (MXFP8) is a
    safe upper bound. The limit is read from the active device. Never returns empty — keeps
    the smallest-footprint config as a fallback.

    ``reduction_dim`` (the K-loop bound arg name, e.g. ``"INTERMEDIATE_DIM"``) additionally drops
    ``dot_scaled`` configs with ``BLOCK_SIZE_K >= reduction_dim`` — a single-trip K-loop, which on
    sm_100 trips a Triton ``optimize-accumulator-init`` bug (uninitialized TMEM alloc must be
    mutable) and never compiles. The autotuner discards them anyway; pruning up front just skips
    the doomed compile and its MLIR error spam. ``None`` (default) disables the check."""

    def prune(configs, named_args, **kwargs):
        # The estimate needs all three tile dims. Each is either an autotuned config meta
        # (MX) or a launch arg (block-dynamic), so look in both; raise clearly if missing.
        all_args = {**named_args, **kwargs}
        dev = triton.runtime.driver.active.get_active_torch_device()
        limit = sm_shared_memory_limit(dev.index if dev.index is not None else 0)

        def dim(c, name):
            v = c.kwargs.get(name, all_args.get(name))
            if v is None:
                raise ValueError(
                    f"smem_config_pruner needs {name} (autotune config meta or launch arg) "
                    "to estimate shared memory; none found."
                )
            return v

        def smem(c):
            BM, BN, BK = (
                dim(c, n) for n in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
            )
            return (
                c.num_stages
                * BK
                * (act_bytes * BM + weight_bytes * n_weight_tiles * BN)
            )

        def single_trip_dot_scaled(c):
            return (
                reduction_dim is not None
                and c.kwargs.get("COMPUTE_MODE") == "dot_scaled"
                and c.kwargs.get("BLOCK_SIZE_K", 0) >= all_args[reduction_dim]
            )

        kept = [
            c for c in configs if smem(c) <= limit and not single_trip_dot_scaled(c)
        ]
        return kept or [min(configs, key=smem)]

    return prune


def grf_config_pruner(
    n_accumulators: int,
    n_weight_tiles: int,
    output_reg_factor: float = 1.0,
    weight_reg_factor: float = 1.0,
    grf_regs: int = 256,
    headroom: float = 0.82,
):
    """Build an ``early_config_prune`` that drops configs whose per-thread register footprint
    would spill even after the XPU large-GRF (256) recompile — the source of the ``Detected N
    spills ... recompiling using large GRF mode ... Kernel has now M spills`` autotune noise and
    the slow, wasteful double-compile + bench of a spilling kernel.

    XPU-only: register/GRF pressure is the Intel spill knob (``maxnreg`` is CUDA-only and ignored
    by the Intel backend), and the estimate was fit on the Intel codegen. On every other backend
    this is a no-op that returns the configs unchanged.

    Per-thread register estimate (fp32-register-equivalents) =
    ``(n_accumulators·BM·BN + stages·(BM·BK + n_weight_tiles·BN·BK)) / (num_warps·warp_size)``:
    the fp32 accumulator(s) ``[BM, BN]`` (gate_up holds 2, down 1) plus the operand working set —
    one activation tile ``[BM, BK]`` and ``n_weight_tiles`` weight tiles ``[BN, BK]`` (gate_up
    fuses 2; down has 1) — spread across the block's threads (``num_warps·warp_size``) since on XPU
    the operands are GRF-resident. That operand term is the same working set the shared-memory
    estimate uses.

    ``stages`` depends on the compute mode: the pipelined MMA paths (``dot``/``dot_scaled``)
    multi-buffer the operands ``num_stages`` deep, so they use ``stages = num_stages``; the
    ``scalar`` decode path is *not* software-pipelined, so it is single-buffered (``stages = 1``)
    — without that distinction a healthy w8 scalar config (recovers to 0 spills) would score the
    same as a w2/w4 one that keeps spilling. ``BM`` is the real tile (``1`` for M=1 decode); the
    MMA pads M to 16 for the systolic array, but the padding rows don't cost real GRF, so BM is
    *not* floored — flooring over-counts the decode configs and wrongly prunes healthy ones. A
    config is dropped when the estimate exceeds ``grf_regs · headroom``. The default budget (~210
    fp32-reg-equivalents) is tuned to keep every observed *healthy* config while dropping the
    *catastrophic* spillers (estimates well above it, with 10k-80k residual spills after the
    large-GRF recompile — the real double-compile time sink). It deliberately does NOT try to catch
    the moderate gray zone (residual ~0.1k-8k): that estimate range overlaps healthy configs and is
    not separable by this working-set model — observed counterexamples include two gate_up configs
    with the *same* estimate (200) but opposite outcomes (``BK=128,s2=4`` healthy vs
    ``BK=256,s2=2`` spilling — equal ``stages·BK`` but the wider single-stage tile spills), and a
    cross-kernel inversion (a decode config healthy at 198 above a gate_up config spilling at 176).
    Chasing them needs a tile-width/codegen-aware model that would overfit; those gray-zone configs
    recompile to working kernels and don't win the tune anyway. ``warp_size`` and the GRF budget are
    hardware assumptions; ``headroom`` trades tuning speed (prune more) against safety (keep
    borderline configs). Never returns empty — keeps the lightest-footprint config as a fallback."""

    def prune(configs, named_args, **kwargs):
        configs = list(configs)
        # Register/GRF spilling is an Intel-backend concern; the estimate was fit on its
        # codegen, so leave every other backend's configs untouched.
        if get_active_device_type() != "xpu":
            return configs
        all_args = {**named_args, **kwargs}
        ws = warp_size()
        budget = grf_regs * headroom

        def dim(c, name):
            v = c.kwargs.get(name, all_args.get(name))
            if v is None:
                raise ValueError(
                    f"grf_config_pruner needs {name} (autotune config meta or launch arg) "
                    "to estimate register pressure; none found."
                )
            return v

        def regs(c):
            BM, BN, BK = (
                dim(c, n) for n in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
            )
            if c.kwargs.get("COMPUTE_MODE") in ("dot", "dot_scaled"):
                stages = c.num_stages  # pipelined: operands multi-buffered num_stages deep
            elif "COMPUTE_MODE" not in c.kwargs:
                stages = c.num_stages  # no COMPUTE_MODE → always pipelined (block-dynamic FP8)
            else:
                stages = 1  # scalar path is not software-pipelined: single operand buffer
            acc = output_reg_factor * n_accumulators * BM * BN
            # The scalar path (mx_scalar_reduce) additionally materializes the *expanded*
            # weight scale as a full [BN, BK] fp32 tile alongside the fp32 weight tile — a real
            # ~2-3x on its BN*BK working set that this term omits. It is deliberately NOT added:
            # it is a uniform multiplier on BN*BK, so it cancels against the num_warps divisor and
            # cannot separate the observed scalar collision — configs with identical BN*BK/num_warps
            # (e.g. BN128/BK256/w8 healthy vs BN128/BK128/w4 and BN32/BK256/w2 spilling ~2k) land on
            # the same estimate regardless. Folding it in would only inflate the healthy config past
            # budget and prune it. So the model catches the catastrophic scalar spillers (the large
            # BN*BK/num_warps ones, ~16k-48k residual spills) and lets the borderline ~2k-spill gray
            # zone through — cheaper than risking the decode-winning scalar config.
            operand = stages * (BM * BK + weight_reg_factor * n_weight_tiles * BN * BK)
            return (acc + operand) / (c.num_warps * ws)

        kept = [c for c in configs if regs(c) <= budget]
        return kept or [min(configs, key=regs)]

    return prune


def chain_config_pruners(*pruners):
    """Compose several ``early_config_prune`` callables into one. Triton's ``prune_configs_by``
    takes a single ``early_config_prune``, so apply each pruner in turn, feeding the survivors of
    one into the next. Each pruner keeps its own never-empty fallback, so the chain never empties
    the config list."""

    def prune(configs, named_args, **kwargs):
        for p in pruners:
            configs = p(configs, named_args, **kwargs)
        return configs

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
def expand_ue8m0_scale(
    scale_u8,
    ROWS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Decode a UE8M0 group-scale tile ``[ROWS, BLOCK_SIZE_K // SCALE_GROUP_K]`` and broadcast each
    group scale across its ``SCALE_GROUP_K`` elements → ``[ROWS, BLOCK_SIZE_K]`` fp32 (one scale per
    K element). Used by the scalar (CUDA-core FMA) MX paths to dequant the whole K chunk."""
    s = decode_ue8m0_scale(scale_u8)
    return tl.reshape(
        tl.broadcast_to(
            s[:, :, None], (ROWS, BLOCK_SIZE_K // SCALE_GROUP_K, SCALE_GROUP_K)
        ),
        (ROWS, BLOCK_SIZE_K),
    )


@triton.jit
def mx_dot_scaled(acc, a, a_scale, w, w_scale, VALUES_PER_BYTE: tl.constexpr):
    """MX 'dot_scaled' path: scaled MMA folding the UE8M0 group scales into the tensor core —
    ``a`` (E4M3) @ ``w`` (packed E2M1 if MXFP4, else E4M3). The rhs format is picked from
    ``VALUES_PER_BYTE``. Caller pre-shapes ``w``/``w_scale`` (e.g. ``tl.trans(gu)``)."""
    rhs_format: tl.constexpr = "e2m1" if VALUES_PER_BYTE == 2 else "e4m3"
    return tl.dot_scaled(a, a_scale, "e4m3", w, w_scale, rhs_format, acc)


@triton.jit
def mx_dot_rescale(acc, a, w, a_scale, w_scale, VALUES_PER_BYTE: tl.constexpr):
    """MX 'dot' path (BK == group): unpack MXFP4 weights to E4M3, fp8 ``tl.dot`` + per-group
    software rescale (decoding both UE8M0 scales internally), accumulating into ``acc`` (returned
    updated). Single weight — for the gate∪up pair (shared activation scale) use
    ``mx_dot_rescale_gate_up``."""
    wq = mxfp4_e2m1_to_e4m3(w) if VALUES_PER_BYTE == 2 else w
    return acc + tl.dot(a, wq) * decode_ue8m0_scale(a_scale) * tl.trans(
        decode_ue8m0_scale(w_scale)
    )


@triton.jit
def mx_dot_rescale_gate_up(
    acc_gate, acc_up, a, w_gate, w_up, a_scale, gate_scale, up_scale
):
    """Gate∪up 'dot' path: decode the SHARED activation scale once, rescale both projections
    and accumulate into ``acc_gate``/``acc_up`` (returned updated)."""
    a_s = decode_ue8m0_scale(a_scale)
    acc_gate += tl.dot(a, w_gate) * a_s * tl.trans(decode_ue8m0_scale(gate_scale))
    acc_up += tl.dot(a, w_up) * a_s * tl.trans(decode_ue8m0_scale(up_scale))
    return acc_gate, acc_up


@triton.jit
def mx_scalar_reduce(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    BLOCK_SIZE_M: tl.constexpr,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
):
    """MX 'scalar' path: CUDA-core FMA GEMV, unpacking MXFP4 weights to E4M3 then dequantizing
    activation + weight per-element by their expanded group scales, reducing and accumulating into
    ``acc`` (returned updated). No tensor core (so no M→16 MMA pad) — wins for the memory-bound
    decode GEMV (M=1). Single weight — for the gate∪up pair use ``mx_scalar_reduce_gate_up``."""
    wq = mxfp4_e2m1_to_e4m3(w) if VALUES_PER_BYTE == 2 else w
    a_deq = tl.trans(
        a.to(tl.float32)
        * expand_ue8m0_scale(a_scale, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K)
    )
    w_exp = tl.trans(expand_ue8m0_scale(w_scale, ROWS_W, BLOCK_SIZE_K, SCALE_GROUP_K))
    return acc + tl.sum(a_deq * (wq.to(tl.float32) * w_exp), axis=0)[None, :]


@triton.jit
def mx_scalar_reduce_gate_up(
    acc_gate,
    acc_up,
    a,
    w_gate,
    w_up,
    a_scale,
    gate_scale,
    up_scale,
    BLOCK_SIZE_M: tl.constexpr,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Gate∪up 'scalar' path: dequant the SHARED activation once, reduce both projections and
    accumulate into ``acc_gate``/``acc_up`` (returned updated)."""
    a_deq = tl.trans(
        a.to(tl.float32)
        * expand_ue8m0_scale(a_scale, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K)
    )
    acc_gate += tl.sum(
        a_deq
        * (
            w_gate.to(tl.float32)
            * tl.trans(
                expand_ue8m0_scale(gate_scale, ROWS_W, BLOCK_SIZE_K, SCALE_GROUP_K)
            )
        ),
        axis=0,
    )[None, :]
    acc_up += tl.sum(
        a_deq
        * (
            w_up.to(tl.float32)
            * tl.trans(
                expand_ue8m0_scale(up_scale, ROWS_W, BLOCK_SIZE_K, SCALE_GROUP_K)
            )
        ),
        axis=0,
    )[None, :]
    return acc_gate, acc_up


@triton.jit
def mx_compute_gate_up(
    acc_gate,
    acc_up,
    a,
    a_scale,
    b_gate,
    b_up,
    gate_scale,
    up_scale,
    COMPUTE_MODE: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Gate∪up MMA step dispatched on ``COMPUTE_MODE``: scaled-MMA on the raw weights
    (``b_gate``/``b_up``), or fp8 ``tl.dot`` + per-group rescale / scalar reduce on the
    E4M3-decoded weights. Returns the updated ``(acc_gate, acc_up)``."""
    w_gate = mxfp4_e2m1_to_e4m3(b_gate) if VALUES_PER_BYTE == 2 else b_gate
    w_up = mxfp4_e2m1_to_e4m3(b_up) if VALUES_PER_BYTE == 2 else b_up
    if COMPUTE_MODE == "dot_scaled":
        acc_gate = mx_dot_scaled(
            acc_gate, a, a_scale, b_gate, gate_scale, VALUES_PER_BYTE
        )
        acc_up = mx_dot_scaled(acc_up, a, a_scale, b_up, up_scale, VALUES_PER_BYTE)
    elif COMPUTE_MODE == "dot":
        acc_gate, acc_up = mx_dot_rescale_gate_up(
            acc_gate, acc_up, a, w_gate, w_up, a_scale, gate_scale, up_scale
        )
    else:  # scalar
        acc_gate, acc_up = mx_scalar_reduce_gate_up(
            acc_gate,
            acc_up,
            a,
            w_gate,
            w_up,
            a_scale,
            gate_scale,
            up_scale,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
        )
    return acc_gate, acc_up


@triton.jit
def mx_compute(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Single-projection MMA step dispatched on ``COMPUTE_MODE``: scaled-MMA on the raw weight
    (``w``), or fp8 ``tl.dot`` + per-group rescale / scalar reduce on the E4M3-decoded weight.
    Returns the updated ``acc``."""
    if COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled(acc, a, a_scale, w, w_scale, VALUES_PER_BYTE)
    elif COMPUTE_MODE == "dot":
        acc = mx_dot_rescale(acc, a, w, a_scale, w_scale, VALUES_PER_BYTE)
    else:  # scalar
        acc = mx_scalar_reduce(
            acc,
            a,
            a_scale,
            w,
            w_scale,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
            VALUES_PER_BYTE,
        )
    return acc


@triton.jit
def glu(
    gate,
    up,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    OUT_DTYPE: tl.constexpr = tl.float32,
    SIMULATE_UNFUSED: tl.constexpr = False,
):
    """Gated linear unit on the gate/up matmul accumulators. ``SWIGLU_LIMIT`` clamps gate above and up
    to ``[-LIMIT, LIMIT]``; ``SWIGLU_ALPHA`` gives the clamped/scaled SwiGLU ``(up + 1) * gate * sigmoid(ALPHA *
    gate)`` (GPT-OSS / MiniMax), else ``ACT_FN(gate) * up`` (``ACT_FN`` in {silu, gelu, relu}, gelu exact
    via erf). ``SIMULATE_UNFUSED`` rounds each materialized value through ``OUT_DTYPE`` to match the
    unfused (separate-kernel) path, where every intermediate lands in that dtype."""
    g = gate
    u = up

    if SIMULATE_UNFUSED:
        g = g.to(OUT_DTYPE).to(tl.float32)
        u = u.to(OUT_DTYPE).to(tl.float32)

    if SWIGLU_LIMIT is not None:
        g = tl.minimum(g, SWIGLU_LIMIT)
        u = tl.minimum(tl.maximum(u, -SWIGLU_LIMIT), SWIGLU_LIMIT)

    if SWIGLU_ALPHA is not None:
        gate_scaled = g * SWIGLU_ALPHA
        if SIMULATE_UNFUSED:
            gate_scaled = gate_scaled.to(OUT_DTYPE).to(tl.float32)
        sig = tl.sigmoid(gate_scaled)
        if SIMULATE_UNFUSED:
            sig = sig.to(OUT_DTYPE).to(tl.float32)
        act = g * sig
        u = u + 1.0
    elif ACT_FN == "silu":
        act = g * tl.sigmoid(g)
    elif ACT_FN == "gelu":
        act = 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))
    elif ACT_FN == "relu":
        act = tl.maximum(g, 0.0)
    else:
        tl.static_assert(
            False, "unsupported ACT_FN; expected 'silu', 'gelu', or 'relu'"
        )

    if SIMULATE_UNFUSED:
        act = act.to(OUT_DTYPE).to(tl.float32)
        u = u.to(OUT_DTYPE).to(tl.float32)

    gated = act * u

    if SIMULATE_UNFUSED:
        gated = gated.to(OUT_DTYPE).to(tl.float32)

    return gated


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
