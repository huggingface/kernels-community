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
from typing import NamedTuple


import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune

# ── Format constants ──────────────────────────────────────────────────────────

# FP8 (E4M3) is the main format for weights and activations;
FP8_DTYPE = torch.float8_e4m3fn
# FP4 (E2M1) packs two 4-bit nibbles per byte. MX formats (MXFP4 weights, MXFP8
# E4M3 weights/activations) share one UE8M0 scale per 32-element K-group — the OCP
# MX block size, consumed by ``tl.dot_scaled``. Format constants, not tunables.
NIBBLES_PER_BYTE = 2
MX_SCALE_GROUP_K = 32


# ── Host-side helpers ─────────────────────────────────────────────────────────


def compile_time_only_triton_wrap(kernel):
    """``wrap_triton`` while torch.compile is tracing (required to capture a raw Triton
    launch into the graph), the bare kernel in eager. Every kernel launch goes through
    this: eager skips ``wrap_triton``'s per-call dispatch overhead, and it dodges a
    correctness trap — eager ``wrap_triton`` of a stock ``@triton.autotune`` kernel that
    has ``prune_configs_by`` re-runs the FULL tune on every call (torch's wrapper rebuilds
    a fresh autotuner around the pruned configs per call, ``triton_kernel_wrap.py``:
    ``autotune(configs=pruned, key=[])``), so nothing ever lands in the original
    ``Autotuner.cache`` (~2.3s/call for ``_mxfp_act_quant_kernel``) and the mid-capture
    tune invalidates cudagraph capture. Pruner-less kernels and the ``bayesian_autotune``
    kernels reuse their cache through the wrapper (measured) — but eager never needs the
    wrapper at all."""
    return wrap_triton(kernel) if torch.compiler.is_compiling() else kernel


def compile_time_only_triton_op(name, mutates_args=()):
    """``@triton_op`` under torch.compile (the registered custom op is what dynamo
    captures), the plain function in eager: the torch.library op dispatch stack costs
    ~160µs per eager call — the dominant decode CPU cost in the eager-breakdown probe —
    and eager needs none of it. The op is still registered at import time, so compiled
    callers and ``torch.ops`` introspection see it unchanged. Sibling of
    ``compile_time_only_triton_wrap`` (same rule one level down, at the kernel launch)."""

    def decorator(fn):
        op = triton_op(name, mutates_args=mutates_args)(fn)

        @functools.wraps(fn)
        def dispatch(*args, **kwargs):
            if torch.compiler.is_compiling():
                return op(*args, **kwargs)
            return fn(*args, **kwargs)

        dispatch._triton_op = op
        return dispatch

    return decorator


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


@bayesian_autotune(
    [
        triton.Config({"BLOCK_H": block_h}, num_warps=warps)
        for block_h in (256, 512, 1024, 2048)
        for warps in (4, 8)
    ],
    # the H tile width trades off against grid occupancy: at few groups (decode) narrow
    # tiles spread more H-blocks across SMs, at many groups (prefill) wide tiles amortize
    # the per-row weight load — so key on H and the group-count bucket.
    ["H", "num_groups_bit_length"],
    n_trials=8,
)
@triton.jit
def weighted_reduce_kernel(
    Rows,  # (num_groups * NUM_TOP_K, H) — rows to reduce, group-major
    Out,  # (num_groups, H) — one reduced row per group
    Ids,  # (num_groups, NUM_TOP_K) — per-row id; a row is skipped when its id >= NUM_EXPERTS
    Weights,  # (num_groups * NUM_TOP_K,) — per-row scale
    H,
    stride_rows_m,
    stride_rows_h,
    stride_o_m,
    stride_o_h,
    stride_ids_m,
    stride_ids_k,
    num_groups_bit_length,  # autotune key only (log2 group-count bucket); unused in body
    NUM_TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr = False,
):
    """Per group ``g``, the weighted sum of its ``NUM_TOP_K`` rows into ``Out[g]``:
    ``sum_k Weights[g*NUM_TOP_K + k] * Rows[g*NUM_TOP_K + k]``, skipping rows whose id is
    ``>= NUM_EXPERTS`` (out-of-range rows are never written upstream and contribute 0).
    fp32 accumulate; ~2.8x a generic ``view(g, k, H).sum(1)``. ``SIMULATE_UNFUSED`` rounds
    each weighted row to ``Out``'s dtype before summing, matching a reference that weights
    in that dtype; production leaves the accumulation in fp32."""
    g = tl.program_id(0)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for k in tl.static_range(NUM_TOP_K):
        flat = g * NUM_TOP_K + k
        valid = tl.load(Ids + g * stride_ids_m + k * stride_ids_k) < NUM_EXPERTS
        weight = tl.load(Weights + flat)
        contrib = weight * tl.load(
            Rows + flat * stride_rows_m + offs_h * stride_rows_h,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)
        if SIMULATE_UNFUSED:
            contrib = contrib.to(Out.dtype.element_ty).to(tl.float32)
        acc += contrib
    tl.store(
        Out + g * stride_o_m + offs_h * stride_o_h,
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
    (``int8``, two E2M1 codes/byte), both with UE8M0 group-32 scales. The dispatchers
    route on this; the op picks the format from ``weight.dtype``."""
    return is_mxfp8(weight, scale) or is_mxfp4(weight, scale)


def is_tensor_wide(block_size, weight: torch.Tensor) -> bool:
    """True when ``block_size`` selects per-tensor (tensor-dynamic) scaling: ``None`` or
    equal to the weight's full ``(N, K)`` — one scale block spanning the whole matrix.
    Handles 2D ``(N, K)`` and 3D ``(E, N, K)`` weights via the last two dims."""
    return block_size is None or (
        block_size[0] == weight.shape[-2] and block_size[1] == weight.shape[-1]
    )


def tl_dtype(dtype: torch.dtype) -> tl.dtype:
    """The ``tl`` dtype matching a torch dtype (``torch.bfloat16`` → ``tl.bfloat16``) — the
    attribute names line up, so no table. For passing a tensor's dtype as a kernel constexpr
    when the kernel can't read it off a pointer argument."""
    return getattr(tl, str(dtype).removeprefix("torch."))


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


def resolve_memory_modes(memory_modes):
    """Resolve the generic ``"descriptor"`` MEMORY_MODE to the device's flavor: host-built
    (NVIDIA TMA) on CUDA, device-built in-kernel tensormap on XPU. Installs the XPU
    scratch allocator device-built tensormaps need."""
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
    return memory_modes


def get_accelerator_autotuning_configs(
    *,
    mx: bool = False,
    tune_block_m: bool = False,
    tune_block_n: bool = False,
    tune_block_nk: bool = False,
    swap_ab: bool = False,
    warp_spec: bool = False,
    compute_modes=None,
    memory_modes=None,
    pre_hook=None,
):
    """Autotune search grid for the current accelerator — the single generator for every
    GEMM kernel. One union span serves both families: num_warps {2,4,8,16} x num_stages
    {2..6} (XPU: warps {8,16}); the tile comes from the caller's ``block_size``, or
    ``tune_block_nk`` spans BN {32,64,128,256} x BK {64,128,256}. ``mx=True`` (group-32
    scale formats, MXFP4/MXFP8) requires ``compute_modes`` (BK is coupled per mode) and
    ``tune_block_nk`` (MX kernels have no caller block_size — the tile is always tuned).

    ``compute_modes`` emits the ``COMPUTE_MODE`` axis (``None`` — the default — emits no
    axis, like ``memory_modes``), tuner-selected per workload (token count is in the
    key): ``"dot_scaled"`` native group-32 scaled MMA (wide BK {128,256}; wins once the
    grid saturates, ~S>=32), ``"dot"`` fp8 ``tl.dot`` + per-group software rescale
    (under ``mx`` its BK is pinned to the 32-group), ``"scalar"`` CUDA-core FMA reduce
    (no MMA M->16 pad; wins memory-bound decode, only emit where implemented — it is
    pinned to BM=1 here because a bigger-BM scalar config at BN == BM COMPILES and
    silently computes garbage the tuner would time and pick).

    Axes, crossed per flag:

    - ``tune_block_m``: BLOCK_SIZE_M in {16,32,64,128} instead of the
      ``adaptive_block_size_m`` launch heuristic.
    - ``tune_block_n``: independent BLOCK_SIZE_N in {64,128,256} for the block-scale
      kernels whose BK is pinned by the caller (activation scale groups are per
      block_k) but whose N tile may grow past the scale granularity.
    - ``swap_ab``: marks the single-token decode GEVM kernels — crosses in the coupled
      decode ``(BLOCK_SIZE_M, SWAP_AB)`` pairs (see below) and, for MX, adds BK=512
      (at decode's tiny memory-parallelism-bound grids, longer per-iteration bursts are
      the one in-block lever that helps: +12% on the dsv4 swap GEMV; deeper num_stages
      and split-K both measured WORSE).
    - ``warp_spec`` (CUDA only): warp-specialize the K-loop, crossed over every mode —
      pair it with ``warp_spec_compile_guard_pruner``, which owns the can't-compile
      regions (including dot_scaled + WS, a PassManager failure). Compile support is
      (shape, config)-dependent on Triton 3.7.1, so it is a tuner axis — failures score
      inf and self-prune; where it compiles it is both faster and (bd grouped gate_up)
      load-bearing for correctness.
    - ``memory_modes``: the MEMORY_MODE weight-load axis — pass
      ``("descriptor", "pointer")`` to let the tuner pick the device's tensor-descriptor
      flavor vs explicit pointers; ``None`` (default) emits no axis. In the plain-dot
      kernels the descriptor arm is the SWAPPED loop (see the bd 2D kernel).

    A given ``pre_hook`` is attached to every config and must self-guard on
    ``MEMORY_MODE``.

    Winner census over 743 tunes (B200, all kernels/shapes), useful when reading tuner
    logs: every value wins somewhere — MX lives in stages 5/6 (69 wins) and never picked
    warps 16; plain-dot picked warps 16 (19 wins, e.g. the bd gate_up decode) and warps 2
    (24 wins, the tensor batched decode 9/11) and concentrated in stages 3/4.
    """
    assert not (mx and compute_modes is None), (
        "the MX grid couples BLOCK_SIZE_K to COMPUTE_MODE — pass compute_modes"
    )
    assert not (mx and not tune_block_nk), (
        "MX kernels have no caller block_size — their (BN, BK) tile is always tuned"
    )
    is_xpu = get_active_device_type() == "xpu"

    # ONE union span for both families — the per-family sets were stock-tuner-era cost
    # caps; the Bayesian tuner samples a fixed trial budget, values measured dead for
    # one family (MX warps=16, MX BK=64; plain-dot stages 5/6 were simply never tested)
    # just never win, and can't-compile configs (smem) score inf and stay out of the
    # TPE densities. The 743-tune winner census (see docstring) maps where winners
    # actually live per family. The decode GEVMs (swap_ab) extend BK to 512: at their
    # tiny memory-parallelism-bound grids, longer per-iteration bursts are the one
    # in-block lever that helps (+12% on the dsv4 swap GEMV).
    num_warps = [8, 16] if is_xpu else [2, 4, 8, 16]
    num_stages = [2, 3, 4, 5, 6]
    bn_span = (128,) if is_xpu else (32, 64, 128, 256)
    bk_span = (128,) if is_xpu else (64, 128, 256, 512) if swap_ab else (64, 128, 256)

    # no tuned tile -> one empty meta-dict (the tile comes from the launch kwargs)
    blocks = (
        [{"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk} for bn in bn_span for bk in bk_span]
        if tune_block_nk
        else [{}]
    )

    if compute_modes is not None:
        # generic COMPUTE_MODE axis; MX "dot"'s BK is STRUCTURALLY the UE8M0 scale group
        # (one software rescale per K-step — a format constraint, not a tile choice), so
        # it overrides the wide-BK span and the dedup below collapses the copies.
        blocks = [
            {
                **b,
                "COMPUTE_MODE": mode,
                **({"BLOCK_SIZE_K": MX_SCALE_GROUP_K} if mx and mode == "dot" else {}),
            }
            for b in blocks
            for mode in compute_modes
        ]

    if tune_block_m:
        blocks = [{**b, "BLOCK_SIZE_M": bm} for b in blocks for bm in (16, 32, 64, 128)]

    if tune_block_n:
        blocks = [{**b, "BLOCK_SIZE_N": bn} for b in blocks for bn in (64, 128, 256)]

    if warp_spec and get_active_device_type() == "cuda":
        blocks = [{**b, "WARP_SPEC": ws} for b in blocks for ws in (False, True)]

    # (avoidance of the descriptor modes' can't-win regions lives in descriptor_config_pruner)
    if memory_modes is not None:
        blocks = [
            {**b, "MEMORY_MODE": mm}
            for b in blocks
            for mm in resolve_memory_modes(memory_modes)
        ]

    # The coupled decode (BLOCK_SIZE_M, SWAP_AB) pairs — the single source of the swap/BM
    # coupling. At M=1 the MMA 16-atom is filled either by replicating the token (non-swap
    # BM=16, ~40% over the degenerate BM=1 on plain tl.dot) or by putting the weight's
    # output rows in M (swap, BM=1). Swap REQUIRES BM=1 (its rhs is the single token
    # padded to the N=16 atom), so (16, swap) never exists; the tuner picks. Prefill
    # kernels leave SWAP_AB unset — the prefill swap arms measured losers everywhere
    # (see the bd 2D kernel's tuner note).
    if swap_ab:
        blocks = [
            {**b, "BLOCK_SIZE_M": bm, "SWAP_AB": sw}
            for b in blocks
            for bm, sw in ((1, False), (16, False), (1, True))
        ]

    # scalar's BM is STRUCTURALLY 1 (a single-row GEVM reduce — a bigger-BM scalar
    # config at BN == BM even COMPILES and silently computes garbage the tuner would
    # time and pick); the dedup below collapses the pinned copies of the BM sweep.
    blocks = [
        {**b, "BLOCK_SIZE_M": 1}
        if b.get("COMPUTE_MODE") == "scalar" and "BLOCK_SIZE_M" in b
        else b
        for b in blocks
    ]

    configs = [
        triton.Config(b, num_warps=w, num_stages=s, pre_hook=pre_hook)
        for b in blocks
        for w in num_warps
        for s in num_stages
    ]
    # dedup: the scalar BM=1 pin and the MX "dot" BK override collapse swept axes onto
    # one value, leaving identical copies that would each be benched.
    return list({str(c): c for c in configs}.values())


@functools.cache
def is_sm10x() -> bool:
    """Whether the CURRENT device is Blackwell-datacenter (sm_10x, TMEM scaled-MMA) —
    the target the ``dot_scaled`` compiler-bug guards are scoped to. False off-CUDA;
    cached (pruners call this per tune, and a process is pinned to one device)."""
    return (
        get_active_device_type() == "cuda"
        and torch.cuda.get_device_capability()[0] == 10
    )


@functools.lru_cache(maxsize=None)
def sm_shared_memory_limit() -> int:
    """Max dynamic shared memory per block (bytes) for the CURRENT device — the cap
    Triton reports as the 'Hardware limit' on an ``out of resource: shared memory``
    failure (~232 KB on B200, ~227 KB on H100, much less on older/consumer parts).
    Queried from the driver so the prune adapts to the hardware instead of hardcoding
    one GPU; cached (a process is pinned to one device)."""
    device_index = (
        torch.cuda.current_device()
        if get_active_device_type() == "cuda"
        else torch.xpu.current_device()
        if get_active_device_type() == "xpu"
        else 0
    )
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


# ── config pruners ────────────────────────────────────────────────────────────
# Every guard exists for one of three reasons; the map (kernel -> pruner -> rule):
#
#   SILENTLY-WRONG configs (must be pruned, the tuner would time and pick them):
#     block_k_within_k_pruner   BK not dividing K over-reads rows (maskless K-loops)
#     scalar "scalar" -> BM=1   the scalar GEVM broadcasts wrong at BM>1 (config builder)
#   COMPILER BUGS / unsupported combos on this triton+arch (benign inf, pruned to save
#   compiles and to keep can't-win configs from poisoning the TPE densities):
#     warp_spec_compile_guard_pruner   WS compiles iff BM>=64 & warps in {4,8} (measured
#                                      identical on four plain-dot loops, Triton 3.7.1);
#                                      dot_scaled+WS never compiles (PassManager)
#     batched_mx_pruner sm_10x guards  dot_scaled shape gates (fp4-scalar dead; swapped
#                                      rows<128 = bf16 fallback; no-swap width>256 traps;
#                                      BK<128 traps -> sticky 716)
#     smem_config_pruner               operand smem overflow; single-trip dot_scaled bug
#   RACE guards (Triton 3.7.1 pipeliner, per-loop-structure flake maps):
#     block_dynamic_grouped_gate_up_pruner  non-WS sound only at (w>=8, BM<=64)
#     block_dynamic_grouped_matmul_pruner   WS-only at BM>=64, non-WS below (disjoint)
#   TPE-POISON fences (valid configs that can't win in a regime):
#     scalar_max_m_pruner              scalar above M=64 (prefill GEVM)
#     descriptor_config_pruner         orientation couplings: descriptor requires swap
#                                      (no-swap descriptor races), swap drops WS (3-4x
#                                      slower + unprobed loop structure), descriptor warp
#                                      floor (warps<8 at BN>=128 = 3.6x slower)
#
# Compose with compose_pruners; every pruner keeps a non-empty fallback.


def block_dynamic_grouped_gate_up_pruner(n_weight_tiles: int):
    """``early_config_prune`` for the block-dynamic grouped gate_up: the smem estimate (see
    ``smem_config_pruner``), the shared WS compile guard, and the Triton 3.7.1
    pipeliner-race guard. The kernel's six-load-stream dual-dot K-loop RACES under the
    default pipeliner at ``num_warps < 8`` or ``BLOCK_SIZE_M > 64`` (nondeterministic
    wrong output) — ``warp_specialize`` fixes those configs and is faster. WS configs
    outside the measured compile region (BM >= 64 with warps in {4, 8} — identical across
    every mapped plain-dot loop, see ``warp_spec_compile_guard_pruner``) are dropped up
    front instead of burning compiles into benign infs; WS=False survives only in the
    flake-verified sound (w >= 8, BM <= 64) region. Configs in neither region (w < 8 at
    BM <= 32) are correctly unrepresentable: non-WS races there and WS cannot compile.
    CUDA-only — the race is a CUDA pipeliner artifact and the WS axis isn't emitted
    elsewhere."""
    smem_prune = smem_config_pruner(n_weight_tiles=n_weight_tiles)
    ws_guard = warp_spec_compile_guard_pruner()

    def prune(configs, named_args, **kwargs):
        kept = smem_prune(configs, named_args, **kwargs)
        if get_active_device_type() == "cuda":
            kept = ws_guard(kept, named_args, **kwargs)
            kept = [
                c
                for c in kept
                if c.kwargs.get("WARP_SPEC")
                or (c.num_warps >= 8 and c.kwargs.get("BLOCK_SIZE_M", 128) <= 64)
            ] or kept
        return kept

    return prune


def block_dynamic_grouped_matmul_pruner():
    """``early_config_prune`` for the block-dynamic grouped matmul: the Triton 3.7.1 pipeliner-race
    guard, sized to THIS kernel's four-load-stream single-dot K-loop (the fused gate_up's
    six-stream loop has a different sound region — see ``block_dynamic_grouped_gate_up_pruner``).
    Measured on sm_100, 15-fresh-process flake runs per cell, big and tiny shapes:

    - ``warp_specialize`` compiles iff ``BLOCK_SIZE_M >= 64`` with ``num_warps`` in {4, 8}
      (PassManager failure otherwise) and is race-free everywhere it compiles.
    - The default pipeliner RACES at ``BLOCK_SIZE_M >= 64`` (3/15 wrong at BM64/w8) and is
      clean at BM16/32.

    The regions are disjoint, so per launch-``BLOCK_SIZE_M`` (a launch arg here, not a tuned
    axis): BM >= 64 keeps only the compilable WS configs, BM < 64 keeps only non-WS. CUDA-only
    — the race is a CUDA pipeliner artifact and the WS axis isn't emitted elsewhere."""

    def prune(configs, named_args, **kwargs):
        if get_active_device_type() != "cuda":
            return configs
        launch_bm = {**named_args, **kwargs}.get("BLOCK_SIZE_M", 128)

        def ok(c):
            # BLOCK_SIZE_M is a tuner axis since the metadata port — config-first
            bm = c.kwargs.get("BLOCK_SIZE_M", launch_bm)
            if bm >= 64:
                return c.kwargs.get("WARP_SPEC") and c.num_warps in (4, 8)
            return not c.kwargs.get("WARP_SPEC")

        return [c for c in configs if ok(c)] or configs

    return prune


def warp_spec_compile_guard_pruner():
    """``early_config_prune`` dropping ``warp_specialize`` configs that can never compile.
    Measured on sm_100 for the plain-``tl.dot`` single-dot loops (matrix probe + 15-process
    flake runs, big and tiny shapes, on both the 2D and grouped bd matmuls): WS compiles iff
    ``BLOCK_SIZE_M >= 64`` with ``num_warps`` in {4, 8} (PassManager failure otherwise), and
    ``dot_scaled`` + WS never compiles at all (PassManager, probed separately — MX "dot" and
    "scalar" rows keep the axis). Non-WS configs all stay — on kernels using this guard the default pipeliner is the
    validated existing state and WS is purely a perf axis (compiling WS configs are
    deterministic by construction; off-region survivors self-prune as inf). ``BLOCK_SIZE_M``
    is read from the config when tuned, else from the launch args. CUDA-only."""

    def prune(configs, named_args, **kwargs):
        if get_active_device_type() != "cuda":
            return configs
        launch_bm = {**named_args, **kwargs}.get(
            "BLOCK_SIZE_M", 128
        )  # config-first below

        def ok(c):
            bm = c.kwargs.get("BLOCK_SIZE_M", launch_bm)
            if not c.kwargs.get("WARP_SPEC"):
                return True
            if c.kwargs.get("COMPUTE_MODE") == "dot_scaled":
                return False
            bm = c.kwargs.get("BLOCK_SIZE_M", launch_bm)
            return bm >= 64 and c.num_warps in (4, 8)

        return [c for c in configs if ok(c)] or configs

    return prune


def descriptor_config_pruner():
    """``early_config_prune`` coupling the (MEMORY_MODE, SWAP_AB, WARP_SPEC) orientation
    axes to their validated regions (B200, bd 2D loop, M=8192):

    - descriptor modes REQUIRE ``SWAP_AB``: the natural orientation needs a per-iteration
      ``tl.trans`` on the descriptor tile, which RACES without WS (Triton 3.7.1 pipeliner)
      and loses 2.3x with it.
    - ``SWAP_AB`` drops ``WARP_SPEC``: descriptor+WS measured 3-4x slower at every
      (BM, stages) probed, and the WS compile/race map was only measured on the
      non-swapped plain-dot loops — the swapped loop is a different structure.
    - descriptor modes keep a warp floor: ``num_warps < 8`` under-subscribes the swapped
      dot's M-operand (the full BN weight tile), 3.6x slower at BN=128. Applied only at
      ``BLOCK_SIZE_N >= 128`` (config when tuned, else launch), where it was measured."""

    def prune(configs, named_args, **kwargs):
        launch_bn = {**named_args, **kwargs}.get("BLOCK_SIZE_N", 128)

        def ok(c):
            descriptor = c.kwargs.get("MEMORY_MODE", "pointer") != "pointer"
            swapped = c.kwargs.get("SWAP_AB", descriptor)
            if descriptor and not swapped:
                return False
            if swapped and c.kwargs.get("WARP_SPEC"):
                return False
            bn = c.kwargs.get("BLOCK_SIZE_N", launch_bn)
            return not descriptor or c.num_warps >= 8 or bn < 128

        return [c for c in configs if ok(c)] or configs

    return prune


def compose_pruners(*pruners):
    """Chain ``early_config_prune`` callbacks left to right (each sees the previous
    survivors)."""

    def prune(configs, named_args, **kwargs):
        for p in pruners:
            configs = p(configs, named_args, **kwargs)
        return configs

    return prune


def block_k_within_k_pruner(k_arg: str):
    """``early_config_prune`` dropping configs whose ``BLOCK_SIZE_K`` does not divide the
    launch's contraction dim (``k_arg`` names the launch argument): the K-loops load
    unmasked, so a non-dividing BK's last trip reads past the row — silently wrong results
    the tuner would happily time and pick. A contraction dim smaller than every grid BK is
    a hard error. Used standalone by the tensor-dynamic kernels (``BLOCK_SIZE_K`` is a
    tuned axis there) and as the first stage of ``batched_mx_pruner``."""

    def prune(configs, named_args, **kwargs):
        k = {**named_args, **kwargs}[k_arg]
        kept = [
            c
            for c in configs
            if c.kwargs.get("BLOCK_SIZE_K", 0) == 0 or k % c.kwargs["BLOCK_SIZE_K"] == 0
        ]
        if not kept:
            min_bk = min(c.kwargs.get("BLOCK_SIZE_K", 0) for c in configs)
            raise ValueError(
                f"{k_arg}={k} is not a multiple of any BLOCK_SIZE_K in the autotune grid; "
                f"the unmasked K-loop would read past the row. Pad the problem along "
                f"{k_arg} (smallest grid BK: {min_bk})."
            )
        return kept

    return prune


def scalar_max_m_pruner(m_arg: str, max_m: int = 64):
    """``early_config_prune`` dropping ``scalar`` configs when the launch's row count
    (``m_arg``) exceeds ``max_m``: scalar is a BM=1 GEVM — sensible for decode-sized M,
    hopeless at prefill, and hopeless-but-benched configs poison the TPE's per-dimension
    densities (measured: with scalar in the M=8192 grid the 2D MX attn prefill tune
    landed 0.48x vs hub; without it, 2.06x)."""

    def prune(configs, named_args, **kwargs):
        if {**named_args, **kwargs}[m_arg] <= max_m:
            return configs
        return [
            c for c in configs if c.kwargs.get("COMPUTE_MODE") != "scalar"
        ] or configs

    return prune


def batched_mx_pruner(k_arg: str, stacked_gate_up: bool = False):
    """``early_config_prune`` for the batched MX kernels: a BK-within-K veto plus two sm_10x
    MMA-shape guards (no-ops elsewhere and for scalar configs). With ``stacked_gate_up`` the
    kernel computes gate|up as one stacked 2*BN extent, so the swapped dot_scaled lhs has
    ``2*BN`` rows and the no-swap combined dot is ``2*BN`` wide; the down kernel's counts are
    just ``BN``.

    - ``BLOCK_SIZE_K`` not dividing the launch's contraction dim (``k_arg`` names it: ``"K"``
      for the batched matmul, ``HIDDEN_DIM`` / ``INTERMEDIATE_DIM`` for the fused gate_up /
      down) → dropped: the K-loop loads are unmasked, so any non-dividing BK's last trip reads
      past the row — silently wrong results the tuner would happily time and pick (bit us when
      BK=512 met a K=256 test problem).
    - MXFP4 ``scalar`` configs → dropped (sm_10x only — the 1.8x-dead evidence is B200;
      other targets lower dot_scaled differently and keep the mode): fp4 scalar decode is ALU-bound in the E2M1 unpack
      and measured 1.8x SLOWER than dot_scaled (twice, incl. the no-pad form) — it never wins,
      and its swapped variants poison the TPE's per-dimension model into writing off SWAP_AB
      (a 100-trial dsv4 down tune benched 3 swap configs — two dead-slow swap-scalar, one inf —
      and shipped a 38.9µs no-swap winner, missing the ~24µs swap dot_scaled basin entirely).
    - Swapped ``dot_scaled`` rows < 128 → dropped: the native mxfp scaled-MMA gates on the M
      operand being exactly 128, so smaller rows run the bf16-upcast fallback and never win —
      the same poison mechanism (an earlier dsv4 gate_up tune shipped 63µs missing the 43µs
      swap winner).
    - No-swap tensor-core width > 256 → dropped: sm_10x caps an MMA at N=256 and Triton
      miscompiles wider ones into a sticky device trap (see ``smem_config_pruner``'s
      ``wide_dot_scaled``).

    Never returns empty — dot_scaled no-swap configs pass every guard (the ``or kept``
    fallbacks cover the pathological cases). A contraction dim smaller than every grid BK is
    a hard error: any config would over-read past the row and return silently wrong results."""
    n_blocks = 2 if stacked_gate_up else 1
    bk_within_k = block_k_within_k_pruner(k_arg)

    def prune(configs, named_args, **kwargs):
        all_args = {**named_args, **kwargs}
        kept = bk_within_k(configs, named_args, **kwargs)
        on_sm10x = is_sm10x()
        if on_sm10x and all_args.get("VALUES_PER_BYTE") == 2:
            kept = [c for c in kept if c.kwargs.get("COMPUTE_MODE") != "scalar"] or kept
        if on_sm10x:

            def ok(c):
                if c.kwargs.get("COMPUTE_MODE") == "scalar":
                    return True
                # dot_scaled BK < 128 performs MISALIGNED accesses (verified: every
                # isolated BK=64 launch prints the cudaErrorMisalignedAddress signature;
                # BK>=128 never does). The UB is intermittently fatal — two long tuning
                # runs died with the sticky CUDA 716 context corruption while isolated
                # launches survive — so the rows can't even be benched safely.
                if c.kwargs["BLOCK_SIZE_K"] < 128:
                    return False
                rows = n_blocks * c.kwargs["BLOCK_SIZE_N"]
                if c.kwargs.get("SWAP_AB"):
                    return rows >= 128
                return rows <= 256

            kept = [c for c in kept if ok(c)] or kept
        return kept

    return prune


def smem_config_pruner(
    n_weight_tiles: int,
    weight_bytes: int = 1,
    reduction_dim: str | None = None,
    double_mma: bool = False,
):
    """Build an ``early_config_prune`` that drops configs whose pipelined operand shared
    memory would overflow the SM — the source of ``out of resource: shared memory`` autotune
    failures (and the wasted compiles they cause).

    Per-stage estimate (bytes) = ``BK · (act_bytes·BM + weight_bytes·n_weight_tiles·BN)``:
    one activation tile ``[BM, BK]`` plus ``n_weight_tiles`` weight tiles ``[BN, BK]``
    (gate_up loads 2; down has 1), times ``num_stages``. ``BM`` is the routing-derived tile
    and the act element size is read off the first launch arg (the activation tensor by
    kernel convention), so both track the launch. MXFP4 packs 2 weights/byte, so
    ``weight_bytes=1`` (MXFP8) is a safe upper bound. The limit is read from the active device. Never returns empty — keeps
    the smallest-footprint config as a fallback.

    Two sm_10x (Blackwell-datacenter, TMEM scaled-MMA) ``dot_scaled`` compiler-bug guards, both
    no-ops off sm_10x:

    ``reduction_dim`` (the K-loop bound arg name, e.g. ``"INTERMEDIATE_DIM"``) drops ``dot_scaled``
    configs with ``BLOCK_SIZE_K >= reduction_dim`` — a single-trip K-loop, which trips a Triton
    ``optimize-accumulator-init`` bug (uninitialized TMEM alloc must be mutable) and never
    compiles. ``None`` (default) disables the check.

    ``double_mma`` marks a kernel that folds its ``n_weight_tiles`` into ONE wide MMA rather than
    dotting them separately — the fused gate_up kernel does a single ``[BM, n_weight_tiles*BN]``
    dot (n_weight_tiles=2 → double-width N=2*BN). Its ``dot_scaled`` configs whose MMA width
    ``n_weight_tiles * BLOCK_SIZE_N`` exceeds sm_10x's N=256 cap are dropped (Triton miscompiles
    wider ones: packed-E2M1 rhs → sticky "misaligned address" device trap). Off by default: a
    kernel can hold 2 weight tiles yet dot them *separately* (each ``[BM, BN]``), e.g. the batched
    gate_up — there ``n_weight_tiles=2`` but ``double_mma=False``, so N stays BN."""

    def prune(configs, named_args, **kwargs):
        """``early_config_prune`` callback. Drops configs that (a) overflow the SM's shared memory
        or run a ``dot_scaled`` that miscompiles on sm_100 — (b) a single-trip K-loop or (c) an
        MMA wider than N=256. Falls back to the smallest-smem config so the autotuner is never
        handed an empty list."""
        # Tile dims live either in the autotuned config meta (MX) or the launch args (block-dynamic),
        # so `dim` looks in both.
        all_args = {**named_args, **kwargs}
        limit = sm_shared_memory_limit()
        # The dot_scaled guards below are Blackwell-datacenter (sm_10x, TMEM scaled-MMA) compiler
        # bugs — off on every other target, so they never over-prune there.
        on_sm10x = is_sm10x()

        def dim(c, name):
            """A dimension for config ``c`` (a ``BLOCK_SIZE_*`` tile dim or a reduction extent like
            ``HIDDEN_DIM``) — from its config meta or the launch args; raises if present in neither."""
            v = c.kwargs.get(name, all_args.get(name))
            if v is None:
                raise ValueError(
                    f"smem_config_pruner needs {name} (autotune config meta or launch arg) "
                    "to estimate shared memory; none found."
                )
            return v

        def smem(c):
            """Peak pipelined-operand shared memory (bytes) for ``c``: ``num_stages`` copies of one
            ``[BM, BK]`` activation tile plus ``n_weight_tiles`` ``[BN, BK]`` weight tiles.
            The act element size is read off the FIRST launch arg — the activation tensor
            by kernel convention — so it tracks the dtype per launch (fp8 offline arm vs
            raw bf16 inline arm below the ``maybe_act_quant`` threshold)."""
            BM, BN, BK = (
                dim(c, n) for n in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
            )
            act_bytes = next(iter(named_args.values())).element_size()
            return (
                c.num_stages
                * BK
                * (act_bytes * BM + weight_bytes * n_weight_tiles * BN)
            )

        def single_trip_dot_scaled(c):
            """True if ``c`` is a ``dot_scaled`` config whose K-loop is a single trip
            (``BLOCK_SIZE_K >= reduction_dim``) — the sm_10x accumulator-init miscompile (see the
            ``reduction_dim`` note above). No-op when ``reduction_dim`` is None or off sm_10x."""
            return (
                on_sm10x
                and reduction_dim is not None
                and c.kwargs.get("COMPUTE_MODE") == "dot_scaled"
                and dim(c, "BLOCK_SIZE_K") >= dim(c, reduction_dim)
            )

        def wide_dot_scaled(c):
            """True if ``c`` is a tensor-core config whose single MMA is wider than N=256 —
            sm_10x caps an MMA there and Triton miscompiles wider ones (sticky illegal-address
            device trap that poisons the context mid-autotune). Hits ``dot_scaled`` AND plain
            ``dot``: an exhaustive forced-config sweep (M3 MXFP8 gate_up, 1440 configs) trapped
            both modes at width 512, config-dependently (BM/warps/stages decide whether the
            compiler splits the wide MMA), so the whole width class is dropped. The MMA width is
            ``n_weight_tiles * BLOCK_SIZE_N`` when the kernel fuses its tiles into one dot
            (``double_mma``); otherwise each dot is N=BN and can't exceed the cap. No-op off
            sm_10x or when ``double_mma`` is False."""
            return (
                on_sm10x
                and double_mma
                and c.kwargs.get("COMPUTE_MODE") in ("dot_scaled", "dot")
                and n_weight_tiles * dim(c, "BLOCK_SIZE_N") > 256
            )

        # The smem estimate CANNOT classify near the limit: vs compiled ground truth (2880-config
        # sweep) configs ran fine up to est=1.41x limit while real failures dipped to est=0.21x —
        # the distributions overlap, so any tight threshold trims viable configs (a 1.0x cutoff
        # measurably discarded 138 that compile and run). It only vetoes the impossible: >2x is
        # 42% above the largest surviving over-estimate ever observed, and still skips the deep
        # overflows (up to 5x). Everything else is left to the compiler — a config that doesn't
        # fit fails its one benching compile and is scored inf.
        kept = [
            c
            for c in configs
            if smem(c) <= 2 * limit
            and not single_trip_dot_scaled(c)
            and not wide_dot_scaled(c)
        ]
        return kept or [min(configs, key=smem)]

    return prune


# ── Triton-side helpers (inlined by ``@triton.jit`` callers) ──────────────────


@triton.jit
def fp8_act_quant_inline(a_raw, TRANSPOSED: tl.constexpr = False):
    """Inline FP8 (E4M3) activation quant for the W8A8 block-scale path.

    Per-token amax → fp32 scale ``amax/448`` (floored at 1e-12 against zero rows)
    → cast values to FP8. Returns ``(a_fp8, a_s)`` with ``a_s`` shaped ``(M,)``;
    ``TRANSPOSED`` marks a ``(K, M)`` tile (the swapped descriptor arm), where the
    token axis is 0 instead of 1.
    """
    if TRANSPOSED:
        a_s = tl.max(tl.abs(a_raw), axis=0) / 448.0
        a_fp8 = (a_raw / tl.maximum(a_s[None, :], 1e-12)).to(tl.float8e4nv)
    else:
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
def load_block_fp8_act_tile(a_ptrs, as_ptrs, TRANSPOSED: tl.constexpr = False):
    """Block-FP8 counterpart of ``load_mx_act_tile``: load one activation K-tile as
    ``(a_fp8, a_scale_f32)`` — the arm folds off the pointer dtype at compile time
    (fp8 = pre-quantized offline + per-K-block scales, raw bf16/fp16 = quantize inline;
    ``as_ptrs`` is then a constexpr-dead placeholder). ``TRANSPOSED`` marks a ``(K, M)``
    tile (the swapped descriptor arm) so the inline amax reduces the token axis either
    way. Unmasked: every caller's rows are %-wrapped, expert-advanced, or
    token-replicated."""
    if a_ptrs.dtype.element_ty == tl.float8e4nv:  # pre-quantized offline
        a = tl.load(a_ptrs)
        a_s = tl.load(as_ptrs)
    else:  # raw bf16/fp16 — quantize inline
        a, a_s = fp8_act_quant_inline(tl.load(a_ptrs).to(tl.float32), TRANSPOSED)
    return a, a_s


@triton.jit
def load_mx_act_tile(
    a_ptrs,
    as_ptrs,
    row_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Load one MX activation K-tile as ``(a_fp8, a_scale_u8)`` — the arm is picked
    off the pointer dtype at compile time: fp8 pointers load pre-quantized values +
    UE8M0 scales (``maybe_act_quant``'s offline arm), raw bf16/fp16 pointers load and
    quantize inline (``as_ptrs`` then points at a dead placeholder and is never read).
    ``row_mask`` may be ``None`` (unmasked tiles, e.g. the %-wrapped 2D matmul).
    Callers advance both pointers unconditionally."""
    if a_ptrs.dtype.element_ty == tl.float8e4nv:  # pre-quantized offline
        if row_mask is None:
            a = tl.load(a_ptrs)
            a_scale = tl.load(as_ptrs)
        else:
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0)
    else:  # raw bf16/fp16 — quantize inline
        if row_mask is None:
            a_raw = tl.load(a_ptrs).to(tl.float32)
        else:
            a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a, a_scale = mxfp_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
    return a, a_scale


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
    updated). The batched gate_up kernel passes the stacked
    gate|up tile (2*BN columns) — per-column independence keeps that bit-exact."""
    wq = mxfp4_e2m1_to_e4m3(w) if VALUES_PER_BYTE == 2 else w
    return acc + tl.dot(a, wq) * decode_ue8m0_scale(a_scale) * tl.trans(
        decode_ue8m0_scale(w_scale)
    )


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
    decode GEMV (M=1). The batched gate_up kernel passes the stacked gate|up tile (ROWS_W = 2*BN).

    The UE8M0 scale is constant within each group of ``SCALE_GROUP_K``, so it factors out of the
    inner sum: instead of expanding it to every K element and doing ``BLOCK_SIZE_K`` scale-muls,
    reduce the raw products within each group, then apply ONE combined (act × weight) scale per
    group — ``SCALE_GROUP_K``× fewer scale-muls. Measured ~18% faster on the decode reduce
    (the per-element expand was pure overhead), bit-identical to the expanded form (rel 1e-7)."""
    wq = mxfp4_e2m1_to_e4m3(w) if VALUES_PER_BYTE == 2 else w
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    prod = tl.trans(a.to(tl.float32)) * wq.to(tl.float32)  # [BK, ROWS_W]
    grp = tl.sum(
        tl.reshape(prod, (NG, SCALE_GROUP_K, ROWS_W)), axis=1
    )  # per-group partial
    scale = tl.trans(decode_ue8m0_scale(a_scale)) * tl.trans(
        decode_ue8m0_scale(w_scale)
    )
    return acc + tl.sum(grp * scale, axis=0)[None, :]


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
    SWAP_AB: tl.constexpr = False,
):
    """Single-projection MMA step. Under ``SWAP_AB`` the swapped decode path runs (weight output rows
    in the MMA M dim — different acc shape/finalize; see ``mx_swap_compute``); otherwise dispatch on
    ``COMPUTE_MODE``: scaled-MMA on the raw weight (``w``), or fp8 ``tl.dot`` + per-group rescale /
    scalar reduce on the E4M3-decoded weight. Single return — only the taken branch compiles."""
    if SWAP_AB:
        acc = mx_swap_compute(
            acc,
            a,
            a_scale,
            w,
            w_scale,
            COMPUTE_MODE,
            VALUES_PER_BYTE,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
        )
    elif COMPUTE_MODE == "dot_scaled":
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


# ── swap-AB decode compute: M=1 batched GEMV with output rows in the MMA M dim ───────
#
# The batched (decode) kernels are structurally M=1, where the sm_100 scaled MMA pads M→128.
# Putting the WEIGHT's output rows in M (fully used) and the single token in N cuts that pad to
# the N-atom (16) — a ~1.5× decode win for fp4 (dot_scaled), neutral-to-worse for fp8 (scalar
# stays memory-bound and wins, so the tuner keeps it). Weight is loaded output-rows-major
# ``[BN, BK]`` for every mode, so the kernel does ONE load; each helper returns ``[1, BN]``.


# The sm_100 MMA's minimum N tile (16). In the swap path the single decode token sits in the MMA's
# N dim, so it must be padded up to this width (col 0 = the token, cols 1..15 = zero). It is NOT a
# block size — BLOCK_SIZE_M stays 1 under swap; this is the token's *padded N extent*, fixed by the
# hardware. Assigned via tl.constexpr(...), the only module-global form a @triton.jit fn can read.
MMA_N_ATOM = tl.constexpr(16)


@triton.jit
def mx_dot_scaled_swapped_rhs(a, a_scale, BLOCK_SIZE_K: tl.constexpr):
    """Build the swap-AB rhs (activation) + its scale: the [BK] E4M3 token becomes an
    ``[BK, MMA_N_ATOM]`` tile with only column 0 real (16 is Triton's tcgen05-selection gate,
    not the hardware floor: PTX shows N=8 drops dot_scaled to the bf16-upcast fallback —
    zero tcgen05 ops — costing +30%; bare-1 was 1.83x; plain dot is indifferent), and its
    UE8M0 scale is broadcast to the same padded shape."""
    rhs = swap_pad_rhs(a, BLOCK_SIZE_K)
    asc = tl.trans(
        a_scale[:, None] + tl.zeros((BLOCK_SIZE_K // 32, MMA_N_ATOM), tl.uint8)
    )
    return rhs, asc


@triton.jit
def mx_dot_scaled_swapped(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    VALUES_PER_BYTE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Swapped ``dot_scaled`` decode step: weight ``w`` [BN, BK] (E2M1 packed if MXFP4 else E4M3)
    is the MMA lhs (output rows in M); the activation is the N=16 rhs (col 0 real). ``acc`` is the
    persistent ``[BN, MMA_N_ATOM]`` MMA accumulator (accumulated across the K-loop, then the caller
    takes column 0) — NOT a fresh per-step init, which trips the sm_100 accumulator-init pass."""
    fmt: tl.constexpr = "e2m1" if VALUES_PER_BYTE == 2 else "e4m3"
    rhs, asc = mx_dot_scaled_swapped_rhs(a, a_scale, BLOCK_SIZE_K)
    return tl.dot_scaled(w, w_scale, fmt, rhs, asc, "e4m3", acc)


@triton.jit
def mx_dot_rescale_swapped(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    VALUES_PER_BYTE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Swapped MX 'dot' step (BK == one scale group): weight output rows in the MMA M dim
    (``[ROWS, BK]`` after the column-unpack for MXFP4), the [BK] token padded to the N=16
    atom — the well-shaped fp8 MMA at M=1 (M quantizes to 64/128, N only to 8, so weight
    rows fill the big atom). Both UE8M0 scales factor out of the single-group step: the
    weight's per-output-row scale broadcasts down the acc columns, the token's group scale
    is a scalar. ``acc`` is the persistent ``[ROWS, MMA_N_ATOM]`` accumulator (col 0 taken
    by the caller's ``acc_finalize``)."""
    if VALUES_PER_BYTE == 2:  # column-unpack E2M1 -> E4M3 (K order: low nibble first)
        wq = tl.interleave(_e2m1_code_to_f32(w & 0xF), _e2m1_code_to_f32(w >> 4)).to(
            tl.float8e4nv
        )
    else:
        wq = w
    rhs = swap_pad_rhs(a, BLOCK_SIZE_K)
    a_s = decode_ue8m0_scale(a_scale)  # [1] — the single group's token scale
    w_s = decode_ue8m0_scale(w_scale)  # [ROWS, 1] — per output row
    return acc + tl.dot(wq, rhs) * w_s * a_s


@triton.jit
def mx_scalar_reduce_swapped(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
):
    """Swapped scalar reduce: weight ``w`` output-rows-major ``[ROWS_W, BK]``, ``a`` the [BK]
    activation. No transpose (vs ``mx_scalar_reduce``); MXFP4 unpacks along columns (K). Per-group
    scale factored out of the reduce (grpscale). Reduces over K; returns ``acc + [1, ROWS_W]``."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    if VALUES_PER_BYTE == 2:  # column-unpack E2M1 -> f32, K-order via interleave
        wq = tl.interleave(_e2m1_code_to_f32(w & 0xF), _e2m1_code_to_f32(w >> 4))
    else:
        wq = w.to(tl.float32)
    prod = a.to(tl.float32)[None, :] * wq  # [ROWS_W, BK]
    grp = tl.sum(tl.reshape(prod, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)  # [ROWS_W, NG]
    scale = decode_ue8m0_scale(a_scale)[None, :] * decode_ue8m0_scale(w_scale)
    return acc + tl.reshape(tl.sum(grp * scale, axis=1), (1, ROWS_W))


@triton.jit
def mx_swap_compute(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Swapped-AB counterpart to ``mx_compute``: weight output-rows in the MMA M dim, the single
    decode token flattened to the [BK] rhs. Dispatches the three swapped modes — ``dot_scaled``
    and ``dot`` (persistent ``[BLOCK_SIZE_N, MMA_N_ATOM]`` MMA acc, col 0 taken by the caller)
    and ``scalar`` (``[1, BLOCK_SIZE_N]`` reduce). The acc shapes diverge, but only the taken
    constexpr branch compiles so the single return never has to unify them. ``BLOCK_SIZE_N`` is the weight tile's row count — the gate_up kernel passes ``2*BN``
    with its STACKED gate|up tile (gate rows first, split back via ``split_gate_up``): one
    load and one MMA for both projections keeps the native mxfp M=128 operand at BN=64, doubling
    the CTAs on the parallelism-starved decode grid (dsv4 gate_up 1.34x, bit-exact)."""
    a1 = tl.reshape(a, (BLOCK_SIZE_K,))
    as1 = tl.reshape(a_scale, (BLOCK_SIZE_K // SCALE_GROUP_K,))
    if COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled_swapped(
            acc, a1, as1, w, w_scale, VALUES_PER_BYTE, BLOCK_SIZE_K
        )
    elif COMPUTE_MODE == "dot":
        acc = mx_dot_rescale_swapped(
            acc, a1, as1, w, w_scale, VALUES_PER_BYTE, BLOCK_SIZE_K
        )
    elif COMPUTE_MODE == "scalar":
        acc = mx_scalar_reduce_swapped(
            acc,
            a1,
            as1,
            w,
            w_scale,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
            VALUES_PER_BYTE,
        )
    else:
        tl.static_assert(False, "unknown COMPUTE_MODE under SWAP_AB")
    return acc


@triton.jit
def swap_pad_rhs(a, BLOCK_SIZE_K: tl.constexpr):
    """Pad the ``[BLOCK_SIZE_K]`` M=1 token to the ``[BLOCK_SIZE_K, MMA_N_ATOM]`` swap-AB MMA rhs —
    only column 0 is the real token (16 is measured, not a hardware floor — see acc_init). Used
    by the M=1 batched / fused-MoE fp8 ``tl.dot`` swap paths (weight output rows in the MMA M dim);
    the caller takes column 0 of the ``[BN, MMA_N_ATOM]`` result after the K-loop."""
    return tl.where(
        tl.arange(0, MMA_N_ATOM)[None, :] == 0,
        a[:, None],
        tl.zeros((BLOCK_SIZE_K, MMA_N_ATOM), a.dtype),
    )


@triton.jit
def fp8_dot(a, b, SWAP_AB: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """Swap-aware plain ``tl.dot`` for the fp8 (block/tensor-dynamic) paths — no scaling; the caller
    applies its own per-block/per-tensor scales. Swap: weight ``b`` ``[N, BK]`` × the single token
    padded to the N=16 atom → ``[N, MMA_N_ATOM]`` (col 0 real). No-swap: token ``a`` ``[M, BK]`` ×
    weight ``b`` ``[BK, N]`` → ``[M, N]``. ``BLOCK_SIZE_K`` is the contraction tile (the down
    projection passes its intermediate tile). Single return: only the taken branch compiles."""
    if SWAP_AB:
        out = tl.dot(b, swap_pad_rhs(tl.reshape(a, (BLOCK_SIZE_K,)), BLOCK_SIZE_K))
    else:
        out = tl.dot(a, b)
    return out


@triton.jit
def swizzle_offsets(
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """2D-grid tile scheduling shared by the kernels below: swizzle the
    ``(pid_m, pid_n)`` program ids for L2 locality on B, then build the operand
    offset vectors. Returns ``(pid_m, pid_n, offs_am, offs_bn, offs_k)`` — the
    swizzled ids (reused by the output store) and the ``%``-wrapped row/col offsets
    plus the K range."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    return pid_m, pid_n, offs_am, offs_bn, offs_k


@triton.jit
def store_masked(
    C,
    accumulator,
    pid_m,
    pid_n,
    M,
    N,
    stride_c_m,
    stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Shared output epilogue of the kernels below: cast the fp32 accumulator to
    ``C``'s dtype and store the ``(BLOCK_SIZE_M, BLOCK_SIZE_N)`` tile at the swizzled
    ``(pid_m, pid_n)``, masked to the ``(M, N)`` bounds."""
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_c_m * offs_cm[:, None] + stride_c_n * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def store_masked_oriented(
    C,
    accumulator,
    pid_m,
    pid_n,
    M,
    N,
    stride_c_m,
    stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """``store_masked`` with the (m, n) roles swapped under ``SWAP_AB``, where the
    accumulator is ``(BLOCK_SIZE_N, BLOCK_SIZE_M)``. Only the taken branch compiles."""
    if SWAP_AB:
        store_masked(
            C,
            accumulator,
            pid_n,
            pid_m,
            N,
            M,
            stride_c_n,
            stride_c_m,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
        )
    else:
        store_masked(
            C,
            accumulator,
            pid_m,
            pid_n,
            M,
            N,
            stride_c_m,
            stride_c_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )


# Flat-slot tile per program for the O(S) routing kernels (count + scatter). These are small
# latency-bound atomic kernels that want many programs: a sweep over {256..4096} x prefill shapes
# put 256 best (or within ~1%) for both, with 1024 up to ~1.5x slower. The grid derives from it
# so the two can't drift. Power of 2.
_ROUTING_BLOCK_SIZE = 256


@triton.jit
def _exclusive_offsets_kernel(
    ExpertFreq, ExpertStart, Counters, NUM_EXPERTS: tl.constexpr
):
    """Exclusive cumsum of per-expert token counts → ``expert_start`` (leading 0, trailing
    S), and zero the scatter counters — one launch."""
    offs = tl.arange(0, NUM_EXPERTS)
    incl = tl.cumsum(tl.load(ExpertFreq + offs), 0)
    tl.store(ExpertStart, 0)
    tl.store(ExpertStart + 1 + offs, incl)
    tl.store(Counters + offs, tl.zeros([NUM_EXPERTS], tl.int32))


@triton.jit
def _scatter_kernel(
    ExpertIds,
    Perm,
    PermToken,
    ExpertStart,
    Counters,
    S,
    NUM_TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Counting-sort scatter: each flat slot atomically claims the next slot of its expert
    (``expert_start[e] + counter[e]++``). O(S), replaces an O(S·logS) argsort. Within-expert
    order is arbitrary (atomic race) — fine, the per-token reduce is order-invariant. Slots whose
    expert is non-local (EP sentinel id ``>= NUM_EXPERTS``) are skipped — matches ``_count_kernel``,
    and avoids the atomic/store landing at an out-of-range (invalid) global address."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_id = tl.load(ExpertIds + offs, mask=offs < S, other=NUM_EXPERTS)
    valid = expert_id < NUM_EXPERTS
    dest = tl.load(ExpertStart + expert_id, mask=valid, other=0) + tl.atomic_add(
        Counters + expert_id, 1, mask=valid, sem="relaxed"
    )
    tl.store(Perm + dest, offs, mask=valid)
    tl.store(PermToken + dest, offs // NUM_TOP_K, mask=valid)


@triton.jit
def _count_kernel(
    ExpertIds, ExpertFreq, S, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Per-expert token count via atomics — replaces ``torch.histc`` (no float cast), fixed
    ``(NUM_EXPERTS,)`` output stays CUDA-graph friendly. ``ExpertFreq`` is pre-zeroed."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < S
    expert_id = tl.load(ExpertIds + offs, mask=mask, other=NUM_EXPERTS)
    tl.atomic_add(
        ExpertFreq + expert_id, 1, mask=mask & (expert_id < NUM_EXPERTS), sem="relaxed"
    )


class GroupedScheduling(NamedTuple):
    """One routing pass over ``expert_ids``, shared by every grouped GEMM of a layer.
    ``perm_routed`` maps each expert-sorted position to its token-major routed row
    ``(t*K + j)`` — the same ``perm = torch.sort(expert_ids)`` indices as transformers'
    moe.py. ``perm_token`` is its row-of-hidden form ``perm_routed // num_top_k``
    (many-to-one for
    top_k > 1 — the gather that reads hidden without replication), NOT ``inv_perm``:
    the inverse is never materialized here, because the kernels un-permute by
    SCATTERING through ``perm_routed`` at store time instead of gathering through ``inv_perm``
    on the host. ``expert_start`` is ``(E+1,)`` cumulative sorted-row starts padded with
    S — what the kernels build their register-resident tile layout from."""

    perm_token: torch.Tensor
    perm_routed: torch.Tensor
    expert_start: torch.Tensor


def compute_grouped_scheduling(
    expert_ids: torch.Tensor, num_experts: int, num_top_k: int
) -> GroupedScheduling:
    """On-device routing: expert-sorted index (no copy of the activations) via two Triton
    launches — exclusive offsets + an atomic counting-sort scatter (replaces host
    ``argsort``). Public: run it once per layer and pass the handle to every grouped GEMM
    of that layer (``matmul_grouped``'s ``input_ordered``/``output_ordered`` flags pick
    the right map) and to the fused MoE ops. E must be a power of 2 (the scheduling kernels
    hold the per-expert vectors in one ``tl.arange`` block)."""
    # the scheduling kernels hold the (E,) frequency/offset vectors in one tl.arange
    # block, which requires a power of 2 — fail here with a clear message instead of a
    # Triton compile error from an internal kernel
    assert num_experts & (num_experts - 1) == 0, (
        f"num_experts ({num_experts}) must be a power of 2"
    )
    perm_token, perm, expert_start = _compute_grouped_scheduling(
        expert_ids, num_experts, num_top_k
    )
    return GroupedScheduling(perm_token, perm, expert_start)


@compile_time_only_triton_op(
    add_op_namespace_prefix("compute_grouped_scheduling"), mutates_args=()
)
def _compute_grouped_scheduling(
    expert_ids: torch.Tensor, num_experts: int, num_top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = expert_ids.device
    expert_ids = expert_ids.int().contiguous()  # routing kernels index with unit stride
    num_routed_tokens = expert_ids.numel()  # S = num_tokens * num_top_k
    expert_freq = torch.zeros(num_experts, dtype=torch.int32, device=device)
    expert_start = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    counters = torch.empty(num_experts, dtype=torch.int32, device=device)
    perm = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
    perm_token = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
    with device_context(device):
        compile_time_only_triton_wrap(_count_kernel)[
            (triton.cdiv(num_routed_tokens, _ROUTING_BLOCK_SIZE),)
        ](
            expert_ids,
            expert_freq,
            num_routed_tokens,
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE=_ROUTING_BLOCK_SIZE,
        )
        compile_time_only_triton_wrap(_exclusive_offsets_kernel)[(1,)](
            expert_freq,
            expert_start,
            counters,
            NUM_EXPERTS=num_experts,
        )
        compile_time_only_triton_wrap(_scatter_kernel)[
            (triton.cdiv(num_routed_tokens, _ROUTING_BLOCK_SIZE),)
        ](
            expert_ids,
            perm,
            perm_token,
            expert_start,
            counters,
            num_routed_tokens,
            NUM_TOP_K=num_top_k,
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE=_ROUTING_BLOCK_SIZE,
        )
    return perm_token, perm, expert_start


@triton.jit
def build_tile_layout(
    ExpertStart, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    """Load ``expert_start`` once and derive the per-BM tile layout vectors (kept in
    registers for the whole persistent loop): per-expert first sorted row, token count,
    exclusive tile-start cumsum, and the total M-tile count. ``ExpertStart`` is
    ``(NUM_EXPERTS + 1,)`` with a trailing ``S`` sentinel (``expert_start[E] == S``)."""
    e_offs = tl.arange(0, NUM_EXPERTS)
    exp_start = tl.load(ExpertStart + e_offs)
    exp_end = tl.load(ExpertStart + e_offs + 1)
    freqs = exp_end - exp_start
    tiles_per_e = (freqs + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_start_excl = (
        tl.cumsum(tiles_per_e, 0) - tiles_per_e
    )  # first tile index of expert e
    total_m_tiles = tl.sum(tiles_per_e, 0)
    return exp_start, freqs, tile_start_excl, total_m_tiles, e_offs


@triton.jit
def resolve_tile_inline(
    pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M: tl.constexpr
):
    """Map an M-tile id to its owning expert + the tile's sorted row range, from the
    register-resident layout (no global loads). Returns ``(expert_id, sorted_indices,
    row_mask)``."""
    # Bucketize via the exclusive tile cumsum: #experts whose tile-start <= pid_m, minus 1.
    expert_id = tl.sum((tile_start_excl <= pid_m).to(tl.int32), 0) - 1
    sel = (
        e_offs == expert_id
    )  # scalar-index the E-vectors via mask-sum (no dynamic index)
    e_start = tl.sum(tl.where(sel, exp_start, 0), 0)
    e_tile_start = tl.sum(tl.where(sel, tile_start_excl, 0), 0)
    freq = tl.sum(tl.where(sel, freqs, 0), 0)
    within = pid_m - e_tile_start
    m_start = e_start + within * BLOCK_SIZE_M
    offs = tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs < freq - within * BLOCK_SIZE_M
    sorted_indices = tl.max_contiguous(m_start + offs, BLOCK_SIZE_M)
    return expert_id, sorted_indices, row_mask


@triton.jit
def resolve_grouped_tile(
    tile_id,
    num_n_tiles,
    exp_start,
    freqs,
    tile_start_excl,
    e_offs,
    InputPerm,
    OutputPerm,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_INPUT_PERM: tl.constexpr,
    HAS_OUTPUT_PERM: tl.constexpr,
):
    """One persistent grouped tile: split the flat ``tile_id`` into (M-tile, N-tile), map
    the M-tile to its expert + rows via ``resolve_tile_inline`` (on the register-resident
    layout ``build_tile_layout`` builds once per program, passed in), and apply the virtual
    sort — rows load from ``in_row`` and store to ``out_row``, mapped by ``InputPerm`` /
    ``OutputPerm`` when present else the expert-sorted position itself.

    Returns ``(pid_n, expert_id, expert_id64, in_row, out_row, row_mask, offs_bn)`` — both
    expert-id widths: ``expert_id`` (int32, e.g. TMA descriptor row indices, bounded by the
    expert count) and ``expert_id64`` (int64, for byte-offset pointer arithmetic — ``expert
    * stride`` overflows int32 at full expert counts). Shared by the base grouped GEMMs and
    the fused kernels; callers ``_``-ignore whichever width they don't use."""
    pid_m = tile_id // num_n_tiles
    pid_n = tile_id % num_n_tiles
    expert_id, offs_global_m, row_mask = resolve_tile_inline(
        pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
    )
    if HAS_INPUT_PERM:
        in_row = tl.load(InputPerm + offs_global_m, mask=row_mask, other=0)
    else:
        in_row = offs_global_m
    if HAS_OUTPUT_PERM:
        out_row = tl.load(OutputPerm + offs_global_m, mask=row_mask, other=0)
    else:
        out_row = offs_global_m
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    return pid_n, expert_id, expert_id.to(tl.int64), in_row, out_row, row_mask, offs_bn


@triton.jit
def store_tile(
    C, accumulator, offs_global_m, offs_bn, row_mask, stride_c_m, stride_c_n
):
    """Output epilogue shared by the grouped kernels: cast the fp32 accumulator to
    ``C``'s dtype and store the tile at expert-sorted global rows ``offs_global_m`` ×
    columns ``offs_bn``, masked to the expert's valid rows (``row_mask``)."""
    c = accumulator.to(C.dtype.element_ty)
    c_ptrs = C + stride_c_m * offs_global_m[:, None] + stride_c_n * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=row_mask[:, None])


@triton.jit
def weight_tile_descriptor(
    HostDescriptor,
    W,
    N,
    K,
    stride_n,
    stride_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MEMORY_MODE: tl.constexpr,
):
    """Resolve the weight-tile descriptor once per program: the host-built TMA descriptor
    as passed, a device-built in-kernel tensormap, or 0 under "pointer" (never read — the
    constexpr branch folds it out of ``load_weight_tile``). Single return — only the taken
    branch compiles."""
    if MEMORY_MODE == "host_descriptor":
        descriptor = HostDescriptor
    elif MEMORY_MODE == "device_descriptor":
        descriptor = tl.make_tensor_descriptor(
            W,
            shape=(N, K),
            strides=(stride_n, stride_k),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
    else:  # pointer
        descriptor = 0
    return descriptor


@triton.jit
def load_weight_tile(w_ptrs, w_descriptor, row_off, k_off, MEMORY_MODE: tl.constexpr):
    """One weight K-tile: the ``(BN, BK)`` descriptor box at ``(row_off, k_off)`` under
    the descriptor modes, else the explicit-pointer tile (whatever orientation ``w_ptrs``
    was built with). Single return — only the taken branch compiles."""
    if MEMORY_MODE == "pointer":
        w = tl.load(w_ptrs)
    else:
        w = w_descriptor.load([row_off, k_off])
    return w


@triton.jit
def block_scaled_fp8_dot(a, a_scale, w, w_scale, SWAP_AB: tl.constexpr):
    """Plain fp8 ``tl.dot`` with the per-block scales applied, oriented: no-swap
    ``(M, N) = (a @ w) * a_s[:, None] * w_s[None, :]``; swapped the weight rows sit in the
    MMA M dim, ``(N, M) = (w @ a) * w_s[:, None] * a_s[None, :]``. Single return — only
    the taken branch compiles."""
    if SWAP_AB:
        out = tl.dot(w, a) * w_scale[:, None] * a_scale[None, :]
    else:
        out = tl.dot(a, w) * a_scale[:, None] * w_scale[None, :]
    return out


@triton.jit
def stacked_gate_up_ptrs(
    base, offs_n, offs_k, block_stride, stride_n, stride_k, SWAP_AB: tl.constexpr
):
    """Gate|up stacked 3D weight-tile pointers, oriented by ``SWAP_AB`` — the gate_up
    counterpart of ``oriented_tile_ptrs``. One axis indexes the {gate, up} row block (up
    offset by ``block_stride``), placed so ``stacked_gate_up_flatten``'s plain reshape yields
    the 2D stacked tile: swap ``[2, N, K]`` (output rows in the MMA M dim), no-swap
    ``[K, 2, N]`` (K-major, gate|up along the MMA N dim — the grouped kernel's combined form).
    The per-step K-advance is the same scalar stride step in both orientations."""
    blk = tl.arange(0, 2) * block_stride
    if SWAP_AB:
        ptrs = base + (
            blk[:, None, None]
            + offs_n[None, :, None] * stride_n
            + offs_k[None, None, :] * stride_k
        )
    else:
        ptrs = base + (
            offs_k[:, None, None] * stride_k
            + blk[None, :, None]
            + offs_n[None, None, :] * stride_n
        )
    return ptrs


@triton.jit
def stacked_gate_up_flatten(
    w3, N2: tl.constexpr, KB: tl.constexpr, SWAP_AB: tl.constexpr
):
    """Flatten a loaded 3D gate|up tile (see ``stacked_gate_up_ptrs``) to the stacked 2D tile:
    swap ``[N2, KB]`` (rows-major MMA lhs), no-swap ``[KB, N2]`` (K-major rhs). Rows/columns
    0..N-1 are gate, N..2N-1 up — ``split_gate_up`` undoes the stacking after the K-loop."""
    if SWAP_AB:
        w2 = tl.reshape(w3, (N2, KB))
    else:
        w2 = tl.reshape(w3, (KB, N2))
    return w2


@triton.jit
def oriented_tile_ptrs(
    base, offs_rows, offs_k, stride_rows, stride_k, SWAP_AB: tl.constexpr
):
    """Operand-tile pointers oriented by whether these rows sit in the MMA M dim
    (``SWAP_AB``, from the weight's viewpoint — activation callers pass the flag inverted):
    rows-major ``[rows, K]`` when they do, else K-major ``[K, rows]``. Only the taken
    constexpr branch compiles, so the divergent shapes never meet. The per-step K-advance
    is identical for both layouts, so the caller advances the returned pointer the same
    way regardless of orientation."""
    if SWAP_AB:
        ptrs = base + (offs_rows[:, None] * stride_rows + offs_k[None, :] * stride_k)
    else:
        ptrs = base + (offs_k[:, None] * stride_k + offs_rows[None, :] * stride_rows)
    return ptrs


@triton.jit
def acc_init(
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """Zero accumulator shaped for the layout: swapped scalar reduces into ``[1, N]``; any other
    swapped mode keeps the ``[N, M]`` MMA acc (weight rows in M, act tile in N — padded up to
    the ``MMA_N_ATOM`` when the caller is the single-token decode GEVM, col 0 taken after the
    K-loop); no-swap uses ``[M, N]``. ``COMPUTE_MODE`` matters only under swap — kernels with no
    mode axis (fp8 ``tl.dot``) pass ``"dot"``. ``N`` is the weight-output tile (``BLOCK_SIZE_H``
    for the fp8 down projection). Single return: only the taken branch compiles."""
    if SWAP_AB and COMPUTE_MODE == "scalar":
        acc = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)
    elif SWAP_AB and BLOCK_SIZE_M < MMA_N_ATOM:
        acc = tl.zeros((BLOCK_SIZE_N, MMA_N_ATOM), dtype=tl.float32)
    elif SWAP_AB:
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    return acc


@triton.jit
def acc_finalize(
    acc, COMPUTE_MODE: tl.constexpr, ROWS: tl.constexpr, SWAP_AB: tl.constexpr
):
    """Bookend to ``acc_init``: when the acc was built as the persistent ``[ROWS, MMA_N_ATOM]`` MMA
    tile (any swapped non-scalar mode), collapse the padded token dim to column 0 → ``[1, ROWS]``.
    Swapped scalar (already ``[1, ROWS]``) and no-swap pass through unchanged. ``COMPUTE_MODE``
    matches ``acc_init``'s (fp8 ``tl.dot`` kernels, which have no mode axis, pass ``"dot"``)."""
    if SWAP_AB and COMPUTE_MODE != "scalar":
        # take column 0: the padded token dim collapses back to the single real token
        acc = tl.reshape(
            tl.sum(acc * (tl.arange(0, MMA_N_ATOM)[None, :] == 0), axis=1), (1, ROWS)
        )
    return acc


@triton.jit
def split_gate_up(
    acc,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """Bookend to the stacked gate|up accumulator: finalize it (swap MMA col-0 collapse or
    pass-through, via ``acc_finalize``) and split the stacked N extent back into the
    ``(gate, up)`` pair, each ``[rows, BN]`` (rows = 1 under swap, else BM). Gate was stacked
    first (see ``stacked_gate_up_flatten``)."""
    rows: tl.constexpr = 1 if SWAP_AB else BLOCK_SIZE_M
    flat = acc_finalize(acc, COMPUTE_MODE, 2 * BLOCK_SIZE_N, SWAP_AB)
    pair = tl.permute(tl.reshape(flat, (rows, 2, BLOCK_SIZE_N)), (0, 2, 1))
    g, u = tl.split(pair)
    return g, u


@triton.jit
def glu(
    gate,
    up,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.float32,
):
    """Gated linear unit on the gate/up matmul accumulators. ``SWIGLU_LIMIT`` clamps gate above and up
    to ``[-LIMIT, LIMIT]``; ``SWIGLU_ALPHA`` gives the clamped/scaled SwiGLU ``(up + 1) * gate * sigmoid(ALPHA *
    gate)`` (GPT-OSS / MiniMax), else ``ACT_FN(gate) * up`` (``ACT_FN`` in {silu, gelu, relu}, gelu exact
    via erf). ``SIMULATE_UNFUSED`` rounds each materialized value through ``INTERMEDIATE_DTYPE`` (the dtype the unfused path lands intermediates in) to match the
    unfused (separate-kernel) path, where every intermediate lands in that dtype."""
    g = gate
    u = up

    if SIMULATE_UNFUSED:
        g = g.to(INTERMEDIATE_DTYPE).to(tl.float32)
        u = u.to(INTERMEDIATE_DTYPE).to(tl.float32)

    if SWIGLU_LIMIT is not None:
        g = tl.minimum(g, SWIGLU_LIMIT)
        u = tl.minimum(tl.maximum(u, -SWIGLU_LIMIT), SWIGLU_LIMIT)

    if SWIGLU_ALPHA is not None:
        gate_scaled = g * SWIGLU_ALPHA
        if SIMULATE_UNFUSED:
            gate_scaled = gate_scaled.to(INTERMEDIATE_DTYPE).to(tl.float32)
        sig = tl.sigmoid(gate_scaled)
        if SIMULATE_UNFUSED:
            sig = sig.to(INTERMEDIATE_DTYPE).to(tl.float32)
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
        act = act.to(INTERMEDIATE_DTYPE).to(tl.float32)
        u = u.to(INTERMEDIATE_DTYPE).to(tl.float32)

    gated = act * u

    if SIMULATE_UNFUSED:
        gated = gated.to(INTERMEDIATE_DTYPE).to(tl.float32)

    return gated


@triton.jit
def _e2m1_code_to_f32(code):
    """One E2M1 4-bit code -> fp32. Layout ``[sign | exp(2) | mant(1)]``; the 8
    magnitudes are ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` (exp==0 is the 0/0.5 subnormal)."""
    code = code.to(tl.int32)
    s = (code >> 3) & 1
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    pow2 = (1 << e).to(
        tl.float32
    ) * 0.5  # e in 0..3 -> 0.5, 1, 2, 4 (int shift, no exp2)
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


def _quant_block_k_pruner(configs, named_args, **kwargs):
    """Keep configs whose BLOCK_K divides K (the quant grid is K // BLOCK_K programs per row;
    K is always a multiple of 32, so the BLOCK_K=32 configs guarantee a non-empty list)."""
    k = {**named_args, **kwargs}["K"]
    return [c for c in configs if k % c.kwargs["BLOCK_K"] == 0]


@bayesian_autotune(
    [
        triton.Config({"BLOCK_K": bk}, num_warps=w)
        for bk in (32, 64, 128, 256, 512, 1024)
        for w in (2, 4, 8)
    ],
    # t_bucket (log2 of the token count) is in the key: the grid is (T, K // BLOCK_K), so at
    # small T the block size is the only parallelism lever while at prefill scale it isn't —
    # same bucketing as the grouped kernels' tokens_per_sm_bit_length (raw T would retune per
    # unique token count).
    ["K", "t_bucket"],
    n_trials=100,
    prune_configs_by={"early_config_prune": _quant_block_k_pruner},
)
@triton.jit
def _mxfp_act_quant_kernel(
    X,
    Y,
    S,
    stride_x_t,
    stride_x_k,
    t_bucket,  # autotune key only (log2 token-count bucket); unused in body
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """One-pass MX activation quant: bf16 rows → E4M3 + UE8M0 group-32 scales. Grid
    ``(T, K // BLOCK_K)``; group boundaries are identical to the kernels' inline quant
    (32 | BLOCK_K | K), so consumers are bit-exact with the inline form. Arbitrary input
    strides (no host-side copy); Triton's ==1 specialization keeps the contiguous fast path."""
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(X + t * stride_x_t + offs * stride_x_k)[None, :].to(tl.float32)
    y, s = mxfp_act_quant_inline(x, 1, BLOCK_K, SCALE_GROUP_K)
    tl.store(Y + t * K + offs, tl.reshape(y, (BLOCK_K,)))
    sg = tl.program_id(1) * (BLOCK_K // SCALE_GROUP_K) + tl.arange(
        0, BLOCK_K // SCALE_GROUP_K
    )
    tl.store(
        S + t * (K // SCALE_GROUP_K) + sg, tl.reshape(s, (BLOCK_K // SCALE_GROUP_K,))
    )


def maybe_act_quant(x, act_quant, min_m):
    """Row-count-gated offline activation pre-quant. Apply ``act_quant`` (a one-pass
    quant kernel, e.g. ``mxfp_act_quant``) when the GEMM consuming ``x`` is
    compute-bound (``rows >= min_m``) — the inline form re-quantizes per N-tile there.
    ``min_m`` is the consumer kernel's crossover, defined next to its wrapper with its
    provenance (measured sweep or inherited estimate). Below the threshold return ``x``
    raw: the weight-bandwidth-bound GEMM quantizes its
    one thin tile inline for free (the UE8M0 inline quant is exponent-only), and a
    separate quant kernel is pure added latency (M=1 graph attn decode measured
    0.66-0.85x offline). Bit-exact either way. Returns ``(a, a_scale)``; the consumer
    kernel picks its arm off ``a``'s dtype at compile time (fp8 = pre-quantized, raw
    bf16/fp16 = quantize inline), so in the inline arm ``a_scale`` is a constexpr-dead
    placeholder."""
    if x.shape[0] >= min_m:
        return act_quant(x)
    return x, x


def mxfp_act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to MX once (E4M3 values + UE8M0 group-32 uint8 scales)
    instead of inline per weight-tile — the fused gate_up re-ran the inline quant per N-tile
    (16x redundant amax/convert ALU + 2x act bytes), which held it at ~380 TFLOPS while the
    pre-quantized down kernel ran at ~1080. One pass costs ~50µs at 8k tokens. Bit-exact with
    the inline form (same group boundaries)."""
    T, K = x.shape
    y = torch.empty(T, K, device=x.device, dtype=FP8_DTYPE)
    s = torch.empty(T, K // MX_SCALE_GROUP_K, device=x.device, dtype=torch.uint8)
    with device_context(x.device):
        compile_time_only_triton_wrap(_mxfp_act_quant_kernel)[
            lambda META: (T, K // META["BLOCK_K"])
        ](
            x,
            y,
            s,
            x.stride(0),
            x.stride(1),
            T.bit_length(),
            K=K,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
        )
    return y, s


@bayesian_autotune(
    [triton.Config({}, num_warps=w) for w in (1, 2, 4)],
    ["K", "BLOCK_K", "t_bucket"],
    n_trials=100,
)
@triton.jit
def _fp8_act_quant_2d_kernel(
    X,
    Y,
    S,
    stride_x_t,
    stride_x_k,
    t_bucket,  # autotune key only (log2 token-count bucket); unused in body
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One-pass block-FP8 activation quant: rows → E4M3 + one fp32 ``amax/448`` scale per
    ``BLOCK_K`` span. Grid ``(T, K // BLOCK_K)``; the span equals the consumer's
    ``BLOCK_SIZE_K``, so results are bit-exact with the kernels' inline quant. Arbitrary
    input strides (no host-side copy). ``BLOCK_K`` is fixed by the scale layout — only
    warps are tuned."""
    t = tl.program_id(0).to(tl.int64)
    kb = tl.program_id(1)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(X + t * stride_x_t + offs * stride_x_k)[None, :].to(tl.float32)
    y, s = fp8_act_quant_inline(x)
    tl.store(Y + t * K + offs, tl.reshape(y, (BLOCK_K,)))
    tl.store(S + t * (K // BLOCK_K) + kb + tl.arange(0, 1), tl.reshape(s, (1,)))


def fp8_act_quant_2d(
    x: torch.Tensor, block_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to block-FP8 once (E4M3 + fp32 per-``block_k`` scales)
    instead of inline per weight-tile — same rationale and layout as ``mxfp_act_quant`` (a
    GEMM re-reads its activation once per N-tile). Bit-exact with the inline form."""
    T, K = x.shape
    y = torch.empty(T, K, device=x.device, dtype=FP8_DTYPE)
    s = torch.empty(T, K // block_k, device=x.device, dtype=torch.float32)
    with device_context(x.device):
        compile_time_only_triton_wrap(_fp8_act_quant_2d_kernel)[(T, K // block_k)](
            x, y, s, x.stride(0), x.stride(1), T.bit_length(), K=K, BLOCK_K=block_k
        )
    return y, s


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


@compile_time_only_triton_op(add_op_namespace_prefix("fp8_act_quant"), mutates_args=())
def fp8_act_quant(
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
