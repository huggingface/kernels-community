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


def decode_bm_swap_pairs(swap_ab: bool = True):
    """The coupled decode ``(BLOCK_SIZE_M, SWAP_AB)`` pairs — the single source of the swap/BM
    coupling. At M=1 the MMA 16-atom is filled either by replicating the token (non-swap BM=16,
    ~40% over the degenerate BM=1 on plain tl.dot) or by putting the weight's output rows in M
    (swap, BM=1). Swap REQUIRES BM=1 (its rhs is the single token padded to the N=16 atom), so the
    pairs are ``{(1,no-swap), (16,no-swap), (1,swap)}`` — never ``(16, swap)``. The tuner picks."""
    pairs = [(1, False), (16, False)]
    if swap_ab:
        pairs.append((1, True))
    return pairs


def get_accelerator_autotuning_configs(
    *,
    tune_block_m: bool = False,
    tune_block_nk: bool = False,
    for_decode: bool = False,
    swap_ab: bool = False,
):
    """Autotune search grid for the current accelerator.

    ``num_warps``, ``num_stages`` and ``blocks`` (the ``(BLOCK_SIZE_N, BLOCK_SIZE_K)``
    tile shapes) are fixed up front from ``(is_xpu, tune_block_nk)``, then crossed
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
            [(128, 128)] if is_xpu else [(128, 128), (256, 128), (128, 64), (64, 256)]
        )
        blocks = [{"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk} for bn, bk in tiles]
    else:
        blocks = [{}]

    if tune_block_m:
        blocks = [{**b, "BLOCK_SIZE_M": bm} for b in blocks for bm in (16, 32, 64, 128)]

    # (BLOCK_SIZE_M, SWAP_AB) axis: for_decode crosses in the coupled decode pairs (see
    # ``decode_bm_swap_pairs``); else SWAP_AB is a plain axis (BM from the launch), if swap_ab.
    if for_decode:
        pairs = decode_bm_swap_pairs(swap_ab)
        blocks = [
            {**b, "BLOCK_SIZE_M": bm, **({"SWAP_AB": sw} if swap_ab else {})}
            for b in blocks
            for bm, sw in pairs
        ]
    elif swap_ab:
        blocks = [{**b, "SWAP_AB": sw} for b in blocks for sw in (False, True)]

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
    for_decode=False,
    swap_ab=False,
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
    # (BLOCK_SIZE_M, SWAP_AB) axis, orthogonal to COMPUTE_MODE (swap is a low-M GEMV win that fades
    # once the grid saturates):
    #   - for_decode (M=1 batched/fused): the coupled decode pairs (see ``decode_bm_swap_pairs``).
    #   - tune_block_m (grouped/prefill): BM ∈ {16,32,64,128}, crossed independently with swap.
    #   - neither: BM from the launch; SWAP_AB a plain axis if swap_ab.
    if for_decode:
        bm_sw = decode_bm_swap_pairs(swap_ab)
    else:
        bms = (16, 32, 64, 128) if tune_block_m else (None,)
        sws = (False, True) if swap_ab else (False,)
        bm_sw = [(bm, sw) for bm in bms for sw in sws]
    return [
        triton.Config(
            {
                "COMPUTE_MODE": mode,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                **({"MEMORY_MODE": mm} if mm is not None else {}),
                **({"BLOCK_SIZE_M": bm} if bm is not None else {}),
                **({"SWAP_AB": sw} if swap_ab else {}),
            },
            num_warps=w,
            num_stages=s,
            pre_hook=pre_hook,
        )
        for mm in memory_modes
        for mode in compute_modes
        for bn in [32, 64, 128, 256]
        # only "dot" pins BK to the group (one scale group per fp8 tl.dot step); "dot_scaled"
        # and "scalar" handle multiple groups per K-chunk, so they use a wide BK. Decode adds
        # BK=512: at its tiny grids (~256 blocks, memory-parallelism-bound) longer per-iteration
        # bursts are the one in-block lever that helps (+12% measured on the dsv4 swap GEMV; deeper
        # num_stages and split-K both measured WORSE — stages waste smem, split-K shortens loops).
        for bk in (
            [32] if mode == "dot" else [128, 256, 512] if for_decode else [128, 256]
        )
        for s in [2, 3, 4, 5, 6]
        for w in [2, 4, 8]
        for (bm, sw) in bm_sw
        # scalar is a true M=1 reduce — the BM=16 fill-atom is only for the tensor-core modes.
        if not (mode == "scalar" and bm == 16)
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


def bk_within_k_pruner(k_arg: str):
    """Build an ``early_config_prune`` that drops configs with ``BLOCK_SIZE_K`` larger than the
    launch's contraction dim (``k_arg`` names it: ``"K"`` for the batched matmul, ``HIDDEN_DIM`` /
    ``INTERMEDIATE_DIM`` for the fused gate_up / down). The decode kernels' K-loop loads are
    unmasked (K is always a multiple of BK on real models), so an oversized BK reads past the row —
    silently wrong results the tuner would happily time and pick (bit us when BK=512 met a K=256
    test problem). Never returns empty — falls back to the smallest-BK configs."""

    def prune(configs, named_args, **kwargs):
        k = {**named_args, **kwargs}[k_arg]
        kept = [c for c in configs if c.kwargs.get("BLOCK_SIZE_K", 0) <= k]
        if kept:
            return kept
        min_bk = min(c.kwargs.get("BLOCK_SIZE_K", 0) for c in configs)
        return [c for c in configs if c.kwargs.get("BLOCK_SIZE_K", 0) == min_bk]

    return prune


def smem_config_pruner(
    act_bytes: int,
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
    (gate_up loads 2; down has 1), times ``num_stages``. ``BM`` is the routing-derived tile,
    read from the launch args. MXFP4 packs 2 weights/byte, so ``weight_bytes=1`` (MXFP8) is a
    safe upper bound. The limit is read from the active device. Never returns empty — keeps
    the smallest-footprint config as a fallback.

    Two sm_10x (Blackwell-datacenter, TMEM scaled-MMA) ``dot_scaled`` compiler-bug guards, both
    no-ops off sm_10x:

    ``reduction_dim`` (the K-loop bound arg name, e.g. ``"INTERMEDIATE_DIM"``) drops ``dot_scaled``
    configs with ``BLOCK_SIZE_K >= reduction_dim`` — a single-trip K-loop, which trips a Triton
    ``optimize-accumulator-init`` bug (uninitialized TMEM alloc must be mutable) and never
    compiles. ``None`` (default) disables the check.

    ``double_mma`` marks a kernel that folds its ``n_weight_tiles`` into ONE wide MMA rather than
    dotting them separately — the fused gate∪up kernel does a single ``[BM, n_weight_tiles*BN]``
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
        dev = triton.runtime.driver.active.get_active_torch_device()
        dev_index = dev.index if dev.index is not None else 0
        limit = sm_shared_memory_limit(dev_index)
        # The dot_scaled guards below are Blackwell-datacenter (sm_10x, TMEM scaled-MMA) compiler
        # bugs — off on every other target, so they never over-prune there.
        is_sm10x = (
            dev.type == "cuda" and torch.cuda.get_device_capability(dev_index)[0] == 10
        )

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
            ``[BM, BK]`` activation tile plus ``n_weight_tiles`` ``[BN, BK]`` weight tiles."""
            BM, BN, BK = (
                dim(c, n) for n in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
            )
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
                is_sm10x
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
                is_sm10x
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
    decode GEMV (M=1). Single weight — for the gate∪up pair use ``mx_scalar_reduce_gate_up``.

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
    accumulate into ``acc_gate``/``acc_up`` (returned updated). Per-group scale factored out of
    the inner sum (see ``mx_scalar_reduce``): reduce raw products within each group of 32, then
    one combined (act × weight) scale per group — 32× fewer scale-muls, bit-identical."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    a_t = tl.trans(a.to(tl.float32))  # [BK, BM]
    a_s = tl.trans(decode_ue8m0_scale(a_scale))  # [NG, BM]
    grp_gate = tl.sum(
        tl.reshape(a_t * w_gate.to(tl.float32), (NG, SCALE_GROUP_K, ROWS_W)), axis=1
    )
    acc_gate += tl.sum(
        grp_gate * a_s * tl.trans(decode_ue8m0_scale(gate_scale)), axis=0
    )[None, :]
    grp_up = tl.sum(
        tl.reshape(a_t * w_up.to(tl.float32), (NG, SCALE_GROUP_K, ROWS_W)), axis=1
    )
    acc_up += tl.sum(grp_up * a_s * tl.trans(decode_ue8m0_scale(up_scale)), axis=0)[
        None, :
    ]
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
    SWAP_AB: tl.constexpr,
):
    """Gate∪up MMA step. Under ``SWAP_AB`` the swapped decode path runs (weight output rows in the MMA
    M dim — see ``mx_swap_compute_gate_up``); otherwise dispatch on ``COMPUTE_MODE``: scaled-MMA on the
    raw weights (``b_gate``/``b_up``), or fp8 ``tl.dot`` + per-group rescale / scalar reduce on the
    E4M3-decoded weights. Returns the updated ``(acc_gate, acc_up)`` — only the taken branch compiles."""
    if SWAP_AB:
        acc_gate, acc_up = mx_swap_compute_gate_up(
            acc_gate, acc_up, a, a_scale, b_gate, b_up, gate_scale, up_scale,
            COMPUTE_MODE, VALUES_PER_BYTE, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K,
        )
    elif COMPUTE_MODE == "dot_scaled":
        acc_gate = mx_dot_scaled(
            acc_gate, a, a_scale, b_gate, gate_scale, VALUES_PER_BYTE
        )
        acc_up = mx_dot_scaled(acc_up, a, a_scale, b_up, up_scale, VALUES_PER_BYTE)
    elif COMPUTE_MODE == "dot":
        acc_gate, acc_up = mx_dot_rescale_gate_up(
            acc_gate, acc_up, a,
            mxfp4_e2m1_to_e4m3(b_gate) if VALUES_PER_BYTE == 2 else b_gate,
            mxfp4_e2m1_to_e4m3(b_up) if VALUES_PER_BYTE == 2 else b_up,
            a_scale, gate_scale, up_scale,
        )
    else:  # scalar
        acc_gate, acc_up = mx_scalar_reduce_gate_up(
            acc_gate,
            acc_up,
            a,
            mxfp4_e2m1_to_e4m3(b_gate) if VALUES_PER_BYTE == 2 else b_gate,
            mxfp4_e2m1_to_e4m3(b_up) if VALUES_PER_BYTE == 2 else b_up,
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
    SWAP_AB: tl.constexpr,
):
    """Single-projection MMA step. Under ``SWAP_AB`` the swapped decode path runs (weight output rows
    in the MMA M dim — different acc shape/finalize; see ``mx_swap_compute``); otherwise dispatch on
    ``COMPUTE_MODE``: scaled-MMA on the raw weight (``w``), or fp8 ``tl.dot`` + per-group rescale /
    scalar reduce on the E4M3-decoded weight. Single return — only the taken branch compiles."""
    if SWAP_AB:
        acc = mx_swap_compute(
            acc, a, a_scale, w, w_scale, COMPUTE_MODE, VALUES_PER_BYTE,
            BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K,
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
    ``[BK, MMA_N_ATOM]`` tile with only column 0 real (the tensor core can't do N<16), and its
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
def mx_dot_scaled_swapped_gate_up(
    acc_gate,
    acc_up,
    a,
    a_scale,
    w_gate,
    w_up,
    gate_scale,
    up_scale,
    VALUES_PER_BYTE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Gate∪up swapped ``dot_scaled``: build the shared activation rhs once (N=16, col 0 real) and
    run both projections with the weights (output rows) in the MMA M dim. ``acc_gate``/``acc_up``
    are persistent ``[BN, MMA_N_ATOM]`` MMA accumulators (col 0 taken by the caller)."""
    fmt: tl.constexpr = "e2m1" if VALUES_PER_BYTE == 2 else "e4m3"
    rhs, asc = mx_dot_scaled_swapped_rhs(a, a_scale, BLOCK_SIZE_K)
    acc_gate = tl.dot_scaled(w_gate, gate_scale, fmt, rhs, asc, "e4m3", acc_gate)
    acc_up = tl.dot_scaled(w_up, up_scale, fmt, rhs, asc, "e4m3", acc_up)
    return acc_gate, acc_up


@triton.jit
def mx_scalar_reduce_swapped_gate_up(
    acc_gate,
    acc_up,
    a,
    a_scale,
    w_gate,
    w_up,
    gate_scale,
    up_scale,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
):
    """Gate∪up swapped scalar reduce (see ``mx_scalar_reduce_swapped``): decode the SHARED
    activation once, reduce both projections over K with per-group scale. ``acc*`` are ``[1, ROWS_W]``."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    if VALUES_PER_BYTE == 2:
        wg = tl.interleave(
            _e2m1_code_to_f32(w_gate & 0xF), _e2m1_code_to_f32(w_gate >> 4)
        )
        wu = tl.interleave(_e2m1_code_to_f32(w_up & 0xF), _e2m1_code_to_f32(w_up >> 4))
    else:
        wg = w_gate.to(tl.float32)
        wu = w_up.to(tl.float32)
    af = a.to(tl.float32)[None, :]
    a_s = decode_ue8m0_scale(a_scale)[None, :]
    grp_g = tl.sum(tl.reshape(af * wg, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)
    grp_u = tl.sum(tl.reshape(af * wu, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)
    acc_gate += tl.reshape(
        tl.sum(grp_g * a_s * decode_ue8m0_scale(gate_scale), axis=1), (1, ROWS_W)
    )
    acc_up += tl.reshape(
        tl.sum(grp_u * a_s * decode_ue8m0_scale(up_scale), axis=1), (1, ROWS_W)
    )
    return acc_gate, acc_up


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
    decode token flattened to the [BK] rhs. Dispatches the two swapped modes — ``dot_scaled``
    (persistent ``[BN, MMA_N_ATOM]`` MMA acc, col 0 taken by the caller) and ``scalar`` (``[1, BN]``
    reduce). The acc shapes diverge, but only the taken constexpr branch compiles so the single
    return never has to unify them. Swap has no ``dot`` path (dead at M=1 decode)."""
    a1 = tl.reshape(a, (BLOCK_SIZE_K,))
    as1 = tl.reshape(a_scale, (BLOCK_SIZE_K // SCALE_GROUP_K,))
    if COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled_swapped(acc, a1, as1, w, w_scale, VALUES_PER_BYTE, BLOCK_SIZE_K)
    elif COMPUTE_MODE == "scalar":
        acc = mx_scalar_reduce_swapped(
            acc, a1, as1, w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, VALUES_PER_BYTE
        )
    else:
        tl.static_assert(False, "SWAP_AB supports only dot_scaled/scalar")
    return acc


@triton.jit
def mx_swap_compute_gate_up(
    acc_gate,
    acc_up,
    a,
    a_scale,
    w_gate,
    w_up,
    gate_scale,
    up_scale,
    COMPUTE_MODE: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Gate∪up swapped-AB counterpart to ``mx_compute_gate_up`` (see ``mx_swap_compute``): the shared
    activation is flattened once, then both projections run with weight rows in the MMA M dim."""
    a1 = tl.reshape(a, (BLOCK_SIZE_K,))
    as1 = tl.reshape(a_scale, (BLOCK_SIZE_K // SCALE_GROUP_K,))
    if COMPUTE_MODE == "dot_scaled":
        acc_gate, acc_up = mx_dot_scaled_swapped_gate_up(
            acc_gate, acc_up, a1, as1, w_gate, w_up, gate_scale, up_scale,
            VALUES_PER_BYTE, BLOCK_SIZE_K,
        )
    elif COMPUTE_MODE == "scalar":
        acc_gate, acc_up = mx_scalar_reduce_swapped_gate_up(
            acc_gate, acc_up, a1, as1, w_gate, w_up, gate_scale, up_scale,
            BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, VALUES_PER_BYTE,
        )
    else:
        tl.static_assert(False, "SWAP_AB supports only dot_scaled/scalar")
    return acc_gate, acc_up


@triton.jit
def swap_pad_rhs(a, BLOCK_SIZE_K: tl.constexpr):
    """Pad the ``[BLOCK_SIZE_K]`` M=1 token to the ``[BLOCK_SIZE_K, MMA_N_ATOM]`` swap-AB MMA rhs —
    only column 0 is the real token (the tensor core can't do N<16, so the pad is the N-atom). Used
    by the M=1 batched / fused-MoE fp8 ``tl.dot`` swap paths (weight output rows in the MMA M dim);
    the caller takes column 0 of the ``[BN, MMA_N_ATOM]`` result after the K-loop."""
    return tl.where(
        tl.arange(0, MMA_N_ATOM)[None, :] == 0,
        a[:, None],
        tl.zeros((BLOCK_SIZE_K, MMA_N_ATOM), a.dtype),
    )


@triton.jit
def swap_take_col0(acc, ROWS: tl.constexpr):
    """Extract column 0 of a ``[ROWS, MMA_N_ATOM]`` swap-AB accumulator → ``[1, ROWS]`` (the padded
    token dim collapses back to the single real token)."""
    return tl.reshape(
        tl.sum(acc * (tl.arange(0, MMA_N_ATOM)[None, :] == 0), axis=1), (1, ROWS)
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
def oriented_weight_ptrs(base, offs_rows, offs_k, stride_rows, stride_k, SWAP_AB: tl.constexpr):
    """Weight-tile pointers oriented by ``SWAP_AB``: output-rows-major ``[rows, K]`` when swapped
    (output rows are the MMA M dim), else K-major ``[K, rows]``. Only the taken constexpr branch
    compiles, so the divergent shapes never meet. The per-step K-advance is identical for both
    layouts, so the caller advances the returned pointer the same way regardless of orientation."""
    if SWAP_AB:
        ptrs = base + (offs_rows[:, None] * stride_rows + offs_k[None, :] * stride_k)
    else:
        ptrs = base + (offs_k[:, None] * stride_k + offs_rows[None, :] * stride_rows)
    return ptrs


@triton.jit
def acc_init(
    IS_SCALAR: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """Zero accumulator shaped for the layout: swapped scalar reduces into ``[1, N]``; any other
    swapped mode keeps the persistent ``[N, MMA_N_ATOM]`` MMA acc (weight rows in M, token in the
    padded N, col 0 taken after the K-loop); no-swap uses ``[M, N]``. ``IS_SCALAR`` matters only under
    swap — kernels with no scalar path (fp8 ``tl.dot``) pass ``False``. ``N`` is the weight-output
    tile (``BLOCK_SIZE_H`` for the fp8 down projection). Single return: only the taken branch compiles."""
    if SWAP_AB and IS_SCALAR:
        acc = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)
    elif SWAP_AB:
        acc = tl.zeros((BLOCK_SIZE_N, MMA_N_ATOM), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    return acc


@triton.jit
def acc_finalize(acc, IS_SCALAR: tl.constexpr, ROWS: tl.constexpr, SWAP_AB: tl.constexpr):
    """Bookend to ``acc_init``: when the acc was built as the persistent ``[ROWS, MMA_N_ATOM]`` MMA
    tile (any swapped non-scalar mode), collapse the padded token dim to column 0 → ``[1, ROWS]``.
    Swapped scalar (already ``[1, ROWS]``) and no-swap pass through unchanged. ``IS_SCALAR`` matches
    the flag given to ``acc_init`` (fp8 ``tl.dot`` kernels, which have no scalar path, pass False)."""
    if SWAP_AB and not IS_SCALAR:
        acc = swap_take_col0(acc, ROWS)
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
