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


import contextvars
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal


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

NVFP4_SCALE_GROUP_K = 16



# ── Host-side helpers ─────────────────────────────────────────────────────────


# set ONLY while an opaque op's registered fake impl runs (shape inference); read by
# compile_time_only_triton_wrap to no-op the kernel launches within
_SKIP_LAUNCHES = contextvars.ContextVar("finegrained_fp8_skip_launches", default=False)



def compile_time_only_triton_wrap(kernel):
    """``wrap_triton`` while torch.compile is tracing (required to capture a raw Triton
    launch into the graph), the bare kernel in eager. Every kernel launch goes through
    this: eager skips ``wrap_triton``'s per-call dispatch overhead, and it dodges a
    correctness trap — eager ``wrap_triton`` of a stock ``@triton.autotune`` kernel that
    has ``prune_configs_by`` re-runs the FULL tune on every call (torch's wrapper rebuilds
    a fresh autotuner around the pruned configs per call, ``triton_kernel_wrap.py``:
    ``autotune(configs=pruned, key=[])``), so nothing ever lands in the original
    ``Autotuner.cache`` (~2.3s/call for ``_mx_act_quant_kernel``) and the mid-capture
    tune invalidates cudagraph capture. Pruner-less kernels and the ``bayesian_autotune``
    kernels reuse their cache through the wrapper (measured) — but eager never needs the
    wrapper at all.

    Inside an OPAQUE op's fake impl (see ``compile_time_only_triton_op``) the launch is
    skipped outright: the op function has already allocated the correctly-shaped
    outputs, which is all fake mode needs. The skip is an EXPLICIT contextvar, never
    ambient fake-mode detection — ``triton_op``'s capture also runs under FakeTensor
    mode, and skipping there would compile graphs with the kernel MISSING (uninitialized
    outputs; found as identical garbage across recipes, 2026-07-16). The skip is also
    load-bearing in the other direction — probed: without it the opaque ops' fake impls
    reach the launcher with FakeTensors ("RuntimeError when making fake tensor call";
    ``is_compiling`` is still true during fake prop, so they'd take the wrap_triton
    branch and hit the pre_hook assert that made them opaque in the first place)."""
    if _SKIP_LAUNCHES.get():

        class _NoLaunch:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: None

        return _NoLaunch()
    return wrap_triton(kernel) if torch.compiler.is_compiling() else kernel



def compile_time_only_triton_op(name, mutates_args=(), opaque=False):
    """``@triton_op`` under torch.compile (the registered custom op is what dynamo
    captures), the plain function in eager: the torch.library op dispatch stack costs
    ~160µs per eager call — the dominant decode CPU cost in the eager-breakdown probe —
    and eager needs none of it. The op is still registered at import time, so compiled
    callers and ``torch.ops`` introspection see it unchanged. Sibling of
    ``compile_time_only_triton_wrap`` (same rule one level down, at the kernel launch)."""

    def decorator(fn):
        if opaque:
            # kernels whose configs carry a pre_hook (TMA descriptor rebinders) cannot
            # be traced by triton_op's capture; register OPAQUE instead — the fake impl
            # is the op fn itself, whose launches no-op under FakeTensor mode (see
            # compile_time_only_triton_wrap), leaving just the shape-correct allocations
            op = torch.library.custom_op(name, mutates_args=mutates_args)(fn)

            @functools.wraps(fn)
            def fake_impl(*args, **kwargs):
                token = _SKIP_LAUNCHES.set(True)
                try:
                    return fn(*args, **kwargs)
                finally:
                    _SKIP_LAUNCHES.reset(token)

            op.register_fake(fake_impl)
        else:
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



def weighted_reduce(
    rows: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Routing-weighted top-k reduce — the bookend of the fused-MoE chain. Folds each token's
    ``num_top_k`` expert-output rows (``rows``, group-major, scaled by ``top_k_weights``, with
    EP-sentinel rows ``id >= num_experts`` skipped) from the routed-row layout back to
    ``(num_tokens, H)``. See ``weighted_reduce_kernel``."""
    num_tokens, num_top_k = top_k_index.shape
    H = rows.size(1)
    reduced = torch.empty(num_tokens, H, device=rows.device, dtype=rows.dtype)
    with device_context(rows.device):
        weighted_reduce_kernel[
            lambda meta: (num_tokens, triton.cdiv(H, meta["BLOCK_H"]))
        ](
            rows,
            reduced,
            top_k_index,
            top_k_weights,
            H,
            rows.stride(0),
            rows.stride(1),
            reduced.stride(0),
            reduced.stride(1),
            top_k_index.stride(0),
            top_k_index.stride(1),
            num_groups_bit_length=int(num_tokens).bit_length(),
            NUM_TOP_K=num_top_k,
            NUM_EXPERTS=num_experts,
            SIMULATE_UNFUSED=simulate_unfused,
        )
    return reduced



def tl_dtype(dtype: torch.dtype) -> tl.dtype:
    """The ``tl`` dtype matching a torch dtype (``torch.bfloat16`` → ``tl.bfloat16``) — the
    attribute names line up, so no table. For passing a tensor's dtype as a kernel constexpr
    when the kernel can't read it off a pointer argument."""
    return getattr(tl, str(dtype).removeprefix("torch."))



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
    """Resolve the generic ``"descriptor"`` B_MEMORY_MODE to the device's flavor: host-built
    (NVIDIA TMA) on CUDA, device-built in-kernel tensormap on XPU. Installs the XPU
    scratch allocator device-built tensormaps need."""
    descriptor = (
        "device_descriptor" if get_active_device_type() == "xpu" else "host_descriptor"
    )
    memory_modes = tuple(descriptor if m == "descriptor" else m for m in memory_modes)
    if "device_descriptor" in memory_modes:
        # device-side descriptors build in-kernel tensormaps that need a scratch
        # allocator — on whichever accelerator is active (XPU always; CUDA when a kernel
        # requests "device_descriptor" explicitly to skip the host-descriptor plumbing).
        device_type = get_active_device_type()
        triton.set_allocator(
            lambda size, alignment, stream: torch.empty(
                size, dtype=torch.int8, device=device_type
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
    a_memory_modes=None,
    b_memory_modes=None,
    pre_hook=None,
):
    """Autotune search grid for the current accelerator — the single generator for every
    GEMM kernel. One union span serves both families: num_warps {2,4,8,16} x num_stages
    {2..6} (XPU: warps {8,16}); the tile comes from the caller's ``block_size``, or
    ``tune_block_nk`` spans BN {32,64,128,256} x BK {64,128,256}. ``mx=True`` (group-32
    scale formats, MXFP4/MXFP8) requires ``compute_modes`` (BK is coupled per mode) and
    ``tune_block_nk`` (MX kernels have no caller block_size — the tile is always tuned).

    ``compute_modes`` emits the ``COMPUTE_MODE`` axis (``None`` — the default — emits no
    axis, like ``b_memory_modes``), tuner-selected per workload (token count is in the
    key): ``"dot_scaled"`` native group-32 scaled MMA (wide BK {128,256}; wins once the
    grid saturates, ~S>=32), ``"dot"`` fp8 ``tl.dot`` + per-group software rescale
    (under ``mx`` its BK is pinned to the 32-group), ``"scalar"`` CUDA-core FMA reduce
    (no MMA M->16 pad; wins memory-bound decode, only emit where implemented — it is
    pinned to BM=1 here because a bigger-BM scalar config at BN == BM COMPILES and
    silently computes garbage the tuner would time and pick).

    Axes, crossed per flag:

    - ``tune_block_m``: BLOCK_SIZE_M in {16,32,64,128} instead of a fixed
      launch-time choice.
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
    - ``a_memory_modes``: the A_MEMORY_MODE activation-load axis (descriptor legal only
      without a gather — the tile's rows are the contiguous sorted positions; the pruner
      fences descriptor rows when GatherIdx is passed).
    - ``b_memory_modes``: the B_MEMORY_MODE weight-load axis — pass
      ``("descriptor", "pointer")`` to let the tuner pick the device's tensor-descriptor
      flavor vs explicit pointers; ``None`` (default) emits no axis. In the plain-dot
      kernels the descriptor arm is the SWAPPED loop (see the bd 2D kernel).

    A given ``pre_hook`` is attached to every config and must self-guard on
    ``B_MEMORY_MODE``.

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
    bk_span = (128,) if is_xpu else (64, 128, 256, 512)

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
    if a_memory_modes is not None:
        blocks = [
            {**b, "A_MEMORY_MODE": mm}
            for b in blocks
            for mm in resolve_memory_modes(a_memory_modes)
        ]
    if b_memory_modes is not None:
        blocks = [
            {**b, "B_MEMORY_MODE": mm}
            for b in blocks
            for mm in resolve_memory_modes(b_memory_modes)
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
    the target the ``dot_scaled`` compiler-bug guards are scoped to. False off-CUDA and on
    driverless build boxes (``torch.cuda.is_available()`` short-circuits before the
    capability query, which would otherwise raise); cached (pruners call this per tune, and
    a process is pinned to one device)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10



@functools.lru_cache(maxsize=None)
def sm_shared_memory_limit() -> int:
    """Max dynamic shared memory per block (bytes) for the CURRENT device — the cap
    Triton reports as the 'Hardware limit' on an ``out of resource: shared memory``
    failure (~232 KB on B200, ~227 KB on H100, much less on older/consumer parts).
    Queried from the driver so the prune adapts to the hardware instead of hardcoding
    one GPU; cached (a process is pinned to one device)."""
    dev = get_active_device_type()
    device_index = (
        torch.cuda.current_device()
        if dev == "cuda"
        else torch.xpu.current_device()
        if dev == "xpu"
        else 0
    )
    try:
        return triton.runtime.driver.active.utils.get_device_properties(device_index)[
            "max_shared_mem"
        ]
    except Exception:
        if dev == "cuda":
            return torch.cuda.get_device_properties(
                device_index
            ).shared_memory_per_block_optin
        elif dev == "xpu":
            return torch.xpu.get_device_properties(
                device_index
            ).shared_memory_per_block_optin
        else:
            raise RuntimeError(
                f"Unsupported device type {dev} for sm_shared_memory_limit; only cuda/xpu are supported."
            )
