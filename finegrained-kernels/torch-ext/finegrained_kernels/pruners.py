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


from .compat import *  # noqa: F401,F403
from .recipes import *  # noqa: F401,F403
from .swizzle import *  # noqa: F401,F403
from .tile_layout import *  # noqa: F401,F403
from .quant import *  # noqa: F401,F403
from .scales import *  # noqa: F401,F403
from .mma import *  # noqa: F401,F403
from .scheduling import *  # noqa: F401,F403
from .tiles import *  # noqa: F401,F403
from .epilogue import *  # noqa: F401,F403



# ── config pruners ────────────────────────────────────────────────────────────
# Every guard exists for one of four reasons; the map (pruner -> rule -> attached to):
#
#   SILENTLY-WRONG configs (must be pruned, the tuner would time and pick them):
#     block_within_dim_pruner   BK not dividing K over-reads rows (maskless K-loops) —
#                               tensor/weight-only/full-precision 2D+batched+grouped, and the
#                               first stage of mx_config_pruner, whose BN|N stage is scoped
#                               to the mx-dynamic AFFINE arm (its grouped scale gather has no
#                               wrap; swizzled masks its N tail). The weight-only batched and
#                               grouped kernels carry their own ungated BN|N stages.
#     scalar "scalar" -> BM=1   the scalar GEVM broadcasts wrong at BM>1 (config builder)
#     descriptor_needs_prequant_pruner  raw-A inline-quant arm reads a_ptrs, dead under a
#                               descriptor mode — static 2D
#     gate_pointer_only_pruner  one contiguous descriptor box can't span the gate|up rows
#                               N apart — 2D kernels (bd/static/mx/weight-only), GATE-gated
#     mma_trap_ok               swapped dot_scaled BK<128 (sticky 716), single-trip
#                               accumulator-init miscompile, width > 256 (sticky trap),
#                               packed-fp4 GATE BN<64 silent zeros — mx 2D/batched/grouped;
#                               raises when empty (never falls back to a miscompile)
#     block_dynamic_mma_width_pruner  the same width trap on the bd implicit dot_scaled
#                               arm (GATE 2*BN) — bd 2D only (grouped/batched bd pin their
#                               N tile to the quant block, within the cap); UE8M0-As-gated
#     scale_subblock_pruner (whole-block clause)  a tile wider than the quant block reads
#                               one scale across two blocks; raises when the grid has no
#                               fitting tile — bd/static batched
#   COMPILER BUGS / unsupported combos on this triton+arch (benign inf, pruned to save
#   compiles and to keep can't-win configs from poisoning the TPE densities):
#     warp_spec_compile_guard_pruner   WS compiles iff (BN if swapped else BM)>=64 &
#                               num_warps%4==0 — bd/tensor/mx/weight-only/fp 2D+grouped
#     gated_pointer_weight_warp_spec_pruner  GATE+pointer-weight WS lowering failures —
#                               weight-only grouped
#     affine_scale_warp_spec_pruner    affine scale gather's other=0.0 can't take
#                               ttg.partition under WS — mx-dynamic grouped
#     nvfp4_native_ok (mx_config_pruner)  E4M3-scale dot_scaled below M=128 PassManager-fails
#     descriptor_box_pruner     TMA box > 256/dim is illegal — all descriptor-axis kernels
#     smem_pruner               empirical smem model (B200-sampled, keep-side error) —
#                               bd/static/mx/weight-only 2D; tensor/mx/weight-only/fp grouped;
#                               mx batched
#   RACE guards (Triton 3.7.1 pipeliner, per-loop-structure flake maps):
#     block_dynamic_grouped_matmul_pruner   WS-only at BM>=64, non-WS below (disjoint) —
#                               bd + static grouped (plain and GATE arms, same loop)
#   TPE-POISON fences (valid configs that can't win in a regime):
#     scalar_max_m_pruner       scalar above M=64 (prefill GEVM) — mx 2D
#     mx_config_pruner poison fences   sm_10x fp4-scalar (1.8x dead), the dot arm where the
#                               native MMA exists (A/B'd 2026-07-14; weight-only kernels
#                               keep dot), mma_poison_ok: no-swap BK<128 where 128 divides K
#                               (basin claim) + swapped rows<128 (bf16 fallback never wins)
#     scale_subblock_pruner     sub-quant-block BN only when the grid starves — bd/static
#                               batched
#     mx_2d_swap_scope_pruner   2D mx SWAP_AB rows only for E4M3-scale decode (BM=1,
#                               M<=16) — swap loses everywhere else (18-cell sweep)
#     packed_schedule_scope_pruner  PACKED_SCHEDULE rows only at BM>=128 (register relief
#                               is the whole win) — full-precision grouped
#     swizzled_scales_bm_pruner grouped mx swizzled arm pinned to BM=BN=128 (slab layout);
#                               affine arm keeps every BM/BN
#     swizzled_scale_config_pruner  swizzled-arm BK%128, BN<=128, GATE block pairs —
#                               mx 2D/batched
#     matched_memory_modes_pruner   mixed pointer/descriptor never wins a DENSE 2D GEMM —
#                               bd/static 2D (grouped keeps mixed: it wins there)
# Every pruner is a ``config_filter``: a per-config ``ok(config, args)`` predicate, an
# optional ``when(args)`` regime gate, and an empty-survivor policy (advisory guards fall
# back to the untouched grid; the BK-within-K correctness veto raises via ``on_empty`` —
# silence there means wrong results). Ordered below: shared infrastructure, then the four
# categories above.

# sm_10x caps a single scaled MMA (dot_scaled) at N=256; Triton miscompiles wider ones
# into a sticky illegal-address device trap. Plain-dot MMAs have NO such cap: a forced
# width-512 GATE sweep on the tensor grouped kernel (2026-07-14) ran bit-exact wherever
# it fit shared memory and failed only as benign launch-time smem overflows.
SM10X_SCALED_MMA_MAX_N = 256



def config_dim(c, all_args, name):
    """A tile dimension for config ``c`` — from its tuned meta, else the launch args.
    No default: a pruner reasoning about a dimension the kernel doesn't have is a wiring
    bug, and a made-up value would silently mis-prune."""
    v = c.kwargs.get(name, all_args.get(name))
    if v is None:
        raise ValueError(
            f"pruner needs {name} (autotune config meta or launch arg); none found"
        )
    return v



def config_filter(ok, when=None, on_empty=None):
    """Wrap a per-config predicate into an ``early_config_prune``: keeps the configs where
    ``ok(config, all_args)`` (``all_args`` = launch args + autotune kwargs). Every pruner
    below is one of these — the predicate is always named ``ok``, so the factories differ
    only in their rule, not their plumbing. ``when(all_args)`` gates the whole filter —
    hardware/launch-regime checks live there, per-config logic in ``ok``. When nothing
    survives, the untouched grid is returned (guards are advisory) unless ``on_empty(configs,
    all_args)`` is given — correctness vetoes pass one that raises, because falling back to
    configs that compute wrong results must be loud."""

    def prune(configs, named_args, **kwargs):
        all_args = {**named_args, **kwargs}
        if when is not None and not when(all_args):
            return configs
        kept = [c for c in configs if ok(c, all_args)]
        if kept:
            return kept
        return configs if on_empty is None else on_empty(configs, all_args)

    return prune



def compose_pruners(*pruners):
    """Chain ``early_config_prune`` callbacks left to right (each sees the previous
    survivors)."""

    def prune(configs, named_args, **kwargs):
        for p in pruners:
            configs = p(configs, named_args, **kwargs)
        return configs

    return prune



def block_within_dim_pruner(dim_arg: str, block_key: str = "BLOCK_SIZE_K", when=None):
    """``early_config_prune`` dropping configs whose ``block_key`` tile does not divide the
    launch dim named by ``dim_arg``: the K-loops load unmasked and the batched/grouped
    N-tiles store row-masked only, so a non-dividing tile's last trip reads or writes past
    the row — silently wrong results the tuner would happily time and pick (a BN=256
    winner at N=128 corrupted 40/64 rows before the N veto existed). A dim smaller than
    every grid tile is a hard error. ``when`` gates the veto to a launch regime (e.g. the
    grouped swizzled arm masks its N-tail — %-wrapped value load + ``N_COLS`` store mask — so
    BN needn't divide N there). Used standalone by the tensor-dynamic kernels and as the first
    stages of ``mx_config_pruner``."""

    def ok(c, args):
        block = c.kwargs.get(block_key, 0)
        return block == 0 or args[dim_arg] % block == 0

    def raise_no_dividing_block(configs, args):
        min_block = min(c.kwargs.get(block_key, 0) for c in configs)
        raise ValueError(
            f"{dim_arg}={args[dim_arg]} is not a multiple of any {block_key} in the "
            f"autotune grid; the unmasked tile would run past the row. Pad the problem "
            f"along {dim_arg} (smallest grid tile: {min_block})."
        )

    return config_filter(ok, when=when, on_empty=raise_no_dividing_block)



def require_moe_dims_aligned(N: int, K: int, block_n: int, block_k: int) -> None:
    """Routed MoE (grouped/batched) GEMMs tile expert dims by the quant block — N-tiles store
    column-unmasked and K loads unmasked — so ``N`` and ``K`` must each be a multiple of the block.
    Non-aligned dims are a dense capability: use ``matmul_2d`` (its single GEMM masks the N/K tail).
    Raised early with a clear message rather than letting a tile run past the row."""
    if N % block_n != 0 or K % block_k != 0:
        raise ValueError(
            f"routed MoE GEMM needs N ({N}) and K ({K}) each a multiple of the quant block "
            f"(block_n={block_n}, block_k={block_k}); non-aligned dims are supported only by the "
            f"dense matmul_2d op."
        )



def warp_spec_compile_guard_pruner():
    """``early_config_prune`` dropping ``warp_specialize`` configs that can never compile: WS needs
    ``num_warps % 4 == 0`` (its async producer/consumer partitions) and enough M work — the MMA's M
    operand, which ``SWAP_AB`` moves to ``BLOCK_SIZE_N`` (so BM=1 decode can WS when swapped), so
    ``(BN if swapped else BM) >= 64``. ``dot_scaled`` + WS + TMA + gate all compile and win here
    (measured 1995 TFLOPS on a clean context) — the earlier "dot_scaled/gate never WS-compiles"
    verdicts were a ``num_warps=2`` / trap-contaminated-GPU misdiagnosis. Non-WS configs all stay —
    WS is purely a perf axis. CUDA-only."""

    def ok(c, args):
        if not c.kwargs.get("WARP_SPEC"):
            return True
        # WS needs num_warps a multiple of 4 (its async producer/consumer partitions) and enough M
        # work — the MMA's M operand, which SWAP_AB moves to BLOCK_SIZE_N (so BM=1 decode can still
        # WS when swapped). dot_scaled + WS + TMA DOES compile here (measured 1995 on a clean
        # context) — the old "dot_scaled never WS-compiles" was a num_warps=2 misdiagnosis on a
        # trap-contaminated GPU. Callers needing dot_scaled+WS off (block-dynamic swap arm) fence it.
        mma_m = "BLOCK_SIZE_N" if c.kwargs.get("SWAP_AB") else "BLOCK_SIZE_M"
        return config_dim(c, args, mma_m) >= 64 and c.num_warps % 4 == 0

    return config_filter(ok, when=lambda args: get_active_device_type() == "cuda")



def gated_pointer_weight_warp_spec_pruner():
    """``early_config_prune`` dropping the ``warp_specialize`` configs that cannot lower on the
    weight-only grouped MX kernel's GATE arm, where the stacked gate|up tile doubles the weight
    extent to ``2 * BLOCK_SIZE_N``. With a pointer-mode weight (its masked load carries the
    ``other=0.0`` constant the WS partitioner cannot tag) the automatic-warp-specialization pass
    fails in ``TritonGPUOptimizePartitionWarps`` with ``RuntimeError: PassManager::run failed``,
    but only where the partitioner also has to split a descriptor-mode A, 16 warps, or a
    sub-128 M tile — plain pointer/pointer at BM=128 and w4/w8 compiles and is often the winner.

    Region measured by tuning the GPT-OSS W4A16 problem (N=K=2880) with a fresh Triton cache and
    ``FINEGRAINED_AUTOTUNE_LOG``: of 129 WS configs, the 12 that failed are exactly the ones this
    predicate names, and it drops none of the 117 that compiled. Every failure sat in the GATE
    shape — the ungated shape's 63 WS configs all compiled. Unfenced these are wasted compiles per
    new shape (the tuner memoizes them only after paying once). The ``BM < 128`` clause rests on a
    single observation, so it is the clause to revisit if a later probe contradicts it. CUDA-only;
    a no-op for ungated launches.

    NOT covered: the later ``dot_scaled`` arm adds 4 UNGATED failures (descriptor-A +
    pointer-B at num_warps>=8), outside this predicate's GATE scope. Left unfenced on
    purpose — the region separates only on a GATE x COMPUTE_MODE x A-mode x warps
    interaction, and every predicate wide enough to catch it also dropped working configs
    across the five probes. The tuner memoizes the 4 after paying once per shape, and this
    fence was re-checked against the two-arm grid: 0/207 working configs dropped."""

    def ok(c, args):
        if not c.kwargs.get("WARP_SPEC"):
            return True
        if c.kwargs.get("B_MEMORY_MODE", "pointer") != "pointer":
            return True
        return not (
            c.kwargs.get("A_MEMORY_MODE", "pointer") != "pointer"
            or c.num_warps >= 16
            or config_dim(c, args, "BLOCK_SIZE_M") < 128
        )

    return config_filter(
        ok, when=lambda args: args.get("GATE", False) and get_active_device_type() == "cuda"
    )



def affine_scale_warp_spec_pruner():
    """``early_config_prune`` dropping ``warp_specialize`` on the grouped MX AFFINE (row-major,
    non-``SWIZZLED_SCALES``) scale path. Its per-(gathered-row, K-group) 2D pointer-gather scale
    load carries an ``other=0.0`` constant the WS partitioner cannot tag — the automatic-warp-
    specialization pass fails with ``'arith.constant' op does not have expected attribute
    ttg.partition``. The swizzled arm (gate + non-gate) loads the scale through a TMA descriptor
    (no such constant) and WS-compiles + wins, so the fused / deployment path keeps WS. CUDA-only;
    a no-op where ``SWIZZLED_SCALES`` isn't a kernel arg (defaults to allowing WS)."""

    def ok(c, args):
        if not c.kwargs.get("WARP_SPEC"):
            return True
        return args.get("SWIZZLED_SCALES", True)

    return config_filter(ok, when=lambda args: get_active_device_type() == "cuda")



def mx_config_pruner(k_arg: str, n_arg: str | None = None):
    """``early_config_prune`` for the MX kernels (2D, batched, grouped — their tile is always
    tuned): a BK-within-K veto plus sm_10x MMA-shape guards (no-ops elsewhere and for scalar
    configs). GATE-aware: under the GATE constexpr (read from the launch args) the kernel
    computes gate|up as one stacked 2*BN extent, so the swapped dot_scaled lhs has ``2*BN``
    rows and the no-swap combined dot is ``2*BN`` wide; a plain launch's counts are just
    ``BN``.

    - ``BLOCK_SIZE_K`` not dividing the launch's contraction dim (``k_arg`` names it) →
      dropped: the K-loop loads are unmasked, so any non-dividing BK's last trip reads
      past the row — silently wrong results the tuner would happily time and pick (bit us when
      BK=512 met a K=256 test problem).
    - MXFP4 ``scalar`` configs → dropped (sm_10x only — the 1.8x-dead evidence is B200;
      other targets lower dot_scaled differently and keep the mode): fp4 scalar decode is
      ALU-bound in the E2M1 unpack and measured 1.8x SLOWER than dot_scaled (twice, incl.
      the no-pad form) — it never wins, and its swapped variants poison the TPE's
      per-dimension model into writing off SWAP_AB (a 100-trial dsv4 down tune benched 3
      swap configs — two dead-slow swap-scalar, one inf — and shipped a 38.9µs no-swap
      winner, missing the ~24µs swap dot_scaled basin entirely).
    - Swapped ``dot_scaled`` rows < 128 → dropped: the native microscaled MMA gates on the M
      operand being exactly 128, so smaller rows run the bf16-upcast fallback and never win —
      the same poison mechanism (an earlier dsv4 gate_up tune shipped 63µs missing the 43µs
      swap winner).
    - No-swap ``dot_scaled`` width > ``SM10X_SCALED_MMA_MAX_N`` → dropped: Triton
      miscompiles wider scaled MMAs into a sticky illegal-address device trap that poisons
      the context mid-autotune.

    Packed-E2M1 activations (a ``uint8`` activation tile, W4A4 — caller-packed or
    quantized inline from a raw tile) are consumed by EVERY arm: dot_scaled natively
    (the ``kind::mxf4`` MMA was probed bit-exact: BK down to 64, single-trip, and
    BN=256 all clean; BM<128 falls back to the correct-but-slow f16 kind), and the
    dot/scalar/swap arms column-unpack them to E4M3 (lossless) first — no arm is
    structurally packed-incompatible, so W4A4 needs no shape gate of its own.

    The shape gates above are scoped to ``dot_scaled`` — they are native scaled-MMA bug
    gates. The ``dot`` arm (BK structurally the UE8M0 group, 32) is CORRECT everywhere
    probed (forced-config sweep 2026-07-14, GATE and plain, MXFP4/MXFP8, incl. width-512
    stacked dots) but is fenced on sm_10x as TPE POISON: an A/B of fresh 100-trial tunes
    (mx grouped, S=4096 E=32 N=1024 K=2048 MXFP8) shipped dot winners 2-3x SLOWER with the
    dot rows in the grid (gate 0.297ms vs 0.095ms, plain 0.147ms vs 0.079ms) — the ~2x
    larger grid dilutes the trial budget and the early-sampled dot basin hijacks the TPE
    densities, so the native scaled-MMA winner is never found. dot never wins on sm_10x
    (native dot_scaled owns prefill, scalar owns M=1 decode); other targets keep the arm.
    Revisit if a dot win materializes (e.g. the batched decode BM=16 observation).

    The fence applies only where the kernel carries an ``As`` activation-scale operand, so
    the weight-only (W4A16/W8A16) kernels keep their dot rows while still getting the shape
    gates. Their ``dot`` is a weight upcast + plain ``tl.dot`` over a free BK — with raw
    activations there is no native microscaled MMA to lose to, so both arms upcast and this
    one WINS the stacked gate|up tile (gpt-oss K=2880: dot 4645.7us vs dot_scaled 5011.3us,
    while the down projection goes the other way, 2348.1us vs 2882.6us).

    The gates are two classes with opposite empty-survivor policies, composed trap-first:
    ``mma_trap_ok`` (miscompiles: swapped BK<128, single-trip, width, fp4 GATE zeros) raises
    when nothing survives — falling back to configs that compute wrong results must be loud —
    while the poison fences (dot/scalar arms, ``mma_poison_ok``) stay advisory, so emptying
    one of them re-admits CORRECT-but-slow configs, never a miscompile. A contraction dim
    smaller than every grid BK is a hard error: any config would over-read past the row and
    return silently wrong results."""

    def scaled_mma_available(args):
        # dot_scaled rows exist for this launch iff the grid's smallest BK (64)
        # divides the contraction dim with at least two trips; off the 64 grid no
        # row survives the shape gates, so the software arms are the only correct
        # ones and their poison fences lift.
        return args[k_arg] % 64 == 0 and args[k_arg] > 64

    def mma_trap_ok(c, args):
        # MISCOMPILE gates only — every rule here names a config that runs and returns
        # wrong results or traps the device; the composed filter raises when they empty.
        if c.kwargs.get("COMPUTE_MODE") != "dot_scaled":
            return True
        # Swapped dot_scaled below BK=128 is a Triton sm_10x lowering bug — a STICKY
        # misaligned-address device trap — on the UE8M0 (group-32) scale family. Probed
        # per-arm 2026-07-15 (fresh processes): batched W4A4 swap BK=64 traps at K=2880
        # AND K=2944 alike (K-alignment is NOT the variable), swap BK=128 runs clean and
        # correct, no-swap BK=64 is compute-sanitizer-clean, and the grouped kernel (no
        # swap axis) is clean at BK=64 in every memory mode. The E4M3 (NVFP4 group-16)
        # variant does NOT trap: 40 raw-grid sweep configs + 6 forced fresh-process runs
        # at K∈{256,2048,2944} all bit-correct (2026-08-05) — and swap BN=64 GATE is the
        # native M=128 staging for gate decode, so it stays admitted. All probing on
        # B200 — re-chart on H100 or the target device before trusting the fence there.
        if (
            c.kwargs.get("SWAP_AB")
            and config_dim(c, args, "BLOCK_SIZE_K") < 128
            and getattr(args.get("Bs"), "dtype", None) != torch.float8_e4m3fn
        ):
            return False
        # Single-trip dot_scaled (BK >= contraction dim) trips the sm_10x
        # accumulator-init miscompile (uninitialized TMEM alloc must be mutable).
        # Bites only small K (e.g. a K=512 gate_up with BK=512): silently wrong results
        # the tuner would happily time and pick (surfaced as a 35% MXFP4 fused-MoE error).
        if config_dim(c, args, "BLOCK_SIZE_K") >= args[k_arg]:
            return False
        # tcgen05 microscaled-fp4 MMA silently no-ops at the GATE stacked N=2*BN=64 shape (BN<64):
        # the IR emits a correct tc_gen5_mma_scaled (128x64 acc, 64x2 scale) but it never writes the
        # accumulator, which reads back its zero-init → EXACTLY-ZERO output. N-independent (probed
        # zero at both non-aligned N=320 and aligned N=512; BN=64 / stacked N=128 is correct
        # everywhere), so this is a LATENT poison config: any packed-fp4 gate|up whose tuner crowned
        # BN=32 (favoured at small N) would silently return zeros — larger shapes pass only because
        # BN>=64 wins. (BN=32 is recoverable only by emitting gate/up as two separate BN dots
        # instead of one stacked 2*BN dot; not worth it — a sub-64 gated tile rarely wins over BN=64.)
        if (
            args.get("GATE")
            and config_dim(c, args, "BLOCK_SIZE_N") < 64
            and getattr(args.get("B"), "dtype", None) == torch.uint8
        ):
            return False
        # no-swap width cap: Triton miscompiles a scaled MMA wider than 256 into a sticky
        # illegal-address device trap; the stacked 2*BN extent under GATE counts in full
        rows = (2 if args.get("GATE") else 1) * config_dim(c, args, "BLOCK_SIZE_N")
        return c.kwargs.get("SWAP_AB") or rows <= SM10X_SCALED_MMA_MAX_N

    def raise_all_mma_trapped(configs, args):
        raise ValueError(
            f"every autotune config hits an sm_10x scaled-MMA miscompile for "
            f"{k_arg}={args[k_arg]} (GATE={bool(args.get('GATE'))}); no correct config "
            f"exists in this grid — extend the grid (a no-swap BK strictly inside the "
            f"contraction dim) rather than lifting the fence: the pruned configs return "
            f"wrong results or trap the device."
        )

    def mma_poison_ok(c, args):
        # CORRECT-but-can't-win fences (advisory: emptying re-admits correct configs)
        if c.kwargs.get("COMPUTE_MODE") != "dot_scaled":
            return True
        # no-swap BK<128 rows are CORRECT but admitted only where BK=128 can't divide the
        # contraction dim (e.g. gpt-oss K=2880). "The BK>=128 basin owns the shape" is a
        # basin claim, not an A/B — the fence only trims trial-budget dilution, so lifting
        # it costs tune time, not results.
        if (
            not c.kwargs.get("SWAP_AB")
            and config_dim(c, args, "BLOCK_SIZE_K") < 128
            and args[k_arg] % 128 == 0
        ):
            return False
        # swapped rows < 128 run the bf16-upcast fallback (the native microscaled MMA
        # gates on the M operand being exactly 128) — correct, never wins, TPE poison
        rows = (2 if args.get("GATE") else 1) * config_dim(c, args, "BLOCK_SIZE_N")
        return not c.kwargs.get("SWAP_AB") or rows >= 128

    def fp4_scalar_ok(c, args):
        return c.kwargs.get("COMPUTE_MODE") != "scalar"

    def dot_arm_ok(c, args):
        return c.kwargs.get("COMPUTE_MODE") != "dot"

    def acts_are_scaled(args):
        # The dot-arm poison verdict holds only where dot_scaled can reach the NATIVE
        # microscaled MMA, which needs both operands microscaled — i.e. the kernel carries an
        # activation-scale operand. The weight-only (W4A16/W8A16) kernels have no ``As``: their
        # activations are raw bf16/fp16, so no native low-precision MMA exists, both arms upcast,
        # and their "dot" (weight upcast + plain tl.dot) genuinely competes. Keyed on the operand,
        # not A's dtype — the MX-dynamic inline-quant arm takes a raw bf16 A and still reaches the
        # native MMA.
        return "As" in args

    def nvfp4_native_ok(c, args):
        # mxf4nvf4 (E4M3 scales) has NO fallback below the native M=128 staging —
        # PassManager fails outright (charted BM {16,32,64} x BK {128,256}, 2026-07-15).
        # dot_scaled-only (dot/scalar decode E4M3 scales in software at any M); the MMA's
        # M operand is BLOCK_SIZE_M, or the weight rows when SWAP_AB puts the token in N.
        if c.kwargs.get("COMPUTE_MODE") != "dot_scaled":
            return True
        if c.kwargs.get("SWAP_AB"):
            return (2 if args.get("GATE") else 1) * config_dim(c, args, "BLOCK_SIZE_N") >= 128
        return config_dim(c, args, "BLOCK_SIZE_M") >= 128

    def scales_are_e4m3(args):
        return getattr(args.get("Bs"), "dtype", None) == torch.float8_e4m3fn

    stages = [block_within_dim_pruner(k_arg)]
    if n_arg is not None:
        # BN | N is needed only on the mx-dynamic AFFINE arm: its weight-scale gather has no
        # column wrap. The swizzled arm masks its N-tail (grouped/batched: %-wrapped value load
        # + N_COLS store mask; 2D: swizzle-offset wrap), and the weight-only kernels (no
        # SWIZZLED_SCALES axis) clamp their scale pointers in-bounds and mask the tail in the
        # epilogue (N_COLS), so both keep non-dividing BN.
        stages.append(
            block_within_dim_pruner(
                n_arg, "BLOCK_SIZE_N", when=lambda args: not args.get("SWIZZLED_SCALES", True)
            )
        )
    # trap gates run FIRST (on the full grid, raising when empty); the poison fences after
    # them are advisory, so an emptying poison fence falls back to trap-clean survivors —
    # never to a miscompile (the old single mixed filter fell back to the untouched grid,
    # re-admitting the single-trip/width traps it existed to remove).
    return compose_pruners(
        *stages,
        config_filter(nvfp4_native_ok, when=scales_are_e4m3),
        config_filter(
            mma_trap_ok, when=lambda args: is_sm10x(), on_empty=raise_all_mma_trapped
        ),
        config_filter(
            fp4_scalar_ok,
            when=lambda args: is_sm10x()
            and scaled_mma_available(args)
            and getattr(args.get("B"), "dtype", None) == torch.uint8
            and not scales_are_e4m3(args),
        ),
        config_filter(
            dot_arm_ok,
            when=lambda args: is_sm10x()
            and scaled_mma_available(args)
            and acts_are_scaled(args),
        ),
        config_filter(mma_poison_ok, when=lambda args: is_sm10x()),
    )



def block_dynamic_mma_width_pruner():
    """``early_config_prune`` for the block-dynamic kernels' implicit ``dot_scaled`` arm
    (UE8M0 scales on a native-M tile — ``USE_DOT_SCALED`` in the kernel): the sm_10x
    width cap ``mx_config_pruner``'s trap gate applies to the MX kernels, which the bd
    family never got. Under GATE the stacked gate|up dot is ``2 * BLOCK_SIZE_N`` wide, so
    a BN=256 GATE config emits a width-512 scaled MMA — Triton miscompiles those into a
    sticky illegal-address device trap the tuner trips mid-tune. Scoped to launches whose
    ``As`` is UE8M0 (uint8) on sm_10x; fp32-scale launches never emit the scaled MMA.
    Configs below the native-M tile keep the plain ``tl.dot`` + software rescale and pass
    untouched; the swapped N side is the token tile (within the cap on every grid)."""

    def ok(c, args):
        native_m = "BLOCK_SIZE_N" if c.kwargs.get("SWAP_AB") else "BLOCK_SIZE_M"
        if config_dim(c, args, native_m) < 128 or c.kwargs.get("SWAP_AB"):
            return True
        width = (2 if args.get("GATE") else 1) * config_dim(c, args, "BLOCK_SIZE_N")
        return width <= SM10X_SCALED_MMA_MAX_N

    def raise_all_too_wide(configs, args):
        raise ValueError(
            "every autotune config emits a scaled MMA wider than "
            f"{SM10X_SCALED_MMA_MAX_N} (GATE={bool(args.get('GATE'))}) — sm_10x "
            "miscompiles those into a sticky device trap; extend the grid below the cap."
        )

    return config_filter(
        ok,
        when=lambda args: is_sm10x()
        and getattr(args.get("As"), "dtype", None) == torch.uint8,
        on_empty=raise_all_too_wide,
    )



def scale_subblock_pruner():
    """``early_config_prune`` for kernels whose compute tile (``BLOCK_SIZE_N``) may subdivide the
    quant block (``BLOCK_N``). One scale covers the whole block, so a narrower tile just reads its
    own block — and it multiplies the N grid, the lever once the whole-block grid no longer fills
    the device (a gate|up fusion halves that grid again by folding both into one program).

    Sub-block tiles are admitted ONLY there. A grid that already fills the device gains nothing from
    a narrower tile, and carrying the extra tiles into every launch would spend one trial budget
    over a 3x config space — measurably worse winners on the launches that never needed it.

    A fused requant always pins the tile to the whole block: its output scale is one amax per
    (row, ``BLOCK_N``) group, which a partial tile can't compute. Raises when nothing
    survives: a tile wider than the quant block reads ONE scale across two blocks — silently
    wrong results — so a checkpoint whose ``BLOCK_N`` is off the tile grid must fail loudly,
    not fall back to the full grid."""

    def ok(c, args):
        block_n = args.get("BLOCK_N")
        if block_n is None:
            return True
        bn = config_dim(c, args, "BLOCK_SIZE_N")
        starved = args["S"] * triton.cdiv(args["N"], block_n) < sm_count(
            args["B"].device.index
        )
        if args.get("OUTPUT_RECIPE") is not None or not starved:
            return bn == block_n
        return bn <= block_n and args["N"] % bn == 0

    def raise_no_block_tile(configs, args):
        raise ValueError(
            f"no BLOCK_SIZE_N in the autotune grid fits the weight quant block "
            f"(BLOCK_N={args.get('BLOCK_N')}); a wider tile would read one scale across "
            f"two quant blocks and return wrong results — extend the grid or requantize "
            f"the checkpoint to a gridded block size."
        )

    return config_filter(ok, on_empty=raise_no_block_tile)



def packed_schedule_scope_pruner(min_bm: int = 128):
    """``early_config_prune`` scoping ``PACKED_SCHEDULE`` rows to big-BM configs — the only
    regime the packed resolve wins (register relief unlocking pressure-walled tiles: DSV3
    bf16 forced pairs +10.7% at BM128 vs −12.5% at BM64/BN256 and a wash at BM32,
    2026-08-06; both tuned crowns chose BM128). Everywhere else the inline resolve's free
    ALU beats the packed arm's per-tile dependent loads, so those rows are tune-budget
    dilution (the axis would double the grid; this keeps it +25%)."""

    def ok(c, args):
        if not c.kwargs.get("PACKED_SCHEDULE"):
            return True
        return config_dim(c, args, "BLOCK_SIZE_M") >= min_bm

    return config_filter(ok)


def mx_2d_swap_scope_pruner(max_m: int = 16):
    """``early_config_prune`` scoping the 2D mx kernel's ``SWAP_AB`` rows to the ONE regime
    they exist for: E4M3-scale (NVFP4) single-token decode. NVFP4 has no other native M=1
    arm — ``dot_scaled`` has no sub-M=128 form for E4M3 scales (PassManager) and scalar is
    E2M1-unpack-bound (measured 60-142µs vs swap's 22-50µs on the GLM attn shape,
    2026-08-05). Everywhere else swap is a proven loser (an 18-cell forced-swap sweep:
    UE8M0 recipes' scalar/dot arms win, M3 attn swap −38%), so its rows are dropped:
    UE8M0-scale launches, M above the decode band, and any non-BM=1 row (the mx swap
    compute flattens the token tile — BM=1 is structural, enforced by a static_assert)."""

    def ok(c, args):
        if not c.kwargs.get("SWAP_AB"):
            return True
        return (
            config_dim(c, args, "BLOCK_SIZE_M") == 1
            and getattr(args.get("Bs"), "dtype", None) == torch.float8_e4m3fn
            and args["M"] <= max_m
        )

    return config_filter(ok)



def swizzled_scales_bm_pruner():
    """``early_config_prune`` pinning the grouped MX pre-swizzled (``SWIZZLED_SCALES``) arm to
    ``BLOCK_SIZE_M == BLOCK_SIZE_N == 128``. BM: the offline act-quant lays each expert's scale
    slab out 128-padded, and the kernel's per-tile scale-block index (``pid_m``) only lines up
    when the M tile is exactly 128 (``build_tile_layout`` pads experts on the same granularity).
    BN: the scale is read as whole 128-row SWIZZLE_32_4_4 blocks off the descriptor (under GATE,
    the gate 128-block + the up 128-block stacked into the 2*BN tile); a sub-128 BN would need a
    partial-block read the descriptor can't express.

    Both pins are swizzled-slab properties, so the un-swizzled (affine) arm keeps every BM and
    BN: ``build_tile_layout`` tiles any BM with a row mask, and the affine act-scale read is
    indexed by the gathered row, not ``pid_m``."""

    def ok(c, args):
        return (
            config_dim(c, args, "BLOCK_SIZE_M") == 128
            and config_dim(c, args, "BLOCK_SIZE_N") == 128
        )

    return config_filter(ok, when=lambda args: args.get("SWIZZLED_SCALES"))



def swizzled_scale_config_pruner(allow_gate_subblock=False):
    """Drop pre-swizzled-scale configs the SWIZZLE_32_4_4 descriptor load can't serve, gated on
    ``SWIZZLED_SCALES`` (the un-swizzled arm loads scales per group and takes any tile):

    - ``BLOCK_SIZE_K < 4 * SCALE_GROUP_K``: the un-swizzle reads whole 4-group K bands
      (``REP_K = (BK // SCALE_GROUP_K) // 4``), so a smaller BK collapses ``REP_K`` to 0 and the
      reshape traps — 128 for group-32 (MX), 64 for group-16 (NVFP4). BK must also tile whole
      bands (``BK % (4 * group) == 0``).
    - ``BLOCK_SIZE_N > 128``: the descriptor's TMA box is created at one 128-row block; the load
      reads that block (BN=128) or a sub-tile of it (BN<128, scalar slice). A BN>128 tile would
      need a box grown past its creation shape, which the tensormap does not honor. Decode never
      wants BN>128 anyway (M=1 grid occupancy), so this costs no win.
    - under ``GATE``, ``BLOCK_SIZE_N != 128``: the gate|up scale is interleaved as whole 128-row
      block pairs [g0,u0,g1,u1,...], read as one 2*BN tile; a sub-128 BN can't index a block pair
      off the descriptor. ``allow_gate_subblock`` (batched decode) admits BN in (32, 64): its
      scale leaf pointer-GATHERs the sub-128 gate|up tile out of the interleaved buffer, which is
      the raw path's BN=32 gate|up tile without un-swizzling (the whole swizzled-vs-raw decode
      gap). Packed fp4 is admitted too: the historical sub-tile fp4->bf16 lowering trap
      (``xVals.size() % 4``) no longer reproduces — mxfp4 AND nvfp4 forced at BN∈{32,64},
      BK=128, scalar, K=2048 all run bit-correct (2026-08-05); a resurfacing trap is a loud
      compile failure (forgiven inf), never silent."""

    def ok(c, args):
        band = 4 * args["SCALE_GROUP_K"]  # one whole swizzled K band (4 scale groups)
        if config_dim(c, args, "BLOCK_SIZE_K") % band != 0:
            return False
        bn = config_dim(c, args, "BLOCK_SIZE_N")
        if args.get("GATE"):
            return bn == 128 or (allow_gate_subblock and bn in (32, 64))
        return bn <= 128

    def raise_no_swizzled_tile(configs, args):
        raise ValueError(
            "no autotune config can serve pre-swizzled scales for this launch (the "
            "SWIZZLE_32_4_4 read needs BLOCK_SIZE_K % 128 == 0 and a <=128-row N tile; "
            f"GATE={bool(args.get('GATE'))}) — the contraction dim likely has no "
            "128-dividing tile; pass affine (row-major) scales for this shape."
        )

    return config_filter(
        ok, when=lambda args: args.get("SWIZZLED_SCALES"), on_empty=raise_no_swizzled_tile
    )



def block_dynamic_grouped_matmul_pruner():
    """``early_config_prune`` for the block-dynamic grouped kernel: the Triton 3.7.1
    pipeliner-race guard, sized to its four-load-stream single-dot K-loop. The GATE arm
    shares the guard: post-activation-prequant it dots the stacked gate|up as the same
    single-dot loop (one wider weight tile, not a second dot/load stream) and was
    flake-verified race-free cross-process. Measured on sm_100, 15-fresh-process flake
    runs per cell, big and tiny shapes:

    - ``warp_specialize`` compiles iff ``num_warps % 4 == 0`` (its async partitions) and
      ``BLOCK_SIZE_M >= 64``, and is race-free everywhere it compiles.
    - The default pipeliner RACES at ``BLOCK_SIZE_M >= 64`` (3/15 wrong at BM64/w8) and is
      clean at BM16/32.

    The regions are disjoint, so per ``BLOCK_SIZE_M``: BM >= 64 keeps only the compilable
    WS configs, BM < 64 keeps only non-WS. CUDA-only — the race is a CUDA pipeliner
    artifact and the WS axis isn't emitted elsewhere.

    Also attached to the STATIC grouped kernel as an INHERITED guard: its three-load-stream
    K-loop (scalar act scale preloaded, no per-row ``as`` stream) was never flake-mapped —
    race soundness is per-loop-structure, so this map may over-fence it (perf cost) or, if
    the 3-stream loop races elsewhere, under-fence it. Re-map with multi-process flake runs
    before trusting the static attachment."""

    def ok(c, args):
        if config_dim(c, args, "BLOCK_SIZE_M") >= 64:
            return c.kwargs.get("WARP_SPEC") and c.num_warps % 4 == 0
        return not c.kwargs.get("WARP_SPEC")

    return config_filter(ok, when=lambda args: get_active_device_type() == "cuda")



def scalar_max_m_pruner(m_arg: str, max_m: int = 64):
    """``early_config_prune`` dropping ``scalar`` configs when the launch's row count
    (``m_arg``) exceeds ``max_m``: scalar is a BM=1 GEVM — sensible for decode-sized M,
    hopeless at prefill, and hopeless-but-benched configs poison the TPE's per-dimension
    densities (measured: with scalar in the M=8192 grid the 2D MX attn prefill tune
    landed 0.48x vs hub; without it, 2.06x)."""

    def ok(c, args):
        return c.kwargs.get("COMPUTE_MODE") != "scalar"

    return config_filter(ok, when=lambda args: args[m_arg] > max_m)



def matched_memory_modes_pruner():
    """Drop configs that mix pointer and descriptor operand loads — for a DENSE 2D GEMM
    only (contiguous A, whole M in one tile). Measured across M=1..8192 on the block-scale
    2D matmul: mixed only ever TIES the matched combos (<1us at small M) and loses at large
    M, so pruning it costs no win and halves the descriptor search. Do NOT use this on the
    GROUPED kernels: there M is distributed across experts (S/E small), A is token-GATHERED,
    and A=pointer + B=descriptor (mixed) is the dominant winner — those keep both axes free.
    Non-empty (the matched configs remain)."""

    def ok(c, args):
        a_ptr = c.kwargs.get("A_MEMORY_MODE", "pointer") == "pointer"
        b_ptr = c.kwargs.get("B_MEMORY_MODE", "pointer") == "pointer"
        return a_ptr == b_ptr

    return config_filter(ok)



def gate_pointer_only_pruner():
    """Keep only pointer-mode weight configs under gate|up fusion. The stacked gate|up weight has
    the gate rows at ``[0,N)`` and up at ``[N,2N)`` — N apart — which a single contiguous descriptor
    box can't span; the pointer arm (``weight_tile_ptrs``) folds the N-offset in. (A ``(2,N,K)`` box
    would let the descriptor arm span both — a perf follow-up.) Gated on the launch's ``GATE``."""

    def ok(c, args):
        return c.kwargs.get("B_MEMORY_MODE", "pointer") == "pointer"

    return config_filter(ok, when=lambda args: args.get("GATE", False))



def descriptor_needs_prequant_pruner():
    """Keep only pointer-mode configs when ``A`` is raw (bf16/fp16 — the ``maybe_act_quant``
    M<threshold inline-quant arm). A descriptor operand load needs a pre-quantized fp8 ``A``: the
    inline arm reads ``a_ptrs``, which ``operand_tile_ptrs`` leaves a dead placeholder under a
    descriptor mode, so a descriptor config there would compute garbage yet bench fast (a broken
    config the tuner would crown). Gated on ``A``'s dtype so the offline (fp8) path keeps both
    memory axes; a no-op for callers that always pre-quantize (``A`` already fp8)."""

    def ok(c, args):
        return (
            c.kwargs.get("A_MEMORY_MODE", "pointer") == "pointer"
            and c.kwargs.get("B_MEMORY_MODE", "pointer") == "pointer"
        )

    return config_filter(
        ok, when=lambda args: args["A"].dtype not in (torch.float8_e4m3fn, torch.int8, torch.uint8)
    )



def descriptor_box_pruner(k_dim="BLOCK_SIZE_K"):
    """Drop descriptor configs whose TMA box exceeds the tensormap's 256-per-dim limit:
    the weight inner dim is the K tile in BYTES over values-per-byte (packed E2M1
    halves it; E4M3 is one byte per value), so BK=512 e4m3 boxes are illegal while
    BK=512 packed ones fit. Pointer configs pass untouched. ``k_dim`` names the K-tile
    axis — ``BLOCK_SIZE_K`` for the tune_block_nk kernels, ``block_k`` for the block-scale
    kernels whose K tile is the caller's fixed scale block."""

    def ok(c, args):
        # tiles may be tuned (config kwargs) or pinned launch args — config_dim reads both
        bk = config_dim(c, args, k_dim)
        if c.kwargs.get("B_MEMORY_MODE", "pointer") != "pointer":
            weight_vpb = (
                2 if getattr(args.get("B"), "dtype", None) == torch.uint8 else 1
            )
            if bk // weight_vpb > 256 or config_dim(c, args, "BLOCK_SIZE_N") > 256:
                return False
        if c.kwargs.get("A_MEMORY_MODE", "pointer") != "pointer":
            # descriptor A, both arms tuner-routed per the tokens-per-expert key:
            # contiguous boxes (no gather) win at dense routing (-7.1/-7.4%) and lose at
            # skew; tma gather4 (gathered rows, 1-row box, sm_100 only) is MONOTONIC in
            # tokens/expert — 2x loss at 32/expert, parity at 1024, -5.3/-4.7% at 4096
            # (32k tokens, 2026-07-16). Only the arch is fenced.
            act_vpb = 2 if getattr(args.get("A"), "dtype", None) == torch.uint8 else 1
            if args.get("GatherIdx") is not None and not is_sm10x():
                return False
            if bk // act_vpb > 256:
                return False
        return True

    return config_filter(ok)



def smem_pruner(k_dim="BLOCK_SIZE_K"):
    """``early_config_prune`` dropping configs whose shared memory certainly cannot fit,
    with the bound picked by the operand dtypes (sampled from ``metadata.shared`` across
    every kernel family, 22 cells):

    - both operands >= 2-byte floats (the unquantized loops): ``num_stages`` full
      buffers of both tiles (exact) plus the fixed barrier terms at their sampled
      MINIMA (+16 base; +28 warp-spec, observed up to 48 varying with warps; +32
      host-descriptor) — minima keep the bound below the true usage, so a config is
      only ever dropped when its operand buffers alone cannot fit.
    - quantized ``dot_scaled`` (the only quantized arm whose tiles can reach the limit):
      the per-allocation law, every term named by its TTGIR ``local_alloc`` and exact to
      the allocator's alignment/packing slack (<= 16 B observed; barriers pack into
      padding, so no fixed constant is added — the slack is keep-side) —
      ``num_stages - 1`` buffers of operands+scales at the native M=128 staging (weights
      full-width even when fp4-packed; GATE shifts the A tile to a full ``num_stages``),
      or the bf16-upcast fallback below the gate (one single-buffered upcast tile,
      packed weights, no A-scale buffers).
    - quantized ``dot``/``scalar``: the raw-operand floor (their 32-wide K tiles cannot
      reach the limit).

    Both arms are EMPIRICAL models of this Triton's allocator, fit to and verified
    against ``metadata.shared`` samples (not derived from a spec) — chosen so the error
    direction under the sampled behavior is always toward KEEPING a config, which then
    merely compiles and self-prunes as inf. All samples taken on B200 (sm_100) —
    re-verify on a Triton bump AND on H100 or the target device (smem capacity and
    allocator behavior are per-arch). Non-empty fallback."""

    def ok(c, args):
        a, b = args["A"], args["B"]
        tiles = 2 if args.get("GATE") else 1
        bk = config_dim(c, args, k_dim)
        unquantized = a.element_size() >= 2 and b.element_size() >= 2
        stages = c.num_stages if unquantized else c.num_stages - 1
        packed = 2 if b.dtype == torch.uint8 else 1
        bm = config_dim(c, args, "BLOCK_SIZE_M")
        bn = config_dim(c, args, "BLOCK_SIZE_N")
        if unquantized:
            # sampled minima of the fixed allocations beside the operand buffers
            # (the barrier terms vary by tens of bytes with warps; minima only ever
            # err toward keeping a config)
            fixed = (
                16
                + (28 if c.kwargs.get("WARP_SPEC") else 0)
                + (32 if c.kwargs.get("B_MEMORY_MODE", "pointer") != "pointer" else 0)
            )
            need = fixed + stages * bk * (
                a.element_size() * bm + b.element_size() * tiles * bn
            )
        elif c.kwargs.get("COMPUTE_MODE") == "dot_scaled":
            # exact per-allocation law (every term named by its TTGIR local_alloc)
            scale_cols = bk // 32

            def scale_bufs(tile_bytes):
                # the pipeliner multi-buffers a load only when its tile is big enough
                # (sampled: 512 B pipelined, 256 B single; the open band is UNREACHABLE —
                # every emitted grid's scale tiles are powers of two)
                return stages if tile_bytes >= 512 else 1

            if bm >= 128:
                # native scaled-MMA staging: weights full-width even when fp4-packed;
                # GATE shifts the pipeliner so the A tile gets a full num_stages buffers
                a_bufs = c.num_stages if args.get("GATE") else c.num_stages - 1
                need = (
                    a_bufs * bk * bm
                    + stages * bk * tiles * bn
                    + scale_bufs(scale_cols * bm) * scale_cols * bm
                    + scale_bufs(scale_cols * tiles * bn) * scale_cols * tiles * bn
                )
            else:
                # bf16-upcast fallback (below the native M=128 gate): one single-buffered
                # [BK, tiles*BN] bf16 upcast tile, weights staged packed, no A-scale buffers
                need = (
                    2 * bk * tiles * bn
                    + stages * (bk * bm + (bk // packed) * tiles * bn)
                    + scale_bufs(scale_cols * tiles * bn) * scale_cols * tiles * bn
                )
        else:
            # dot (BK = the 32-group) / scalar: tiles too small to reach the limit —
            # the raw-operand floor suffices
            need = stages * (bk * bm + (bk // packed) * tiles * bn)
        return need <= sm_shared_memory_limit()

    return config_filter(ok, when=lambda args: get_active_device_type() == "cuda")

