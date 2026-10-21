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

from .compat import FP8_DTYPE, get_active_device_type, is_sm10x, is_sm90, sm_count, sm_shared_memory_limit, triton_version
from .mma import MMA_N_ATOM_WIDTH

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
#     weight_only_swap_scope_pruner  swapped dot is implemented for batched integer FP8;
#                               the other batched weight-only kernels swap only for scalar
#                               decode, which requires the weight-rows-major layout
#     mma_trap_ok               swapped dot_scaled BK<128 (sticky 716), single-trip
#                               accumulator-init miscompile, width > 256 (sticky trap),
#                               packed-fp4 GATE BN<64 silent zeros — mx 2D/batched/grouped;
#                               raises when empty (never falls back to a miscompile)
#     gate_stacked_tmem_trap_pruner  GATE 2*BN=512 acc overflows tensor memory undetected
#                               at exactly BN=256/BM=64 (checker gap -> sticky 716) —
#                               weight-only grouped; raises when empty
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
#     nvfp4_native_ok (mx_config_pruner)  E4M3-scale dot_scaled PassManager-fails off the
#                               native staging: M operand < 128, num_warps outside {4, 8},
#                               or a weight-only kernel (no As — never native)
#     descriptor_box_pruner     Descriptor strides and gather architecture — descriptor-axis kernels
#     smem_pruner               IR allocation bounds (B200-verified, keep-side error) —
#                               bd/static/mx/weight-only 2D; tensor/mx/weight-only/fp grouped;
#                               mx batched
#   RACE guards (Triton 3.7.1 pipeliner, per-loop-structure flake maps):
#     block_dynamic_grouped_matmul_pruner   WS-only at BM>=64, non-WS below (disjoint) —
#                               bd + static grouped (plain and GATE arms, same loop)
#     block_dynamic_2d_warp_spec_pruner  bd 2D WS compile-fail laws: w16+WS at any tile,
#                               and the implicit dot_scaled arm (UE8M0 As, native-M tile)
#                               + WS — bd 2D only (COMPILER BUG class, listed here next to
#                               its sibling)
#   TPE-POISON fences (valid configs that can't win in a regime):
#     scalar_max_m_pruner       scalar above M=64 (prefill GEVM) — mx 2D
#     mx_config_pruner poison fences   the dot arm where the
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

# Widest N tile the raw-activation (per-expert calibrated) arm survives; see
# raw_activation_pointer_pruner for the bisect.
RAW_ACT_MAX_BN = 128

# Branch axes that RELOCATE the tile optimum (compute unit / operand orientation): the
# tuner's guaranteed max-tile anchors group by these (``path_anchor_axes`` — a declaration,
# so the tuner itself stays independent of configuration details). Scheduling axes
# (WARP_SPEC, PACKED_SCHEDULE, memory modes) share the tile basin shape and stay out —
# per-basin anchors over the full branch cross starved a 100-trial budget (see
# _basin_anchor_indices).
PATH_ANCHOR_AXES = ("COMPUTE_MODE", "SWAP_AB", "PIPELINE")


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


def int_scale_group_tile_pruner(divides_k: bool = True):
    """``early_config_prune`` for the integer weight-only kernels' K tile: whole scale groups (each group's scale
    broadcasts over its columns of the dequantized tile) or a whole number of tiles per group (a group wider than the
    tile, per-channel included), and a tile that divides K wherever the K-loop has no tail (``divides_k``: the 2D and
    grouped kernels) or a group spans tiles (a slid-back tail tile would straddle two groups). The batched loop slides
    its last tile back in bounds otherwise. Newly admitted within-group tiles start at 128, preserving
    the existing group-128 grid; narrower tiles add K-loop and metadata traffic."""

    def ok(c, args):
        # The packed scalar decode helpers support bf16/fp16 and INT8, not FP8 operands.
        if args["A"].dtype == torch.float8_e4m3fn and c.kwargs.get("COMPUTE_MODE") == "scalar":
            return False
        # INT32 partials must stay integral through the final K sum; the split-K workspace is FP32.
        if args["A"].dtype == torch.int8 and c.kwargs.get("SPLIT_K", 1) != 1:
            return False
        bk, group = c.kwargs["BLOCK_SIZE_K"], args["SCALE_GROUP_K"]
        if group > bk:
            return group % bk == 0 and args["K"] % bk == 0
        return bk % group == 0 and (not divides_k or args["K"] % bk == 0)

    def wide_tile(c, args):  # advisory: within-group tiles from 128, unless only narrower ones are correct
        return c.kwargs["BLOCK_SIZE_K"] >= 128 or args["SCALE_GROUP_K"] <= c.kwargs["BLOCK_SIZE_K"]

    def raise_no_tile(configs, args):
        raise ValueError(
            f"no K tile fits K={args['K']} with scale group {args['SCALE_GROUP_K']}: every remaining config "
            f"would skip or straddle part of K"
        )

    # correctness vetoes raise when they empty the grid (the fallback would drop the K tail);
    # the 128 preference only narrows what survives
    return compose_pruners(config_filter(ok, on_empty=raise_no_tile), config_filter(wide_tile))


def block_fits_dim_pruner(dim_arg: str, block_key: str = "BLOCK_SIZE_K"):
    """``early_config_prune`` for a loop whose tail tile SLIDES back in bounds instead of
    running past the row (the weight-only batched K-loop): the tile need not divide the dim,
    it only has to fit inside it, since the clamp would otherwise walk backwards off the
    buffer. Everywhere the tail is unmasked, use ``block_within_dim_pruner`` — this one keeps
    non-dividing tiles, and on a maskless loop those read past the row."""

    def ok(c, args):
        block = c.kwargs.get(block_key, 0)
        return block == 0 or block <= args[dim_arg]

    def raise_no_fitting_block(configs, args):
        min_block = min(c.kwargs.get(block_key, 0) for c in configs)
        raise ValueError(
            f"{dim_arg}={args[dim_arg]} is smaller than every {block_key} in the autotune "
            f"grid (smallest: {min_block}); the sliding tail tile has nowhere to sit."
        )

    return config_filter(ok, on_empty=raise_no_fitting_block)


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
        # The tile-loop WS candidates require Triton 3.8's partitioning passes.
        if c.kwargs.get("PIPELINE", "k_loop") != "k_loop" and triton_version() < (3, 8):
            return False
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
    an autotune ``log_path``: of 129 WS configs, the 12 that failed are exactly the ones this
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


def gate_tile_cap_pruner(max_bn: int = 32):
    """``early_config_prune`` capping ``BLOCK_SIZE_N`` on the XPU GATE arm, whose tile spans
    ``2 * BLOCK_SIZE_N`` weight columns and so covers the output in half as many programs.
    Measured on BMG, batched MXFP8 decode: BN=32 beats 64 by 10% and 128 by 60%.

    A cap rather than a tuner hint because the default trial budget does not reach BN=32 on its
    own — it crowns BN=64, ~6% of decode. Sub-32 tiles stay available, so no N is unschedulable.

    XPU + GATE only: CUDA's wider tile is the measured winner there, and ungated launches prefer
    it too (down is ~25% faster at BN=64). The win is grid-, not register-bound — BN=32 still
    spills under ``IGC_ShaderDumpEnable``, more than BN=64 does.

    Probed on BMG (Xe2) only — best to re-probe on other Xe parts before taking this as a Xe-wide law."""

    def ok(c, args):
        return config_dim(c, args, "BLOCK_SIZE_N") <= max_bn

    return config_filter(
        ok, when=lambda args: args.get("GATE", False) and get_active_device_type() == "xpu"
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


def mx_config_pruner(k_arg: str, n_arg: str | None = None, block_within_k: bool = True):
    """``early_config_prune`` for the MX kernels (2D, batched, grouped — their tile is always
    tuned): a BK-within-K veto plus sm_10x MMA-shape guards (no-ops elsewhere and for scalar
    configs). GATE-aware: under the GATE constexpr (read from the launch args) the kernel
    computes gate|up as one stacked 2*BN extent, so the swapped dot_scaled lhs has ``2*BN``
    rows and the no-swap combined dot is ``2*BN`` wide; a plain launch's counts are just
    ``BN``.

    ``block_within_k=False`` drops the BK veto for a caller whose K-loop masks or slides its
    tail (the weight-only batched kernel); the MMA-shape guards still apply.

    - ``BLOCK_SIZE_K`` not dividing the launch's contraction dim (``k_arg`` names it) →
      dropped: the K-loop loads are unmasked, so any non-dividing BK's last trip reads
      past the row — silently wrong results the tuner would happily time and pick (bit us when
      BK=512 met a K=256 test problem).
    - MXFP4 ``scalar`` configs are KEPT. They used to be dropped on sm_10x: fp4 scalar decode
      was ALU-bound in the E2M1 unpack and measured 1.8x slower than dot_scaled. That verdict
      was against the INTEGER unpack. Decoding E2M1 through the sm_100 CVT unit
      (``_e2m1_pair_to_e4m3``) inverts it — on the dsv4 W4A8 decode, scalar+swap beats the
      dot_scaled crown 20.7 vs 41.8us (gate_up) and 11.3 vs 22.0us (down), bit-identical —
      because at BM=1 dot_scaled cannot reach the native microscaled MMA either (it needs an
      M=128 operand) and emulates through the same upcast, while scalar skips the MMA
      entirely. Re-fence only against a fresh measurement.
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

    A weight-only launch whose raw activation is fp32 drops ``dot_scaled`` outright: the op's
    lhs must be bf16/fp16, so the arm cannot compile at all there (arch-independent, unlike the
    shape gates below).

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
        # Triton <= 3.8.0 mis-lowers native dot_scaled with UE8M0 scales at BM=128 / BN=64 /
        # BK=64 / 8 warps: wrong results at any K (fixed upstream after 3.8.0). Admitted wherever
        # BK=128 cannot divide K.
        if (
            not c.kwargs.get("SWAP_AB")
            and c.kwargs.get("BLOCK_SIZE_M", args.get("BLOCK_SIZE_M")) == 128
            and config_dim(c, args, "BLOCK_SIZE_K") == 64
            and config_dim(c, args, "BLOCK_SIZE_N") == 64
            and c.num_warps == 8
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
        # A gate|up tile overhanging N (the 2D kernel's non-aligned N: N=320 at BN=128) returns a
        # wrong GLU / requant on every M, warp, stage, memory-mode and WS variant (all 157 such
        # admitted configs, W4A4 gate requant N=K=320, B200 2026-10-20); a BN dividing N is correct.
        if args.get("GATE") and args["N"] % config_dim(c, args, "BLOCK_SIZE_N"):
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
        # mxf4nvf4 (E4M3 scales) has NO fallback below the native staging — the decompose
        # path materializes the scale through an integer-only DenseElementsAttr and
        # PassManager fails outright (charted BM {16,32,64} x BK {128,256}, 2026-07-15;
        # warps/operand law 2026-08-12). Native staging needs the M operand >= 128 AND
        # num_warps in {4, 8} (2 and 16 both fail across every BN/BK/stages cell) AND
        # both operands microscaled — the weight-only kernels (no ``As``: raw bf16 acts,
        # in-register inline quant) never stage it, so their E4M3-scale dot_scaled fails
        # at every tile (their dot/scalar arms compete instead). dot_scaled-only
        # (dot/scalar decode E4M3 scales in software at any M);
        # the MMA's M operand is BLOCK_SIZE_M, or the weight rows when SWAP_AB puts the
        # token in N.
        if c.kwargs.get("COMPUTE_MODE") != "dot_scaled":
            return True
        if args.get("SCALE_GROUP_K") == 32:
            # Mixed tensor-FP8 / group-32 E4M3 FP4 is not the group-16 FP4xFP4 MMA.
            return False
        if not acts_are_scaled(args) or not 4 <= c.num_warps <= 8:
            return False
        if c.kwargs.get("SWAP_AB"):
            return (2 if args.get("GATE") else 1) * config_dim(c, args, "BLOCK_SIZE_N") >= 128
        return config_dim(c, args, "BLOCK_SIZE_M") >= 128

    def dot_scaled_act_dtype_ok(c, args):
        # `tl.dot_scaled` takes a bf16/fp16 lhs. A weight-only launch hands it the RAW
        # activation, so an fp32 one cannot reach the arm at all: every dot_scaled config dies
        # in "Unexpected dtype for bf16. Got fp32" and the tuner books the whole arm as compile
        # FAILURES rather than as a declared fence (87 of them on one grouped launch). A model
        # running in fp32 reaches this through the transformers integration. The dot and scalar
        # arms upcast the weight and take any float lhs, so they serve the launch — drop rather
        # than raise. Not arch-gated: the lhs dtype is the op's contract, not an sm_10x quirk.
        if c.kwargs.get("COMPUTE_MODE") != "dot_scaled" or acts_are_scaled(args):
            return True
        return getattr(args.get("A"), "dtype", None) != torch.float32

    def scales_are_e4m3(args):
        return getattr(args.get("Bs"), "dtype", None) == torch.float8_e4m3fn

    stages = [block_within_dim_pruner(k_arg)] if block_within_k else []
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
        config_filter(dot_scaled_act_dtype_ok),
        config_filter(nvfp4_native_ok, when=scales_are_e4m3),
        config_filter(
            mma_trap_ok, when=lambda args: is_sm10x(), on_empty=raise_all_mma_trapped
        ),
        config_filter(
            dot_arm_ok,
            when=lambda args: is_sm10x()
            and scaled_mma_available(args)
            and not (scales_are_e4m3(args) and args.get("SCALE_GROUP_K") == 32)
            and acts_are_scaled(args),
        ),
        config_filter(mma_poison_ok, when=lambda args: is_sm10x()),
    )


def dot_scaled_staging_pruner():
    """``early_config_prune`` dropping ``dot_scaled`` where the MMA's M operand degenerates to a
    single row: ``BLOCK_SIZE_M == 1`` with no ``SWAP_AB`` to move the operand to ``BLOCK_SIZE_N``.
    The scaled MMA has no 1-row staging and the config dies in ``PassManager::run failed`` at every
    warps/stages cell — the decode arms that DO serve BM=1 are scalar, or dot_scaled swapped (which
    puts the weight rows in M). Measured on the batched mx kernel: the last 6 dead compiles per
    tune after the swapped-scalar act fence.

    Two staging laws, both of which ``mx_config_pruner`` already enforces for NVFP4/E4M3 and
    neither of which was applied to the UE8M0 half:

    - ``num_warps`` outside {4, 8}: the scaled MMA has no staging for it (w16 s6 measured failing).
    - a 1-row M operand, per above.

    Measured on the batched mx kernel: 128 dead compiles per tune on HEAD, 2 after both clauses."""

    def ok(c, args):
        if c.kwargs.get("COMPUTE_MODE") != "dot_scaled":
            return True
        if c.num_warps not in (4, 8):
            return False
        if c.kwargs.get("SWAP_AB"):
            return True
        return config_dim(c, args, "BLOCK_SIZE_M") > 1

    return config_filter(ok)


def block_dynamic_2d_warp_spec_pruner():
    """``early_config_prune`` for the block-dynamic 2D kernel's two WARP_SPEC compile-fail
    laws (Triton 3.7.1 PassManager, TritonGPUPipeline; charted 2026-08-13 at the
    DeepSeek-V4-Flash ue8m0 checkpoint shapes, forced-config matrices):
      - WS + num_warps=16 fails at EVERY tile/scale-dtype (w8 compiles everywhere). The mx
        2D kernel WS-compiles w16 fine; the bd GROUPED kernel does NOT on its pointer arm
        (same law, see block_dynamic_grouped_matmul_pruner) — so this tracks the loop's
        structure and memory mode, not a global WS law.
      - the implicit ``dot_scaled`` arm (``USE_DOT_SCALED``: UE8M0/uint8 ``As`` on a
        native-M tile — (BN if SWAP_AB else BM) >= 128) + WS fails at every warps/stages/
        memory-mode cell; BM=64 and non-WS all compile. fp32-scale launches never emit the
        scaled MMA and keep the (w<=8) WS grid. On Triton 3.8 these compile (124/180, bitwise) but
        never win: best 555 us vs the 518 us non-WS crown (DeepSeek-V4 attn, M=8192, B200), so the
        fence stays as TPE hygiene.
    The output-tile pipeline (``PIPELINE`` != k_loop) is fp32-scale only: with UE8M0 scales it never wins (DeepSeek-V4
    attn M=8192, B200: best forced pipeline 637 us vs the 510 us k_loop crown; tuned A/B 1.00-1.02x), while fp32
    scales gain 0.82-0.95x (DeepSeek-V3 attn, M=512-8192, bitwise).
    Mostly 2D-only: the grouped bd loop WS-compiles from its descriptor arm and its own
    race guard REQUIRES WS at BM >= 64, but it shares the dot_scaled+WS failure on the
    POINTER arm (see block_dynamic_grouped_matmul_pruner). Compile soundness is
    per-loop-structure AND per-memory-mode — chart both before reusing a law."""

    def ok(c, args):
        if not c.kwargs.get("WARP_SPEC"):
            return True
        if c.num_warps > 8:
            return False
        if getattr(args.get("As"), "dtype", None) == torch.uint8:
            if c.kwargs.get("PIPELINE", "k_loop") != "k_loop":
                return False
            native_m = "BLOCK_SIZE_N" if c.kwargs.get("SWAP_AB") else "BLOCK_SIZE_M"
            return config_dim(c, args, native_m) < 128
        return True

    return config_filter(ok, when=lambda args: is_sm10x())


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


def fp8_dot_warp_pruner():
    """SILENTLY-WRONG fence: the fp8 ``tl.dot`` paths lose accuracy above ``num_warps=2`` at both
    ends of the M-tile range, on sm_90. Two clauses, one phenomenon.

    (1) The single-token swapped GEVM. ``swap_pad_rhs`` pads the lone token to the ``MMA_N_ATOM``
        rhs and the epilogue takes column 0 after the K-loop (``acc_init``'s ``[BN, MMA_N_ATOM]``
        accumulator). Above 2 warps that column comes back wrong.

    (2) Large M tiles, ``BLOCK_SIZE_M >= 64``, in the grouped and 2D kernels.

    Measured on H100 (sm_90), triton 3.6 AND 3.8, by forcing every admitted config against the
    torch oracle and scoring error in bf16-ULP units (the floor — what perfect bf16 output
    storage costs — is 0.49 mean / 1.00 max; the torch reference itself sits exactly there)::

        BLOCK_SIZE_M 16 / 32, any num_warps   0.49 mean / 1.0 max ULP   bit-optimal
        BLOCK_SIZE_M 64 / 128, num_warps 8    0.58 mean / 52.1 max ULP  WRONG   (grouped)
        BLOCK_SIZE_M 64 / 128, num_warps >2   280 of 578 configs WRONG          (2D)
        SWAP_AB + BLOCK_SIZE_M=1, num_warps >2                         WRONG   (batched)

    The tuner benches by SPEED, so these get crowned silently: block-static crowned a wrong one
    outright, while block-dynamic / tensor-dynamic (batched) and the grouped kernel carried
    30-90 wrong configs each behind a GREEN test, because the winner happened to be correct.

    In the grouped kernel ``BLOCK_SIZE_M >= 64`` is admitted only at ``num_warps=8``, so those two
    axes are confounded there and the data cannot separate them; the 2D kernel does separate them
    (BM=64 at num_warps=2 is clean), which is why the rule is written on num_warps.

    Scoped by a POSITIVE ``is_sm90()`` test, not by ``not is_sm10x()``: the latter is also true on
    XPU, ROCm and driverless build boxes, which do not run the NVIDIA codegen this was probed
    against, and restricting them on sm_90 evidence would be a guess. sm_10x is likewise untouched
    — it is where this repo's tuning and validation were done and its tcgen05 MMA lowers these
    paths differently. Other CUDA arches (sm_89, sm_120) are unmeasured and so unfenced; each
    needs its own probe. The sm_90 cost is real — the grouped kernel is the prefill path, where
    the large-M tiles are the throughput — and is accepted: the alternative is a crowned config
    that returns wrong results."""

    def ok(c, args):
        bm = c.kwargs.get("BLOCK_SIZE_M", 1)
        if c.num_warps <= 2:
            return True
        if bm >= 64:  # (2) large-M tile
            return False
        # (1) swapped single-token GEVM; native-M tiles need no pad/column-extract
        return not (c.kwargs.get("SWAP_AB") and bm < MMA_N_ATOM_WIDTH)

    def raise_all_wrong(configs, args):
        raise ValueError(
            "every autotune config is an fp8 tl.dot at num_warps>2 with a BLOCK_SIZE_M that "
            "returns WRONG results off sm_10x; no correct config exists in this grid — extend it "
            "with a num_warps=2 or a 16/32-row M tile rather than lifting the fence."
        )

    return config_filter(ok, when=lambda args: is_sm90(), on_empty=raise_all_wrong)


def scale_subblock_pruner(min_ctas_per_sm: int = 4):
    """``early_config_prune`` for kernels whose compute tile (``BLOCK_SIZE_N``) may subdivide the
    quant block (``BLOCK_N``). One scale covers the whole block, so a narrower tile just reads its
    own block — and it multiplies the N grid, the lever while the whole-block grid is too thin to
    hide the loop's latency (a gate|up fusion halves that grid again by folding both into one program).

    Sub-block tiles are admitted ONLY there: below ``min_ctas_per_sm`` whole-block programs per SM.
    "Fills the device" (one program per SM) is NOT enough at decode — the loop is latency-bound, so
    a 192-448-program grid still gains from more, narrower tiles: forced BN=64 vs the BN=128 crown,
    bit-identical, DSV4 fp8 decode gate 24.9 -> 23.4us and down 13.4 -> 12.2, DSV3 down 22.3 -> 21.7
    but DSV3 gate 0.88x (2026-08-29) — a tuner call, so the fence only trims prefill, where a
    thousands-of-programs grid never wants the 3x wider config space (measurably worse winners
    when it carried it).

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
        starved = args["S"] * triton.cdiv(args["N"], block_n) < min_ctas_per_sm * sm_count(
            args["B"].device.index
        )
        if args.get("OUTPUT_FORMAT") is not None or not starved:
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


def tile_order_pruner(min_m_tiles_per_expert: int = 2):
    """``early_config_prune`` fixing ``TILE_ORDER`` per config instead of searching it: of each
    otherwise-identical pair it keeps the ordered row iff an expert averages at least two M-tiles
    (``ceil(S / num_experts) >= 2 * BLOCK_SIZE_M``, the quantity ``tokens_per_expert_bit_length``
    buckets, so one key never mixes orders), else the row-major one: the space size equals the
    row-major space. Below that no adjacent M-tiles share a weight tile and the remap is pure
    overhead (DSV3 E256 block-FP8 down T2048, BM16: 1.016x). Wired only on the kernels where
    ordering measured a win at the same forced config (B200, T8192): block-dynamic/static FP8
    gate -6..-11%, block-FP8 weight-only gate -4..-6%, full-precision down -4..-5%; MX
    (+6..+9%) and INT weight-only (+14..+18% down) lose and stay row-major. Searching the axis
    instead doubled the space and let a 100-trial tune crown a 1.45x slower config."""

    def ok(c, args):
        bm = config_dim(c, args, "BLOCK_SIZE_M")
        ordered = -(-args["S"] // args["num_experts"]) >= min_m_tiles_per_expert * bm
        return bool(c.kwargs.get("TILE_ORDER")) == ordered

    return config_filter(ok)


# Pointer/pointer warp counts that trap on the WEIGHT-ONLY kernel under Triton < 3.8. Charted
# 2026-08-26 and REPRODUCED on 3.7.1 by the 2026-09-17 two-version sweep, so these rows are real
# measurements of that compiler, not the mis-chart the descriptor rows turned out to be. Triton
# 3.8 fixes every one of them, which is why `warp_spec_memory_mode_pruner` only consults the
# table below that version.
_WEIGHT_ONLY_POINTER_TRAPS = {
    ("dot", "pointer"): {16},
    ("dot_scaled", "pointer"): {4, 8, 16},
}


def _warp_spec_needs_descriptor_tile() -> bool:
    """Triton >= 3.8, where a TMA-descriptor operand narrows WS to one tile (see the pruner)."""
    return triton_version() >= (3, 8)


def warp_spec_memory_mode_pruner(weight_only: bool = False, gather_only: bool = False):
    """``early_config_prune`` dropping ``warp_specialize`` configs the WS passes cannot lower
    (``TritonGPUOptimizePartitionWarps`` / ``RelayoutTritonGPU`` -> ``PassManager::run failed``).

    The rule is COMPILER-DEPENDENT, and the two versions are near mirror images. Charted
    2026-09-17 on B200 by filtering each kernel's OWN tuner configs (never hand-built — that skips
    the descriptor pre_hooks and fails every descriptor cell for the wrong reason), BN held at 128,
    BM and ``num_warps`` both axes, gpt-oss 2048x2048, on three kernels across two families:
    grouped MX weight-only, grouped MX dynamic, and block-dynamic FP8.

        operands          triton 3.8.0              triton 3.7.1
        pointer/pointer   every warp count and BM   mostly FAILS (see the table above)
        desc/desc         ONLY w4 AND BM == 128     most cells lower; w2 never does
        mixed             identical to desc/desc    identical to desc/desc

    Cell for cell across all three kernels, 3.8 rescues NO descriptor cell that 3.7.1 lowered
    (0 of 96) and 3.7.1 rescues NO pointer cell that 3.8 lowered (0 of 80): descriptor+WS
    REGRESSED in 3.8 and pointer+WS was FIXED. So each version gets the rule its own sweep
    measured, rather than an intersection that would fence off a whole memory mode on both.

    ``gather_only`` applies this restriction only to gathered descriptor A. The raw-activation
    grouped kernel reproduces the TMA gather layout failure with both BF16 and rescaled FP8
    weights, while non-gathered descriptor boxes compile at the other warp counts. Pointer A
    and non-gathered launches retain their existing search space.

    Weight-only dot prefill additionally supports the measured w8 128x128x128 tile at
    stages 2/3 with both host descriptors, inline scheduling and an already expert-sorted A. Both original
    and swapped operand orientations lower, except the swapped MXFP8 down K loop; descriptor gather is
    outside this exception.
    Integer outer pipelines also support eight warps for the 128x64x256 gate tile and
    the packed, flattened 128x128x256 down tile at stage 2; MX outer pipelines support the
    swapped 128x128x128 down tile at stage 2 (its gate|up tile fails to lower).

    ``weight_only`` adds the rows that kernel charted for itself and the sweep did not re-probe:
    ``BN == 32`` with WS, which failed 9/9 cells there, and (below 3.8) its trapping
    pointer/pointer warp counts.

    NOT covered here, deliberately: ``mx_dynamic``'s ``dot`` arm dies with a descriptor operand on
    BOTH versions, and it is not a WS failure at all — it raises ``descriptor gather of uint8 must
    have at least 32 columns, but got 16`` with ``warp_specialize`` on AND off. That is a gather
    width law and belongs wherever gather width is decided."""

    def ok(c, args):
        if gather_only and (
            args.get("GatherIdx") is None or c.kwargs.get("A_MEMORY_MODE", "pointer") == "pointer"
        ):
            return True
        if not c.kwargs.get("WARP_SPEC"):
            return True
        if weight_only and config_dim(c, args, "BLOCK_SIZE_N") == 32:
            return False
        a, b = c.kwargs.get("A_MEMORY_MODE"), c.kwargs.get("B_MEMORY_MODE")
        descriptor = a != "pointer" or b != "pointer"
        if _warp_spec_needs_descriptor_tile():
            prefill_tile = (
                weight_only and c.kwargs.get("COMPUTE_MODE") == "dot"
                and a == b == "host_descriptor" and args.get("GatherIdx") is None
                and not c.kwargs.get("PACKED_SCHEDULE")
                and c.kwargs.get("PIPELINE", "k_loop") == "k_loop"
                and not (
                    c.kwargs.get("SWAP_AB") and args.get("WEIGHT_VALUES_PER_BYTE") == 1
                    and not args.get("GATE", False)
                )
                and config_dim(c, args, "BLOCK_SIZE_N") == 128
                and config_dim(c, args, "BLOCK_SIZE_K") == 128
                and c.num_stages in (2, 3)
            )
            pipeline_tile = (
                weight_only and c.kwargs.get("PIPELINE") in ("tile_loop", "flattened")
                and c.kwargs.get("SWAP_AB")
                and a == b == "host_descriptor" and args.get("GatherIdx") is None
                and c.num_stages == 2
                and (
                    config_dim(c, args, "BLOCK_SIZE_K") == 256 and (
                        args.get("GATE", False) and config_dim(c, args, "BLOCK_SIZE_N") == 64
                        or (
                            args.get("INT_BITS") in (2, 4, 8) and not args.get("GATE", False)
                            and config_dim(c, args, "BLOCK_SIZE_N") == 128
                            and c.kwargs.get("PACKED_SCHEDULE") and c.kwargs.get("PIPELINE") == "flattened"
                        )
                    )
                    or (
                        "WEIGHT_VALUES_PER_BYTE" in args and not args.get("GATE", False)
                        and config_dim(c, args, "BLOCK_SIZE_N") == 128
                        and config_dim(c, args, "BLOCK_SIZE_K") == 128
                    )
                )
            )
            # FP8-activation dot loops pipelined across output tiles lower at w8 too (tensor-wide grouped
            # gate|up, Mistral-4: BM128 BN128 BK128 tile_loop w8 s4 runs, 0.85x the k_loop crown)
            fp8_pipeline_tile = (
                not weight_only and getattr(args.get("A"), "dtype", None) == FP8_DTYPE
                and c.kwargs.get("PIPELINE", "k_loop") != "k_loop" and a == b == "host_descriptor"
            )
            return not descriptor or (
                config_dim(c, args, "BLOCK_SIZE_M") == 128
                and (c.num_warps == 4 or ((prefill_tile or pipeline_tile or fp8_pipeline_tile) and c.num_warps == 8))
            )
        if descriptor:
            return True
        traps = _WEIGHT_ONLY_POINTER_TRAPS if weight_only else {}
        return c.num_warps not in traps.get((c.kwargs.get("COMPUTE_MODE"), a), set())

    return config_filter(ok, when=lambda args: is_sm10x())


def int_2d_warp_spec_pruner():
    """``early_config_prune`` for the integer 2D kernel: ``warp_specialize`` with a host-TMA activation and a pointer
    weight (A descriptor, B pointer) fails ``TritonGPUOptimizePartitionWarps`` (``PassManager::run failed``).
    Charted on B200 / Triton 3.8 over the kernel's own WS neighbourhood (4- and 8-bit, warps {4, 8, 16}, BM {64, 128},
    BK {128, 256, 512}, BN {32, 64, 128}, stages {2, 4}, all four operand-mode pairs): every lowering failure is in
    that pair, at warps 4 and 8 (68 of the 100 cells there that fit shared memory; warps 16 lowers), and none in the
    other pairs. The pair is fenced whole: its cells that do lower never came within 1.6x of a crown in any recorded
    tune (decode 5-10x off), so the exact boundary (tile bytes, width, warps) buys nothing. CUDA-only."""

    def ok(c, args):
        mixed = c.kwargs.get("A_MEMORY_MODE", "pointer") != "pointer" and c.kwargs.get("B_MEMORY_MODE", "pointer") == "pointer"
        return not (c.kwargs.get("WARP_SPEC") and mixed)

    return config_filter(ok, when=lambda args: get_active_device_type() == "cuda")


def weight_only_swap_scope_pruner():
    """Batched FP8 integer-weight dots support the swapped MMA; the other batched weight-only
    kernels swap only for scalar decode. The scalar reduce requires the swapped row layout."""

    def ok(c, args):
        if args["A"].dtype == FP8_DTYPE and c.kwargs.get("COMPUTE_MODE") == "dot" and c.kwargs.get("SWAP_AB"):
            return c.kwargs["BLOCK_SIZE_M"] <= MMA_N_ATOM_WIDTH
        return bool(c.kwargs.get("SWAP_AB")) == (c.kwargs.get("COMPUTE_MODE") == "scalar")

    return config_filter(ok)


def raw_activation_pointer_pruner():
    """``early_config_prune`` scoping the RAW-activation arm to the pointer load and away from
    warp specialization. A calibrated scale held per expert leaves ``A`` unquantized for the
    kernel to quantize per tile (a gathered row serves several experts, so there is no single
    pre-quantized form), which puts a quantize INSIDE the K-loop and changes its structure:

    - the TMA gather cannot serve it at all (``async_tma_gather`` wants 4 contiguous elements
      per thread and the lowering fails), which at a deployment shape left the tuner no config;
    - ``BLOCK_SIZE_N`` above 128 traps the device at prefill scale — a sticky misaligned
      address. Bisected on B200 / Triton 3.8 with a COLD tune cache (a warm one replays a safe
      crown and hides it): holding BN <= 128 runs clean, while fencing warp specialization, the
      packed schedule, or BK < 128 each still trap. So the tile width is the variable, not the
      schedule, the memory mode or WS.

    A pre-quantized ``A`` keeps the full grid, unchanged — which is what the grouped prefill
    now hands it (``quantize_routed_rows_per_expert`` lays the routed rows out quantized once
    the row count pays for the copy), so the raw arm is reached below that regime, or above it
    when ``FINEGRAINED_FORCE_GATHER`` holds the gather. The BN fence is load-bearing there."""

    def ok(c, args):
        a = args.get("A")
        if getattr(a, "dtype", None) == FP8_DTYPE or args.get("As") is None:
            return True
        return (
            c.kwargs.get("A_MEMORY_MODE", "pointer") == "pointer"
            and config_dim(c, args, "BLOCK_SIZE_N") <= RAW_ACT_MAX_BN
        )

    return config_filter(ok)


def native_mma_m_pruner():
    """Keep native tcgen05 M tiles for plain FP8 on sm_10x. Narrow unswapped
    tiles lower to warp MMA and lose warm-cache performance; swapped decode tiles
    use the weight-row dimension as MMA M and remain eligible."""

    def ok(c, args):
        return bool(c.kwargs.get("SWAP_AB")) or config_dim(c, args, "BLOCK_SIZE_M") >= 64

    return config_filter(ok, when=lambda args: is_sm10x())


def single_row_tile_pruner():
    """Avoid replicating a single token across a BM=16 warp-MMA tile on sm_10x.
    BM=1 retains the same compute path with less replicated work."""

    def ok(c, args):
        return config_dim(c, args, "BLOCK_SIZE_M") == 1

    return config_filter(ok, when=lambda args: is_sm10x())


def swap_decode_scope_pruner(max_m: int = 16):
    """``early_config_prune`` scoping a 2D plain-dot kernel's ``SWAP_AB`` rows (the weight's output
    rows in the MMA M dim, the token tile padded to the N=16 atom) to decode M, where they run
    one program row per token, and to a pointer-mode activation (an A-side descriptor under
    ``SWAP_AB`` traps on Triton 3.8, see ``mx_2d_swap_scope_pruner``). Unswapped rows pass."""

    def ok(c, args):
        if not c.kwargs.get("SWAP_AB"):
            return True
        return args["M"] <= max_m and c.kwargs.get("A_MEMORY_MODE", "pointer") == "pointer"

    return config_filter(ok)


def mx_2d_swap_scope_pruner(max_m: int = 16):
    """``early_config_prune`` scoping the 2D mx kernel's ``SWAP_AB`` rows to the ONE regime
    they exist for: E4M3-scale (NVFP4) single-token decode. NVFP4 has no other native M=1
    arm — ``dot_scaled`` has no sub-M=128 form for E4M3 scales (PassManager) and scalar is
    E2M1-unpack-bound (measured 60-142µs vs swap's 22-50µs on the GLM attn shape,
    2026-08-05). Everywhere else swap is a proven loser (an 18-cell forced-swap sweep:
    UE8M0 formats' scalar/dot arms win, M3 attn swap −38%), so its rows are dropped:
    UE8M0-scale launches, M above the decode band, and any non-BM=1 row (the mx swap
    compute flattens the token tile — BM=1 is structural, enforced by a static_assert).

    Also drops UNSWAPPED ``scalar``: the arm exists only in its swapped form here, so those
    configs cannot build an accumulator and fail to compile."""

    def ok(c, args):
        swap = bool(c.kwargs.get("SWAP_AB"))
        # Under GATE, scalar exists only as the swapped arm: unswapped it has no accumulator
        # shape (``acc_init`` is handed "dot" when not swapped) and every such config dies in
        # CompilationError (3 dead compiles per gated mxfp4 tune). NOT true ungated — unswapped
        # scalar is a legitimate (and sometimes winning) W4A4 arm there, and fencing it globally
        # crowned a wrong config at N=320.
        if c.kwargs.get("COMPUTE_MODE") == "scalar" and not swap and args.get("GATE"):
            return False
        if not swap:
            return True
        return (
            config_dim(c, args, "BLOCK_SIZE_M") == 1
            and getattr(args.get("Bs"), "dtype", None) == torch.float8_e4m3fn
            and args["M"] <= max_m
            # An A-side TMA descriptor under SWAP_AB traps the device on Triton 3.8 — a sticky
            # misaligned address, after which every later launch reports it wherever it came
            # from. Observed on B200 in a GLM-5.2-NVFP4 forward (16 tokens; clean at 4) at
            # dot_scaled, BM=1, BN ∈ {128, 256}, BK ∈ {64, 128}: the tile WIDTH is not the
            # variable (BN=128 traps too), the A descriptor is. Narrowed by probe — fencing the
            # B side as well was NOT needed, so the weight descriptor stays available to the
            # crown, and pointer-mode A keeps the decode arm this pruner exists for.
            and c.kwargs.get("A_MEMORY_MODE", "pointer") == "pointer"
        )

    return config_filter(ok)


def swizzled_out_bm_pruner():
    """``early_config_prune`` pinning the SWIZZLE_32_4_4 requant OUTPUT arm (``SWIZZLED_OUT``) to
    ``BLOCK_SIZE_M == BLOCK_SIZE_N == 128`` — the output-side twin of
    ``swizzled_scales_bm_pruner``.

    ``_epilogue_requant_mx`` writes the group scales straight into the down proj's swizzled
    layout as ``q_s.reshape(1, 4, 32, REP_K_CS, 4)`` with ``REP_K_CS = (BN // group) // 4``.
    ``q_s`` is ``[BM, BN // group]``, so the reshape balances only at ``BM == 128``
    (``BM * (BN//group) == 128 * (BN//group)``); every other BM raises "reshape() cannot change
    total number of elements". BN is pinned with it because the descriptor's block shape is the
    tuned ``REP_K_CS``. The code comment there already said "BM/BN pinned 128" — this makes the
    config space agree, instead of handing the tuner configs that cannot compile.

    Measured 2026-08-26 on the gated fp4-requant cells: 288 dead compiles per tune, i.e. every
    non-128 config of the swizzled-out key (``SWIZZLED_OUT`` is in the tune key, so the key is
    entirely swizzled). Un-swizzled requant keeps every BM/BN — it stores Cs row-major."""

    def ok(c, args):
        return (
            config_dim(c, args, "BLOCK_SIZE_M") == 128
            and config_dim(c, args, "BLOCK_SIZE_N") == 128
        )

    return config_filter(ok, when=lambda args: args.get("SWIZZLED_OUT"))


def swizzled_scales_bm_pruner():
    """``early_config_prune`` pinning the grouped MX pre-swizzled (``SWIZZLED_SCALES``) arm to
    ``BLOCK_SIZE_M == 128`` and a whole-128-block ``BLOCK_SIZE_N``. BM: the offline act-quant lays each expert's scale
    slab out 128-padded, and the kernel's per-tile scale-block index (``pid_m``) only lines up
    when the M tile is exactly 128 (``build_tile_layout`` pads experts on the same granularity).
    BN: the scale is read as whole 128-row SWIZZLE_32_4_4 blocks off the descriptor (under GATE,
    the gate 128-block + the up 128-block stacked into the 2*BN tile); a sub-128 BN would need a
    partial-block read the descriptor can't express, while any whole multiple is a multi-block bulk
    load the pre_hook sizes per config — BN=256 wins the down GEMM (665 -> 576us at DSV4 prefill,
    bit-identical; GATE at 256 is the scaled-MMA width cap's veto, ``mx_config_pruner``).

    Both pins are swizzled-slab properties, so the un-swizzled (affine) arm keeps every BM and
    BN: ``build_tile_layout`` tiles any BM with a row mask, and the affine act-scale read is
    indexed by the gathered row, not ``pid_m``."""

    def ok(c, args):
        return (
            config_dim(c, args, "BLOCK_SIZE_M") == 128
            and config_dim(c, args, "BLOCK_SIZE_N") % 128 == 0
        )

    return config_filter(ok, when=lambda args: args.get("SWIZZLED_SCALES"))


def swizzled_scale_config_pruner(allow_gate_subblock=False):
    """Drop pre-swizzled-scale configs the SWIZZLE_32_4_4 descriptor load can't serve, gated on
    ``SWIZZLED_SCALES`` (the un-swizzled arm loads scales per group and takes any tile):

    - ``BLOCK_SIZE_K < 4 * SCALE_GROUP_K``: the un-swizzle reads whole 4-group K bands
      (``REP_K = (BK // SCALE_GROUP_K) // 4``), so a smaller BK collapses ``REP_K`` to 0 and the
      reshape traps — 128 for group-32 (MX), 64 for group-16 (NVFP4). BK must also tile whole
      bands (``BK % (4 * group) == 0``).
    - ``BLOCK_SIZE_N > 128`` not a whole-block multiple: the descriptor arm bulk-loads
      ``BN // 128`` swizzled blocks (each kernel's pre_hook sizes the box per config), a sub-128 BN
      reads a scalar slice of one block; a BN between is neither, and a multi-block BN whose
      last tile passes the 128-padded N extent zero-fills (see the fit bound below). BN=256 is
      a legal two-block load where it fits (the GATE tile has always read one) and wins at
      prefill — grouped down 665 -> 576us, dense 2D 703 -> 552us at M=49152, bit-identical; the
      scaled-MMA width cap is ``mx_config_pruner``'s.
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
        if bn > 128:
            # BN=256 pipelines OOM smem at num_stages >= 5 (253 KB at s5, 304 KB at s6 for the
            # fp8 x fp4 and fp8 x fp8 dot_scaled tiles, BK=128; s4 fits) — below smem_pruner's
            # keep-side bound, so without this the TPE burns whole 200-trial budgets on dead
            # compiles (observed 2026-08-29).
            if c.num_stages >= 5:
                return False
            # A multi-block tile must FIT the swizzled grid: the bulk load reads BN // 128 whole
            # blocks with no row mask, and the grid is padded to 128 rows — a tile whose blocks
            # pass the padded extent TMA-zero-fills the scales: silent garbage, never a trap
            # (mxfp4_swizzled N=128 forced at BN=256 fails parity on triton 3.7.1 and 3.8 alike;
            # the BN=256 prefill wins were measured at N % 256 == 0 shapes).
            n = args.get("N")
            if n is not None and -(-n // bn) * bn > -(-n // 128) * 128:
                return False
        return bn <= 128 or bn % 128 == 0

    def raise_no_swizzled_tile(configs, args):
        raise ValueError(
            "no autotune config can serve pre-swizzled scales for this launch (the "
            "SWIZZLE_32_4_4 read needs BLOCK_SIZE_K % 128 == 0 and a <=128-row or 128-multiple N "
            f"tile; N={args.get('N')}, K={args.get('K')}, GATE={bool(args.get('GATE'))}) — say "
            "which dim is short rather than guessing: pass affine (row-major) scales for it."
        )

    return config_filter(
        ok, when=lambda args: args.get("SWIZZLED_SCALES"), on_empty=raise_no_swizzled_tile
    )


def block_static_grouped_matmul_pruner():
    """``early_config_prune`` for the static block-FP8 grouped kernel's own dead cell, on top of the
    inherited block-dynamic map: a warp-specialized ``BLOCK_SIZE_M = 128`` gate|up tile with a
    pointer A and a descriptor B overflows tensor memory (576 > 512 columns) at every stage count,
    while pointer/pointer at the same tile compiles (B200, Triton 3.8, BN = 128 stacked)."""

    def ok(c, args):
        return not (
            args.get("GATE") and c.kwargs.get("WARP_SPEC")
            and config_dim(c, args, "BLOCK_SIZE_M") == 128
            and c.kwargs.get("A_MEMORY_MODE", "pointer") == "pointer"
            and c.kwargs.get("B_MEMORY_MODE", "pointer") != "pointer"
        )

    return config_filter(ok, when=lambda args: get_active_device_type() == "cuda")


def block_dynamic_grouped_matmul_pruner():
    """``early_config_prune`` for the block-dynamic grouped kernel: the Triton 3.7.1
    pipeliner-race guard, sized to its four-load-stream single-dot K-loop. The GATE arm
    shares the guard: post-activation-prequant it dots the stacked gate|up as the same
    single-dot loop (one wider weight tile, not a second dot/load stream) and was
    flake-verified race-free cross-process. Measured on sm_100, 15-fresh-process flake
    runs per cell, big and tiny shapes:

    - ``warp_specialize`` compiles iff ``num_warps % 4 == 0`` (its async partitions) and
      ``BLOCK_SIZE_M >= 64``, and is race-free everywhere it compiles — but on the POINTER
      arm it hits a lowering wall: "'arith.constant' op does not have expected attribute
      ttg.partition" -> PassManager::run failed (TritonGPUPipeline). The descriptor arm
      always compiles. The obvious suspect — the masked activation load's ``other=0.0``,
      which is the known cause of the analogous mx-grouped fence — was PROBED and is NOT
      it: dropping the mask entirely leaves every pointer cell below still failing, so the
      trigger is structural to the pointer arm, not that constant. Do not re-try the
      clamp-not-mask rewrite on this basis. Two INDEPENDENT legs, forced-config matrices,
      2026-08-21 B200:
        * ``num_warps >= 16`` fails at BM 64 AND 128, with fp32 scales as well as UE8M0 —
          a property of this loop under WS, not of the scaled MMA.
        * uint8/UE8M0 ``As`` on a native-M ``BLOCK_SIZE_M >= 128`` tile (the implicit
          ``dot_scaled`` arm) fails at every warps/stages cell: pointer 0/6, descriptor
          4/6 + 2 smem.
      Combined with the race guard below, the descriptor arm is the only BM=128 route for
      UE8M0 grouped — which is what makes the native tcgen05 tile reachable at all here.
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
        bm = config_dim(c, args, "BLOCK_SIZE_M")
        if bm < 64:
            return not c.kwargs.get("WARP_SPEC")
        if not (c.kwargs.get("WARP_SPEC") and c.num_warps % 4 == 0):
            return False
        if c.kwargs.get("A_MEMORY_MODE") == "pointer":
            if c.num_warps >= 16:
                return False
            if getattr(args.get("As"), "dtype", None) == torch.uint8 and bm >= 128:
                return False
        return True

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


def swap_max_m_pruner(m_arg: str, max_m: int):
    """Keep measured small-row swaps in one M tile, retaining BM1/2 candidates for M1."""

    def ok(c, args):
        if not c.kwargs.get("SWAP_AB"):
            return True
        m = args[m_arg]
        bm = c.kwargs["BLOCK_SIZE_M"]
        return m <= max_m and m <= bm <= triton.next_power_of_2(max(2, m))

    return config_filter(ok)


def block_m_cap_pruner(m_arg: str, min_bm: int = 16):
    """``early_config_prune`` dropping configs whose ``BLOCK_SIZE_M`` exceeds the launch's row count rounded up to a
    power of two (floor ``min_bm``, the MMA atom): a dense GEMM with M below the tile has one M tile either way, so
    the extra rows only pad the MMA and shrink the grid, and benching them dilutes a decode tune (int4 2D, B200:
    the 4096x12288 M=16 tune crowned 28.2us against the 25.2us its BM=16 basin holds). Non-empty: ``min_bm`` stays."""

    def ok(c, args):
        bm = c.kwargs.get("BLOCK_SIZE_M")
        return bm is None or bm <= max(min_bm, triton.next_power_of_2(args[m_arg]))

    return config_filter(ok)


def scalar_block_m_pruner(m_arg: str):
    """``early_config_prune`` keeping only the ``scalar`` configs whose ``BLOCK_SIZE_M`` is the launch's row count
    rounded up to a power of two: the reduce dequantizes each weight word once for its BM rows, so a smaller BM
    re-dequantizes per program and a larger one reduces padding rows (int4 2D, B200, 4096x4096: M=2 BM 1/2/4 =
    9.0/7.7/10.2us, M=4 14.2/11.2/9.9us), and the TPE missed the matched tile at the default budget. Non-empty:
    the dot configs stay."""

    def ok(c, args):
        return c.kwargs.get("COMPUTE_MODE") != "scalar" or c.kwargs["BLOCK_SIZE_M"] == triton.next_power_of_2(args[m_arg])

    return config_filter(ok)


def mx_auxiliary_pruner():
    """Split the single-token decode reduction (the launch allocates the partials for M=1 without GATE), or persist
    across SVDQuant prefill tiles. Plain M=1 splits only the scalar BM=1 reduce: the native split tile pads the token
    to 128 MMA rows and never crowns there (B200 graph, NVFP4: 5120x5120 scalar 11.3us, scalar split 10.2us, native
    split 26.2us; N1024 K14336 22.6 / 7.7 / 31.5us); SVDQuant keeps the native split (down: 27.4us vs FlashInfer 33.1us)."""
    def ok(c, args):
        split = c.kwargs.get("SPLIT_K", 1)
        persistent = c.kwargs.get("PERSISTENT", False)
        if split == 1 and not persistent:
            return True
        if not is_sm10x() or args["GATE"]:
            return False
        if persistent:
            return args.get("LORA_RANK", 0) > 0 and args["M"] >= 128
        if c.kwargs["COMPUTE_MODE"] != "scalar" and args.get("LORA_RANK", 0) == 0:
            return False
        return args["M"] == 1 and args["K"] // c.kwargs["BLOCK_SIZE_K"] >= 2 * split
    return config_filter(ok)


def int_split_k_pruner(m_arg: str, max_m: int):
    """``early_config_prune`` for a split-K axis: splits only in the decode band (``m_arg`` <= ``max_m``, where the
    launch allocates the partials) and only where the split divides the K tiles. Sub-16 dot tiles cover the
    launch's rows in one tile, keeping the counter grid within its allocation and avoiding repeated weight reads."""

    def ok(c, args):
        bm = c.kwargs.get("BLOCK_SIZE_M", 16)
        if bm < 16 and c.kwargs.get("COMPUTE_MODE", "dot") != "scalar" and bm != triton.next_power_of_2(args[m_arg]):
            return False
        split = c.kwargs.get("SPLIT_K", 1)
        if split == 1:
            return True
        k_tiles = args["K"] // c.kwargs["BLOCK_SIZE_K"]
        return args[m_arg] <= max_m and k_tiles % split == 0 and k_tiles >= split

    return config_filter(ok)


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
    """Fence descriptor strides and gather support. Triton splits logical tiles into
    legal physical TMA boxes; the logical tile may exceed 256 elements per dimension.
    Pointer configs pass untouched."""

    def ok(c, args):
        if c.kwargs.get("B_MEMORY_MODE", "pointer") != "pointer":
            # A TMA descriptor needs a unit-stride innermost dim; a transposed operand
            # (column-major view, as a backward pass hands in) has none, so only the
            # pointer arm can serve it. The kernel itself is stride-general.
            b = args.get("B")
            if b is not None and getattr(b, "ndim", 0) == 2 and b.stride(-1) != 1:
                return False
            if "BDescriptor" in args and args["BDescriptor"] is None:
                return False
        if c.kwargs.get("A_MEMORY_MODE", "pointer") != "pointer":
            if "ADescriptor" in args and args["ADescriptor"] is None:
                return False
            # descriptor A, both arms tuner-routed per the tokens-per-expert key:
            # contiguous boxes (no gather) win at dense routing (-7.1/-7.4%) and lose at
            # skew; tma gather4 (gathered rows, 1-row box, sm_100 only) is MONOTONIC in
            # tokens/expert — 2x loss at 32/expert, parity at 1024, -5.3/-4.7% at 4096
            # (32k tokens, 2026-07-16). Only the arch is fenced.
            if args.get("GatherIdx") is not None and not is_sm10x():
                return False
        return True

    return config_filter(ok)


def gate_stacked_tmem_trap_pruner():
    """Miscompile gate: drop GATE configs at exactly ``BLOCK_SIZE_N == 256, BLOCK_SIZE_M == 64``
    — the stacked gate|up accumulator (``2*BN = 512`` columns) exceeds sm_100 tensor memory, and
    at BM=64 Triton's resource checker misses the overflow, so the config runs and TRAPS the
    device (sticky CUDA 716 "misaligned address"; one such trial poisons the whole tune and
    every launch after it). Probed per-arm in fresh processes (2026-08-10, B200, weight-only
    grouped, K=2880 and K=2048 alike — N-divisibility is NOT the variable): BM=64 traps in every
    memory mode, WS on and off, BK 64 and 128; BM=128 fails CLEANLY (OutOfResources: tensor
    memory 544 > 512) and self-prunes; BM=32 and BN<=128 run correct. The checker gap is the
    only reason this needs a fence at all — drop precisely the probed trap point and let every
    neighbor keep self-reporting.

    ON sm_90 the whole ``2*BN = 512`` stacked tile goes, not just the BM=64 point. The
    self-pruning above is a TENSOR-MEMORY overflow, and sm_90 has no tensor memory to overflow:
    nothing raises OutOfResources, so the config runs and returns garbage instead. Measured on
    H100, triton 3.8, nvfp4 gate|silu with fused requant through the 2D kernel — sampling the
    admitted grid, ``BLOCK_SIZE_N`` splits cleanly::

        BN 32 / 64 / 128   60 correct, 0 wrong
        BN 256             0 correct, 9 wrong   (BM=128; the crowned config varies per tune,
                                                 so the scenario test passes or fails by luck)

    The mechanism is the stacked accumulator width, which is format-independent, so this fences
    every GATE config at BN=256 rather than only the nvfp4 cell that exposed it. The BM=64 trap
    point stays fenced on every arch (it was probed on B200); the wider sm_90 clause is scoped by
    a positive arch test so XPU/ROCm, which never ran this codegen, keep their full grid."""

    def ok(c, args):
        if not args.get("GATE") or config_dim(c, args, "BLOCK_SIZE_N") != 256:
            return True
        if config_dim(c, args, "BLOCK_SIZE_M") == 64:
            return False  # the probed sm_100 trap point — fenced on every arch
        return not is_sm90()  # sm_90 has no tensor memory, so nothing self-reports the overflow

    def raise_all_trap(configs, args):
        raise ValueError(
            "every config in the grid is the (GATE, BN=256, BM=64) tensor-memory trap point; "
            "falling back would run configs that trap the device (sticky CUDA 716)"
        )

    return config_filter(ok, on_empty=raise_all_trap)


def smem_pruner(
    k_dim="BLOCK_SIZE_K", min_sets: float = 0.5, full_stages: bool = False, pointer_ws_extra_stage: bool = False,
    ws_full_stages: bool = False,
):
    """``early_config_prune`` dropping configs whose shared memory certainly cannot fit,
    with bounds derived from TTGIR allocations and checked against ``metadata.shared``:

    - both operands >= 2-byte floats (the unquantized loops): ``num_stages`` full
      buffers of both tiles (exact) plus the fixed barrier terms at their sampled
      MINIMA (+16 base; +28 warp-spec, observed up to 48 varying with warps; +32
      host-descriptor) — minima keep the bound below the true usage, so a config is
      only ever dropped when its operand buffers alone cannot fit.
    - quantized ``dot_scaled``:
      the per-allocation law, every term named by its TTGIR ``local_alloc`` and exact to
      the allocator's alignment/packing slack (<= 16 B observed; barriers pack into
      padding, so no fixed constant is added — the slack is keep-side) —
      ``num_stages - 1`` buffers of operands+scales at the native M=128 staging (weights
      full-width even when fp4-packed; GATE shifts the A tile to a full ``num_stages``),
      or the bf16-upcast fallback below the gate (one single-buffered upcast tile,
      packed weights, no A-scale buffers).
    - FP8 weight-only, natural-orientation K loop with host descriptors: ``num_stages``
      BF16/FP16 A buffers, ``num_stages - 1`` raw FP8 B buffers (``num_stages`` under
      warp specialization), plus one decoded BF16/FP16 B buffer. Scales, barriers and
      alignment are omitted from this lower bound. Swapped/output-tile pipelines use
      different storage and retain the raw floor.
    - other quantized ``dot``/``scalar``: the raw-operand floor.

    ``min_sets`` scales the quantized ``dot_scaled`` bound: the default HALF a set is the
    family-wide keep-side envelope (the 2D nvfp4 arm sub-tiles a request into half a set);
    the GROUPED kernel never compiles below TWO full sets — fresh-tune logs at the DSV4 W4A8
    gate_up key (244 trials, 2026-08-29): 0 of 82 compiling configs had 2*set > smem, while
    120 of 162 dead ones did (every BK=512 request, 59/59), so ``min_sets=2`` there halves the
    dead compiles a 100-trial tune pays without dropping a single compiling config.

    ``full_stages``: the loop stages every one of ``num_stages`` raw-operand buffers (no
    pipeliner clamp), so the raw floor counts them all. Measured on the static block-FP8 grouped
    gate|up loop (B200, BN=128 stacked, BK=128): BM=128 s5 262152 B and s6 327688 B, BM=64 s6
    245768 B, each above ``num_stages * BK * (BM + 2*BN)``; over the block-dynamic grouped tune
    logs (DSV3/E16 keys, ~4k trials) that floor drops 109 failed configs and no compiling one.
    ``pointer_ws_extra_stage``: a warp-specialized pointer/pointer loop holds one more set (the
    block-dynamic kernel: BM=128 s4 gate|up 245784 B = 5 sets); with it the dynamic floor drops all
    155 logged failures and none of 3302 compiling configs. The static kernel does not (its
    pointer/pointer WS s4 tiles compile), so it is off there.
    ``ws_full_stages``: ``full_stages`` for warp-specialized configs only (the K-loop pipeliner still
    clamps the others): the tensor-wide grouped gate|up tune at Mistral-4 drops all 32 WS
    shared-memory failures and none of 105 compiling configs.

    These are lower bounds on the observed IR allocations, not a complete allocator
    model: omitted storage keeps borderline configs for the compiler to check. Samples
    were taken on B200 (sm_100); the FP8 weight-only extension is scoped to that arch —
    re-verify on a Triton bump AND on H100 or the target device (smem capacity and
    allocator behavior are per-arch). Non-empty fallback."""

    def ok(c, args):
        a, b = args["A"], args["B"]
        tiles = 2 if args.get("GATE") else 1
        bk = config_dim(c, args, k_dim)
        unquantized = a.element_size() >= 2 and b.element_size() >= 2
        # an output-tile pipeline keeps every stage live (no K-loop pipeliner clamp): 2D block-FP8
        # BM64xBN256 s6 needs 254084 B and BM128xBN256 s5 262276 B, both above that full-stage floor
        pipelined = c.kwargs.get("PIPELINE", "k_loop") != "k_loop"
        all_live = full_stages or pipelined or (ws_full_stages and c.kwargs.get("WARP_SPEC"))
        stages = c.num_stages if unquantized or all_live else c.num_stages - 1
        if pointer_ws_extra_stage and c.kwargs.get("WARP_SPEC") and (
            c.kwargs.get("A_MEMORY_MODE", "pointer") == c.kwargs.get("B_MEMORY_MODE", "pointer") == "pointer"
        ):
            stages += 1
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
            # an integer weight is uint32 words of 32 // bits K values: bits / 8 bytes per K per row
            b_bytes_per_k = 4 * b.shape[-1] / args["K"] if b.dtype == torch.uint32 else b.element_size()
            need = fixed + stages * bk * (a.element_size() * bm + b_bytes_per_k * tiles * bn)
        elif c.kwargs.get("COMPUTE_MODE") == "dot_scaled":
            # KEEP-SIDE bound = HALF of one buffer set: operands stored PACKED (fp4 = 2
            # values/byte) plus both group scale tiles (BK/group cols; group 32 UE8M0, 16
            # E4M3). Requested-stages models over-estimate — the pipeliner CLAMPS stages
            # (an s3 request can compile with an s2 footprint) and the 2D nvfp4 arm can
            # compile half a set — so any tighter bound drops configs the compiler would
            # have rescued. Proven est<=actual on 1,099 metadata.shared samples across the
            # family (see OPTIMIZATION_LOG: smem-keep-side-refit).
            group = 16 if getattr(args.get("Bs"), "dtype", None) == torch.float8_e4m3fn else 32
            # A's smem bytes follow its LAUNCH dtype: packed E2M1 (int8/uint8) = 2 values/byte,
            # E4M3 = 1 B/value, raw bf16/fp16 (weight-only) = element_size B/value
            a_dtype = a.dtype
            if a_dtype in (torch.int8, torch.uint8):
                a_bytes = (bk // 2) * bm
            else:
                a_bytes = bk * bm * a.element_size()
            # HALF a set: the sampled lower envelope. The 2D kernel's nvfp4 arm can compile an
            # s2 request into half a buffer set (sub-tile pipelining: BN=64/128/256 x BK=512
            # all landed at set/2 + <=8 B), so the full set is not a lower bound; the half set
            # is, over all 1099 samples (largest remaining margin 68 B on a 4 KB decode tile,
            # covered by the halving). A config whose HALF set exceeds the limit cannot compile
            # in any observed form — everything else is kept and self-reports as a benign inf.
            # The family-wide keep-side model counts fp4 weights PACKED; the grouped pipeline stages
            # them at full E4M3 width (BN=256, BK=128, s5 OOMs at 253 KB = 5 x 50 KB sets, which only
            # a full-width B tile adds up to), so the whole-set bound (min_sets >= 1) uses full width.
            b_bytes = bk * tiles * bn if min_sets >= 1 else (bk // packed) * tiles * bn
            need = int(
                (a_bytes + b_bytes + (bk // group) * bm + (bk // group) * tiles * bn) * min_sets
            )
        elif (
            is_sm10x()
            and a.dtype in (torch.bfloat16, torch.float16) and b.dtype == FP8_DTYPE
            and "BLOCK_K" in args
            and not c.kwargs.get("SWAP_AB", False)
            and c.kwargs.get("PIPELINE", "k_loop") == "k_loop"
            and c.kwargs.get("A_MEMORY_MODE") == "host_descriptor"
            and c.kwargs.get("B_MEMORY_MODE") == "host_descriptor"
        ):
            # Concurrent local_alloc buffers: staged raw A/B and one decoded rhs.
            # Pointer loads can use fewer buffers; a swapped lhs can live in tmem.
            b_stages = c.num_stages if c.kwargs.get("WARP_SPEC") else c.num_stages - 1
            need = bk * (c.num_stages * a.element_size() * bm
                         + (b_stages + a.element_size()) * tiles * bn)
        else:
            # Other dot/scalar pipelines retain their raw-operand lower bound.
            need = stages * (bk * bm + (bk // packed) * tiles * bn)
        return need <= sm_shared_memory_limit()

    return config_filter(ok, when=lambda args: get_active_device_type() == "cuda")

