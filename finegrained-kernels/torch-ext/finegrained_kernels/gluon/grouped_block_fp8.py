"""Persistent grouped block-FP8 GEMM (128x128 weight blocks, 1x128 activation blocks) for the small-batch MoE regime:
the contract of ``w8a8_block_dynamic_fp8_matmul_grouped`` (gate|up GLU + FP8 requant, down with scatter), in the
warp roles of TRT-LLM's batched GEMM.

- one CTA per SM loops over (16-row token tile, 128-row weight block) items, so the TMA ring and the MMAs run across
  item boundaries;
- swapped tile: 128 weight rows on the tcgen05 M side, the token tile on N;
- loader (1 warp): NS-stage TMA ring of the weight tile (gate|up: the gate and up rows, one strided box each) and the
  token tile;
- MMA (1 warp): per K step a fresh fp32 partial per projection in tensor memory;
- two rescale warpgroups (4 warps each) take alternate items: each accumulates its partials times the activation x
  weight block scales in registers, then runs GLU / requant / store through the Triton epilogue helpers, while the
  other rescales the next item.

64-token MoE batch, B200, CUDA graphs, op times (Triton grouped crown in brackets): DeepSeek-V4-Base (UE8M0)
gate_up 511 (550), down 252 (285); DeepSeek-V3 (fp32 scales) gate_up 1016 (1042), down 484 (512)."""
import itertools
from typing import List

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory,
                                                                 tcgen05_commit, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .._ops import add_op_namespace_prefix
from ..compat import (FP8_DTYPE, compile_time_only_triton_op, compile_time_only_triton_wrap, device_context,
                      launches_skipped, sm_count, tl_dtype)
from ..epilogue import glu
from ..formats import (expert_weight_shape, gate_out_cols, resolve_output_dtype, routed_rows,
                       tokens_per_expert_bucket, ue8m0_as_uint8)
from ..loading.scales import decode_group_scale
from ..pruners import require_moe_dims_aligned
from ..quant import fp8_act_quant_block_dynamic, fp8_act_quant_inline
from ..bayesian_autotuner import bayesian_autotune
from .common import (TOKEN_TILES, gather_rows, grouped_pre_hook, load_token_tile, schedule_item, smem_fit_pruner,
                     token_tile_descriptor, token_tile_schedule, token_tile_schedules)

BLOCK_N = 128  # weight rows per item (the 128-row scale block)
BLOCK_K = 128  # one scale block per K step


@gluon.jit
def _decode_act_scale(scale):
    """An activation block scale as fp32. UE8M0 bytes skip ``decode_group_scale``'s 0 / 255 special cases: they come
    from our quantizers, whose amax floor (1e-12) keeps the exponent byte >= 88, and every thread decodes all the
    tile's tokens each K step, on the rescale's critical path."""
    if scale.dtype == gl.uint8:
        scale = (scale.to(gl.int32) << 23).to(gl.float32, bitcast=True)
    return scale


@gluon.jit
def _loader(w_desc, a_desc, w_bufs, u_bufs, a_bufs, ready, slot_free, GatherIdx, Schedule, n_items, k_tiles,
            NB: gl.constexpr, BN: gl.constexpr, BT: gl.constexpr, BKL: gl.constexpr, NS: gl.constexpr,
            KSUB: gl.constexpr, GATE: gl.constexpr):
    GATHER: gl.constexpr = GatherIdx is not None
    nbytes: gl.constexpr = ((2 if GATE else 1) * w_desc.block_type.nbytes
                            + (BT if GATHER else 1) * a_desc.block_type.nbytes)
    c = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        expert, m_start, row_end, nb = schedule_item(it, Schedule, NB)
        if GATHER:
            src_rows = gather_rows(GatherIdx, m_start, row_end, a_desc.shape[0], BT)
        else:
            src_rows = m_start
        for k in range(gl.cdiv(k_tiles, KSUB)):  # each stage carries KSUB 128-wide K steps
            s = c % NS
            mbarrier.wait(slot_free.index(s), ((c // NS) & 1) ^ 1)
            mbarrier.expect(ready.index(s), nbytes)
            if GATE:  # the stack interleaves gate (even) and up (odd) rows: one strided box each
                tma.async_load(w_desc, [expert, nb * BN, 0, k * BKL], ready.index(s), w_bufs.index(s))
                tma.async_load(w_desc, [expert, nb * BN, 1, k * BKL], ready.index(s), u_bufs.index(s))
            else:
                tma.async_load(w_desc, [expert, nb * BN, k * BKL], ready.index(s), w_bufs.index(s))
            load_token_tile(a_desc, src_rows, m_start, k * BKL, ready.index(s), a_bufs.index(s), GATHER)
            c += 1


@gluon.jit
def _mma(w_bufs, u_bufs, a_bufs, gate_parts, up_parts, ready, slot_free, part_ready, part_free, n_items, k_tiles,
         BN: gl.constexpr, BK: gl.constexpr, NS: gl.constexpr, NBUF: gl.constexpr, KSUB: gl.constexpr,
         GATE: gl.constexpr):
    c = 0
    i = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        # items alternate between the two rescale warpgroups, each with its own NBUF partials
        base = (i % 2) * NBUF
        for kl in range(gl.cdiv(k_tiles, KSUB)):
            s = c % NS
            mbarrier.wait(ready.index(s), (c // NS) & 1)
            w_all = w_bufs.index(s).reshape([BN, KSUB * BK])
            u_all = u_bufs.index(s).reshape([BN, KSUB * BK])
            for sub in gl.static_range(KSUB):  # one fresh partial per 128-wide scale block of the stage
                k = kl * KSUB + sub
                if k < k_tiles:
                    cw = (i // 2) * k_tiles + k  # the owning warpgroup's step count
                    p = base + cw % NBUF
                    mbarrier.wait(part_free.index(p), ((cw // NBUF) & 1) ^ 1)
                    a = a_bufs.index(s).slice(sub * BK, BK, dim=1).permute((1, 0))
                    if GATE:  # one barrier covers both: tcgen05 completions are ordered
                        tcgen05_mma(w_all.slice(sub * BK, BK, dim=1), a, gate_parts.index(p), use_acc=False)
                        tcgen05_mma(u_all.slice(sub * BK, BK, dim=1), a, up_parts.index(p), use_acc=False,
                                    mbarriers=[part_ready.index(p)])
                    else:
                        tcgen05_mma(w_all.slice(sub * BK, BK, dim=1), a, gate_parts.index(p), use_acc=False,
                                    mbarriers=[part_ready.index(p)])
            tcgen05_commit(slot_free.index(s))  # the stage's MMAs all read it
            c += 1
        i += 1


@gluon.jit
def _rescale_epilogue(gate_parts, up_parts, part_ready, part_free, As, Bs, C, Cs, GatherIdx, ScatterIdx, Schedule,
                      n_items,
                      k_tiles, stride_as_m, stride_bs_e, stride_bs_n, stride_c_m, stride_cs_m,
                      NB: gl.constexpr, BN: gl.constexpr, BT: gl.constexpr, NBUF: gl.constexpr, GATE: gl.constexpr,
                      ACT_FN: gl.constexpr, SWIGLU_ALPHA: gl.constexpr, SWIGLU_LIMIT: gl.constexpr,
                      SIMULATE_UNFUSED: gl.constexpr, INTERMEDIATE_DTYPE: gl.constexpr, REQUANT: gl.constexpr,
                      WG: gl.constexpr):
    """Warpgroup ``WG`` (0 or 1) takes every other item of the CTA, so one item's GLU / requant / store overlaps the
    other warpgroup's rescale of the next item."""
    reg: gl.constexpr = gate_parts.index(0).get_reg_layout()
    tok = gl.arange(0, BT, layout=gl.SliceLayout(0, reg))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(1, reg))
    c = 0
    for it in range(gl.program_id(0) + WG * gl.num_programs(0), n_items, 2 * gl.num_programs(0)):
        expert, m_start, row_end, nb = schedule_item(it, Schedule, NB)
        rows = m_start + tok
        valid = rows < row_end
        if GatherIdx is not None:  # gathered tiles read their activation scales at the source rows
            a_row = As + gl.load(GatherIdx + rows, mask=valid, other=0) * stride_as_m
        else:
            a_row = As + rows * stride_as_m
        # gate|up: the item's (gate, up) scale blocks are the adjacent pair at 2 * nb (the stack's scales pair the
        # projections' blocks, its rows interleave them)
        w_row = Bs + expert * stride_bs_e + (2 * nb if GATE else nb) * stride_bs_n
        acc_g = gl.zeros([BN, BT], gl.float32, reg)
        acc_u = gl.zeros([BN, BT], gl.float32, reg)
        # the scales are loaded one K step ahead: a load issued after the previous step's barrier left its latency
        # (~300 ns) on every step (UE8M0 down, DeepSeek-V4-Base T=64, B200: 477 -> see the module docstring)
        a_next = gl.load(a_row, mask=valid, other=0)
        w_next = gl.load(w_row)
        u_next = gl.load(w_row + stride_bs_n) if GATE else w_next
        for k in range(k_tiles):
            p = WG * NBUF + c % NBUF
            a_raw, w_raw, u_raw = a_next, w_next, u_next
            kn = gl.minimum(k + 1, k_tiles - 1)
            a_next = gl.load(a_row + kn, mask=valid, other=0)
            w_next = gl.load(w_row + kn)
            if GATE:
                u_next = gl.load(w_row + stride_bs_n + kn)
            a_s = _decode_act_scale(a_raw)
            w_s = decode_group_scale(w_raw)
            mbarrier.wait(part_ready.index(p), (c // NBUF) & 1)
            if GATE:  # the Triton gate|up arm's order: weight scale, then an fma with the activation scale
                u_s = decode_group_scale(u_raw)
                acc_g = gl.fma(gate_parts.index(p).load(reg) * w_s, a_s[None, :], acc_g)
                acc_u = gl.fma(up_parts.index(p).load(reg) * u_s, a_s[None, :], acc_u)
            else:  # one weight scale per tile, folded into the activation scale; one rounding, as the Triton arm
                acc_g = gl.fma(gate_parts.index(p).load(reg), (a_s * w_s)[None, :], acc_g)
            mbarrier.arrive(part_free.index(p))
            c += 1
        out = glu(acc_g, acc_u, ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE) if GATE else acc_g
        if ScatterIdx is not None:
            dst = gl.load(ScatterIdx + rows, mask=valid, other=0)
        else:
            dst = rows
        c_ptrs = C + dst.to(gl.int64)[None, :] * stride_c_m + (nb * BN + nn)[:, None]
        if REQUANT:  # per-(row, 128-column block) FP8 scale over the tile's 128 rows, as the Triton epilogue
            q, q_s = fp8_act_quant_inline(out, TRANSPOSED=True, UE8M0=Cs.dtype.element_ty == gl.uint8)
            gl.store(c_ptrs, q, mask=valid[None, :])
            gl.store(Cs + dst * stride_cs_m + nb, q_s, mask=valid)
        else:
            gl.store(c_ptrs, out.to(C.dtype.element_ty), mask=valid[None, :])


def _configs():
    """Token tile x K width per load x TMA ring depth x partials per rescale warpgroup x its register budget. A
    256-wide load (two 128-wide scale blocks) streams each weight row in 256-byte pieces."""
    return [
        triton.Config(dict(BT=bt, BKL=bkl, NS=ns, NBUF=nbuf, RESCALE_REGS=regs), num_warps=4,
                      pre_hook=grouped_pre_hook("a_desc", "w_desc", k_key="BKL"))
        for bt, bkl, ns, nbuf, regs in itertools.product(TOKEN_TILES, (128, 256), (2, 3, 4, 5, 6, 8), (2, 3, 4),
                                                         (152, 192, 232))
    ]


def _stage_bytes(kwargs, gate):
    """One ring stage: the E4M3 weight rows (both projections under gate|up) and the token tile."""
    return ((2 if gate else 1) * BLOCK_N + kwargs["BT"]) * kwargs["BKL"]


@bayesian_autotune(
    _configs(),
    # BN / BK are the 128x128 quant block (launch-pinned); the token bucket keys the tile choice
    ["N", "K", "GATE", "tokens_per_expert_bit_length"],
    n_trials=24,
    prune_configs_by={"early_config_prune": smem_fit_pruner(_stage_bytes)},
)
@gluon.jit
def block_fp8_grouped_kernel(w_desc, a_desc, As, Bs, C, Cs, GatherIdx, ScatterIdx, Schedules, schedule_stride, N, K,
                             stride_as_m, stride_bs_e, stride_bs_n, stride_c_m, stride_cs_m,
                             tokens_per_expert_bit_length, BN: gl.constexpr, BT: gl.constexpr,
                             BK: gl.constexpr, BKL: gl.constexpr, NS: gl.constexpr, NBUF: gl.constexpr,
                             GATE: gl.constexpr,
                             ACT_FN: gl.constexpr, SWIGLU_ALPHA: gl.constexpr, SWIGLU_LIMIT: gl.constexpr,
                             SIMULATE_UNFUSED: gl.constexpr, INTERMEDIATE_DTYPE: gl.constexpr, REQUANT: gl.constexpr,
                             RESCALE_REGS: gl.constexpr, num_warps: gl.constexpr):
    Schedule = token_tile_schedule(Schedules, schedule_stride, BT)
    NB = gl.cdiv(N, BN)
    n_items = gl.load(Schedule) * NB
    k_tiles = gl.cdiv(K, BK)
    w_bufs = gl.allocate_shared_memory(gl.float8e4nv, [NS] + w_desc.block_shape, w_desc.layout)
    u_bufs = gl.allocate_shared_memory(gl.float8e4nv, [NS if GATE else 1] + w_desc.block_shape, w_desc.layout)
    KSUB: gl.constexpr = BKL // BK  # 128-wide scale blocks per load
    a_bufs = gl.allocate_shared_memory(gl.float8e4nv, [NS, BT, BKL], a_desc.layout)
    tmem: gl.constexpr = TensorMemoryLayout([BN, BT], col_stride=1)
    gate_parts = allocate_tensor_memory(gl.float32, [2 * NBUF, BN, BT], tmem)
    up_parts = allocate_tensor_memory(gl.float32, [2 * NBUF if GATE else 1, BN, BT], tmem)
    ready = mbarrier.allocate_mbarrier(batch=NS)
    slot_free = mbarrier.allocate_mbarrier(batch=NS)
    part_ready = mbarrier.allocate_mbarrier(batch=2 * NBUF)
    part_free = mbarrier.allocate_mbarrier(batch=2 * NBUF)
    for i in gl.static_range(NS):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(slot_free.index(i), count=1)
    for i in gl.static_range(2 * NBUF):
        mbarrier.init(part_ready.index(i), count=1)
        mbarrier.init(part_free.index(i), count=1)
    gl.warp_specialize([
        (_rescale_epilogue, (gate_parts, up_parts, part_ready, part_free, As, Bs, C, Cs, GatherIdx, ScatterIdx, Schedule,
                             n_items,
                             k_tiles, stride_as_m, stride_bs_e, stride_bs_n, stride_c_m, stride_cs_m, NB, BN, BT, NBUF,
                             GATE, ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE, REQUANT,
                             0)),
        (_rescale_epilogue, (gate_parts, up_parts, part_ready, part_free, As, Bs, C, Cs, GatherIdx, ScatterIdx, Schedule,
                             n_items,
                             k_tiles, stride_as_m, stride_bs_e, stride_bs_n, stride_c_m, stride_cs_m, NB, BN, BT, NBUF,
                             GATE, ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE, REQUANT,
                             1)),
        (_loader, (w_desc, a_desc, w_bufs, u_bufs, a_bufs, ready, slot_free, GatherIdx, Schedule, n_items, k_tiles, NB,
                   BN, BT, BKL, NS, KSUB, GATE)),
        (_mma, (w_bufs, u_bufs, a_bufs, gate_parts, up_parts, ready, slot_free, part_ready, part_free, n_items, k_tiles,
                BN, BK, NS, NBUF, KSUB, GATE)),
    ], [4, 1, 1], [RESCALE_REGS, 24, 24])


def block_fp8_grouped_gluon(A, As, B, Bs, expert_start, C, Cs, N, gate, act_fn, swiglu_alpha, swiglu_limit,
                            gather_idx, scatter_idx, simulate_unfused=False, intermediate_dtype=torch.bfloat16):
    """Launch over E4M3 ``A`` + ``As`` (rows, K // 128), expert-sorted or read through ``gather_idx``, and the ``B``
    (E, 2N|N, K) stack with ``Bs`` (E, 2N|N // 128, K // 128). Writes ``C`` (and ``Cs`` when not None, the
    requantized intermediate's scales) at the sorted rows, or at ``scatter_idx`` when given."""
    if launches_skipped():  # an opaque op's fake impl: the outputs are allocated, and descriptors need real pointers
        return
    E, rows, K = B.shape
    S = C.shape[0]
    # gate|up: the (E, 2N, K) stack viewed (E, N, 2, K), so the gate (even) and up (odd) rows are one box each
    w_view, w_box = (B.view(E, rows // 2, 2, K), [1, BLOCK_N, 1, BLOCK_K]) if gate else (B, [1, BLOCK_N, BLOCK_K])
    w_layout = gl.NVMMASharedLayout.get_default_for(w_box, gl.float8e4nv)
    w_desc = TensorDescriptor.from_tensor(w_view, w_box, w_layout)
    a_desc = token_tile_descriptor(A, min(TOKEN_TILES), BLOCK_K, gather_idx is not None)  # rebound per config
    schedules, schedule_stride = token_tile_schedules(expert_start, S)
    with device_context(A.device):
        compile_time_only_triton_wrap(block_fp8_grouped_kernel)[(sm_count(A.device.index),)](
            w_desc, a_desc, As, Bs, C, Cs, gather_idx, scatter_idx, schedules, schedule_stride, N, K, As.stride(0),
            Bs.stride(0), Bs.stride(1), C.stride(0), Cs.stride(0) if Cs is not None else 0,
            tokens_per_expert_bucket(S, E), BN=BLOCK_N, BK=BLOCK_K, GATE=gate, ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha, SWIGLU_LIMIT=swiglu_limit, SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(intermediate_dtype), REQUANT=Cs is not None,
        )


@compile_time_only_triton_op(add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_grouped_gluon"), opaque=True)
def w8a8_block_dynamic_fp8_matmul_grouped_gluon(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
    gate: bool = False,
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    quantize_output: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """``w8a8_block_dynamic_fp8_matmul_grouped`` (128x128 blocks, no bias) on the persistent Gluon kernel. Raw ``A`` is
    quantized once in token order; the kernel gathers each tile's rows with TMA."""
    num_experts, n_rows, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)
    K = A.shape[1]
    require_moe_dims_aligned(N, K, BLOCK_N, BLOCK_K)
    bs_u8 = ue8m0_as_uint8(Bs)
    if As is None:
        A, As = fp8_act_quant_block_dynamic(A, BLOCK_K, use_ue8m0=bs_u8.dtype == torch.uint8)
    output_dtype = resolve_output_dtype(output_dtype, A, As)
    if quantize_output:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        Cs = torch.empty(S, N // BLOCK_N, device=A.device, dtype=bs_u8.dtype)
    else:
        C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
        Cs = None
    block_fp8_grouped_gluon(A, As, B, bs_u8, expert_start, C, Cs, N, gate, act_fn, swiglu_alpha, swiglu_limit, gather_idx,
                            scatter_idx, simulate_unfused, output_dtype)
    return [C, Cs] if quantize_output else [C]
