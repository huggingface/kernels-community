"""Persistent grouped tensor-scale FP8 GEMM (per-tensor / row-wise weight scales; per-token or calibrated activation
scales) for the small-batch MoE regime: the contract of ``w8a8_tensor_dynamic_fp8_matmul_grouped`` (gate|up GLU,
down with scatter) in the warp roles of TRT-LLM's batched GEMM.

- one CTA per SM loops over (16-row token tile, 128-row weight block) items;
- swapped tile: 128 weight rows on the tcgen05 M side, the token tile on N;
- loader (1 warp): NS-stage TMA ring of the weight tile (gate|up: one strided box per projection) and the token tile;
- MMA (1 warp): accumulates each item over K into a double-buffered tensor-memory accumulator per projection;
- epilogue (default, 4 warps): activation x weight scales, bias, GLU and the store, overlapping the next item's K
  loop.

64-token MoE batch, B200, CUDA graphs, op times (Triton grouped crown in brackets): Mistral-4 static per-tensor
gate_up 287 (303), down 142 (155) us."""
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
from ..formats import (expert_weight_shape, gate_out_cols, normalize_per_expert_scale, resolve_output_dtype,
                       routed_rows, row_scale_stride, tokens_per_expert_bucket)
from ..quant import quantize_routed_rows_per_expert, tensor_wide_act_operands
from ..bayesian_autotuner import bayesian_autotune
from ..scheduling import packed_schedule_buffer
from .common import (TOKEN_TILES, gather_rows, grouped_pre_hook, load_token_tile, schedule_item, smem_fit_pruner,
                     token_tile_descriptor)

BLOCK_N = 128  # weight rows per item
BLOCK_K = 128  # the launch's K step; the tuned configs set their own (BK)


@gluon.jit
def _loader(w_desc, a_desc, w_bufs, u_bufs, a_bufs, ready, slot_free, GatherIdx, Schedule, n_items, k_tiles,
            NB: gl.constexpr, BN: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr, NS: gl.constexpr,
            GATE: gl.constexpr):
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
        for k in range(k_tiles):
            s = c % NS
            mbarrier.wait(slot_free.index(s), ((c // NS) & 1) ^ 1)
            mbarrier.expect(ready.index(s), nbytes)
            if GATE:  # the stack interleaves gate (even) and up (odd) rows: one strided box each
                tma.async_load(w_desc, [expert, nb * BN, 0, k * BK], ready.index(s), w_bufs.index(s))
                tma.async_load(w_desc, [expert, nb * BN, 1, k * BK], ready.index(s), u_bufs.index(s))
            else:
                tma.async_load(w_desc, [expert, nb * BN, k * BK], ready.index(s), w_bufs.index(s))
            load_token_tile(a_desc, src_rows, m_start, k * BK, ready.index(s), a_bufs.index(s), GATHER)
            c += 1


@gluon.jit
def _mma(w_bufs, u_bufs, a_bufs, acc_g, acc_u, ready, slot_free, acc_ready, acc_free, n_items, k_tiles,
         BN: gl.constexpr, BK: gl.constexpr, NS: gl.constexpr, GATE: gl.constexpr):
    c = 0
    i = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        a = i % 2
        mbarrier.wait(acc_free.index(a), ((i // 2) & 1) ^ 1)
        for k in range(k_tiles):
            s = c % NS
            mbarrier.wait(ready.index(s), (c // NS) & 1)
            x = a_bufs.index(s).permute((1, 0))
            if GATE:  # one barrier covers both: tcgen05 completions are ordered
                tcgen05_mma(w_bufs.index(s).reshape([BN, BK]), x, acc_g.index(a), use_acc=k > 0)
                tcgen05_mma(u_bufs.index(s).reshape([BN, BK]), x, acc_u.index(a), use_acc=k > 0,
                            mbarriers=[slot_free.index(s)])
            else:
                tcgen05_mma(w_bufs.index(s).reshape([BN, BK]), x, acc_g.index(a), use_acc=k > 0,
                            mbarriers=[slot_free.index(s)])
            c += 1
        tcgen05_commit(acc_ready.index(a))
        i += 1


@gluon.jit
def _epilogue(acc_g, acc_u, acc_ready, acc_free, As, Bs, Y, Bias, GatherIdx, ScatterIdx, Schedule, n_items, N, stride_as_m,
              stride_as_e, stride_bs_e, stride_bs_n, stride_y_t, stride_bias_e, NB: gl.constexpr, BN: gl.constexpr,
              BT: gl.constexpr, GATE: gl.constexpr, ACT_FN: gl.constexpr, SWIGLU_ALPHA: gl.constexpr,
              SWIGLU_LIMIT: gl.constexpr, SIMULATE_UNFUSED: gl.constexpr, INTERMEDIATE_DTYPE: gl.constexpr):
    reg: gl.constexpr = acc_g.index(0).get_reg_layout()
    tok = gl.arange(0, BT, layout=gl.SliceLayout(0, reg))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(1, reg))
    i = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        a = i % 2
        expert, m_start, row_end, nb = schedule_item(it, Schedule, NB)
        rows = m_start + tok
        valid = rows < row_end
        n = nb * BN + nn
        # per token (at the source row of a gathered tile), or the item's expert under a calibrated scale
        if GatherIdx is not None:
            a_rows = gl.load(GatherIdx + rows, mask=valid, other=0)
        else:
            a_rows = rows
        a_s = gl.load(As + a_rows * stride_as_m + expert * stride_as_e, mask=valid, other=1.0)
        w_row = Bs + expert * stride_bs_e
        mbarrier.wait(acc_ready.index(a), (i // 2) & 1)
        # the Triton epilogue's order: activation scale, then the weight row's (stack row 2n / 2n + 1 under gate|up)
        out = acc_g.index(a).load(reg) * a_s[None, :]
        if GATE:
            up = acc_u.index(a).load(reg) * a_s[None, :]
        mbarrier.arrive(acc_free.index(a))
        if GATE:
            out = out * gl.load(w_row + 2 * n * stride_bs_n)[:, None]
            up = up * gl.load(w_row + (2 * n + 1) * stride_bs_n)[:, None]
        else:
            out = out * gl.load(w_row + n * stride_bs_n)[:, None]
        if Bias is not None:  # pre-activation, interleaved like the stack's rows under gate|up
            bias_row = Bias + expert * stride_bias_e
            if GATE:
                out += gl.load(bias_row + 2 * n).to(gl.float32)[:, None]
                up += gl.load(bias_row + 2 * n + 1).to(gl.float32)[:, None]
            else:
                out += gl.load(bias_row + n).to(gl.float32)[:, None]
        if GATE:
            out = glu(out, up, ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE)
        if ScatterIdx is not None:
            dst = gl.load(ScatterIdx + rows, mask=valid, other=0)
        else:
            dst = rows
        gl.store(Y + dst.to(gl.int64)[None, :] * stride_y_t + n[:, None], out.to(Y.dtype.element_ty),
                 mask=valid[None, :] & (n < N)[:, None])
        i += 1


def _configs():
    """Token tile x K step x TMA ring depth. A 256-wide K step streams each weight row in 256-byte pieces (Mistral-4
    T=64 gate|up, B200: 285 -> 272 us against 128)."""
    return [triton.Config(dict(BT=bt, BK=bk, NS=ns), num_warps=4, pre_hook=grouped_pre_hook("a_desc", "w_desc"))
            for bt, bk, ns in itertools.product(TOKEN_TILES, (128, 256), (2, 3, 4, 5, 6, 8))]


def _stage_bytes(kwargs, gate):
    """One ring stage: the E4M3 weight rows (both projections under gate|up) and the token tile."""
    return ((2 if gate else 1) * BLOCK_N + kwargs["BT"]) * kwargs["BK"]


@bayesian_autotune(
    _configs(),
    ["N", "K", "GATE", "tokens_per_expert_bit_length"],
    n_trials=24,
    prune_configs_by={"early_config_prune": smem_fit_pruner(_stage_bytes)},
)
@gluon.jit
def tensor_fp8_grouped_kernel(w_desc, a_desc, As, Bs, Y, Bias, GatherIdx, ScatterIdx, Schedule, ExpertStart, N, K,
                              stride_as_m, stride_as_e, stride_bs_e, stride_bs_n, stride_y_t, stride_bias_e,
                              tokens_per_expert_bit_length, BN: gl.constexpr,
                              BT: gl.constexpr, BK: gl.constexpr, NS: gl.constexpr, GATE: gl.constexpr,
                              ACT_FN: gl.constexpr, SWIGLU_ALPHA: gl.constexpr, SWIGLU_LIMIT: gl.constexpr,
                              SIMULATE_UNFUSED: gl.constexpr, INTERMEDIATE_DTYPE: gl.constexpr,
                              num_warps: gl.constexpr):
    NB = gl.cdiv(N, BN)
    n_items = gl.load(Schedule) * NB
    k_tiles = gl.cdiv(K, BK)
    w_bufs = gl.allocate_shared_memory(gl.float8e4nv, [NS] + w_desc.block_shape, w_desc.layout)
    u_bufs = gl.allocate_shared_memory(gl.float8e4nv, [NS if GATE else 1] + w_desc.block_shape, w_desc.layout)
    a_bufs = gl.allocate_shared_memory(gl.float8e4nv, [NS, BT, BK], a_desc.layout)
    accl: gl.constexpr = TensorMemoryLayout([BN, BT], col_stride=1)
    acc_g = allocate_tensor_memory(gl.float32, [2, BN, BT], accl)
    acc_u = allocate_tensor_memory(gl.float32, [2 if GATE else 1, BN, BT], accl)
    ready = mbarrier.allocate_mbarrier(batch=NS)
    slot_free = mbarrier.allocate_mbarrier(batch=NS)
    acc_ready = mbarrier.allocate_mbarrier(batch=2)
    acc_free = mbarrier.allocate_mbarrier(batch=2)
    for i in gl.static_range(NS):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(slot_free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(acc_ready.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=1)
    gl.warp_specialize([
        (_epilogue, (acc_g, acc_u, acc_ready, acc_free, As, Bs, Y, Bias, GatherIdx, ScatterIdx, Schedule, n_items, N, stride_as_m,
                     stride_as_e, stride_bs_e, stride_bs_n, stride_y_t, stride_bias_e, NB, BN, BT, GATE, ACT_FN,
                     SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE)),
        (_loader, (w_desc, a_desc, w_bufs, u_bufs, a_bufs, ready, slot_free, GatherIdx, Schedule, n_items, k_tiles, NB,
                   BN, BT, BK, NS, GATE)),
        (_mma, (w_bufs, u_bufs, a_bufs, acc_g, acc_u, ready, slot_free, acc_ready, acc_free, n_items, k_tiles, BN, BK,
                NS, GATE)),
    ], [1, 1], [24, 24])


def tensor_fp8_grouped_gluon(A, As, stride_as_m, stride_as_e, B, Bs, expert_start, Y, N, gate, act_fn, swiglu_alpha,
                             swiglu_limit, bias, gather_idx, scatter_idx, simulate_unfused=False):
    """Launch over E4M3 ``A`` (rows, K), expert-sorted or read through ``gather_idx``, with its activation scales
    ``As`` (strides per row / per expert), the ``B`` (E, 2N|N, K) stack and its ``(E, rows, 1)`` weight scales.
    Writes ``Y`` at the sorted rows, or at ``scatter_idx`` when given."""
    if launches_skipped():  # an opaque op's fake impl: the outputs are allocated, and descriptors need real pointers
        return
    E, rows, K = B.shape
    S = Y.shape[0]
    w_view, w_box = (B.view(E, rows // 2, 2, K), [1, BLOCK_N, 1, BLOCK_K]) if gate else (B, [1, BLOCK_N, BLOCK_K])
    w_desc = TensorDescriptor.from_tensor(w_view, w_box, gl.NVMMASharedLayout.get_default_for(w_box, gl.float8e4nv))
    a_desc = token_tile_descriptor(A, min(TOKEN_TILES), BLOCK_K, gather_idx is not None)  # rebound per config
    schedule = packed_schedule_buffer(S, E, A.device, min(TOKEN_TILES))  # filled per config
    with device_context(A.device):
        compile_time_only_triton_wrap(tensor_fp8_grouped_kernel)[(sm_count(A.device.index),)](
            w_desc, a_desc, As, Bs, Y, bias, gather_idx, scatter_idx, schedule, expert_start, N, K, stride_as_m,
            stride_as_e, Bs.stride(0), row_scale_stride(Bs), Y.stride(0), bias.stride(0) if bias is not None else 0,
            tokens_per_expert_bucket(S, E), BN=BLOCK_N, GATE=gate, ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha, SWIGLU_LIMIT=swiglu_limit, SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(Y.dtype),
        )


@compile_time_only_triton_op(add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_grouped_gluon"), opaque=True)
def w8a8_tensor_dynamic_fp8_matmul_grouped_gluon(
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
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """``w8a8_tensor_dynamic_fp8_matmul_grouped`` on the persistent Gluon kernel. Activations quantized in token order
    (per token, or against one shared calibrated scale) are gathered per tile with TMA; a per-expert calibrated scale
    lays the routed rows out quantized and expert-sorted first."""
    num_experts, _, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)
    Bs = normalize_per_expert_scale(Bs, num_experts)
    output_dtype = resolve_output_dtype(output_dtype, A, As)
    A, As, stride_as_m, stride_as_e = tensor_wide_act_operands(A, As, num_experts)
    if A.dtype != FP8_DTYPE:  # a per-expert calibrated scale on raw rows
        A = quantize_routed_rows_per_expert(A, As, gather_idx, expert_start, num_experts)
        gather_idx = None
    Y = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
    tensor_fp8_grouped_gluon(A, As, stride_as_m, stride_as_e, B, Bs, expert_start, Y, N, gate, act_fn, swiglu_alpha,
                             swiglu_limit, bias, gather_idx, scatter_idx, simulate_unfused)
    return [Y]
