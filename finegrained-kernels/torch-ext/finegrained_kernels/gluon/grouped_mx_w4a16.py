"""Persistent grouped MXFP4-weight x bf16-activation GEMM (W4A16, GPT-OSS) for the small-batch MoE regime: the
contract of ``mx_weight_only_matmul_grouped`` (MXFP4 weights, UE8M0 group-32 scales, gate|up bias + GLU, down with
scatter) in the warp roles of TRT-LLM's batched GEMM, where a cast warpgroup upcasts the weights into tensor memory.

- one CTA per SM loops over (16-row token tile, 128-row weight block) items;
- swapped tile: 128 weight rows on the tcgen05 M side (A operand from tensor memory), the token tile on N;
- loader (1 warp): NS-stage TMA ring of the packed weight words (gate|up: one strided box per projection) and x;
- two decoders (4 warps each) take alternate K steps: E2M1 -> bf16 pairs (``cvt.rn.bf16x2.e2m1x2``) times the
  E8M0 scale (exact), into an NBUF-deep tensor-memory ring;
- MMA (1 warp): accumulates each item over K into a double-buffered tensor-memory accumulator;
- epilogue (default, 4 warps): bias, GLU and the store, overlapping the next item's K loop.

64-token MoE batch, B200, CUDA graphs, op times (Triton grouped crown in brackets): GPT-OSS-120B gate_up 224 (272),
down 119 (156) us."""
from typing import List

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory,
                                                                 tcgen05_commit, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .._ops import add_op_namespace_prefix
from ..compat import compile_time_only_triton_op, compile_time_only_triton_wrap, device_context, launches_skipped
from ..epilogue import glu
from ..formats import e2m1_as_uint8, gate_out_cols, resolve_output_dtype, routed_rows, ue8m0_as_uint8
from ..scheduling import build_packed_schedule

BLOCK_N = 128  # weight rows per item
BLOCK_K = 128  # K values per step: both projections' decoded bf16 tiles fit the tensor-memory ring
BLOCK_T = 16  # token rows per item: the tcgen05 N side
_SM_COUNT = {}


@gluon.jit
def _e2m1x8_to_bf16x2(w):
    """Eight E2M1 values of a word as four bf16x2 words, bytes in order (each byte: low nibble -> low half)."""
    return gl.inline_asm_elementwise(
        "{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, $4; cvt.rn.bf16x2.e2m1x2 $0, b0; "
        "cvt.rn.bf16x2.e2m1x2 $1, b1; cvt.rn.bf16x2.e2m1x2 $2, b2; cvt.rn.bf16x2.e2m1x2 $3, b3; }",
        "=r,=r,=r,=r,r", [w], dtype=(gl.uint32,) * 4, is_pure=True, pack=1)


@gluon.jit
def _mul_bf16x2(a, s):
    """bf16x2 ``a`` times the bf16 ``s`` in both lanes (an E8M0 power of two: exact)."""
    return gl.inline_asm_elementwise(
        "{ .reg .b32 s2; mov.b32 s2, {$2, $2}; mul.rn.bf16x2 $0, $1, s2; }",
        "=r,r,h", [a, s], dtype=gl.uint32, is_pure=True, pack=1)


@gluon.jit
def _item(it, Schedule, NB: gl.constexpr):
    """Item ``it``'s (expert, first sorted row, row end, weight block) from the packed schedule."""
    m = it // NB
    entry = Schedule + 1 + m * 3
    return gl.load(entry), gl.load(entry + 1), gl.load(entry + 2), it % NB


@gluon.jit
def _loader(w_desc, x_desc, w_bufs, u_bufs, x_bufs, ready, slot_free, Schedule, n_items, k_tiles,
            NB: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, NS: gl.constexpr, GATE: gl.constexpr):
    nbytes: gl.constexpr = (2 if GATE else 1) * w_desc.block_type.nbytes + x_desc.block_type.nbytes
    c = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        expert, m_start, _, nb = _item(it, Schedule, NB)
        for k in range(k_tiles):
            s = c % NS
            mbarrier.wait(slot_free.index(s), ((c // NS) & 1) ^ 1)
            mbarrier.expect(ready.index(s), nbytes)
            if GATE:  # the stack interleaves gate (even) and up (odd) rows: one strided box each
                tma.async_load(w_desc, [expert, nb * BN, 0, k * (BK // 8)], ready.index(s), w_bufs.index(s))
                tma.async_load(w_desc, [expert, nb * BN, 1, k * (BK // 8)], ready.index(s), u_bufs.index(s))
            else:
                tma.async_load(w_desc, [expert, nb * BN, k * (BK // 8)], ready.index(s), w_bufs.index(s))
            tma.async_load(x_desc, [m_start, k * BK], ready.index(s), x_bufs.index(s))
            c += 1


@gluon.jit
def _mma(x_bufs, dt_g, dt_u, acc_g, acc_u, dec_ready, dec_free, slot_free, acc_ready, acc_free, n_items, k_tiles,
         NS: gl.constexpr, NBUF: gl.constexpr, GATE: gl.constexpr):
    c = 0
    i = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        a = i % 2
        mbarrier.wait(acc_free.index(a), ((i // 2) & 1) ^ 1)
        for k in range(k_tiles):
            s = c % NS
            d = c % NBUF
            mbarrier.wait(dec_ready.index(d), (c // NBUF) & 1)  # the decoder waited for slot s, so x is in too
            x = x_bufs.index(s).permute((1, 0))
            if GATE:
                tcgen05_mma(dt_g.index(d), x, acc_g.index(a), use_acc=k > 0)
                tcgen05_mma(dt_u.index(d), x, acc_u.index(a), use_acc=k > 0,
                            mbarriers=[dec_free.index(d), slot_free.index(s)])
            else:
                tcgen05_mma(dt_g.index(d), x, acc_g.index(a), use_acc=k > 0,
                            mbarriers=[dec_free.index(d), slot_free.index(s)])
            c += 1
        tcgen05_commit(acc_ready.index(a))
        i += 1


@gluon.jit
def _decode_half(words_buf, scale_ptrs, scale_mask, dst, BN: gl.constexpr, BK: gl.constexpr, L2: gl.constexpr,
                 word_reg: gl.constexpr, wl: gl.constexpr):
    """One projection's [BN, BK] weight tile: packed words -> bf16 x E8M0 scale, stored into tensor memory."""
    C: gl.constexpr = BK // 8
    sc = gl.load(scale_ptrs, mask=scale_mask, other=127)
    sbf = (sc.to(gl.uint16) << 7).to(gl.bfloat16, bitcast=True)
    w = words_buf.reshape([BN, C]).load(L2)
    q0, q1, q2, q3 = _e2m1x8_to_bf16x2(w)
    r = gl.join(gl.join(_mul_bf16x2(q0, sbf), _mul_bf16x2(q2, sbf)), gl.join(_mul_bf16x2(q1, sbf), _mul_bf16x2(q3, sbf)))
    dst.reinterpret(gl.uint32, [BN, BK // 2], wl).store(gl.convert_layout(gl.reshape(r, [BN, BK // 2]), word_reg))


@gluon.jit
def _decoder(w_bufs, u_bufs, dt_g, dt_u, ready, dec_ready, dec_free, S, Schedule, n_items, k_tiles, N, K,
             stride_s_e, stride_s_n, START: gl.constexpr, NB: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
             NS: gl.constexpr, NBUF: gl.constexpr, GATE: gl.constexpr):
    """Decoder ``START`` (0 or 1) takes every other K step."""
    C: gl.constexpr = BK // 8  # words per row per K step
    wl: gl.constexpr = TensorMemoryLayout([BN, BK // 2], col_stride=1)
    # the words load in the TMEM register layout of a [BN, C] tile, so a thread's words expand to its own columns
    L2: gl.constexpr = dt_g.index(0).reinterpret(gl.uint32, [BN, C], TensorMemoryLayout([BN, C], col_stride=1)).get_reg_layout()
    word_reg: gl.constexpr = dt_g.index(0).reinterpret(gl.uint32, [BN, BK // 2], wl).get_reg_layout()
    rr = gl.arange(0, BN, layout=gl.SliceLayout(1, L2))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, L2))
    base = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        expert, _, _, nb = _item(it, Schedule, NB)
        n = nb * BN + rr
        # gate|up: projection row n is stack row 2n (gate) / 2n + 1 (up), each with its own scale row
        srow = S + expert * stride_s_e + ((2 * n) if GATE else n)[:, None] * stride_s_n
        for k in range((START - base % 2 + 2) % 2, k_tiles, 2):
            c = base + k
            s = c % NS
            d = c % NBUF
            g = k * (BK // 32) + cols // 4  # one E8M0 scale per 32 values = per 4 words; past K the values are zeros
            mask = (g < K // 32)[None, :] & (n < N)[:, None]
            mbarrier.wait(ready.index(s), (c // NS) & 1)
            mbarrier.wait(dec_free.index(d), ((c // NBUF) & 1) ^ 1)
            _decode_half(w_bufs.index(s), srow + g[None, :], mask, dt_g.index(d), BN, BK, L2, word_reg, wl)
            if GATE:
                _decode_half(u_bufs.index(s), srow + stride_s_n + g[None, :], mask, dt_u.index(d), BN, BK, L2,
                             word_reg, wl)
            mbarrier.arrive(dec_ready.index(d))
        base += k_tiles


@gluon.jit
def _epilogue(acc_g, acc_u, acc_ready, acc_free, Y, Bias, ScatterIdx, Schedule, n_items, N, stride_y_t,
              stride_bias_e, NB: gl.constexpr, BN: gl.constexpr, BT: gl.constexpr, GATE: gl.constexpr,
              ACT_FN: gl.constexpr, SWIGLU_ALPHA: gl.constexpr, SWIGLU_LIMIT: gl.constexpr,
              SIMULATE_UNFUSED: gl.constexpr):
    reg: gl.constexpr = acc_g.index(0).get_reg_layout()
    tok = gl.arange(0, BT, layout=gl.SliceLayout(0, reg))
    nn = gl.arange(0, BN, layout=gl.SliceLayout(1, reg))
    i = 0
    for it in range(gl.program_id(0), n_items, gl.num_programs(0)):
        a = i % 2
        expert, m_start, row_end, nb = _item(it, Schedule, NB)
        rows = m_start + tok
        valid = rows < row_end
        n = nb * BN + nn
        n_ok = n < N
        mbarrier.wait(acc_ready.index(a), (i // 2) & 1)
        out = acc_g.index(a).load(reg)
        if GATE:
            up = acc_u.index(a).load(reg)
        mbarrier.arrive(acc_free.index(a))
        if Bias is not None:  # pre-activation, interleaved like the stack's rows under gate|up
            bias_row = Bias + expert * stride_bias_e
            if GATE:
                out += gl.load(bias_row + 2 * n, mask=n_ok, other=0.0).to(gl.float32)[:, None]
                up += gl.load(bias_row + 2 * n + 1, mask=n_ok, other=0.0).to(gl.float32)[:, None]
            else:
                out += gl.load(bias_row + n, mask=n_ok, other=0.0).to(gl.float32)[:, None]
        if GATE:
            out = glu(out, up, ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, Y.dtype.element_ty)
        if ScatterIdx is not None:
            dst = gl.load(ScatterIdx + rows, mask=valid, other=0)
        else:
            dst = rows
        gl.store(Y + dst.to(gl.int64)[None, :] * stride_y_t + n[:, None], out.to(Y.dtype.element_ty),
                 mask=valid[None, :] & n_ok[:, None])
        i += 1


@gluon.jit
def mx_w4a16_grouped_kernel(w_desc, x_desc, S, Y, Bias, ScatterIdx, Schedule, N, K, stride_s_e, stride_s_n,
                            stride_y_t, stride_bias_e, BN: gl.constexpr, BT: gl.constexpr, BK: gl.constexpr,
                            NS: gl.constexpr, NBUF: gl.constexpr, GATE: gl.constexpr, ACT_FN: gl.constexpr,
                            SWIGLU_ALPHA: gl.constexpr, SWIGLU_LIMIT: gl.constexpr, SIMULATE_UNFUSED: gl.constexpr,
                            DEC_REGS: gl.constexpr,
                            num_warps: gl.constexpr):
    NB = gl.cdiv(N, BN)
    n_items = gl.load(Schedule) * NB
    k_tiles = gl.cdiv(K, BK)
    w_bufs = gl.allocate_shared_memory(gl.int32, [NS] + w_desc.block_shape, w_desc.layout)
    u_bufs = gl.allocate_shared_memory(gl.int32, [NS if GATE else 1] + w_desc.block_shape, w_desc.layout)
    x_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, BT, BK], x_desc.layout)
    dtl: gl.constexpr = TensorMemoryLayout([BN, BK], col_stride=1)
    dt_g = allocate_tensor_memory(gl.bfloat16, [NBUF, BN, BK], dtl)
    dt_u = allocate_tensor_memory(gl.bfloat16, [NBUF if GATE else 1, BN, BK], dtl)
    accl: gl.constexpr = TensorMemoryLayout([BN, BT], col_stride=1)
    acc_g = allocate_tensor_memory(gl.float32, [2, BN, BT], accl)
    acc_u = allocate_tensor_memory(gl.float32, [2 if GATE else 1, BN, BT], accl)
    ready = mbarrier.allocate_mbarrier(batch=NS)
    slot_free = mbarrier.allocate_mbarrier(batch=NS)
    dec_ready = mbarrier.allocate_mbarrier(batch=NBUF)
    dec_free = mbarrier.allocate_mbarrier(batch=NBUF)
    acc_ready = mbarrier.allocate_mbarrier(batch=2)
    acc_free = mbarrier.allocate_mbarrier(batch=2)
    for i in gl.static_range(NS):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(slot_free.index(i), count=1)
    for i in gl.static_range(NBUF):
        mbarrier.init(dec_ready.index(i), count=1)
        mbarrier.init(dec_free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(acc_ready.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=1)
    gl.warp_specialize([
        (_epilogue, (acc_g, acc_u, acc_ready, acc_free, Y, Bias, ScatterIdx, Schedule, n_items, N, stride_y_t,
                     stride_bias_e, NB, BN, BT, GATE, ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED)),
        (_decoder, (w_bufs, u_bufs, dt_g, dt_u, ready, dec_ready, dec_free, S, Schedule, n_items, k_tiles, N, K,
                    stride_s_e, stride_s_n, 0, NB, BN, BK, NS, NBUF, GATE)),
        (_decoder, (w_bufs, u_bufs, dt_g, dt_u, ready, dec_ready, dec_free, S, Schedule, n_items, k_tiles, N, K,
                    stride_s_e, stride_s_n, 1, NB, BN, BK, NS, NBUF, GATE)),
        (_loader, (w_desc, x_desc, w_bufs, u_bufs, x_bufs, ready, slot_free, Schedule, n_items, k_tiles, NB, BN, BK, NS,
                   GATE)),
        (_mma, (x_bufs, dt_g, dt_u, acc_g, acc_u, dec_ready, dec_free, slot_free, acc_ready, acc_free, n_items,
                k_tiles, NS, NBUF, GATE)),
    ], [4, 4, 1, 1], [DEC_REGS, DEC_REGS, 24, 24])


def mx_w4a16_grouped_gluon(x, B, Bs, schedule, Y, N, gate, act_fn, swiglu_alpha, swiglu_limit, bias, scatter_idx,
                           simulate_unfused=False, num_stages=None, num_part_buffers=3, dec_regs=184):
    """Launch over expert-sorted bf16 ``x`` (S, K) and the packed E2M1 ``B`` (E, 2N|N, K // 2) with UE8M0 ``Bs``
    (E, 2N|N, K // 32); ``schedule`` is the 16-row packed schedule. Writes ``Y`` at the sorted rows, or at
    ``scatter_idx`` when given."""
    if launches_skipped():  # an opaque op's fake impl: the outputs are allocated, and descriptors need real pointers
        return
    E, rows, Kh = B.shape
    K = 2 * Kh
    if num_stages is None:  # GPT-OSS T=64, B200: gate|up best at 6 (8 runs out of headroom), down at 8 (119 vs 121 us)
        num_stages = 6 if gate else 8
    words = B.view(torch.int32)  # (E, rows, K // 8)
    # gate|up: the (E, 2N, K // 8) stack viewed (E, N, 2, K // 8), so the gate (even) and up (odd) rows are one box each
    w_view, w_box = ((words.view(E, rows // 2, 2, Kh // 4), [1, BLOCK_N, 1, BLOCK_K // 8]) if gate
                     else (words, [1, BLOCK_N, BLOCK_K // 8]))
    w_desc = TensorDescriptor.from_tensor(w_view, w_box, gl.NVMMASharedLayout.get_default_for(w_box, gl.int32))
    x_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_T, BLOCK_K], gl.bfloat16)
    x_desc = TensorDescriptor.from_tensor(x, [BLOCK_T, BLOCK_K], x_layout)
    dev = x.device.index
    if dev not in _SM_COUNT:
        _SM_COUNT[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    with device_context(x.device):
        compile_time_only_triton_wrap(mx_w4a16_grouped_kernel)[(_SM_COUNT[dev],)](
            w_desc, x_desc, Bs, Y, bias, scatter_idx, schedule, N, K, Bs.stride(0), Bs.stride(1), Y.stride(0),
            bias.stride(0) if bias is not None else 0, BN=BLOCK_N, BT=BLOCK_T, BK=BLOCK_K, NS=num_stages,
            NBUF=num_part_buffers, GATE=gate, ACT_FN=act_fn, SWIGLU_ALPHA=swiglu_alpha, SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused, DEC_REGS=dec_regs, num_warps=4,
        )


@compile_time_only_triton_op(add_op_namespace_prefix("mx_weight_only_matmul_grouped_gluon"), opaque=True)
def mx_weight_only_matmul_grouped_gluon(
    A: torch.Tensor,
    B: torch.Tensor,
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
    """``mx_weight_only_matmul_grouped`` for MXFP4 weights (UE8M0 group-32 scales, no global scales) on the persistent
    Gluon kernel. ``A`` is gathered into expert order once so every token tile is one TMA box."""
    num_experts, rows, _ = B.shape
    N = rows // 2 if gate else rows
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)
    if gather_idx is not None:
        A = A.index_select(0, gather_idx)
    output_dtype = resolve_output_dtype(output_dtype, A, None)
    Y = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
    schedule = build_packed_schedule(expert_start, S, BLOCK_T)
    mx_w4a16_grouped_gluon(A.contiguous(), e2m1_as_uint8(B), ue8m0_as_uint8(Bs), schedule, Y, N, gate, act_fn,
                           swiglu_alpha, swiglu_limit, bias, scatter_idx, simulate_unfused)
    return [Y]
