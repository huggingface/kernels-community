# Copyright (c) 2026, QuACK team.

import math
from functools import partial
from typing import Type

import torch
from ._ops_compat import add_op_namespace_prefix
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from . import copy_utils
from . import layout_utils
from .cache import jit_cache
from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import torch2cute_dtype_map
from .dsl import cute_op


def _next_power_of_2(n: int) -> int:
    return 1 << math.ceil(math.log2(n))


def _num_threads_for_log_n(log_n: int) -> int:
    # Same schedule as fast-hadamard-transform / TileLang for power-of-two sizes.
    return [0, 1, 1, 1, 2, 4, 8, 16, 32, 32, 32, 32, 32, 256, 256, 1024][log_n]


def _ensure_last_dim_contiguous(t: Tensor) -> Tensor:
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()


@cute.jit
def _hadamard_thread(
    vals: cute.Tensor,
    num_chunks: cutlass.Constexpr[int],
    log_n: cutlass.Constexpr[int],
) -> None:
    for step in cutlass.range_constexpr(log_n):
        stride = const_expr(1 << step)
        for j in cutlass.range_constexpr(1 << (log_n - 1)):
            lo = const_expr(j & (stride - 1))
            idx = const_expr((j - lo) * 2 + lo)
            for c in cutlass.range_constexpr(num_chunks):
                a = vals[idx, c]
                b = vals[idx + stride, c]
                vals[idx, c] = a + b
                vals[idx + stride, c] = a - b


@cute.jit
def _hadamard_warp(
    vals: cute.Tensor,
    tidx: Int32,
    num_chunks: cutlass.Constexpr[int],
    vecsize: cutlass.Constexpr[int],
    log_width: cutlass.Constexpr[int],
) -> None:
    for step in cutlass.range_constexpr(log_width):
        offset = const_expr(1 << step)
        sign_bit = tidx & offset
        sign = Float32(1.0) if sign_bit == 0 else Float32(-1.0)
        for c in cutlass.range_constexpr(num_chunks):
            for i in cutlass.range_constexpr(vecsize):
                x = vals[i, c]
                other = cute.arch.shuffle_sync_bfly(x, offset=offset)
                vals[i, c] = x * sign + other


class HadamardTransform:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        assert N >= 2, "Hadamard transform requires last dimension >= 2"
        assert N <= 32768, "Hadamard transform supports last dimension up to 32768"
        self.dtype = dtype
        self.N = N
        self.N_padded = _next_power_of_2(N)
        assert self.N_padded <= 32768, "Padded Hadamard dimension must be <= 32768"

        self.log_N = int(math.log2(self.N_padded))
        self.num_threads = _num_threads_for_log_n(self.log_N)
        max_vecsize = 4 if dtype.width == 32 else 8
        self.vecsize = min(max_vecsize, self.N_padded)
        # Row stride is the original N, so padded/non-power-of-two rows may not be
        # aligned enough for the full compute vector width in global memory.
        self.copy_vecsize = math.gcd(self.vecsize, self.N)
        assert self.vecsize & (self.vecsize - 1) == 0
        assert self.copy_vecsize & (self.copy_vecsize - 1) == 0
        assert self.N_padded % (self.num_threads * self.vecsize) == 0
        self.num_chunks = self.N_padded // (self.num_threads * self.vecsize)
        assert self.num_chunks & (self.num_chunks - 1) == 0

        self.log_vecsize = int(math.log2(self.vecsize))
        self.log_chunks = int(math.log2(self.num_chunks))
        self.warp_size = min(self.num_threads, cute.arch.WARP_SIZE)
        self.num_warps = self.num_threads // self.warp_size
        self.log_warp_size = int(math.log2(self.warp_size))
        self.log_num_warps = int(math.log2(self.num_warps))
        self.exchange_vec_elems = min(4, self.vecsize)
        assert self.num_warps == 1 or self.exchange_vec_elems == 4
        self.exchange_vecs = self.vecsize // self.exchange_vec_elems

        max_smem_bytes = 32 * 1024
        chunks_per_exchange = max_smem_bytes // (self.vecsize * self.num_threads * 4)
        self.chunks_per_exchange = min(self.num_chunks, max(chunks_per_exchange, 1))
        assert self.num_chunks % self.chunks_per_exchange == 0
        self.num_exchanges = self.num_chunks // self.chunks_per_exchange

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, self.num_threads, self.num_threads, self.copy_vecsize
        )
        tiler_mn = (1, self.N_padded)
        self.kernel(mX, mO, scale, tiler_mn, tiled_copy).launch(
            grid=[mX.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        scale: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, cX = [cute.local_tile(mT, tiler_mn, (row, 0)) for mT in (mX, mO, idX)]

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXgO = thr_copy.partition_D(gO)
        tXcX_full = thr_copy.partition_S(cX)
        tXrX = cute.make_rmem_tensor_like(tXgX)
        tXrO = cute.make_rmem_tensor_like(tXgO)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = None if is_even_N else copy_utils.predicate_k(tXcX_full, limit=shape[1])
        copy = partial(copy_utils.copy, pred=tXpX)

        tXrX.fill(tXrX.element_type.zero)
        copy(tXgX, tXrX)

        x_vals = cute.make_rmem_tensor(
            cute.make_layout(
                ((self.exchange_vec_elems, self.exchange_vecs), self.num_chunks),
                stride=((1, self.exchange_vec_elems), self.vecsize),
            ),
            Float32,
        )
        for c in cutlass.range_constexpr(self.num_chunks):
            for i in cutlass.range_constexpr(self.vecsize):
                x_vals[i, c] = tXrX[c * self.vecsize + i].to(Float32)

        _hadamard_thread(x_vals, self.num_chunks, self.log_vecsize)
        _hadamard_warp(x_vals, tidx, self.num_chunks, self.vecsize, self.log_warp_size)

        if const_expr(self.num_warps > 1):
            smem = cutlass.utils.SmemAllocator()
            # Exchange buffer layout, in Float32 elements:
            #   ((packet_elem, packet), (lane, warp), chunk)
            # with scalar strides:
            #   ((1, 4 * nthreads), (4, 4 * warp_size), 4 * packets * nthreads)
            # The innermost packet is four f32 values, so every exchange copy is a
            # 16-byte STS.128/LDS.128.  The composed swizzle preserves those low
            # two element bits and maps the thread component as:
            #   lane + 32 * warp -> (lane ^ warp) + 32 * warp
            # This is exactly the XOR previously applied to src_smem_tidx and
            # transposed_smem_tidx, but keeping it in the CuTe layout makes the
            # logical tensor indices the actual lane/warp coordinates.
            s_exchange_outer = cute.make_layout(
                (
                    (self.exchange_vec_elems, self.exchange_vecs),
                    (self.warp_size, self.num_warps),
                    self.chunks_per_exchange,
                ),
                stride=(
                    (1, self.exchange_vec_elems * self.num_threads),
                    (
                        self.exchange_vec_elems,
                        self.exchange_vec_elems * self.warp_size,
                    ),
                    self.exchange_vec_elems * self.exchange_vecs * self.num_threads,
                ),
            )
            s_exchange_layout = cute.make_composed_layout(
                cute.make_swizzle(self.log_num_warps, 2, 5),
                0,
                s_exchange_outer,
            )
            s_exchange = smem.allocate_tensor(Float32, s_exchange_layout, byte_alignment=16)
            lane_id = tidx % self.warp_size
            warp_id = tidx // self.warp_size
            transposed_lane = tidx // self.num_warps
            transposed_warp = tidx % self.num_warps
            exchange_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            copy_exch = partial(cute.copy, exchange_copy_atom)

            # First transpose: logical (lane, warp) -> (transposed_lane, transposed_warp).
            for exchange in cutlass.range_constexpr(self.num_exchanges):
                # No leading barrier is needed for exchange 0: no previous smem
                # reads can race with the first writes.
                if const_expr(exchange > 0):
                    cute.arch.barrier()
                for c in cutlass.range_constexpr(self.chunks_per_exchange):
                    chunk = const_expr(exchange * self.chunks_per_exchange + c)
                    copy_exch(x_vals[None, chunk], s_exchange[None, (lane_id, warp_id), c])
                cute.arch.barrier()
                for c in cutlass.range_constexpr(self.chunks_per_exchange):
                    chunk = const_expr(exchange * self.chunks_per_exchange + c)
                    copy_exch(
                        s_exchange[None, (transposed_lane, transposed_warp), c],
                        x_vals[None, chunk],
                    )

            _hadamard_warp(x_vals, tidx, self.num_chunks, self.vecsize, self.log_num_warps)

            # Inverse transpose: (transposed_lane, transposed_warp) -> logical
            # (lane, warp).  The leading barrier prevents a fast warp from
            # overwriting smem while another warp is still finishing the final read
            # from the first transpose.
            for exchange in cutlass.range_constexpr(self.num_exchanges):
                cute.arch.barrier()
                for c in cutlass.range_constexpr(self.chunks_per_exchange):
                    chunk = const_expr(exchange * self.chunks_per_exchange + c)
                    copy_exch(
                        x_vals[None, chunk],
                        s_exchange[None, (transposed_lane, transposed_warp), c],
                    )
                cute.arch.barrier()
                for c in cutlass.range_constexpr(self.chunks_per_exchange):
                    chunk = const_expr(exchange * self.chunks_per_exchange + c)
                    copy_exch(s_exchange[None, (lane_id, warp_id), c], x_vals[None, chunk])

        if const_expr(self.num_chunks > 1):
            # x_vals is laid out as (vec, chunk), while _hadamard_thread transforms its
            # first mode and iterates over its second. Select modes [1, 0] to get a
            # metadata-only (chunk, vec) view and reuse the same butterfly loop nest.
            _hadamard_thread(layout_utils.select(x_vals, [1, 0]), self.vecsize, self.log_chunks)

        for c in cutlass.range_constexpr(self.num_chunks):
            for i in cutlass.range_constexpr(self.vecsize):
                tXrO[c * self.vecsize + i] = (x_vals[i, c] * scale).to(tXrO.element_type)
        copy(tXrO, tXgO)


@jit_cache
def _compile_hadamard_transform_fwd(dtype, N):
    batch_sym = cute.sym_int()
    div = math.gcd(N, 128 // dtype.width)
    x_cute = fake_tensor(dtype, (batch_sym, N), div)
    out_cute = fake_tensor(dtype, (batch_sym, N), div)
    return cute.compile(
        HadamardTransform(dtype, N),
        x_cute,
        out_cute,
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cute_op(
    add_op_namespace_prefix("_hadamard_transform_fwd"),
    mutates_args={"out"},
    device_types="cuda",
)
def _hadamard_transform_fwd(x: Tensor, out: Tensor, scale: float) -> None:
    assert x.dim() == 2, "Input must be 2D"
    assert out.shape == x.shape, "Output shape must match input"
    assert x.dtype == out.dtype, "Output dtype must match input dtype"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    if x.numel() == 0:
        return
    N = x.size(1)
    assert 2 <= N <= 32768, "Hadamard transform supports last dimension in [2, 32768]"
    dtype = torch2cute_dtype_map[x.dtype]
    _compile_hadamard_transform_fwd(dtype, N)(x, out, scale)


def hadamard_transform_fwd(x: Tensor, scale: float = 1.0) -> Tensor:
    assert x.dim() >= 1, "Input must have at least one dimension"
    x = _ensure_last_dim_contiguous(x)
    N = x.size(-1)
    assert 1 <= N <= 32768, "Hadamard transform supports last dimension in [1, 32768]"
    if x.numel() == 0:
        return torch.empty_like(x)
    if N == 1:
        return x * float(scale)
    x_2d = x.reshape(-1, N)
    out_2d = torch.empty_like(x_2d)
    _hadamard_transform_fwd(x_2d, out_2d, float(scale))
    return out_2d.reshape(x.shape)


def hadamard_transform_ref(x: Tensor, scale: float = 1.0) -> Tensor:
    """PyTorch reference with the same zero-padding convention as fast-hadamard-transform."""
    assert x.dim() >= 1, "Input must have at least one dimension"
    N = x.size(-1)
    assert 1 <= N <= 32768, "Hadamard transform supports last dimension in [1, 32768]"
    N_padded = _next_power_of_2(N)
    y = x.float().reshape(-1, N)
    if N_padded != N:
        y = torch.nn.functional.pad(y, (0, N_padded - N))
    h = 1
    while h < N_padded:
        y = y.reshape(-1, N_padded // (2 * h), 2, h)
        y0 = y[:, :, 0, :]
        y1 = y[:, :, 1, :]
        y = torch.stack((y0 + y1, y0 - y1), dim=2).reshape(-1, N_padded)
        h *= 2
    return (y[:, :N] * scale).reshape(x.shape).to(x.dtype)


class HadamardTransformFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: float = 1.0):
        ctx.scale = float(scale)
        return hadamard_transform_fwd(x, ctx.scale)

    @staticmethod
    def backward(ctx, dout: Tensor):
        return hadamard_transform_fwd(dout, ctx.scale), None


def hadamard_transform(x: Tensor, scale: float = 1.0) -> Tensor:
    """Apply a Sylvester Hadamard transform along the last dimension."""
    return HadamardTransformFunction.apply(x, scale)
