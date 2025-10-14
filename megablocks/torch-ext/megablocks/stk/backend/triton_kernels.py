import torch
import triton
import triton.language as tl
from dataclasses import dataclass

@dataclass
class TritonConfig:
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    BLOCK_K: int = 32
    BLOCK_SIZE: int = 128
    NUM_STAGES: int = 4
    NUM_WARPS: int = 4

def _validate_matmul_dims(M: int, K: int, N: int):
    error_string = "incompatible dimensions: tensor has dim with length: {}, which must be divisible by {}"
    assert M % TritonConfig.BLOCK_M == 0, error_string.format(M, TritonConfig.BLOCK_M)
    assert K % TritonConfig.BLOCK_K == 0, error_string.format(K, TritonConfig.BLOCK_K)
    assert N % TritonConfig.BLOCK_N == 0, error_string.format(N, TritonConfig.BLOCK_N)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({
            'BLOCK_M': TritonConfig.BLOCK_M,
            'BLOCK_N': TritonConfig.BLOCK_N,
            'BLOCK_K': TritonConfig.BLOCK_K,
            'BLOCK_SIZE': TritonConfig.BLOCK_SIZE
        }, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _sdd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_m = tl.load(row_indices + pid)
    pid_n = tl.load(column_indices + pid)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    #Store to sparse matrix
    acc = acc.to(C.dtype.element_ty)
    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    cm = tl.arange(0, BLOCK_M)
    cn = tl.arange(0, BLOCK_N)
    C = C + pid * BLOCK_ELEMENTS + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({
            'BLOCK_M': TritonConfig.BLOCK_M,
            'BLOCK_N': TritonConfig.BLOCK_N,
            'BLOCK_K': TritonConfig.BLOCK_K,
            'BLOCK_SIZE': TritonConfig.BLOCK_SIZE
        }, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dsd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):

    # matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)

    start_inx = tl.load(offsets + pid_m)
    end_inx = tl.load(offsets + pid_m + 1)

    # pointers to sparse matrix
    rm =  tl.arange(0, BLOCK_M)
    rak = tl.arange(0, BLOCK_K)

    A += (rm[:, None] * stride_am + rak[None, :] * stride_ak)

    # pointers to dense matrix
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)
    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    ak_sub_incr = BLOCK_K * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk
    bk_block_incr = BLOCK_SIZE * stride_bk

    for k in range(nsub_blocks * (end_inx - start_inx)):
        sub_block_inx = k % nsub_blocks
        block_inx = k // nsub_blocks

        if trans_A:
            ptr_A = A + tl.load(block_offsets_t + start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * ak_sub_incr
        else:
            ptr_A = A + (start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * ak_sub_incr

        ptr_B = B + tl.load(column_indices + start_inx + block_inx) * bk_block_incr + sub_block_inx * bk_sub_incr

        a = tl.load(ptr_A)
        b = tl.load(ptr_B)
        acc += tl.dot(a, b)

    acc = acc.to(C.dtype.element_ty)

    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({
            'BLOCK_M': TritonConfig.BLOCK_M,
            'BLOCK_N': TritonConfig.BLOCK_N,
            'BLOCK_K': TritonConfig.BLOCK_K,
            'BLOCK_SIZE': TritonConfig.BLOCK_SIZE
        }, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dds_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):

    # matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)

    start_inx = tl.load(offsets + pid_n)
    end_inx = tl.load(offsets + pid_n + 1)

    # pointers to dense matrix
    rm =  pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rak = tl.arange(0, BLOCK_K)

    A += (rm[:, None] * stride_am + rak[None, :] * stride_ak)

    # pointers to sparse matrix
    rn = tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)
    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE

    ak_sub_incr = BLOCK_K * stride_ak
    ak_block_incr = BLOCK_SIZE * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk

    for k in range(nsub_blocks * (end_inx - start_inx)):
        sub_block_inx = k % nsub_blocks
        block_inx = k // nsub_blocks

        if trans_B:
            ptr_B = B + (start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr
        else:
            ptr_B = B + tl.load(block_offsets_t + start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr

        ptr_A = A + tl.load(column_indices + start_inx + block_inx) * ak_block_incr + sub_block_inx * ak_sub_incr
        a = tl.load(ptr_A)
        b = tl.load(ptr_B)
        acc += tl.dot(a, b)

    acc = acc.to(C.dtype.element_ty)
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

def dsd(shape,
        data,
        offsets,
        row_indices,
        column_indices,
        offsets_t,
        column_indices_t,
        block_offsets_t,
        transpose_a,
        rhs,
        out
    ):

    device = rhs.device
    trans_A = transpose_a
    trans_B = False

    if rhs.stride(0) > 1 and rhs.stride(1) > 1:
        trans_B = True

    # checks constraints
    assert shape[1] == rhs.shape[0], "incompatible dimensions"
    M, K = shape
    _, N = rhs.shape

    _validate_matmul_dims(M, K, N)

    # accumulator types
    ACC_TYPE = tl.float32 if rhs.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

    stride_am, stride_ak = data.stride(1), data.stride(2)
    stride_bk, stride_bn = rhs.stride(0), rhs.stride(1)
    a_column_indices  = column_indices
    a_offsets = offsets

    # launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    if trans_A:
        stride_am, stride_ak = data.stride(2), data.stride(1)
        a_column_indices, a_offsets = column_indices_t, offsets_t

    if trans_B:
        stride_bk, stride_bn = rhs.stride(1), rhs.stride(0)

    _dsd_kernel[grid](
        data.data, rhs, out, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        row_indices, a_column_indices, a_offsets,
        block_offsets_t, trans_A, trans_B,
        GROUP_M=128, ACC_TYPE=ACC_TYPE
    )
    # return out

def dds(lhs,
        shape,
        data,
        offsets,
        row_indices,
        column_indices,
        offsets_t,
        column_indices_t,
        block_offsets_t,
        transpose_b,
        out
    ):

    device = lhs.device
    trans_B = transpose_b
    trans_A = False

    if lhs.stride(0) > 1 and lhs.stride(1) > 1:
        trans_A = True

    # checks constraints
    assert lhs.shape[1] == shape[0], "incompatible dimensions"
    M, K = lhs.shape
    _, N = shape

    _validate_matmul_dims(M, K, N)

    # accumulator types
    ACC_TYPE = tl.float32 if lhs.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

    stride_am, stride_ak = lhs.stride(0), lhs.stride(1)
    stride_bk, stride_bn = data.stride(1), data.stride(2)
    b_column_indices  = column_indices_t
    b_offsets = offsets_t

    # launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    if trans_A:
        stride_am, stride_ak = lhs.stride(1), lhs.stride(0)
    if trans_B:
        stride_bk, stride_bn = data.stride(2), data.stride(1)
        b_column_indices, b_offsets = column_indices, offsets

    _dds_kernel[grid](
        lhs, data, out, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        row_indices, b_column_indices, b_offsets,
        block_offsets_t, trans_A, trans_B,
        GROUP_M=128, ACC_TYPE=ACC_TYPE
    )

def sdd(lhs,
        rhs,
        shape,
        out,
        offsets,
        row_indices,
        column_indices
    ):

    device = out.device
    trans_A = False
    trans_B = False

    if lhs.stride(0) > 1 and lhs.stride(1) > 1:
        trans_A = True
    if rhs.stride(0) > 1 and rhs.stride(1) > 1:
        trans_B = True

    # checks constraints
    assert lhs.shape[1] == rhs.shape[0], "incompatible dimensions"
    M, K = lhs.shape
    _, N = rhs.shape

    _validate_matmul_dims(M, K, N)

    # accumulator types
    ACC_TYPE = tl.float32 if out.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

    # launch kernel
    nnz_blocks = len(row_indices)
    grid = lambda META: (nnz_blocks,)

    stride_am, stride_ak = lhs.stride(0), lhs.stride(1)
    stride_bk, stride_bn = rhs.stride(0), rhs.stride(1)

    if trans_A:
        stride_am, stride_ak = lhs.stride(1), lhs.stride(0)
    if trans_B:
        stride_bk, stride_bn = rhs.stride(1), rhs.stride(0)

    _sdd_kernel[grid](
        lhs, rhs, out, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        out.stride(1), out.stride(2),
        row_indices, column_indices,
        GROUP_M=128, ACC_TYPE=ACC_TYPE
        )

@triton.jit
def _row_indices_kernel(offsets, out):
    pid = tl.program_id(0)
    row_offset = tl.load(offsets + pid)
    nnz_blocks = tl.load(offsets + pid + 1) - row_offset
    for nnz_block in range(nnz_blocks):
        tl.store(out + row_offset + nnz_block, pid)

def row_indices(
    shape, data, offsets, column_indices, out
):
    block_rows = len(offsets) - 1
    _row_indices_kernel[(block_rows, )](offsets, out)
