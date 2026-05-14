"""
Dual GEMM + SwiGLU following triton_kernels swiglu.py build pattern.

Structure matches Kernel Community Hub swiglu.py:
- repr() and launch_metadata for specialization
- compute_swiglu() style activation (SiLU(gate) * linear)
- Optional Flexpoint/MXFP (stub for standalone, real import in triton_kernels)
- NTokens support for variable M (MoE routing)
- Persistent kernel pattern with tl.range

Usage (standalone fp16):
    from dual_gemm_swiglu_full import dual_gemm_swiglu
    out = dual_gemm_swiglu(a, b1, b2)

For triton_kernels integration: place in dual_gemm_swiglu_details/, add flexpoint import.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

# Allow importing testing.py from same directory (when run from kernels/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Flexpoint stub (standalone). Replace with:
#   from ..numerics_details.flexpoint import load_scale, float_to_flex, update_scale
# when integrating into triton_kernels.
# -----------------------------------------------------------------------------
_HAS_FLEXPOINT = False
try:
    # Only works inside triton_kernels package
    from ..numerics_details.flexpoint import load_scale, float_to_flex, update_scale
    _HAS_FLEXPOINT = True
except ImportError:
    @triton.jit
    def load_scale(scale_ptr):
        return 1.0 if scale_ptr is None else tl.load(scale_ptr)

    def float_to_flex_stub(x, *args, **kwargs):
        """Pass-through for fp16 standalone."""
        return x

    def update_scale_stub(x, scale_ptr, Out):
        pass


# -----------------------------------------------------------------------------
# Helpers (mirroring swiglu.py)
# -----------------------------------------------------------------------------
@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def thread_local_absmax(x, BLOCK_SIZE: tl.constexpr, NUM_THREADS: tl.constexpr):
    return tl.max(
        tl.reshape(tl.abs(x), [NUM_THREADS, BLOCK_SIZE // NUM_THREADS], can_reorder=True),
        axis=1,
    )


@triton.jit
def compute_swiglu(gelu, linear, scale, alpha, limit: tl.constexpr):
    """SwiGLU: silu(gelu) * linear. Matches swiglu.py compute_swiglu style.
    limit > 0 enables clipping; pass 0.0 for no clip.
    """
    gelu = gelu.to(tl.float32) * scale
    if limit > 0:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32) * scale
    if limit > 0:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp(-alpha * gelu))  # SiLU(gelu)
    return s * linear  # SiLU(gate) * linear (standard SwiGLU)


# -----------------------------------------------------------------------------
# Repr and launch_metadata (swiglu.py pattern)
# -----------------------------------------------------------------------------
def dual_gemm_repr(specialization):
    signature = specialization.signature
    constants = specialization.constants
    convert_dtype = lambda dtype: "mxfp4" if "u8" in str(dtype) else str(dtype)
    dtypes = "x".join([convert_dtype(f"{signature.get(i, 'fp16')}") for i in ["Out", "A", "B1", "B2"]])
    blocks = "x".join([f"{constants.get(i, 0)}" for i in ["BLOCK_M", "BLOCK_N", "BLOCK_K"]])
    return f"_dual_gemm_swiglu_{dtypes}_{blocks}"


def dual_gemm_launch_metadata(grid, kernel, args):
    M, N, K = args["M"], args["N"], args["K"]
    ret = dict()
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    A, B1, B2, Out = args["A"], args["B1"], args["B2"], args["Out"]
    ret["bytes"] = (
        A.numel() * A.element_size()
        + B1.numel() * B1.element_size()
        + B2.numel() * B2.element_size()
        + Out.numel() * Out.element_size()
    )
    return ret


# -----------------------------------------------------------------------------
# Dual GEMM + SwiGLU kernel (swiglu.py structure)
# -----------------------------------------------------------------------------
@triton.jit(repr=lambda _: "_dual_gemm_swiglu", launch_metadata=dual_gemm_launch_metadata)
def _dual_gemm_swiglu(
    Out,
    A,
    B1,
    B2,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_b1k,
    stride_b1n,
    stride_b2k,
    stride_b2n,
    stride_outm,
    stride_outn,
    alpha: tl.constexpr,
    limit,
    NTokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if NTokens is not None:
        M = tl.load(NTokens)
    M_BLOCKS = tl.cdiv(M, BLOCK_M)
    N_BLOCKS = tl.cdiv(N, BLOCK_N)
    num_tiles = M_BLOCKS * N_BLOCKS

    # Persistent kernel: each program handles multiple tiles
    grid_size = tl.num_programs(0)
    for pid in range(tl.program_id(0), num_tiles, grid_size):
        pid_m = pid // N_BLOCKS
        pid_n = pid % N_BLOCKS

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b1_ptrs = B1 + (offs_k[:, None] * stride_b1k + offs_n[None, :] * stride_b1n)
        b2_ptrs = B2 + (offs_k[:, None] * stride_b2k + offs_n[None, :] * stride_b2n)

        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN_K:
                a = tl.load(a_ptrs, mask=m_mask, other=0.0) if not EVEN_M else tl.load(a_ptrs)
                b1 = tl.load(b1_ptrs, mask=n_mask, other=0.0) if not EVEN_N else tl.load(b1_ptrs)
                b2 = tl.load(b2_ptrs, mask=n_mask, other=0.0) if not EVEN_N else tl.load(b2_ptrs)
            else:
                k_rem = K - k * BLOCK_K
                k_mask_m = offs_k[None, :] < k_rem
                k_mask_n = offs_k[:, None] < k_rem
                a = tl.load(a_ptrs, mask=m_mask & k_mask_m, other=0.0)
                b1 = tl.load(b1_ptrs, mask=k_mask_n & n_mask, other=0.0)
                b2 = tl.load(b2_ptrs, mask=k_mask_n & n_mask, other=0.0)

            tl.multiple_of(a_ptrs, [16, 16])
            tl.multiple_of(b1_ptrs, [16, 16])
            tl.multiple_of(b2_ptrs, [16, 16])
            acc1 += tl.dot(a, b1)
            acc2 += tl.dot(a, b2)

            a_ptrs += BLOCK_K * stride_ak
            b1_ptrs += BLOCK_K * stride_b1k
            b2_ptrs += BLOCK_K * stride_b2k

        out = compute_swiglu(acc1, acc2, 1.0, alpha, limit)
        out = out.to(tl.float16)

        out_ptrs = Out + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out, mask=c_mask)


# -----------------------------------------------------------------------------
# Autotuned wrapper (backward compatible, uses simpler kernel for reliability)
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _dual_gemm_swiglu_autotuned(
    a_ptr, b1_ptr, b2_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_b1k, stride_b1n, stride_b2k, stride_b2n, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    alpha: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b1_ptrs = b1_ptr + (offs_k[:, None] * stride_b1k + offs_n[None, :] * stride_b1n)
    b2_ptrs = b2_ptr + (offs_k[:, None] * stride_b2k + offs_n[None, :] * stride_b2n)

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs) if EVEN_M else tl.load(a_ptrs, mask=m_mask, other=0.0)
            b1 = tl.load(b1_ptrs) if EVEN_N else tl.load(b1_ptrs, mask=n_mask, other=0.0)
            b2 = tl.load(b2_ptrs) if EVEN_N else tl.load(b2_ptrs, mask=n_mask, other=0.0)
        else:
            k_rem = K - k * BLOCK_K
            k_mask_m = offs_k[None, :] < k_rem
            k_mask_n = offs_k[:, None] < k_rem
            a = tl.load(a_ptrs, mask=m_mask & k_mask_m, other=0.0)
            b1 = tl.load(b1_ptrs, mask=k_mask_n & n_mask, other=0.0)
            b2 = tl.load(b2_ptrs, mask=k_mask_n & n_mask, other=0.0)

        tl.multiple_of(a_ptrs, [16, 16])
        tl.multiple_of(b1_ptrs, [16, 16])
        tl.multiple_of(b2_ptrs, [16, 16])
        acc1 += tl.dot(a, b1)
        acc2 += tl.dot(a, b2)
        a_ptrs += BLOCK_K * stride_ak
        b1_ptrs += BLOCK_K * stride_b1k
        b2_ptrs += BLOCK_K * stride_b2k

    out = compute_swiglu(acc1, acc2, 1.0, alpha, 0.0)  # 0.0 = no clip
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(tl.float16), mask=c_mask)


def dual_gemm_swiglu(
    a: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    n_tokens: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    limit: Optional[float] = None,
) -> torch.Tensor:
    """Fused Dual GEMM + SwiGLU: silu(A @ B1) * (A @ B2)."""
    if a.ndim != 2 or b1.ndim != 2 or b2.ndim != 2:
        raise ValueError("Expected 2D tensors: a[M,K], b1[K,N], b2[K,N].")
    if a.shape[1] != b1.shape[0] or a.shape[1] != b2.shape[0]:
        raise ValueError("Incompatible shapes for dual GEMM.")
    if b1.shape[1] != b2.shape[1]:
        raise ValueError("b1 and b2 must have same N dimension.")
    if not (a.is_cuda and b1.is_cuda and b2.is_cuda):
        raise ValueError("All tensors must be on a CUDA/ROCm device.")
    if a.dtype != torch.float16 or b1.dtype != torch.float16 or b2.dtype != torch.float16:
        raise ValueError("This kernel currently expects float16 inputs.")

    a = a.contiguous()
    b1 = b1.contiguous()
    b2 = b2.contiguous()

    M, K = a.shape
    _, N = b1.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _dual_gemm_swiglu_autotuned[grid](
        a, b1, b2, c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_b1k=b1.stride(0), stride_b1n=b1.stride(1),
        stride_b2k=b2.stride(0), stride_b2n=b2.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        alpha=alpha,
    )
    return c


def reference_dual_gemm_swiglu(a: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    x1 = a @ b1
    x2 = a @ b2
    return torch.nn.functional.silu(x1) * x2


def test_correctness(device: str = "cuda", maxtol: float = 2e-2, rmstol: float = 4e-3) -> bool:
    from testing import assert_close

    torch.manual_seed(42)
    shapes = [
        (128, 64, 128),
        (256, 256, 512),
        (1024, 512, 1024),
        (4096, 3648, 8192),
        (7, 13, 17),
        (100, 200, 150),
    ]
    input_scale = 0.125
    all_pass = True
    for m, n, k in shapes:
        a = torch.randn((m, k), device=device, dtype=torch.float16) * input_scale
        b1 = torch.randn((k, n), device=device, dtype=torch.float16) * input_scale
        b2 = torch.randn((k, n), device=device, dtype=torch.float16) * input_scale
        with torch.no_grad():
            ref = reference_dual_gemm_swiglu(a.float(), b1.float(), b2.float()).to(torch.float16)
            out = dual_gemm_swiglu(a, b1, b2)
        desc = f"[shape ({m},{n},{k})]"
        try:
            assert_close(ref, out, maxtol=maxtol, rmstol=rmstol, description=desc, verbose=True)
            print(f"  {desc} PASS")
        except AssertionError as e:
            print(f"  {desc} FAIL: {e}")
            all_pass = False
    return all_pass


def benchmark(m: int, n: int, k: int, warmup: int, iters: int, input_scale: float) -> None:
    device = "cuda"
    a = torch.randn((m, k), device=device, dtype=torch.float16) * input_scale
    b1 = torch.randn((k, n), device=device, dtype=torch.float16) * input_scale
    b2 = torch.randn((k, n), device=device, dtype=torch.float16) * input_scale
    for _ in range(warmup):
        _ = dual_gemm_swiglu(a, b1, b2)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = dual_gemm_swiglu(a, b1, b2)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / iters
    total_flops = 4 * m * n * k
    tflops = (total_flops / (avg_ms * 1e-3)) / 1e12
    print(f"[kernel] shape=({m}, {n}, {k}) avg={avg_ms:.3f} ms, ~{tflops:.2f} TFLOP/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual GEMM + SwiGLU (swiglu.py build pattern)")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=3648)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--input-scale", type=float, default=0.125)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA/ROCm GPU detected.")
        raise SystemExit(1)

    if not args.bench_only:
        print("Running correctness tests...")
        t0 = time.time()
        ok = test_correctness()
        print(f"Correctness: {'PASS' if ok else 'FAIL'} ({time.time()-t0:.2f}s)")

    if not args.test_only:
        print("\nRunning benchmark...")
        t0 = time.time()
        benchmark(args.m, args.n, args.k, args.warmup, args.iters, args.input_scale)
        print(f"[done] elapsed={time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
