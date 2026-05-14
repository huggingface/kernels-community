    return c
amd_dual_gemm_swiglu.py
ADDED






























































































































































































































































































































"""
AMD Triton fused Dual GEMM + SwiGLU kernel.
Computes: silu(A @ B1) * (A @ B2) in a single fused kernel.
Uses triton-kernels testing.py: assert_close (maxtol=2e-2, rmstol=4e-3).
"""

import argparse
import os
import sys
import time

# Allow importing testing.py from same directory (when run from kernels/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch
import triton
import triton.language as tl

# Optional MXFP4 pre-dequant (option 1: upcast before GEMM)
try:
    from numerics_details.mxfp_details import upcast_mxfp4_to_fp16
    _HAS_MXFP = True
except ImportError:
    upcast_mxfp4_to_fp16 = None
    _HAS_MXFP = False


def _maybe_upcast_mxfp(b, name: str) -> torch.Tensor:
    """If b is MXFP4 (mx_tensor, mx_scale), upcast to fp16. Else return b."""
    if not isinstance(b, (tuple, list)) or len(b) != 2:
        return b
    mx_tensor, mx_scale = b
    if not (isinstance(mx_tensor, torch.Tensor) and isinstance(mx_scale, torch.Tensor)):
        return b
    if mx_tensor.dtype != torch.uint8 or mx_scale.dtype != torch.uint8:
        return b
    if not _HAS_MXFP:
        raise ImportError("MXFP4 weights require numerics_details.mxfp_details")
    return upcast_mxfp4_to_fp16(mx_tensor, mx_scale, verbose=False)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=2,
        ),
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
def dual_gemm_swiglu_kernel(
    a_ptr,
    b1_ptr,
    b2_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_b1k,
    stride_b1n,
    stride_b2k,
    stride_b2n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
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
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=m_mask, other=0.0)

            if EVEN_N:
                b1 = tl.load(b1_ptrs)
                b2 = tl.load(b2_ptrs)
            else:
                b1 = tl.load(b1_ptrs, mask=n_mask, other=0.0)
                b2 = tl.load(b2_ptrs, mask=n_mask, other=0.0)
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

    silu = acc1 * tl.sigmoid(acc1)
    out = silu * acc2

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(tl.float16), mask=c_mask)


def dual_gemm_swiglu(
    a: torch.Tensor,
    b1: torch.Tensor | tuple,
    b2: torch.Tensor | tuple,
) -> torch.Tensor:
    """Fused Dual GEMM + SwiGLU. b1/b2 can be fp16 [K,N] or MXFP4 (mx_tensor, mx_scale)."""
    b1 = _maybe_upcast_mxfp(b1, "b1")
    b2 = _maybe_upcast_mxfp(b2, "b2")

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

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    dual_gemm_swiglu_kernel[grid](
        a, b1, b2, c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_b1k=b1.stride(0), stride_b1n=b1.stride(1),
        stride_b2k=b2.stride(0), stride_b2n=b2.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
    )
    return c


def reference_dual_gemm_swiglu(a: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    x1 = a @ b1
    x2 = a @ b2
    return torch.nn.functional.silu(x1) * x2


def test_correctness(device: str = "cuda", maxtol: float = 2e-2, rmstol: float = 4e-3) -> bool:
    """Run correctness tests using triton-kernels testing.assert_close."""
    from testing import assert_close

    torch.manual_seed(42)
    shapes = [(128, 64, 128), (256, 256, 512), (1024, 512, 1024), (4096, 3648, 8192),
              (7, 13, 17), (100, 200, 150)]
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
        except AssertionError:
            print(f"  {desc} FAIL")
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
    parser = argparse.ArgumentParser(description="AMD Triton fused dual-GEMM + SwiGLU")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=3648)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--input-scale", type=float, default=0.125)
    parser.add_argument("--test-only", action="store_true", help="Run correctness tests only")
    parser.add_argument("--bench-only", action="store_true", help="Run benchmark only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA/ROCm GPU detected. This kernel requires a GPU to run.")
        print("  - Run on a machine with an NVIDIA GPU (CUDA) or AMD GPU (ROCm)")
        print("  - Ensure PyTorch is installed with GPU support.")
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
