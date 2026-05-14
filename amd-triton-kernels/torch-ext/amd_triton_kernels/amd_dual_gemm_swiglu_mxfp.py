"""
Option 1.5 / 2: Dual GEMM + SwiGLU with MXFP4 weights.

- Option 1 (pre-dequant): use dual_gemm_swiglu from amd_dual_gemm_swiglu with (mx_tensor, mx_scale)
- Option 1.5 (tiled pre-dequant): upcast B in K-blocks, never materialize full fp16 B. Saves memory.
- Option 2 (fused): would decode MXFP in-kernel; blocked by ROCm Triton limitations (tl.cat, indexing).
  Currently falls back to option 1.5.

Usage:
    from dual_gemm_swiglu_mxfp import dual_gemm_swiglu_mxfp_tiled
    out = dual_gemm_swiglu_mxfp_tiled(a, (b1_mx, b1_scale), (b2_mx, b2_scale))
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch

try:
    from numerics_details.mxfp_details import upcast_mxfp4_to_fp16
    from amd_dual_gemm_swiglu import reference_dual_gemm_swiglu
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

MXFP_BLOCK = 32  # N must be multiple of 32 for scale


def _upcast_slice(mx_tensor: torch.Tensor, mx_scale: torch.Tensor, k_start: int, k_end: int) -> torch.Tensor:
    """Upcast MXFP4 slice [k_start:k_end, :] to fp16 [k_end-k_start, N]."""
    return upcast_mxfp4_to_fp16(
        mx_tensor[k_start:k_end, :],
        mx_scale[k_start:k_end, :],
        block_m=k_end - k_start,
        block_k=64,  # N must be divisible by block_k
        verbose=False,
    )


def dual_gemm_swiglu_mxfp_tiled(
    a: torch.Tensor,
    b1_mx: torch.Tensor,
    b1_scale: torch.Tensor,
    b2_mx: torch.Tensor,
    b2_scale: torch.Tensor,
    block_k: int = 64,
) -> torch.Tensor:
    """
    Dual GEMM + SwiGLU with MXFP4 B1, B2 using tiled pre-dequant (Option 1.5).
    Upcasts B in K-blocks; never materializes full fp16 B. Saves memory vs full pre-dequant.
    """
    if not _HAS_DEPS:
        raise ImportError("Requires numerics_details and amd_dual_gemm_swiglu")
    M, K = a.shape
    N = b1_mx.shape[1] * 2
    assert b1_mx.shape == (K, N // 2) and b1_scale.shape == (K, N // 32)
    assert b2_mx.shape == (K, N // 2) and b2_scale.shape == (K, N // 32)
    assert K % block_k == 0 and N % MXFP_BLOCK == 0
    assert block_k % MXFP_BLOCK == 0

    a = a.contiguous()
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    # Option 1.5: Accumulate acc1 = A@B1 and acc2 = A@B2 in K-blocks, then out = silu(acc1)*acc2.
    # Never materialize full fp16 B - upcast slice by slice. Saves O(K*N) -> O(block_k*N) memory.
    acc1 = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    acc2 = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    for k_start in range(0, K, block_k):
        k_end = k_start + block_k
        b1_slice = _upcast_slice(b1_mx, b1_scale, k_start, k_end)
        b2_slice = _upcast_slice(b2_mx, b2_scale, k_start, k_end)
        # Partial GEMM: acc1 += A[:, k_start:k_end] @ b1_slice
        # Use a simple matmul - tl.dot in a loop. We need a kernel for this.
        # Actually PyTorch: acc1 += (a[:, k_start:k_end] @ b1_slice.float()).float()
        acc1 += (a[:, k_start:k_end].float() @ b1_slice.float())
        acc2 += (a[:, k_start:k_end].float() @ b2_slice.float())
    # SwiGLU
    silu = torch.nn.functional.silu(acc1.to(torch.float16))
    out = (silu * acc2.to(torch.float16)).to(torch.float16)
    return out


def dual_gemm_swiglu_mxfp_predequant(a, b1_mx, b1_scale, b2_mx, b2_scale):
    """Option 1: full pre-dequant, then standard dual GEMM."""
    from amd_dual_gemm_swiglu import dual_gemm_swiglu
    b1 = upcast_mxfp4_to_fp16(b1_mx, b1_scale, verbose=False)
    b2 = upcast_mxfp4_to_fp16(b2_mx, b2_scale, verbose=False)
    return dual_gemm_swiglu(a, b1, b2)


if __name__ == "__main__":
    if not _HAS_DEPS:
        print("Missing deps")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("No GPU")
        sys.exit(1)
    device = "cuda"
    M, N, K = 256, 128, 512
    torch.manual_seed(42)
    a = torch.randn((M, K), device=device, dtype=torch.float16) * 0.1
    from example_dual_gemm_mxfp import downcast_fp16_to_mxfp4
    b1_fp = torch.randn((K, N), device=device, dtype=torch.float16) * 0.1
    b2_fp = torch.randn((K, N), device=device, dtype=torch.float16) * 0.1
    b1_mx, b1_scale = downcast_fp16_to_mxfp4(b1_fp, block_k=64)
    b2_mx, b2_scale = downcast_fp16_to_mxfp4(b2_fp, block_k=64)
    print("Option 1 (pre-dequant):")
    out1 = dual_gemm_swiglu_mxfp_predequant(a, b1_mx, b1_scale, b2_mx, b2_scale)
    print("Option 1.5 (tiled pre-dequant):")
    out15 = dual_gemm_swiglu_mxfp_tiled(a, b1_mx, b1_scale, b2_mx, b2_scale)
    ref = reference_dual_gemm_swiglu(a.float(), b1_fp.float(), b2_fp.float()).to(torch.float16)
    err1 = (out1.float() - ref.float()).abs().max().item()
    err15 = (out15.float() - ref.float()).abs().max().item()
    print(f"  Option 1 err: {err1:.2e}")
    print(f"  Option 1.5 err: {err15:.2e}")
    print(f"  Option 1 vs 1.5 diff: {(out1.float() - out15.float()).abs().max().item():.2e}")
