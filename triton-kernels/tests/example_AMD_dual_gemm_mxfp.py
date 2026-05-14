#!/usr/bin/env python3
"""
Example: Dual GEMM + SwiGLU with MXFP4 weights (pre-dequant option 1).
Quantizes B1, B2 to MXFP4, then upcasts and runs the fused GEMM.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from amd_dual_gemm_swiglu import dual_gemm_swiglu, reference_dual_gemm_swiglu
from numerics_details.mxfp_details._downcast_to_mxfp import _downcast_to_mxfp
from numerics_details.mxfp_details import upcast_mxfp4_to_fp16

MXFP_BLOCK_SIZE_PY = 32


def downcast_fp16_to_mxfp4(src: torch.Tensor, block_m: int = 128, block_k: int = 64):
    """fp16 [M,K] -> (mx_tensor [M,K//2], mx_scale [M,K//32])."""
    assert block_k % MXFP_BLOCK_SIZE_PY == 0
    M, K = src.shape
    mx_tensor = torch.empty((M, K // 2), device=src.device, dtype=torch.uint8)
    mx_scale = torch.empty((M, K // 32), device=src.device, dtype=torch.uint8)
    grid = ((M + block_m - 1) // block_m, (K + block_k - 1) // block_k)
    _downcast_to_mxfp[grid](
        mx_tensor, mx_tensor.stride(0), 1,
        mx_scale, mx_scale.stride(0), mx_scale.stride(1),
        src, src.stride(0), src.stride(1),
        M, K,
        BLOCK_SIZE_OUT_DIM=block_m,
        BLOCK_SIZE_QUANT_DIM=block_k,
        DEQUANT_SCALE_ROUNDING_MODE=0,
    )
    return mx_tensor, mx_scale


def main():
    if not torch.cuda.is_available():
        print("No CUDA/ROCm device.")
        return
    device = "cuda"
    print("Device:", torch.cuda.get_device_name(0))

    M, N, K = 256, 128, 512
    torch.manual_seed(42)
    a = torch.randn((M, K), device=device, dtype=torch.float16) * 0.1
    b1_fp16 = torch.randn((K, N), device=device, dtype=torch.float16) * 0.1
    b2_fp16 = torch.randn((K, N), device=device, dtype=torch.float16) * 0.1

    # Quantize B1, B2 to MXFP4 (need K, N multiples of 32 for block_k)
    block_k = 64
    b1_mx, b1_scale = downcast_fp16_to_mxfp4(b1_fp16, block_k=block_k)
    b2_mx, b2_scale = downcast_fp16_to_mxfp4(b2_fp16, block_k=block_k)
    print(f"Quantized B1: {b1_mx.shape}, {b1_scale.shape}")

    # Run dual GEMM with MXFP4 weights (pre-dequant)
    out_mxfp = dual_gemm_swiglu(a, (b1_mx, b1_scale), (b2_mx, b2_scale))
    print(f"Output (MXFP4 path): {out_mxfp.shape}")

    # Reference with fp16
    out_ref = reference_dual_gemm_swiglu(a.float(), b1_fp16.float(), b2_fp16.float()).to(torch.float16)
    err = (out_mxfp.float() - out_ref.float()).abs().max().item()
    rel = err / (out_ref.float().abs().max().item() + 1e-6)
    print(f"vs fp16 ref: max abs err={err:.2e}, rel={rel:.2e}")
    print("Done.")


if __name__ == "__main__":
    main()
