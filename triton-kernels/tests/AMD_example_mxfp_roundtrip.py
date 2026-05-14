#!/usr/bin/env python3
"""
Example: use _downcast_to_mxfp (fp16 -> MXFP4) and _upcast_from_mxfp (MXFP4 -> fp16)
for a round-trip on the remote server.

Usage on remote:
  cd /root/kernels
  python example_mxfp_roundtrip.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from numerics_details.mxfp_details._downcast_to_mxfp import _downcast_to_mxfp
from numerics_details.mxfp_details import upcast_mxfp4_to_fp16

MXFP_BLOCK_SIZE_PY = 32  # Python int for checks (tl.constexpr in kernels)


def downcast_fp16_to_mxfp4(src: torch.Tensor, block_m: int = 128, block_k: int = 64):
    """Convert fp16 tensor [M, K] to MXFP4 (uint8 mx_tensor + uint8 mx_scale)."""
    assert src.dim() == 2 and src.dtype in (torch.float16, torch.bfloat16)
    assert block_k % MXFP_BLOCK_SIZE_PY == 0, f"block_k must be multiple of {MXFP_BLOCK_SIZE_PY}"
    M, K = src.shape

    # Outputs: mx_tensor [M, K//2] uint8, mx_scale [M, K//32] uint8
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

    # Create random fp16 tensor
    M, K = 256, 128
    src = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1

    # Downcast fp16 -> MXFP4
    mx_tensor, mx_scale = downcast_fp16_to_mxfp4(src)
    print(f"Downcast OK: mx_tensor {mx_tensor.shape}, mx_scale {mx_scale.shape}")

    # Upcast MXFP4 -> fp16
    recovered = upcast_mxfp4_to_fp16(mx_tensor, mx_scale)
    print(f"Upcast OK: recovered {recovered.shape}")

    # Compare
    err = (src.float() - recovered.float()).abs().max().item()
    rel = err / (src.float().abs().max().item() + 1e-6)
    print(f"Round-trip max abs err: {err:.2e}, rel: {rel:.2e}")
    print("Done.")


if __name__ == "__main__":
    main()
