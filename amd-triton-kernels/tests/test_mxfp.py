#!/usr/bin/env python3
"""
Minimal test for MXFP _downcast_to_mxfp on MI300X (ROCm).
Tests fp16 -> uint8 (fp4 packed) path; float8 path may not work on ROCm yet.
Run on remote: cd /root/kernels && python test_mxfp.py
"""

import sys
import os

# Allow imports from /root/kernels
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

def main():
    print("=== MXFP Import Test ===")
    try:
        from numerics_details.mxfp_details._downcast_to_mxfp import (
            _downcast_to_mxfp,
            _compute_quant_and_scale,
            MXFP_BLOCK_SIZE,
        )
        print("  Import OK: _downcast_to_mxfp, _compute_quant_and_scale, MXFP_BLOCK_SIZE")
    except Exception as e:
        print(f"  Import FAILED: {e}")
        return 1

    print("\n=== Triton + CUDA/ROCm Check ===")
    import torch
    if not torch.cuda.is_available():
        print("  No GPU available. Skipping kernel test.")
        return 0
    print(f"  Device: {torch.cuda.get_device_name(0)}")

    import triton
    import triton.language as tl

    print("\n=== MXFP Downcast Test (fp16 -> fp4 uint8) ===")
    # Use fp4 path (uint8 output) - avoids float8 dtypes which may lack ROCm support
    BLOCK_SIZE_OUT_DIM = 64
    BLOCK_SIZE_QUANT_DIM = 64  # must be multiple of 32
    outer_dim = 128
    quant_dim = 128
    DEQUANT_SCALE_ROUNDING_MODE = 0

    device = "cuda"
    src = torch.randn(outer_dim, quant_dim, device=device, dtype=torch.float16) * 0.1

    # Output shapes for fp4 (uint8): mx_tensor [outer, quant//2], mx_scale [outer, quant//32]
    mx_tensor = torch.empty(outer_dim, quant_dim // 2, device=device, dtype=torch.uint8)
    mx_scale = torch.empty(outer_dim, quant_dim // 32, device=device, dtype=torch.uint8)

    num_outer_blocks = (outer_dim + BLOCK_SIZE_OUT_DIM - 1) // BLOCK_SIZE_OUT_DIM
    num_quant_blocks = (quant_dim + BLOCK_SIZE_QUANT_DIM - 1) // BLOCK_SIZE_QUANT_DIM
    grid = (num_outer_blocks, num_quant_blocks)

    try:
        _downcast_to_mxfp[grid](
            mx_tensor,
            src.stride(0), 1,
            mx_scale,
            mx_scale.stride(0), mx_scale.stride(1),
            src,
            src.stride(0), src.stride(1),
            outer_dim, quant_dim,
            BLOCK_SIZE_OUT_DIM=BLOCK_SIZE_OUT_DIM,
            BLOCK_SIZE_QUANT_DIM=BLOCK_SIZE_QUANT_DIM,
            DEQUANT_SCALE_ROUNDING_MODE=DEQUANT_SCALE_ROUNDING_MODE,
        )
        torch.cuda.synchronize()
        print("  Kernel launch OK")
        print(f"  mx_tensor shape: {mx_tensor.shape}, dtype: {mx_tensor.dtype}")
        print(f"  mx_scale shape: {mx_scale.shape}")
        print(f"  mx_tensor sample (first row): {mx_tensor[0, :8].tolist()}")
    except Exception as e:
        print(f"  Kernel FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
