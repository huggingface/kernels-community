#!/usr/bin/env python3
"""
Example: Mask compaction + Dual GEMM integration (MoE-style).
Before dual GEMM: compact (Yv, Yi) per row based on BitMask.
Then use compacted tensors for routing into expert weights.

ROCm note: tl.store with dynamic write_indx may fail on ROCm Triton.
If so, use the PyTorch fallback in mask_compaction.py.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from mask_compaction import masked_compaction, masked_compaction_torch_fallback
from amd_dual_gemm_swiglu import dual_gemm_swiglu


def example_integration():
    """Sketch: compact routing outputs, then run dual GEMM on routed experts."""
    device = "cuda"
    if not torch.cuda.is_available():
        print("No GPU")
        return

    M, K, N = 256, 64, 128  # tokens, hidden, expert dim
    top_k = 8
    num_experts = 4

    # Simulate routing: Yv [M, K] values, Yi [M, K] expert indices (0..num_experts-1)
    torch.manual_seed(42)
    Yv = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
    Yi = torch.randint(0, num_experts, (M, K), device=device, dtype=torch.int32)

    # BitMask [M, ceil(K/32)]: 1 = use, 0 = discard (e.g. from load balance)
    BitMask = torch.ones(M, (K + 31) // 32, device=device, dtype=torch.int32)
    BitMask[:, 0] = 0x55555555  # example: alternating bits

    # 1) Compact (Yv, Yi) per row based on BitMask
    try:
        RetYv, RetYi = masked_compaction(Yv, Yi, BitMask, sentinel=float("nan"))
        print("Compaction: Triton kernel OK")
    except Exception as e:
        print(f"Compaction: Triton failed ({e}), using PyTorch fallback")
        RetYv, RetYi = masked_compaction_torch_fallback(Yv, Yi, BitMask, sentinel=float("nan"))

    # 2) Use compacted indices for routing into expert weights
    # Expert weights: B1[E,K,N], B2[E,K,N] or similar. For simplicity, flat GEMM:
    # A = routed activations [M, K], B1/B2 = expert weights [K, N]
    # This is a simplified sketch; real MoE has per-expert B.
    B1 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    B2 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1

    # Use RetYv as activations (compacted); pad/truncate to [M, K] if needed
    A = RetYv[:, :K].contiguous()
    if A.shape[1] < K:
        A = torch.nn.functional.pad(A, (0, K - A.shape[1]), value=0)

    # 3) Dual GEMM
    out = dual_gemm_swiglu(A, B1, B2)
    print(f"Dual GEMM output: {out.shape}")
    print("Done.")


if __name__ == "__main__":
    example_integration()
