"""
Masked compaction kernel: compact (Yv, Yi) per row based on BitMask.
Active elements (bit=1) move to front, inactive (bit=0) move to back with sentinel.
For MoE: use before dual GEMM to get dense top-k for routing into expert weights.

ROCm note: tl.store with dynamic write_indx may have limitations. If it fails,
fall back to PyTorch compaction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_compaction(
    Yv, Yi, BitMask, stride_bm, stride_bn,
    RetYv, RetYi, sentinel, K: tl.constexpr
):
    pid_m = tl.program_id(0)
    yv = tl.load(Yv + pid_m * K + tl.arange(0, K))
    yi = tl.load(Yi + pid_m * K + tl.arange(0, K))
    div = yi // 32
    rem = yi % 32
    active_bits = (tl.load(BitMask + pid_m * stride_bm + div * stride_bn) >> rem) & 1
    exc_cumsum = tl.cumsum(active_bits, 0) - active_bits
    active_flags = active_bits.to(tl.int1)
    rev_arange = tl.where(active_flags, 0, K - 1 - tl.arange(0, K))
    write_indx = exc_cumsum + rev_arange
    yv = tl.where(active_flags, yv, sentinel)
    yi = tl.where(active_flags, yi, sentinel)
    tl.store(RetYv + pid_m * K + write_indx, yv)
    tl.store(RetYi + pid_m * K + write_indx, yi)


def masked_compaction(
    Yv: torch.Tensor,   # [M, K] values
    Yi: torch.Tensor,   # [M, K] indices (int32)
    BitMask: torch.Tensor,  # [M, ceil(K/32)] or similar - 1 bit per position
    sentinel: float = float("nan"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compact Yv, Yi per row: active (BitMask=1) to front, inactive to back with sentinel.
    Returns (RetYv, RetYi) same shape as (Yv, Yi).
    """
    M, K = Yv.shape
    assert Yi.shape == (M, K)
    RetYv = torch.empty_like(Yv)
    RetYi = torch.empty_like(Yi)
    grid = (M,)
    _masked_compaction[grid](
        Yv, Yi, BitMask,
        BitMask.stride(0), BitMask.stride(1),
        RetYv, RetYi, sentinel, K=K,
    )
    return RetYv, RetYi


def masked_compaction_torch_fallback(Yv, Yi, BitMask, sentinel=float("nan")):
    """PyTorch fallback if Triton kernel fails on ROCm."""
    M, K = Yv.shape
    RetYv = torch.full_like(Yv, sentinel)
    RetYi = torch.full_like(Yi, -1)
    for m in range(M):
        # Bit per position k: div=k//32, rem=k%32
        div = torch.arange(K, device=Yv.device) // 32
        rem = torch.arange(K, device=Yv.device) % 32
        active = ((BitMask[m, div] >> rem) & 1).bool()
        n_active = active.sum().item()
        RetYv[m, :n_active] = Yv[m, active]
        RetYi[m, :n_active] = Yi[m, active]
    return RetYv, RetYi


masked_compaction_pytorch = masked_compaction_torch_fallback  # alias for import
