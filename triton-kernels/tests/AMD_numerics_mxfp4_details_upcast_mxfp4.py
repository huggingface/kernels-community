"""
Reusable MXFP4 upcast: MXFP4 (uint8 mx_tensor + uint8 mx_scale) -> fp16/bf16.
Uses Triton kernel when available; falls back to PyTorch on ROCm (tl.cat limitation).
"""

import torch

from ._upcast_from_mxfp import _upcast_from_mxfp

try:
    from triton.compiler.errors import CompilationError
except ImportError:
    CompilationError = Exception

MXFP_BLOCK_SIZE_PY = 32


def _upcast_mxfp4_to_fp16_pytorch(
    mx_tensor: torch.Tensor, mx_scale: torch.Tensor, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """PyTorch fallback (used when Triton kernel fails on ROCm)."""
    M, K_half = mx_tensor.shape
    K = K_half * 2
    dst_bias = 15
    dst_0p5 = 0x3800
    dst_m_bits = 10

    tensor = mx_tensor.to(torch.int32)
    em0 = tensor & 0x07
    em1 = tensor & 0x70
    x0 = (em0 << (dst_m_bits - 1)) | ((tensor & 0x08) << 12)
    x1 = (em1 << (dst_m_bits - 5)) | ((tensor & 0x80) << 8)

    x0 = torch.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
    x1 = torch.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
    x0 = torch.where(em0 == 0x01, torch.full_like(x0, dst_0p5) | (x0 & 0x8000), x0)
    x1 = torch.where(em1 == 0x10, torch.full_like(x1, dst_0p5) | (x1 & 0x8000), x1)

    out_u16 = torch.empty((M, K), device=mx_tensor.device, dtype=torch.uint16)
    out_u16[:, 0::2] = (x0 & 0xFFFF).to(torch.uint16)
    out_u16[:, 1::2] = (x1 & 0xFFFF).to(torch.uint16)
    dst_tensor = out_u16.view(dtype)

    scale_u32 = mx_scale.to(torch.int32) << 23
    dst_scale = scale_u32.view(torch.float32).to(dtype)
    dst_scale = dst_scale.unsqueeze(-1).repeat(1, 1, 32).reshape(M, K)

    out_tensor = dst_tensor * dst_scale
    out_tensor = torch.where(
        mx_scale.unsqueeze(-1).expand(-1, -1, 32).reshape(M, K) == 0xFF,
        float("nan"),
        out_tensor,
    )
    return out_tensor


def upcast_mxfp4_to_fp16(
    mx_tensor: torch.Tensor,
    mx_scale: torch.Tensor,
    block_m: int = 128,
    block_k: int = 64,
    dtype: torch.dtype = torch.float16,
    verbose: bool = False,
) -> torch.Tensor:
    """Convert MXFP4 [M,K/2]+[M,K/32] -> fp16/bf16 [M,K]. Falls back to PyTorch if Triton fails."""
    assert mx_tensor.dim() == 2 and mx_tensor.dtype == torch.uint8
    assert mx_scale.dim() == 2 and mx_scale.dtype == torch.uint8
    M = mx_tensor.shape[0]
    K = mx_tensor.shape[1] * 2
    assert mx_scale.shape == (M, K // 32)
    assert block_k % MXFP_BLOCK_SIZE_PY == 0

    try:
        out = torch.empty((M, K), device=mx_tensor.device, dtype=dtype)
        grid = ((M + block_m - 1) // block_m, (K + block_k - 1) // block_k)
        _upcast_from_mxfp[grid](
            out, out.stride(0), 1,
            mx_scale, mx_scale.stride(0), mx_scale.stride(1),
            mx_tensor, mx_tensor.stride(0), 1,
            M, K,
            BLOCK_SIZE_OUT_DIM=block_m,
            BLOCK_SIZE_QUANT_DIM=block_k,
        )
        return out
    except CompilationError:
        if verbose:
            print("Triton upcast failed (e.g. ROCm), using PyTorch fallback.")
        return _upcast_mxfp4_to_fp16_pytorch(mx_tensor, mx_scale, dtype)

