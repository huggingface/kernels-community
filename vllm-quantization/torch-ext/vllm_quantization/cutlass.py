from typing import Optional

import torch

from ._ops import ops
from .platforms import current_platform


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return ops.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return ops.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]

    cutlass_compatible_b = (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    if not cutlass_compatible_b:
        from .triton_scaled_mm import triton_scaled_mm
        return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    ops.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out


def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype
    assert azp is None or azp.numel() == a.shape[0]

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    ops.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj, azp, bias)
    return out
