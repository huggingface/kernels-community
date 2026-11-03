import kernels
import pytest
import torch
import torch.nn.functional as F

grt = kernels.get_kernel("kernels-community/grt", version=1)


def _pearl_norm_ref(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    d = x.shape[-1]
    pairs = x.view(*x.shape[:-1], d // 2, 2)
    radius_sq = (pairs**2).sum(dim=-1)
    rms = torch.sqrt(radius_sq.mean(dim=-1, keepdim=True) + eps)
    return x / rms * scale


def _radial_gelu_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    d = x.shape[-1]
    pairs = x.view(*x.shape[:-1], d // 2, 2)
    radius = torch.norm(pairs, dim=-1, keepdim=True)
    gate = F.gelu(radius) / (radius + eps)
    return (pairs * gate).view(*x.shape)


@pytest.mark.kernels_ci
def test_grt_ops_ci() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    device = torch.device("cuda")
    torch.manual_seed(42)

    # PearlNorm
    x = torch.randn(16, 256, device=device, dtype=torch.float32)
    scale = torch.ones(256, device=device, dtype=torch.float32)
    out_norm = grt.fused_pearl_norm(x, scale, 1e-6)
    ref_norm = _pearl_norm_ref(x, scale, 1e-6)
    torch.testing.assert_close(out_norm, ref_norm, atol=1e-4, rtol=1e-4)

    # RadialGELU
    out_gelu = grt.fused_radial_gelu(x, 1e-6)
    ref_gelu = _radial_gelu_ref(x, 1e-6)
    torch.testing.assert_close(out_gelu, ref_gelu, atol=1e-4, rtol=1e-4)

    # Fused FFN Givens cascade
    angles = torch.randn(3, 128, device=device, dtype=torch.float32) * 0.1
    raw_scales = torch.randn(3, 256, device=device, dtype=torch.float32) * 0.1
    out_ffn = grt.apply_fused_ffn(x, angles, raw_scales)
    assert out_ffn.shape == x.shape
