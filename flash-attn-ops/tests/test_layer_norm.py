# Self-contained tests for the vendored Triton `layer_norm_fn` / `rms_norm_fn`.
#
# Following the convention used elsewhere in this repo (see rmsnorm/tests), the
# reference is computed directly with PyTorch rather than imported from the
# kernel package.

import pytest
import torch
import torch.nn.functional as F

from flash_attn_ops import layer_norm_fn, rms_norm_fn


def rms_norm_ref(x, weight, bias, eps):
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    out = out * weight.float() + (bias.float() if bias is not None else 0.0)
    return out.to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("hidden", [192, 1024, 4096])
def test_layer_norm(hidden, has_bias, dtype):
    torch.manual_seed(0)
    device = "cuda"
    eps = 1e-5
    x = torch.randn(512, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    bias = torch.randn(hidden, device=device, dtype=dtype) if has_bias else None

    out = layer_norm_fn(x, weight, bias, eps=eps)
    ref = F.layer_norm(
        x.float(), (hidden,), weight.float(), bias.float() if has_bias else None, eps
    ).to(dtype)

    atol = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden", [192, 1024, 4096])
def test_rms_norm(hidden, dtype):
    torch.manual_seed(0)
    device = "cuda"
    eps = 1e-6
    x = torch.randn(512, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=dtype)

    out = rms_norm_fn(x, weight, None, eps=eps)
    ref = rms_norm_ref(x, weight, None, eps)

    atol = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layer_norm_prenorm_residual(dtype):
    """prenorm=True returns (out, residual_out); residual is added before norm."""
    torch.manual_seed(0)
    device, hidden, eps = "cuda", 1024, 1e-5
    x = torch.randn(512, hidden, device=device, dtype=dtype)
    res = torch.randn(512, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    bias = torch.randn(hidden, device=device, dtype=dtype)

    out, residual_out = layer_norm_fn(x, weight, bias, residual=res, prenorm=True, eps=eps)

    ref_residual = (x.float() + res.float())
    ref_out = F.layer_norm(ref_residual, (hidden,), weight.float(), bias.float(), eps).to(dtype)
    torch.testing.assert_close(residual_out, ref_residual.to(dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
