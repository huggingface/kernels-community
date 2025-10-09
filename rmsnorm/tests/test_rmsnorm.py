import platform

import torch

from rmsnorm.layers import RMSNorm


def test_rmsnorm():
    device = torch.device("xpu")
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
    rmsnorm_layer = RMSNorm()
    rmsnorm_layer.weight = torch.randn(1024, device=device, dtype=torch.bfloat16)
    rmsnorm_layer.variance_epsilon = 1e-6
    output = rmsnorm_layer(x)

    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rmsnorm_layer.variance_epsilon)
    ref_out = rmsnorm_layer.weight * x.to(torch.bfloat16)
    torch.testing.assert_close(output, ref_out)


def test_rmsnorm_backward():
    device = torch.device("xpu")
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device, requires_grad=True)
    rmsnorm_layer = RMSNorm()
    rmsnorm_layer.weight = torch.randn(1024, device=device, dtype=torch.float32, requires_grad=True)
    rmsnorm_layer.variance_epsilon = 1e-6
    output = rmsnorm_layer(x)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    x_ref = x.detach().requires_grad_()
    w_ref = rmsnorm_layer.weight.detach().requires_grad_()
    variance = x_ref.pow(2).mean(-1, keepdim=True)
    x_norm = x_ref * torch.rsqrt(variance + rmsnorm_layer.variance_epsilon)
    ref_out = w_ref * x_norm
    grad_output_ref = grad_output.detach()
    ref_out.backward(grad_output_ref)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(rmsnorm_layer.weight.grad, w_ref.grad, atol=1e-2, rtol=1e-2)
