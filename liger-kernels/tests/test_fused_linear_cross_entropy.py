import pytest
import torch
import torch.nn.functional as F

import kernels

liger_kernels = kernels.get_kernel("kernels-community/liger-kernels", version=4)

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA or ROCm"),
]


def test_nonuniform_token_gradients():
    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    weight = torch.randn(17, 8, device="cuda", requires_grad=True)
    bias = torch.randn(17, device="cuda", requires_grad=True)
    target = torch.arange(4, device="cuda")
    grad_output = torch.tensor([0.1, 0.3, 0.7, 1.5], device="cuda")

    expected = F.cross_entropy(F.linear(x, weight, bias), target, reduction="none")
    actual = liger_kernels.layers.liger_fused_linear_cross_entropy(x, weight, target, bias=bias, reduction="none")

    expected_grads = torch.autograd.grad(expected, (x, weight, bias), grad_output)
    actual_grads = torch.autograd.grad(actual, (x, weight, bias), grad_output)

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)


def test_zero_scaled_loss_with_transposed_weight():
    x = torch.ones(2, 4, device="cuda", requires_grad=True)
    weight = torch.zeros(4, 3, device="cuda").t().requires_grad_()
    target = torch.zeros(2, device="cuda", dtype=torch.long)

    loss = liger_kernels.layers.liger_fused_linear_cross_entropy(x, weight, target)
    (0.0 * loss).backward()

    torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight), atol=0, rtol=0)
