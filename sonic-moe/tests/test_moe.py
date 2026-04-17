# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import pytest
import random

import numpy as np
import torch
from torch.testing import assert_close

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
    pytest.skip("SonicMoE requires Hopper (SM90) or newer GPU", allow_module_level=True)

try:
    from sonic_moe import KernelBackendMoE, MoE, enable_quack_gemm
    from sonic_moe.enums import ActivationType
except ImportError as e:
    pytest.skip(f"sonicmoe dependencies not available: {e}", allow_module_level=True)

_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


PROBLEM_SHAPES = [
    (8192, 768, 256, 128, 8),
    (8192, 768, 512, 64, 4),
    (8192, 4096, 512, 128, 8),
    (8192, 4096, 1024, 64, 4),
]


@pytest.mark.parametrize("problem_shape", PROBLEM_SHAPES)
@pytest.mark.parametrize("add_bias", [False, True])
def test_moe_forward_backward(problem_shape, add_bias):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    set_seed(_SEED)

    T, H, I, E, K = problem_shape
    with torch.device(device):
        moe = MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=ActivationType.SWIGLU,
            add_bias=add_bias,
            std=0.02,
        ).to(dtype=dtype)

    if add_bias:
        torch.nn.init.normal_(moe.c_fc.bias, 0, 0.01)
        torch.nn.init.normal_(moe.c_proj.bias, 0, 0.01)

    torch.cuda.empty_cache()
    x_torch = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
    x_kernel = x_torch.clone().detach().requires_grad_()

    with torch.autocast(device.type, torch.float32):
        y_kernel = moe(x_kernel, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]
        y_torch = moe(x_torch, kernel_backend_moe=KernelBackendMoE.torch)[0]

        assert_close(y_kernel.float(), y_torch.float(), atol=1.4e-2, rtol=2e-2)

    dy = 0.02 * torch.randn(T, H, device=device, dtype=dtype)
    W = list(moe.parameters())

    with torch.autocast(device.type, torch.float32):
        kernel_grads = torch.autograd.grad(y_kernel, [x_kernel] + W, grad_outputs=dy, retain_graph=True)
        torch_grads = torch.autograd.grad(y_torch, [x_torch] + W, grad_outputs=dy, retain_graph=True)

        for tg, kg in zip(torch_grads, kernel_grads):
            assert_close(kg.float(), tg.float(), atol=2e-2, rtol=2e-2)

    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "problem_shape",
    [(8192, 4096, 512, 128, 8)],
)
def test_moe_quack_gemm(problem_shape):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    set_seed(_SEED)

    T, H, I, E, K = problem_shape
    with torch.device(device):
        moe = MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(dtype=dtype)

    torch.cuda.empty_cache()
    x_torch = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
    x_kernel = x_torch.clone().detach().requires_grad_()

    with torch.autocast(device.type, torch.float32):
        with enable_quack_gemm(True):
            y_kernel = moe(x_kernel, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]

        y_torch = moe(x_torch, kernel_backend_moe=KernelBackendMoE.torch)[0]

        assert_close(y_kernel.float(), y_torch.float(), atol=1.4e-2, rtol=2e-2)

    torch.cuda.empty_cache()
