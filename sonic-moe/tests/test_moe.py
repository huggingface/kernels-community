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
    from sonic_moe import KernelBackendMoE, MoE, enable_quack_gemm, moe_general_routing_inputs
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


# ────────────────── is_concatenated_gate_up tests ──────────────────


def _make_interleaved_and_concatenated_weights(E, I, H, device, dtype):
    """Create matched interleaved and concatenated weight pairs."""
    w1_inter = torch.randn(E, 2 * I, H, device=device, dtype=dtype).permute(1, 2, 0)
    gate = w1_inter[0::2].permute(2, 0, 1)  # (E, I, H)
    up = w1_inter[1::2].permute(2, 0, 1)    # (E, I, H)
    w1_concat = torch.cat([gate, up], dim=1).contiguous().permute(1, 2, 0)
    w2 = torch.randn(E, H, I, device=device, dtype=dtype).permute(1, 2, 0)
    return w1_inter, w1_concat, w2


def _make_routing(T, E, K, device):
    """Create routing inputs for moe_general_routing_inputs."""
    topk_indices = torch.randint(0, E, (T, K), device=device)
    topk_weights = torch.randn(T, K, device=device, dtype=torch.bfloat16).softmax(dim=-1)
    token_idx = torch.arange(T, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, K).reshape(-1)
    expert_ids = topk_indices.reshape(-1).to(torch.int32)
    router_scores = topk_weights.reshape(-1).to(torch.bfloat16)
    return token_idx, expert_ids, router_scores


CONCAT_SHAPES = [
    # (T, H, I, E, K)
    (128, 768, 256, 128, 8),
    (1024, 768, 512, 64, 4),
    (256, 4096, 512, 128, 8),
    (4096, 4096, 1024, 64, 4),
]

CONCAT_ACTIVATIONS = [ActivationType.SWIGLU, ActivationType.GEGLU, ActivationType.REGLU]


@pytest.mark.parametrize("problem_shape", CONCAT_SHAPES)
@pytest.mark.parametrize("activation_type", CONCAT_ACTIVATIONS)
def test_concatenated_gate_up(problem_shape, activation_type):
    """Verify concatenated layout produces bit-exact results vs interleaved."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    T, H, I, E, K = problem_shape

    set_seed(_SEED)

    w1_inter, w1_concat, w2 = _make_interleaved_and_concatenated_weights(E, I, H, device, dtype)
    x = 0.02 * torch.randn(T, H, device=device, dtype=dtype)
    token_idx, expert_ids, router_scores = _make_routing(T, E, K, device)
    stream_id = torch.cuda.current_stream(device).cuda_stream

    out_ref, _ = moe_general_routing_inputs(
        x, router_scores, token_idx, expert_ids,
        w1_inter, None, w2, None,
        E, stream_id, activation_type,
        is_inference_mode_enabled=True,
        is_concatenated_gate_up=False,
    )
    out_test, _ = moe_general_routing_inputs(
        x, router_scores, token_idx, expert_ids,
        w1_concat, None, w2, None,
        E, stream_id, activation_type,
        is_inference_mode_enabled=True,
        is_concatenated_gate_up=True,
    )

    assert torch.equal(out_ref, out_test), (
        f"Mismatch: max_diff={(out_ref.float() - out_test.float()).abs().max().item()}"
    )

    torch.cuda.empty_cache()


CONCAT_BACKWARD_SHAPES = [
    (256, 768, 256, 128, 8),
    (1024, 4096, 512, 64, 4),
]


@pytest.mark.parametrize("problem_shape", CONCAT_BACKWARD_SHAPES)
def test_concatenated_gate_up_backward(problem_shape):
    """Verify gradients match between interleaved and concatenated layouts."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    T, H, I, E, K = problem_shape

    set_seed(_SEED)

    w1_inter, w1_concat, w2 = _make_interleaved_and_concatenated_weights(E, I, H, device, dtype)
    x = 0.02 * torch.randn(T, H, device=device, dtype=dtype, requires_grad=True)
    x_clone = x.clone().detach().requires_grad_()
    token_idx, expert_ids, router_scores = _make_routing(T, E, K, device)
    stream_id = torch.cuda.current_stream(device).cuda_stream

    # Need w1 to require grad for dw1
    w1_inter_param = w1_inter.clone().detach().requires_grad_()
    w1_concat_param = w1_concat.clone().detach().requires_grad_()

    out_ref, _ = moe_general_routing_inputs(
        x, router_scores, token_idx, expert_ids,
        w1_inter_param, None, w2, None,
        E, stream_id, ActivationType.SWIGLU,
        is_inference_mode_enabled=False,
        is_concatenated_gate_up=False,
    )
    out_test, _ = moe_general_routing_inputs(
        x_clone, router_scores, token_idx, expert_ids,
        w1_concat_param, None, w2, None,
        E, stream_id, ActivationType.SWIGLU,
        is_inference_mode_enabled=False,
        is_concatenated_gate_up=True,
    )

    dy = 0.02 * torch.randn_like(out_ref)

    grads_ref = torch.autograd.grad(out_ref, [x, w1_inter_param], grad_outputs=dy)
    grads_test = torch.autograd.grad(out_test, [x_clone, w1_concat_param], grad_outputs=dy)

    # dx should match exactly
    assert_close(grads_ref[0].float(), grads_test[0].float(), atol=1e-2, rtol=1e-2), "dx mismatch"

    # dw1: interleaved grad vs concatenated grad — compare after mapping to same layout
    dw1_inter = grads_ref[1]  # interleaved layout
    dw1_concat = grads_test[1]  # concatenated layout
    # Convert interleaved dw1 to concatenated for comparison
    dw1_inter_as_concat_gate = dw1_inter[0::2].permute(2, 0, 1)
    dw1_inter_as_concat_up = dw1_inter[1::2].permute(2, 0, 1)
    dw1_inter_as_concat = torch.cat([dw1_inter_as_concat_gate, dw1_inter_as_concat_up], dim=1).contiguous().permute(1, 2, 0)

    assert_close(dw1_inter_as_concat.float(), dw1_concat.float(), atol=1e-2, rtol=1e-2), "dw1 mismatch"

    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "problem_shape",
    [(256, 768, 256, 128, 8), (1024, 4096, 512, 64, 4)],
)
def test_concatenated_gate_up_with_bias(problem_shape):
    """Verify concatenated layout with bias produces bit-exact results vs interleaved."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    T, H, I, E, K = problem_shape

    set_seed(_SEED)

    w1_inter, w1_concat, w2 = _make_interleaved_and_concatenated_weights(E, I, H, device, dtype)
    b1 = torch.randn(E, 2 * I, device=device, dtype=dtype)
    b2 = torch.randn(E, H, device=device, dtype=dtype)
    x = 0.02 * torch.randn(T, H, device=device, dtype=dtype)
    token_idx, expert_ids, router_scores = _make_routing(T, E, K, device)
    stream_id = torch.cuda.current_stream(device).cuda_stream

    out_ref, _ = moe_general_routing_inputs(
        x, router_scores, token_idx, expert_ids,
        w1_inter, b1, w2, b2,
        E, stream_id, ActivationType.SWIGLU,
        is_inference_mode_enabled=True,
        is_concatenated_gate_up=False,
    )
    out_test, _ = moe_general_routing_inputs(
        x, router_scores, token_idx, expert_ids,
        w1_concat, b1, w2, b2,
        E, stream_id, ActivationType.SWIGLU,
        is_inference_mode_enabled=True,
        is_concatenated_gate_up=True,
    )

    assert torch.equal(out_ref, out_test), (
        f"Mismatch: max_diff={(out_ref.float() - out_test.float()).abs().max().item()}"
    )

    torch.cuda.empty_cache()
