# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
from typing import Type

import kernels
import pytest
import torch
import torch.nn.functional as F

from .utils import opcheck
from .allclose_default import get_default_atol, get_default_rtol

activation = kernels.get_kernel("kernels-community/activation", version=1)

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]


def _devices() -> list[str]:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        name, count = "xpu", torch.xpu.device_count()
    elif torch.cuda.is_available():
        name, count = "cuda", torch.cuda.device_count()
    else:
        return ["cpu"]
    return [f"{name}:{i}" for i in range(1 if count == 1 else 2)]


DEVICES = _devices()


def gelu_fast(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_quick(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def fatrelu_and_mul(x: torch.Tensor, threshold: float) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    x1 = F.threshold(x1, threshold, 0.0)
    return x1 * x2


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def mul_and_silu(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return x[..., :d] * F.silu(x[..., d:])


def gelu_and_mul(x: torch.Tensor, approximate: str) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)

def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")

def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)

ACT_AND_MUL_NAMES = ["silu_and_mul", "mul_and_silu", "gelu", "gelu_tanh", "fatrelu"]


def act_and_mul_check(
    activation_name: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation_name == "silu_and_mul":
        torch_fn = silu_and_mul
        fn = activation.silu_and_mul
        op = activation.ops.silu_and_mul
        layer = activation.layers.SiluAndMul()
    elif activation_name == "mul_and_silu":
        torch_fn = mul_and_silu
        fn = activation.mul_and_silu
        op = activation.ops.mul_and_silu
        layer = activation.layers.MulAndSilu()
    elif activation_name == "gelu":
        torch_fn = lambda x: gelu_and_mul(x, "none")
        fn = activation.gelu_and_mul
        op = activation.ops.gelu_and_mul
        layer = activation.layers.GeluAndMul()
    elif activation_name == "gelu_tanh":
        torch_fn = lambda x: gelu_and_mul(x, "tanh")
        fn = activation.gelu_tanh_and_mul
        op = activation.ops.gelu_tanh_and_mul
        layer = activation.layers.GeluTanhAndMul()
    elif activation_name == "fatrelu":
        threshold = random.uniform(0, 1)
        torch_fn = lambda x: fatrelu_and_mul(x, threshold)
        fn = lambda out, x: activation.fatrelu_and_mul(out, x, threshold)
        op = activation.ops.fatrelu_and_mul
        layer = activation.layers.FatreluAndMul(threshold)

    out_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out = fn(out, x)
    mod_out = layer(x)
    ref_out = torch_fn(x)

    # The SiLU, GELU and FatReLU implementations are equivalent to the native
    # PyTorch implementations, so we can do exact comparison.
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)
    torch.testing.assert_close(mod_out, ref_out, atol=0.0, rtol=0.0)

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation_name == "fatrelu":
        opcheck(op, (out, x, threshold))
    else:
        opcheck(op, (out, x))


@pytest.mark.parametrize("activation_name", ACT_AND_MUL_NAMES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation_name: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    act_and_mul_check(activation_name, num_tokens, d, dtype, seed, device)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("activation_name", ACT_AND_MUL_NAMES)
@torch.inference_mode()
def test_act_and_mul_ci(activation_name: str) -> None:
    act_and_mul_check(
        activation_name,
        num_tokens=7,
        d=512,
        dtype=torch.half,
        seed=0,
        device=DEVICES[0],
    )


ACTIVATION_FNS = [
    (
        gelu_fast,
        activation.gelu_fast,
        activation.ops.gelu_fast,
        activation.layers.FastGELU,
    ),
    (
        gelu_new,
        activation.gelu_new,
        activation.ops.gelu_new,
        activation.layers.NewGELU,
    ),
    (
        gelu_quick,
        activation.gelu_quick,
        activation.ops.gelu_quick,
        activation.layers.QuickGELU,
    ),
    (
        gelu_tanh,
        activation.gelu_tanh,
        activation.ops.gelu_tanh,
        activation.layers.GeluTanh,
    ),
    (
        silu,
        activation.silu,
        activation.ops.silu,
        activation.layers.Silu,
    ),
    (
        gelu,
        activation.gelu,
        activation.ops.gelu,
        activation.layers.Gelu,
    ),
]


def activation_check(
    activation_fns,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    torch_fn, fn, op, cls = activation_fns
    layer = cls()
    out = fn(torch.empty_like(x), x)
    layer_out = layer(x)
    ref_out = torch_fn(x)
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )
    torch.testing.assert_close(
        out, layer_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )

    out = torch.empty_like(x)
    opcheck(op, (out, x))


@pytest.mark.parametrize("activation_fns", ACTIVATION_FNS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_activation(
    activation_fns,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    activation_check(activation_fns, num_tokens, d, dtype, seed, device)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("activation_fns", ACTIVATION_FNS)
@torch.inference_mode()
def test_activation_ci(activation_fns) -> None:
    activation_check(
        activation_fns,
        num_tokens=7,
        d=512,
        dtype=torch.half,
        seed=0,
        device=DEVICES[0],
    )


# The shapes above are all vector-width multiples over naturally aligned
# storage; these are neither, so a vectorized backend must take its scalar path.
@pytest.mark.parametrize("d", [1, 13, 4097])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_act_and_mul_unaligned(d: int, dtype: torch.dtype, device: str) -> None:
    torch.manual_seed(0)
    torch.set_default_device(device)
    num_tokens = 5
    base = torch.randn(num_tokens * 2 * d + 1, dtype=dtype)
    x = base[1:].view(num_tokens, 2 * d)

    out = torch.empty(num_tokens, d, dtype=dtype)
    out = activation.silu_and_mul(out, x)

    torch.testing.assert_close(out, silu_and_mul(x), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_empty_input(dtype: torch.dtype, device: str) -> None:
    torch.set_default_device(device)
    x = torch.empty(0, 128, dtype=dtype)

    assert activation.silu_and_mul(torch.empty(0, 64, dtype=dtype), x).shape == (0, 64)
    assert activation.silu(torch.empty(0, 128, dtype=dtype), x).shape == (0, 128)
