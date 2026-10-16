# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP8 W8A16 kernels against independent dequantization, routing, and GEMM references."""

import kernels
import pytest
import torch

TEST_DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else None
)
pytestmark = pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator required")


@pytest.fixture(scope="module")
def kernel():
    return kernels.get_kernel("kernels-community/finegrained-kernels", version=0)


def _weight_reference(
    weight, scales, block_n, block_k, gate=False, dtype=torch.bfloat16
):
    if scales.dtype == torch.uint8:
        scales = torch.pow(2.0, scales.float() - 127)
    rows = weight.shape[-2]
    if scales.shape[-2] == 1:
        scale_rows = torch.zeros(rows, device=weight.device, dtype=torch.long)
    elif gate:
        row = torch.arange(rows, device=weight.device)
        scale_rows = (row // 2 // block_n) * 2 + row % 2
    else:
        scale_rows = torch.arange(rows, device=weight.device) // block_n
    return (
        weight.float() * scales[..., scale_rows, :].repeat_interleave(block_k, -1)
    ).to(dtype)


@pytest.mark.parametrize(
    "m,n,k,block_n,block_k,gate,dtype",
    [
        pytest.param(
            7, 256, 256, 128, 128, False, torch.bfloat16, marks=pytest.mark.kernels_ci
        ),
        (33, 192, 256, 128, 128, False, torch.float16),
        pytest.param(
            1, 128, 256, 256, 256, True, torch.bfloat16, marks=pytest.mark.kernels_ci
        ),
        (17, 256, 256, 64, 128, True, torch.bfloat16),
        (7, 128, 256, 256, 128, True, torch.bfloat16),
        (7, 192, 256, 128, 128, True, torch.bfloat16),
    ],
)
def test_block_fp8_weight_only(kernel, m, n, k, block_n, block_k, gate, dtype):
    torch.manual_seed(42)
    a = torch.randn(m, k, device=TEST_DEVICE, dtype=dtype)
    rows = 2 * n if gate else n
    b = torch.randn(rows, k, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    scale_rows = (rows + block_n - 1) // block_n
    if gate and scale_rows > 1:
        scale_rows = 2 * ((n + block_n - 1) // block_n)
    scales = torch.pow(
        2.0, torch.randint(-3, 3, (scale_rows, k // block_k), device=TEST_DEVICE)
    )
    bias = torch.randn(rows, device=TEST_DEVICE, dtype=dtype)
    dequantized = _weight_reference(b, scales, block_n, block_k, gate, dtype)
    reference = a.float() @ dequantized.float().T + bias.float()
    if gate:
        reference = torch.nn.functional.silu(reference[:, 0::2]) * reference[:, 1::2]
    with torch.no_grad():
        actual = kernel.matmul_2d(
            a,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=gate,
            bias=bias,
            output_dtype=dtype,
        )
    torch.testing.assert_close(actual, reference.to(dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("scalar", [False, True])
def test_block_fp8_weight_only_compiles_fullgraph(kernel, scalar):
    torch.manual_seed(42)
    a = torch.randn(5, 256, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 256, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    scales = (
        torch.tensor(0.25, device=TEST_DEVICE)
        if scalar
        else torch.tensor([[0.25, 0.5], [1.0, 2.0]], device=TEST_DEVICE)
    )

    def forward(a, b, scales):
        return kernel.matmul_2d(
            a, b, None, scales, activation_format="bf16", output_dtype=a.dtype
        )

    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
    with torch.no_grad():
        expected = forward(a, b, scales)
        torch.testing.assert_close(compiled(a, b, scales), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            compiled(a + 1, b, scales), forward(a + 1, b, scales), rtol=0, atol=0
        )


@pytest.mark.parametrize("op", ["batched", "grouped"])
@pytest.mark.parametrize("per_tensor", [False, True])
def test_routed_block_fp8_weight_only(kernel, op, per_tensor):
    torch.manual_seed(17)
    experts, rows, n, k = 3, 17, 128, 256
    a = torch.randn(5, k, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(experts, 2 * n, k, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    exponents = torch.randint(
        -3, 2, (experts, 1, 1) if per_tensor else (experts, 2, 2), device=TEST_DEVICE
    )
    scales = (exponents + 127).to(torch.uint8)
    bias = torch.randn(experts, 2 * n, device=TEST_DEVICE, dtype=a.dtype)
    ids = torch.tensor(([0, 2, 0, 2] * 4) + [3], device=TEST_DEVICE, dtype=torch.int32)
    gather = (torch.arange(rows, device=TEST_DEVICE) % 5).to(torch.int32)
    scatter = torch.randperm(rows, device=TEST_DEVICE).to(torch.int32)
    reference = torch.zeros(rows, n, device=TEST_DEVICE, dtype=a.dtype)
    weights = _weight_reference(
        b, scales, 2 * n if per_tensor else 128, k if per_tensor else 128, True
    )
    for row in range(rows - 1):
        expert = ids[row]
        projection = (
            a[gather[row]].float() @ weights[expert].float().T + bias[expert].float()
        )
        reference[scatter[row]] = (
            torch.nn.functional.silu(projection[0::2]) * projection[1::2]
        ).to(a.dtype)
    if op == "batched":
        kwargs = dict(expert_ids=ids, gather_idx=gather, scatter_idx=scatter)
        fn = kernel.matmul_batched
    else:
        order = torch.argsort(ids)
        # Expert 1 is empty; pad E=3 to a power of two. The off-rank route
        # sits past the local schedule, with a valid entry in both maps.
        starts = torch.tensor([0, 8, 8, 16, 16], device=TEST_DEVICE, dtype=torch.int32)
        kwargs = dict(
            expert_start=starts, gather_idx=gather[order], scatter_idx=scatter[order]
        )
        fn = kernel.matmul_grouped
    valid_rows = scatter[:-1].long()
    with torch.no_grad():
        result = fn(
            a,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=True,
            bias=bias,
            output_dtype=a.dtype,
            **kwargs,
        )
        torch.testing.assert_close(
            result[valid_rows], reference[valid_rows], rtol=2e-3, atol=2e-3
        )
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        actual = compiled(
            a,
            b,
            None,
            scales,
            activation_format="bf16",
            gate=True,
            bias=bias,
            output_dtype=a.dtype,
            **kwargs,
        )
        torch.testing.assert_close(
            actual[valid_rows], result[valid_rows], rtol=0, atol=0
        )


@pytest.mark.parametrize("op", ["batched", "grouped"])
def test_weight_only_fp8_moe(kernel, op):
    torch.manual_seed(21)
    e, h, i = 4, 256, 128
    hidden = torch.randn(5, h, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.05
    gate_up = (torch.randn(e, 2 * i, h, device=TEST_DEVICE) * 0.1).to(
        torch.float8_e4m3fn
    )
    down = (torch.randn(e, h, i, device=TEST_DEVICE) * 0.1).to(torch.float8_e4m3fn)
    gate_scales = torch.pow(2.0, torch.randint(-2, 1, (e, 2, 2), device=TEST_DEVICE))
    down_scales = torch.pow(2.0, torch.randint(-2, 1, (e, 2, 1), device=TEST_DEVICE))
    ids = torch.tensor([[0, 2], [2, 4], [0, 2], [4, 4], [0, 0]], device=TEST_DEVICE)
    routing = torch.full((5, 2), 0.5, device=TEST_DEVICE)
    gate_weight = _weight_reference(gate_up, gate_scales, 128, 128, True)
    down_weight = _weight_reference(down, down_scales, 128, 128)
    reference = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(5):
        for slot in range(2):
            expert = int(ids[token, slot])
            if expert >= e:
                continue
            projections = hidden[token].float() @ gate_weight[expert].float().T
            inter = (
                torch.nn.functional.silu(projections[0::2]) * projections[1::2]
            ).to(hidden.dtype)
            contribution = (inter.float() @ down_weight[expert].float().T).to(
                hidden.dtype
            )
            reference[token] += contribution.float() * routing[token, slot]
    fn = getattr(kernel, "moe_fused_" + op)
    with torch.no_grad():
        actual = fn(
            hidden,
            ids,
            routing,
            gate_up,
            down,
            gate_scales,
            down_scales,
            activation_format="bf16",
        )
    torch.testing.assert_close(actual, reference.to(actual.dtype), rtol=2e-2, atol=1e-5)


@pytest.mark.parametrize("op", ["batched", "grouped"])
def test_raw_activation_launch_keeps_full_precision_weights(kernel, op):
    torch.manual_seed(34)
    a = torch.randn(4, 128, device=TEST_DEVICE, dtype=torch.bfloat16)
    b = torch.randn(2, 128, 128, device=TEST_DEVICE, dtype=a.dtype)
    ids = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE, dtype=torch.int32)
    kwargs = (
        dict(expert_ids=ids)
        if op == "batched"
        else dict(
            expert_start=torch.tensor([0, 2, 4], device=TEST_DEVICE, dtype=torch.int32)
        )
    )
    with torch.no_grad():
        actual = getattr(kernel, "matmul_" + op)(
            a, b, None, None, activation_format="bf16", **kwargs
        )
    reference = (
        torch.bmm(a[:, None, :].float(), b[ids.long()].transpose(1, 2).float())
        .squeeze(1)
        .to(a.dtype)
    )
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)
