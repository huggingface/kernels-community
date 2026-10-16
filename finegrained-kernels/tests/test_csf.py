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

import pytest
import torch
from kernels import get_kernel

from utils import TEST_DEVICE


@pytest.fixture(scope="module")
def kernel():
    return get_kernel(
        "kernels-community/finegrained-kernels", version=0, check_arch=False
    )


def make_streams(experts=2, rows=32, cols=6, exceptions=True):
    """Independent fixture encoder; expected bytes do not use the decoder."""
    generator = torch.Generator().manual_seed(2026)
    bases = torch.randint(
        16, 64, (experts, rows, 1), dtype=torch.uint8, generator=generator
    )
    nibbles = torch.randint(
        0, 16, (experts, rows, cols), dtype=torch.uint8, generator=generator
    )
    packed = nibbles[:, :, 0::2] | (nibbles[:, :, 1::2] << 4)
    fixed = torch.cat(
        (
            bases.reshape(experts, rows // 16, 16),
            packed.reshape(experts, rows // 16, 16 * cols // 2),
        ),
        dim=-1,
    )
    expected = bases + nibbles
    streams = []
    for expert in range(experts):
        # First/last columns, first/last rows, slab boundary, multiple replacements per row.
        positions = (
            sorted(
                set(
                    [
                        0,
                        1,
                        cols - 1,
                        15 * cols,
                        16 * cols - 1,
                        16 * cols,
                        rows * cols - 1,
                    ]
                )
            )
            if exceptions
            else []
        )
        replacements = [
            (expert * 41 + index * 43) % 256 for index in range(len(positions))
        ]
        streams.append(
            torch.tensor(
                [(byte << 24) | pos for pos, byte in zip(positions, replacements)],
                dtype=torch.uint32,
            )
        )
        for pos, byte in zip(positions, replacements):
            expected[expert].view(-1)[pos] = byte
    return fixed, streams, expected


@pytest.mark.cpu
@pytest.mark.kernels_ci
@pytest.mark.parametrize("exceptions", [False, True])
def test_csf_cpu_exact_bytes(kernel, exceptions):
    fixed, streams, expected = make_streams(exceptions=exceptions)
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams)
    assert scales.shape == tuple(expected.shape)
    assert scales.fixed.numel() == fixed.numel()
    assert scales.row_offsets.numel() == expected.shape[0] * expected.shape[1] + 1
    actual = kernel.decompress_nvfp4_csf_scales(scales).view(torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.kernels_ci
@pytest.mark.parametrize("positions", [[1, 1], [3, 1], [32 * 6]])
def test_csf_rejects_invalid_exception_positions(kernel, positions):
    fixed, _, _ = make_streams(experts=1)
    with pytest.raises(ValueError, match="sorted, unique and in range"):
        kernel.prepare_nvfp4_csf_scales(
            fixed, torch.tensor(positions, dtype=torch.uint32)
        )


@pytest.mark.cpu
@pytest.mark.kernels_ci
def test_csf_rejects_invalid_fixed_stream(kernel):
    fixed, streams, _ = make_streams(experts=1)
    fixed[0, 0, 0] = 241
    with pytest.raises(ValueError, match="bases"):
        kernel.prepare_nvfp4_csf_scales(fixed, streams)
    with pytest.raises(ValueError, match="slab shape"):
        kernel.prepare_nvfp4_csf_scales(
            torch.zeros((1, 17), dtype=torch.uint8), streams
        )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("exceptions", [False, True])
def test_csf_gpu_exact_bytes(kernel, exceptions):
    fixed, streams, expected = make_streams(exceptions=exceptions)
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams).to(TEST_DEVICE)
    actual = kernel.decompress_nvfp4_csf_scales(scales).cpu().view(torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("activation_format", ["bf16", "nvfp4"])
def test_csf_gemm_matches_dense_scales(
    kernel, grouped, activation_format, gate, compiled
):
    E, N, K = 2, 128, 256
    S = 32 if grouped else 3
    fixed, streams, _ = make_streams(E, N, K // 16)
    # Finite positive scales: the exact byte decoder tests separately cover all sign/NaN codes.
    streams = [
        ((stream.to(torch.int64) & 0xFFFFFF) | (48 << 24)).to(torch.uint32)
        for stream in streams
    ]
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams)
    dense = kernel.decompress_nvfp4_csf_scales(scales).to(TEST_DEVICE)
    scales = scales.to(TEST_DEVICE)
    A = torch.randn((S, K), device=TEST_DEVICE, dtype=torch.bfloat16) * 0.2
    B = torch.randint(-128, 127, (E, N, K // 2), device=TEST_DEVICE, dtype=torch.int8)
    kwargs = dict(
        activation_format=activation_format,
        gate=gate,
        b_global_scale=torch.tensor([0.25, 0.5], device=TEST_DEVICE),
        bias=torch.randn((E, N), device=TEST_DEVICE, dtype=torch.bfloat16),
        gather_idx=torch.arange(S - 1, -1, -1, device=TEST_DEVICE, dtype=torch.int32),
        scatter_idx=torch.arange(S, device=TEST_DEVICE, dtype=torch.int32).roll(1),
    )
    if grouped:
        kwargs["expert_start"] = torch.tensor(
            [0, S // 2, S], device=TEST_DEVICE, dtype=torch.int32
        )
        op, reference = kernel.matmul_csf_grouped, kernel.matmul_grouped
    else:
        kwargs["expert_ids"] = torch.tensor(
            [0, 1, 0], device=TEST_DEVICE, dtype=torch.int32
        )
        op, reference = kernel.matmul_csf_batched, kernel.matmul_batched
    if compiled:
        op = torch.compile(op, fullgraph=True)
    actual = op(A, B, scales, **kwargs)
    expected = reference(A, B, Bs=dense, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


@pytest.mark.cpu
@pytest.mark.kernels_ci
def test_csf_u24_position_boundary(kernel):
    cols = (1 << 24) // 16
    fixed = torch.zeros((1, 16 * (1 + cols // 2)), dtype=torch.uint8)
    scales = kernel.prepare_nvfp4_csf_scales(
        fixed, torch.tensor([0xFFFFFFFF], dtype=torch.uint32)
    )
    assert scales.shape == (1, 16, cols)
    assert scales.search_steps == 1
    with pytest.raises(ValueError, match=r"2\*\*24"):
        kernel.prepare_nvfp4_csf_scales(
            torch.zeros((1, fixed.shape[1] + 16), dtype=torch.uint8),
            torch.empty(0, dtype=torch.uint32),
        )


@pytest.mark.parametrize(
    "device", [pytest.param("cpu", marks=pytest.mark.cpu), TEST_DEVICE]
)
def test_csf_all_byte_patterns_and_full_exception_row(kernel, device):
    fixed = torch.zeros((2, 16 * (1 + 16 // 2)), dtype=torch.uint8)
    positions = torch.arange(32 * 16, dtype=torch.int64)
    expected = (positions % 256).to(torch.uint8).reshape(1, 32, 16)
    exceptions = (positions | ((positions % 256) << 24)).to(torch.uint32)
    scales = kernel.prepare_nvfp4_csf_scales(fixed, exceptions).to(device)
    assert scales.search_steps == 5
    actual = kernel.decompress_nvfp4_csf_scales(scales).cpu().view(torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("grouped", [False, True])
def test_csf_prequantized_activations_and_requant(kernel, grouped):
    E, N, K, S = 2, 128, 256, 32
    fixed, _, _ = make_streams(E, N, K // 16, exceptions=False)
    scales = kernel.prepare_nvfp4_csf_scales(
        fixed, [torch.empty(0, dtype=torch.uint32)] * E
    )
    dense = kernel.decompress_nvfp4_csf_scales(scales).to(TEST_DEVICE)
    scales = scales.to(TEST_DEVICE)
    global_scale = torch.tensor([0.5], device=TEST_DEVICE)
    A, As = kernel.nvfp4_act_quant(
        torch.randn((S, K), device=TEST_DEVICE, dtype=torch.bfloat16) * 0.05,
        global_scale=global_scale,
    )
    B = torch.randint(-128, 127, (E, N, K // 2), device=TEST_DEVICE, dtype=torch.int8)
    kwargs = dict(
        As=As,
        quantize_output=True,
        gate=True,
        a_global_scale=global_scale,
        b_global_scale=torch.tensor([0.25, 0.5], device=TEST_DEVICE),
        output_global_scale=torch.tensor([4.0], device=TEST_DEVICE),
    )
    if grouped:
        kwargs["expert_start"] = torch.tensor(
            [0, S // 2, S], device=TEST_DEVICE, dtype=torch.int32
        )
        op, reference = kernel.matmul_csf_grouped, kernel.matmul_grouped
    else:
        kwargs["expert_ids"] = torch.arange(
            E, device=TEST_DEVICE, dtype=torch.int32
        ).repeat_interleave(S // E)
        op, reference = kernel.matmul_csf_batched, kernel.matmul_batched
    actual, expected = op(A, B, scales, **kwargs), reference(A, B, Bs=dense, **kwargs)
    assert actual[0].dtype == torch.int8
    assert actual[1].dtype == torch.float8_e4m3fn
    # Compare quantized values rather than demanding identical rounding at grid boundaries.
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=TEST_DEVICE,
    )

    def unpack(result):
        packed, scales = result
        bits = packed.view(torch.uint8)
        codes = torch.stack((bits & 15, bits >> 4), dim=-1).flatten(-2)
        return values[codes.long()] * scales.float().repeat_interleave(16, dim=-1)

    torch.testing.assert_close(unpack(actual), unpack(expected), rtol=0.2, atol=0.3)


@pytest.mark.cpu
@pytest.mark.kernels_ci
@pytest.mark.parametrize("gradient_input", ["activation", "bias"])
def test_csf_rejects_training(kernel, gradient_input):
    fixed, streams, _ = make_streams(1, 32, 16)
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams)
    A = torch.zeros((1, 256), requires_grad=gradient_input == "activation")
    B = torch.zeros((1, 32, 128), dtype=torch.int8)
    bias = torch.zeros((1, 32), requires_grad=gradient_input == "bias")
    with pytest.raises(ValueError, match="inference only"):
        kernel.matmul_csf_batched(
            A, B, scales, expert_ids=torch.zeros(1, dtype=torch.int32), bias=bias
        )


@pytest.mark.cpu
@pytest.mark.kernels_ci
def test_csf_rejects_non_gate_requantization(kernel):
    fixed, streams, _ = make_streams(1, 32, 16)
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams)
    with pytest.raises(ValueError, match="requires gate=True"):
        kernel.matmul_csf_batched(
            torch.zeros((1, 256)),
            torch.zeros((1, 32, 128), dtype=torch.int8),
            scales,
            expert_ids=torch.zeros(1, dtype=torch.int32),
            quantize_output=True,
        )


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "device", [pytest.param("cpu", marks=pytest.mark.cpu), TEST_DEVICE]
)
def test_csf_heterogeneous_expert_exception_counts(kernel, device):
    fixed, _, expected = make_streams(experts=3, exceptions=False)
    words = [[(77 << 24) | 191], [], [(255 << 24) | 0, (12 << 24) | 96]]
    streams = [torch.tensor(stream, dtype=torch.uint32) for stream in words]
    for expert, stream in enumerate(words):
        for word in stream:
            expected[expert].view(-1)[word & 0xFFFFFF] = word >> 24
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams).to(device)
    actual = kernel.decompress_nvfp4_csf_scales(scales).cpu().view(torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("grouped", [False, True])
def test_standard_weight_only_autograd_still_works(kernel, grouped):
    # CSF adds optional arguments to shared ops. Standard dense-scale users must
    # retain their existing dgrad registration and positional argument contract.
    E, N, K, S = 2, 128, 256, 4
    A = torch.randn(
        (S, K), device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    B = torch.full(
        (E, N, K // 2), 0x22, device=TEST_DEVICE, dtype=torch.int8
    )  # E2M1 ones
    scales = torch.ones((E, N, K // 16), device=TEST_DEVICE, dtype=torch.float8_e4m3fn)
    kwargs = dict(
        Bs=scales,
        activation_format="bf16",
        b_global_scale=torch.tensor([0.5, 1.0], device=TEST_DEVICE),
    )
    if grouped:
        kwargs["expert_start"] = torch.tensor(
            [0, 2, 4], device=TEST_DEVICE, dtype=torch.int32
        )
        op = kernel.matmul_grouped
    else:
        kwargs["expert_ids"] = torch.tensor(
            [0, 0, 1, 1], device=TEST_DEVICE, dtype=torch.int32
        )
        op = kernel.matmul_batched
    op(A, B, **kwargs).sum().backward()
    expected = torch.tensor(
        [64, 64, 128, 128], device=TEST_DEVICE, dtype=torch.bfloat16
    )[:, None].expand(S, K)
    torch.testing.assert_close(A.grad, expected, rtol=0, atol=0)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="Forces a CUDA tuner configuration")
def test_csf_weight_only_clamped_k_tail(kernel, monkeypatch):
    import importlib

    import triton

    # BK256 slides the final tile back to K640, so scales must apply the
    # matching negative K offset rather than read beyond their 40 columns.
    tuner = importlib.import_module(
        kernel.__name__ + ".batched"
    ).mx_weight_only_matmul_batched_kernel
    monkeypatch.setattr(
        tuner,
        "configs",
        [
            triton.Config(
                dict(
                    BLOCK_SIZE_N=32,
                    BLOCK_SIZE_K=256,
                    BLOCK_SIZE_M=1,
                    COMPUTE_MODE="scalar",
                    SWAP_AB=False,
                ),
                num_warps=4,
                num_stages=2,
            )
        ],
    )
    fixed, streams, _ = make_streams(1, 160, 40)
    streams = [
        ((stream.to(torch.int64) & 0xFFFFFF) | (48 << 24)).to(torch.uint32)
        for stream in streams
    ]
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams)
    dense = kernel.decompress_nvfp4_csf_scales(scales).to(TEST_DEVICE)
    scales = scales.to(TEST_DEVICE)
    A = torch.randn((2, 640), device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    B = torch.randint(-128, 127, (1, 160, 320), device=TEST_DEVICE, dtype=torch.int8)
    kwargs = dict(
        expert_ids=torch.zeros(2, device=TEST_DEVICE, dtype=torch.int32),
        activation_format="bf16",
    )
    actual = kernel.matmul_csf_batched(A, B, scales, **kwargs)
    expected = kernel.matmul_batched(A, B, Bs=dense, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


@pytest.mark.skipif(
    TEST_DEVICE != "cuda" or torch.cuda.get_device_capability()[0] != 10,
    reason="Forces Blackwell SM10x native NVFP4 staging",
)
def test_csf_native_nvfp4_scaled_mma(kernel, monkeypatch):
    import importlib

    import triton

    tuner = importlib.import_module(
        kernel.__name__ + ".batched"
    ).mx_dynamic_matmul_batched_kernel
    monkeypatch.setattr(
        tuner,
        "configs",
        [
            triton.Config(
                dict(
                    BLOCK_SIZE_N=128,
                    BLOCK_SIZE_K=128,
                    BLOCK_SIZE_M=1,
                    COMPUTE_MODE="dot_scaled",
                    SWAP_AB=True,
                ),
                num_warps=4,
                num_stages=2,
            )
        ],
    )
    fixed, streams, _ = make_streams(2, 128, 16)
    streams = [
        ((stream.to(torch.int64) & 0xFFFFFF) | (48 << 24)).to(torch.uint32)
        for stream in streams
    ]
    scales = kernel.prepare_nvfp4_csf_scales(fixed, streams)
    dense = kernel.decompress_nvfp4_csf_scales(scales).to(TEST_DEVICE)
    scales = scales.to(TEST_DEVICE)
    A = torch.randn((2, 256), device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    B = torch.randint(-128, 127, (2, 128, 128), device=TEST_DEVICE, dtype=torch.int8)
    kwargs = dict(
        expert_ids=torch.tensor([0, 1], device=TEST_DEVICE, dtype=torch.int32)
    )
    actual = kernel.matmul_csf_batched(A, B, scales, **kwargs)
    expected = kernel.matmul_batched(A, B, Bs=dense, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
