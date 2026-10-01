# Copyright 2026 Google LLC and The HuggingFace Inc. team. All rights reserved.
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

import importlib.util

import pytest
import torch
import torch.nn.functional as F

if importlib.util.find_spec("kernels") is not None:
    import kernels

    gemma_quant_kernels = kernels.get_kernel("kernels-community/gemma-quant-kernels", version=1)
else:
    import gemma_quant_kernels  # type: ignore


def _make_packed_weights(out_features: int, in_features: int, num_bits: int, device: str = "cuda"):
    torch.manual_seed(42)
    max_int = 127.0 if num_bits == 8 else (7.0 if num_bits == 4 else 2.0)
    scale = (torch.rand(out_features, 1, dtype=torch.float32, device=device) * 0.1 + 0.01) / max_int
    if num_bits == 4:
        int_w = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8, device=device)
        packed_w = gemma_quant_kernels.pack_int4(int_w)
    elif num_bits == 2:
        int_w = torch.randint(-2, 2, (out_features, in_features), dtype=torch.int8, device=device)
        packed_w = gemma_quant_kernels.pack_int2(int_w)
    elif num_bits == 8:
        int_w = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8, device=device)
        packed_w = int_w
    else:
        raise ValueError(f"Unsupported num_bits: {num_bits}")
    return int_w, packed_w, scale


def _ref_linear(x: torch.Tensor, int_w: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None):
    w_fp32 = int_w.float() * scale.float()
    b_fp32 = bias.float() if bias is not None else None
    return F.linear(x.float(), w_fp32, b_fp32).to(x.dtype)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("num_bits", [2, 4, 8])
@pytest.mark.parametrize("has_bias", [False, True])
def test_lowbit_gemv(num_bits: int, has_bias: bool):
    device = "cuda"
    in_features, out_features = 256, 192
    int_w, packed_w, scale = _make_packed_weights(out_features, in_features, num_bits, device=device)
    bias = torch.randn(out_features, dtype=torch.bfloat16, device=device) if has_bias else None

    x = torch.randn(1, in_features, dtype=torch.bfloat16, device=device)
    expected = _ref_linear(x, int_w, scale, bias)

    actual = gemma_quant_kernels.lowbit_gemm(x, packed_w, scale, bias, num_bits=num_bits)
    cos_sim = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert cos_sim.item() > 0.9999
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("num_bits", [2, 4, 8])
@pytest.mark.parametrize("M", [8, 37])
def test_lowbit_gemm(num_bits: int, M: int):
    device = "cuda"
    in_features, out_features = 256, 192
    int_w, packed_w, scale = _make_packed_weights(out_features, in_features, num_bits, device=device)
    bias = torch.randn(out_features, dtype=torch.bfloat16, device=device)

    x = torch.randn(M, in_features, dtype=torch.bfloat16, device=device)
    expected = _ref_linear(x, int_w, scale, bias)

    actual = gemma_quant_kernels.lowbit_gemm(x, packed_w, scale, bias, num_bits=num_bits)
    cos_sim = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert cos_sim.item() > 0.9999
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("num_bits", [2, 4, 8])
def test_grouped_lowbit_gemm(num_bits: int):
    device = "cuda"
    num_experts = 4
    N = 128
    K = 256
    counts = torch.tensor([5, 0, 19, 7], dtype=torch.int32, device=device)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    total_tokens = int(offsets[-1].item())

    permuted_x = torch.randn(total_tokens, K, dtype=torch.bfloat16, device=device)
    max_int = 127.0 if num_bits == 8 else (7.0 if num_bits == 4 else 2.0)
    scales = (torch.rand(num_experts, N, 1, dtype=torch.float32, device=device) * 0.1 + 0.01) / max_int

    if num_bits == 4:
        int_w = torch.randint(-8, 8, (num_experts, N, K), dtype=torch.int8, device=device)
        packed_w = gemma_quant_kernels.pack_int4(int_w)
    elif num_bits == 2:
        int_w = torch.randint(-2, 2, (num_experts, N, K), dtype=torch.int8, device=device)
        packed_w = gemma_quant_kernels.pack_int2(int_w)
    else:
        int_w = torch.randint(-128, 127, (num_experts, N, K), dtype=torch.int8, device=device)
        packed_w = int_w

    expected = torch.empty((total_tokens, N), dtype=permuted_x.dtype, device=device)
    for e in range(num_experts):
        s, end = int(offsets[e].item()), int(offsets[e + 1].item())
        if end > s:
            expected[s:end] = _ref_linear(permuted_x[s:end], int_w[e], scales[e])

    actual = gemma_quant_kernels.grouped_lowbit_gemm(
        permuted_x, packed_w, scales, offsets, num_bits=num_bits
    )
    cos_sim = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert cos_sim.item() > 0.9999
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_unaligned_and_dtypes(dtype: torch.dtype):
    device = "cuda"
    in_features, out_features = 131, 93
    int_w, packed_w, scale = _make_packed_weights(out_features, in_features, num_bits=4, device=device)

    x = torch.randn(2, 5, in_features, dtype=dtype, device=device)
    expected = _ref_linear(x, int_w, scale)

    actual = gemma_quant_kernels.lowbit_gemm(x, packed_w, scale, None, num_bits=4)
    cos_sim = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert cos_sim.item() > 0.9999
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(7, 131), (4, 6, 256)])
def test_pack_unpack_roundtrip(shape: tuple[int, ...]):
    w4 = torch.randint(-8, 8, shape, dtype=torch.int8)
    assert torch.equal(gemma_quant_kernels.unpack_int4(gemma_quant_kernels.pack_int4(w4), shape[-1]), w4)

    w2 = torch.randint(-2, 2, shape, dtype=torch.int8)
    assert torch.equal(gemma_quant_kernels.unpack_int2(gemma_quant_kernels.pack_int2(w2), shape[-1]), w2)
