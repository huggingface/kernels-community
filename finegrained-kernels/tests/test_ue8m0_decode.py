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
"""Byte-boundary regressions for software UE8M0 decoding."""

import importlib

import kernels
import pytest
import torch
import triton
import triton.language as tl


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_uint8_scale_decode_matches_native_ue8m0():
    kernel = kernels.get_kernel("kernels-community/finegrained-kernels", version=0)
    decode = importlib.import_module(
        kernel.__name__ + ".loading.scales"
    ).decode_group_scale

    @triton.jit
    def run(Scale, Out, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        tl.store(Out + i, decode(tl.load(Scale + i)))

    # Include exponent zero (an FP32 subnormal), the normal range endpoints,
    # the multiplicative identity, and the reserved NaN byte.
    values = torch.tensor(
        [0, 1, 126, 127, 128, 253, 254, 255], device="cuda", dtype=torch.uint8
    )
    actual = torch.empty(values.numel(), device="cuda", dtype=torch.float32)
    run[(1,)](values, actual, values.numel())
    expected = torch.tensor(
        [2.0**-127, 2.0**-126, 0.5, 1.0, 2.0, 2.0**126, 2.0**127, float("nan")],
        device="cuda",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    native_dtype = getattr(torch, "float8_e8m0fnu", None)
    if native_dtype is not None:
        native = values.view(native_dtype).float()
        torch.testing.assert_close(actual, native, rtol=0, atol=0, equal_nan=True)
