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

import os
import subprocess
import sys

import kernels
import pytest
import torch


def _kernel():
    return kernels.get_kernel("kernels-community/finegrained-kernels", version=0)


def _table(packed, dtype, sliced=False):
    device = "cuda" if torch.cuda.is_available() else "xpu"
    rows, width = 37, 160
    if packed:
        weight = torch.randint(
            0, 256, (rows, width // 2), device=device, dtype=torch.uint8
        )
        scale = (torch.rand(rows, width // 16, device=device) * 3 + 0.125).to(
            torch.float8_e4m3fn
        )
        global_scale = torch.tensor([0.03125], device=device)
        if sliced:
            weight, scale = weight[:, :40], scale[:, :5]
    else:
        weight = (torch.randn(rows, width, device=device) * 3).to(torch.float8_e4m3fn)
        scale = torch.tensor([0.375], device=device, dtype=dtype)
        global_scale = None
        if sliced:
            weight = weight[:, :80]
    return weight, scale, global_scale


def _reference(weight, ids, scale, global_scale, dtype):
    rows = torch.nn.functional.embedding(ids, weight)
    if global_scale is None:
        return rows.to(scale.dtype) * scale
    codes = rows.to(torch.uint8)
    codes = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
    )
    selected_scales = (
        torch.nn.functional.embedding(ids, scale).float().repeat_interleave(16, -1)
    )
    return (values[codes] * selected_scales * global_scale).to(dtype)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sliced", [False, True])
def test_lookup_matches_selected_rows(packed, dtype, sliced):
    weight, scale, global_scale = _table(packed, dtype, sliced)
    ids = torch.tensor([[3, 0, 3], [36, 4, 0]], device=weight.device).t()
    actual = _kernel().embedding_lookup(weight, ids, scale, global_scale, dtype)
    torch.testing.assert_close(
        actual, _reference(weight, ids, scale, global_scale, dtype), rtol=0, atol=0
    )
    assert actual.dtype == dtype
    default = _kernel().embedding_lookup(weight, ids, scale, global_scale)
    default_dtype = torch.bfloat16 if packed else scale.dtype
    assert default.dtype == default_dtype
    torch.testing.assert_close(
        default,
        _reference(weight, ids, scale, global_scale, default_dtype),
        rtol=0,
        atol=0,
    )
    for ids in (ids.to(torch.int32), ids.new_empty((2, 0)), ids.new_tensor(3)):
        actual = _kernel().embedding_lookup(weight, ids, scale, global_scale, dtype)
        torch.testing.assert_close(
            actual, _reference(weight, ids, scale, global_scale, dtype), rtol=0, atol=0
        )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("packed", [False, True])
def test_lookup_fullgraph_compile(packed):
    weight, scale, global_scale = _table(packed, torch.bfloat16)
    ids = torch.tensor([[3, 0, 3], [36, 4, 0]], device=weight.device).t()
    lookup = torch.compile(_kernel().embedding_lookup, fullgraph=True)
    actual = lookup(weight, ids, scale, global_scale, torch.bfloat16)
    torch.testing.assert_close(
        actual,
        _reference(weight, ids, scale, global_scale, torch.bfloat16),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("index", [-1, 37])
def test_invalid_index_fails_in_separate_process(index):
    # Device assertions poison the context, just as an invalid native CUDA embedding lookup does.
    code = f"""
import kernels, torch
k = kernels.get_kernel('kernels-community/finegrained-kernels', version=0)
device = 'cuda' if torch.cuda.is_available() else 'xpu'
w = torch.ones(37, 32, device=device).to(torch.float8_e4m3fn)
k.embedding_lookup(w, torch.tensor([{index}], device=device), torch.ones(1, device=device))
getattr(torch, device).synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "embedding index out of range" in result.stderr
