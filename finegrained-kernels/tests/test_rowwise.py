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
"""Row-wise FP8 tail rows and compact activation buffers through the public loader."""

import kernels
import pytest
import torch

from utils import DTYPE_TO_TOL, TEST_DEVICE


@pytest.fixture(scope="module")
def kernel():
    return kernels.get_kernel("kernels-community/finegrained-kernels", version=0)


@pytest.mark.parametrize("activation", ["raw", "prequant", "static"])
@pytest.mark.parametrize(
    "compiled", [pytest.param(False, marks=pytest.mark.kernels_ci), True]
)
def test_rowwise_compact_tail_activations(kernel, monkeypatch, activation, compiled):
    """A nonzero storage offset and unpadded tail must work eagerly and in a full graph."""
    monkeypatch.setenv("FINEGRAINED_DISABLE_SCALED_MM", "1")
    torch.manual_seed(0)
    M, N, K = 17, 128, 256
    A = torch.randn(1, M, K, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=TEST_DEVICE, dtype=torch.float32)
    Bs = W.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448
    B = (W / Bs).to(torch.float8_e4m3fn)
    As = (
        torch.full((), 0.02, device=TEST_DEVICE, dtype=torch.float32)
        if activation == "static"
        else None
    )
    scales = (
        A.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448
        if As is None
        else As
    )
    Aq = (A.float() / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref = (Aq.float() * scales) @ (B.float() * Bs).t()
    if activation == "prequant":
        # A compact view with an offset cannot expose padding beyond its final row.
        storage = torch.empty(M + 1, K, device=TEST_DEVICE, dtype=Aq.dtype)
        A = storage[1:].copy_(Aq.reshape(M, K)).reshape(1, M, K)
        As = scales

    def run(x, scale):
        return kernel.matmul_2d(x, B, scale, Bs, output_dtype=torch.bfloat16)

    if compiled:
        run = torch.compile(run, fullgraph=True, dynamic=False)
    out = run(A, As)
    torch.testing.assert_close(
        out.float(),
        ref,
        atol=DTYPE_TO_TOL[torch.bfloat16][0],
        rtol=DTYPE_TO_TOL[torch.bfloat16][1],
    )
