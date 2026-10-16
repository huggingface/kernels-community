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


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "device_type,sm10x", [("cuda", True), ("cuda", False), ("xpu", False)]
)
def test_rowwise_replay_benchmark_uses_device_backend(
    kernel, monkeypatch, device_type, sm10x
):
    """CUDA replay timing includes the whole callback; other devices keep their benchmark."""
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    import triton
    import triton.testing

    compat = importlib.import_module(kernel.__name__ + ".compat")
    matmul = importlib.import_module(kernel.__name__ + ".matmul")
    called = []

    def callback():
        called.append("callback")

    quantiles = (0.5, 0.2, 0.8)

    def bench(fn, **kwargs):
        assert fn is callback
        assert kwargs["quantiles"] == quantiles
        assert kwargs.get("rep") == (60 if device_type == "cuda" and sm10x else None)
        fn()
        return [1.0, 0.9, 1.1]

    monkeypatch.setattr(compat, "get_active_device_type", lambda: device_type)
    monkeypatch.setattr(compat, "is_sm10x", lambda: sm10x)
    monkeypatch.setattr(triton.testing, "do_bench_cudagraph", bench)
    monkeypatch.setattr(triton.testing, "do_bench", bench)
    monkeypatch.setattr(
        torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=callback)
    )
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: nullcontext())
    monkeypatch.setattr(
        triton.runtime,
        "driver",
        SimpleNamespace(active=SimpleNamespace(get_benchmarker=lambda: bench)),
    )
    expected = [2.0, 1.8, 2.2] if device_type == "cuda" and sm10x else [1.0, 0.9, 1.1]
    assert compat.replay_benchmark(callback, quantiles) == expected
    assert called == ["callback"] * (3 if device_type == "cuda" and sm10x else 1)
    assert (
        matmul.w8a8_tensor_dynamic_fp8_matmul_kernel.do_bench is compat.replay_benchmark
    )


@pytest.mark.kernels_ci
def test_tuning_seeds_are_measured_without_bypassing_pruning(kernel, monkeypatch):
    """A coupled optimum is considered at startup; an absent seed cannot reenter the grid."""
    import importlib

    import triton

    module = importlib.import_module(kernel.__name__ + ".bayesian_autotuner")
    tuner = object.__new__(module.BayesianAutotuner)
    configs = [triton.Config({"TILE": i}) for i in range(20)]
    removed = triton.Config({"TILE": 100})
    tuner.seed_configs = (removed, configs[19])
    tuner.n_trials = 2
    tuner.n_startup_trials = 2
    tuner.refine = False
    tuner.max_failures = 2
    tuner.gamma = 0.25
    tuner.log_path = None
    tuner.path_anchor_axes = ()
    tuner.cache = {}
    measured = []

    def bench(*args, config, **kwargs):
        measured.append(config)
        return 1.0 if config is configs[19] else 10.0

    monkeypatch.setattr(tuner, "_bench", bench)
    best = tuner._bayesian_search(configs, (), {}, ())
    assert best is configs[19]
    assert measured[0] is configs[19]
    assert removed not in measured
    assert len(set(measured)) == 2


@pytest.mark.kernels_ci
@pytest.mark.parametrize("gathered", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_rowwise_grouped_tuning_preserves_irregular_routes(
    kernel, monkeypatch, gathered, compiled
):
    """New TMA/packed-schedule starting candidates retain tail masks and expert routing."""
    import importlib

    if TEST_DEVICE != "cuda" or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("B200 tuning candidates")
    module = importlib.import_module(kernel.__name__ + ".grouped")
    tuner = module.w8a8_tensor_dynamic_fp8_matmul_grouped_kernel
    candidate = tuner.seed_configs[0 if gathered else 1]
    # Preserve the real grid's descriptor-rebinding hook while forcing each coupled path.
    actual_config = next(
        c for c in tuner.configs if c.all_kwargs() == candidate.all_kwargs()
    )
    monkeypatch.setattr(tuner, "configs", [actual_config])
    tuner.cache.clear()
    torch.manual_seed(73)
    e, s, n, k = 4, 257, 256, 256
    source_rows = 17 if gathered else s
    a = torch.randn(source_rows, k, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    b = (torch.randn(e, n, k, device=TEST_DEVICE) * 0.1).to(torch.float8_e4m3fn)
    bs = torch.rand(e, n, 1, device=TEST_DEVICE) * 0.3 + 0.1
    bias = torch.randn(e, n, device=TEST_DEVICE) * 0.01
    starts = torch.tensor([0, 1, 130, 130, s], device=TEST_DEVICE, dtype=torch.int32)
    source = torch.arange(s, device=TEST_DEVICE).remainder(source_rows).to(torch.int32)
    scatter = torch.randperm(s, device=TEST_DEVICE).to(torch.int32)
    experts = torch.tensor([0] + [1] * 129 + [3] * 127, device=TEST_DEVICE)
    scales = a.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    aq = (a.float() / scales).to(torch.float8_e4m3fn)
    a_reference = aq.float() * scales
    weights = b.float() * bs
    expected = torch.empty(s, n, device=TEST_DEVICE)
    values = (
        torch.bmm(
            a_reference[source.long()].unsqueeze(1), weights[experts].transpose(1, 2)
        ).squeeze(1)
        + bias[experts]
    )
    expected[scatter.long()] = values

    def forward(x):
        return kernel.matmul_grouped(
            x,
            b,
            None,
            bs,
            expert_start=starts,
            gather_idx=source if gathered else None,
            scatter_idx=scatter,
            bias=bias,
            output_dtype=torch.float32,
        )

    if compiled:
        forward = torch.compile(forward, fullgraph=True, mode="reduce-overhead")
    with torch.no_grad():
        result = forward(a)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
