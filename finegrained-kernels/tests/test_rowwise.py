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
