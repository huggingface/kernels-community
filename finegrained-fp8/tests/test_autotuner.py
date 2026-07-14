# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Autotuner / config-infrastructure regressions hit during development — the hub-v4
crash-config pruner guards, autotuner resilience and reporting for failing configs,
activation-quant arm equivalence, and cross-process determinism; each docstring
carries its incident."""

import logging
import subprocess
import sys

import pytest
import torch
import triton
import triton.language as tl

from utils import TEST_DEVICE, make_weights

import finegrained_fp8  # type: ignore
import finegrained_fp8.matmul  # type: ignore
from finegrained_fp8.bayesian_autotuner import bayesian_autotune  # type: ignore
from finegrained_fp8.utils import is_sm10x, mx_config_pruner  # type: ignore


@pytest.mark.kernels_ci
@pytest.mark.skipif(not is_sm10x(), reason="sm_10x-specific guard")
def test_single_trip_dot_scaled_configs_are_pruned():
    """The hub-v4 crash class: a dot_scaled config whose K-loop is a single trip
    (BLOCK_SIZE_K >= contraction dim) trips the sm_10x accumulator-init miscompile — v4
    benched such configs during tuning and died with sticky device traps."""
    prune = mx_config_pruner("K")
    cfgs = [
        triton.Config(
            {
                "COMPUTE_MODE": "dot_scaled",
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": bk,
            },
            num_warps=4,
            num_stages=2,
        )
        for bk in (128, 256)
    ]
    kept = prune(cfgs, {"K": 256})
    # BK=256 == K -> single trip
    assert {c.kwargs["BLOCK_SIZE_K"] for c in kept} == {128}


@pytest.mark.kernels_ci
@pytest.mark.skipif(not is_sm10x(), reason="sm_10x-specific guard")
def test_wide_gate_dot_scaled_configs_are_pruned():
    """The other v4 crash class: a stacked gate|up dot_scaled wider than sm_10x's N=256
    MMA cap miscompiles (packed-E2M1 rhs -> sticky "misaligned address" trap). The MX
    pruner reads the GATE constexpr off the launch args: the same BN=256 rows must
    survive a plain (GATE-less) launch."""
    prune = mx_config_pruner("K")
    cfgs = [
        triton.Config(
            {
                "COMPUTE_MODE": "dot_scaled",
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": 128,
            },
            num_warps=4,
            num_stages=2,
        )
        for bn in (64, 128, 256)  # GATE width = 2*BN
    ]
    kept = prune(cfgs, {"K": 4096, "GATE": True})
    # 2*256 > the 256 cap
    assert {c.kwargs["BLOCK_SIZE_N"] for c in kept} == {64, 128}
    plain = prune(cfgs, {"K": 4096, "GATE": False})
    assert {c.kwargs["BLOCK_SIZE_N"] for c in plain} == {64, 128, 256}


@pytest.mark.kernels_ci
@pytest.mark.skipif(not is_sm10x(), reason="sm_10x-specific guard")
def test_dot_arm_is_fenced_on_sm10x():
    """The ``dot`` arm computes correctly on sm_10x (forced-config probe, 2026-07-14) but
    is fenced as TPE poison: with dot rows in the grid a fresh 100-trial tune shipped dot
    winners 2-3x slower than the native dot_scaled basin (A/B'd same day). The fence must
    drop dot rows while dot_scaled survivors remain."""
    prune = mx_config_pruner("K")

    def cfg(mode, bk):
        return triton.Config(
            {
                "COMPUTE_MODE": mode,
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": bk,
            },
            num_warps=4,
            num_stages=2,
        )

    kept = prune([cfg("dot", 32), cfg("dot_scaled", 128)], {"K": 4096})
    assert {c.kwargs["COMPUTE_MODE"] for c in kept} == {"dot_scaled"}


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_autotuner_survives_and_reports_failing_configs(caplog):
    """A config that cannot compile must score inf (not kill the tune), the tune must
    still pick a working config, and the failure must be REPORTED — inf-scoring must
    never silently hide a broken path behind a healthy one."""

    @bayesian_autotune(
        [
            triton.Config({"BLOCK": 64, "BAD": False}, num_warps=4),
            triton.Config({"BLOCK": 64, "BAD": True}, num_warps=4),
        ],
        ["N"],
        n_trials=2,  # >= grid size -> stock-exhaustive path benches EVERY config
        cache_results=False,  # disk-cached winners skip benching -> nothing to report
    )
    @triton.jit
    def _copy_kernel(X, Y, N, BLOCK: tl.constexpr, BAD: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        if BAD:
            # non-power-of-2 arange -> generic CompilationError: one of the classes stock
            # Triton does NOT forgive (unlike static_assert/OOR/PTXAS, which it infs
            # silently) — exactly what the _bench override + reporter exist for.
            offs = offs + tl.arange(0, 3)
        tl.store(Y + offs, tl.load(X + offs, mask=offs < N), mask=offs < N)

    x = torch.randn(64, device="cuda")
    y = torch.empty_like(x)
    with caplog.at_level(logging.WARNING, logger="finegrained_fp8.bayesian_autotuner"):
        _copy_kernel[(1,)](x, y, 64)
    torch.cuda.synchronize()
    assert torch.equal(x, y)  # the good config won
    assert any("failed to compile" in r.getMessage() for r in caplog.records)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_act_quant_arms_are_bit_equal():
    """The inline and offline activation-quant arms must produce IDENTICAL bits — the
    dtype-branched kernels rely on it (same scale-group boundaries by construction)."""
    torch.manual_seed(0)
    B, Bs = make_weights(
        256,
        512,
        "cuda",
        [1, 32],
        weight_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float8_e8m0fnu,
    )
    A = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)  # below the M gate
    inline_out = finegrained_fp8.matmul_2d(A, B, Bs, None, torch.bfloat16)
    saved = finegrained_fp8.matmul.MXFP_MATMUL_ACT_PREQUANT_MIN_M
    try:
        finegrained_fp8.matmul.MXFP_MATMUL_ACT_PREQUANT_MIN_M = 1  # force offline
        offline_out = finegrained_fp8.matmul_2d(A, B, Bs, None, torch.bfloat16)
    finally:
        finegrained_fp8.matmul.MXFP_MATMUL_ACT_PREQUANT_MIN_M = saved
    assert torch.equal(inline_out, offline_out)


@pytest.mark.kernels_ci
@pytest.mark.slow
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_cross_process_determinism_block_dynamic_grouped():
    """The Triton pipeliner race class is PER-PROCESS (compile-time scheduling): repeats
    within one process agree, fresh processes can disagree. Compare an output checksum
    across subprocesses — a cheap CI approximation of the 15-process flake harness."""
    script = (
        "import sys; sys.path.insert(0,'torch-ext'); sys.path.insert(0,'tests');\n"
        "import torch; from utils import make_weights; import finegrained_fp8 as fg\n"
        "torch.manual_seed(0)\n"
        "E,N,K,S=8,512,1024,256\n"
        "eids=torch.randint(0,E,(S,),device='cuda',dtype=torch.int32)\n"
        "est,gi,si=fg.compute_grouped_scheduling(eids,E,1)\n"
        "A=torch.randn(S,K,device='cuda',dtype=torch.bfloat16)\n"
        "Aq,As=fg.fp8_act_quant_block_dynamic(A,128)\n"
        "B,Bs=make_weights(N,K,'cuda',[128,128],num_experts=E)\n"
        "out=fg.matmul_grouped(Aq,As,B,Bs,est,[128,128],epilogue=fg.Epilogue(output_dtype=torch.float32),gather_idx=gi,scatter_idx=si)\n"
        "print(out.double().sum().item())\n"
    )
    sums = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(sums) == 1, f"cross-process outputs diverged: {sums}"
