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
"""Exercise real search and benchmarking plumbing without launching device kernels."""

import importlib
from collections import Counter

import kernels
import pytest
import triton


pytestmark = [pytest.mark.cpu, pytest.mark.kernels_ci]


@pytest.fixture(scope="module")
def autotuner_class():
    kernel = kernels.get_kernel(
        "kernels-community/finegrained-kernels", version=0, check_arch=False
    )
    return importlib.import_module(
        kernel.__name__ + ".bayesian_autotuner"
    ).BayesianAutotuner


class HostKernel:
    def __init__(self):
        # Triton's constructor unwraps .fn until it reaches a Python function.
        self.fn = lambda: None
        self.__name__ = "host_budget_probe"
        self.launches = []

    def run(self, **kwargs):
        self.launches.append((kwargs["MODE"], kwargs["BLOCK_SIZE_N"]))


def make_tuner(autotuner_class, first_failure=None):
    kernel = HostKernel()
    attempts, successful = [], []

    def measure(kernel_call, quantiles):
        attempts.append(None)
        if len(attempts) == 1 and first_failure == "exception":
            raise RuntimeError("synthetic rejected configuration")
        kernel_call()
        if len(attempts) == 1 and first_failure == "inf":
            return [float("inf")] * 3
        mode, tile = kernel.launches[-1]
        successful.append((mode, tile))
        return [1.0 / tile] * 3

    configs = [
        triton.Config({"MODE": mode, "BLOCK_SIZE_N": tile}, num_warps=4, num_stages=2)
        for mode in ("plain", "swap", "tma")
        for tile in (16, 32, 64)
    ]
    tuner = autotuner_class(
        kernel,
        [],
        configs,
        [],
        None,
        None,
        n_trials=80,
        refine=True,
        path_anchor_axes=("MODE",),
        cache_results=False,
        do_bench=measure,
    )
    return tuner, kernel, attempts, successful


@pytest.mark.parametrize("first_failure", [None, "exception", "inf"])
def test_single_trial_measures_one_success(autotuner_class, monkeypatch, first_failure):
    """Multiple basins and enabled refinement must not expand the one-trial budget."""
    monkeypatch.setenv("FINEGRAINED_AUTOTUNE_TRIALS", "1")
    tuner, kernel, attempts, successful = make_tuner(autotuner_class, first_failure)
    tuner.run()
    assert len(successful) == 1
    assert len(attempts) == (1 if first_failure is None else 2)
    assert kernel.launches[-1] == successful[0]
    # A repeated key is a normal cached launch, not a second measurement.
    tuner.run()
    assert len(successful) == 1
    assert len(attempts) == (1 if first_failure is None else 2)


def test_larger_budget_keeps_anchors_refinement_and_runoff(
    autotuner_class, monkeypatch
):
    """Budget 2 keeps the established wider search: six anchors, a neighbor and runoff."""
    monkeypatch.setenv("FINEGRAINED_AUTOTUNE_TRIALS", "2")
    tuner, kernel, attempts, successful = make_tuner(autotuner_class)
    tuner.run()
    counts = Counter(successful)
    assert set(mode for mode, _ in successful) == {"plain", "swap", "tma"}
    assert any(
        tile == 16 for _, tile in successful
    )  # refinement beyond median/max anchors
    assert all(counts[(mode, 64)] == 2 for mode in ("plain", "swap", "tma"))  # runoff
    assert len(successful) == len(attempts) == 10
    assert kernel.launches[-1][1] == 64
