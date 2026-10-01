"""Fast CUDA/ROCm smoke tests for the kernels-community CI runner."""

import pytest
import torch

from . import layer_test, parallel_layer_test, test_gg, test_mb_moe
from . import test_mb_moe_shared_expert as shared_expert
from .ops import (
    binned_gather_test,
    binned_scatter_test,
    cumsum_test,
    histogram_test,
    padded_gather_test,
    padded_scatter_test,
    replicate_test,
    sort_test,
    topology_test,
)


pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA or ROCm"),
]


# Public top-level API, including the direct kernel exports.
def test_public_api_ci():
    test_mb_moe.test_import()
    test_mb_moe.test_exclusive_cumsum()
    test_mb_moe.test_inclusive_cumsum()
    test_mb_moe.test_histogram()


# Full sort and a sort limited to the bits needed for `max_val`.
@pytest.mark.parametrize(
    "n, dtype, max_val", [(1024, torch.int32, None), (1024, torch.int16, 128)]
)
def test_sort_ci(n, dtype, max_val):
    sort_test.test_sort(n, dtype, max_val)


def test_histogram_ci():
    histogram_test.test_histogram(1, 1024, torch.int32, 128)


def test_cumsum_ci():
    cumsum_test.test_exclusive_cumsum(8, 1024)
    cumsum_test.test_inclusive_cumsum(8, 1024)


def test_topology_ci():
    topology_test.test_topology(1024, 1536, 4)


# Gather/scatter with top-k > 1, so that tokens are routed to several experts.
def test_binned_gather_scatter_ci():
    binned_gather_test.test_binned_gather(1024, 1536, 4, 2)
    binned_scatter_test.testBinnedScatter(1024, 1536, 4, 2)


def test_padded_gather_scatter_ci():
    padded_gather_test.testPaddedGather(1024, 1536, 4, 2)
    padded_scatter_test.testPaddedScatter(1024, 1536, 4, 2)


def test_replicate_ci():
    replicate_test.test_replicate(16384, 8, 2)
    replicate_test.test_replicate_backward(16384, 8, 2)


def test_gmm_ci():
    test_gg.test_gmm()


# `MegaBlocksMoeMLP` is the layer that transformers maps to this kernel.
def test_moe_mlp_ci(device):
    layer_test.test_megablocks_moe_mlp_functionality(device)


# Expert parallelism over two processes.
def test_moe_mlp_expert_parallel_ci():
    parallel_layer_test.test_megablocks_moe_mlp_functionality()


def test_shared_expert_ci():
    shared_expert.test_shared_expert_weights_dimensions()
