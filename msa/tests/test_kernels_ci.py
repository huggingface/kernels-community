"""Fast subset of the upstream tests for the kernels-community CI runner.

The upstream suites sweep thousands of configurations; these call the same test
bodies with a handful of small configurations so ``pytest -m kernels_ci`` stays
inexpensive. The kernels are built for sm100 only, so everything is skipped on
other devices.
"""

import pytest
import torch

import test_fp4_indexer as _fp4_indexer
import test_sparse_atten as _sparse_atten

_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not _SM100, reason="SM100-class CUDA device required"),
]


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_reorder_scales_for_mma_matches_public_layout(fmt):
    _fp4_indexer.test_reorder_scales_for_mma_matches_public_layout(fmt)


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("scale_layout", ["public", "preordered_mma"])
def test_cute_block_scores_random_fp4_matches_reference(fmt, scale_layout):
    _fp4_indexer.test_cute_block_scores_random_fp4_matches_reference(
        fmt, 1, 130, 2, 1, 0, True, scale_layout
    )


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_cute_decode_packed_q_matches_reference(fmt):
    _fp4_indexer.test_cute_decode_packed_q_matches_reference(
        1, 1, 128, 0, 16, 1, fmt, False, "public"
    )


def test_scheduler_builds_packed_qsplit_metadata():
    _sparse_atten.test_scheduler_builds_packed_qsplit_metadata()


def test_k2q_csr_builder_matches_torch_reference():
    _sparse_atten.test_k2q_csr_builder_matches_torch_reference(
        batch=1, seqlen_q=1024, seqlen_kv=4096, head_kv=2, topk=8, causal=True
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("qhead_per_kv", [1, 8])
def test_sparse_atten(dtype, qhead_per_kv):
    _sparse_atten.test_sparse_atten(
        batch=3,
        seqlen_q=1024,
        seqlen_kv=4096,
        head_kv=2,
        qhead_per_kv=qhead_per_kv,
        topk=8,
        causal=True,
        dtype=dtype,
        lse_temperature_scale=None,
        check_empty=False,
    )
