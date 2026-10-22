"""Fast CUDA/ROCm smoke tests for the kernels-community CI runner."""

import pytest
import torch

from . import test_attention as attention
from . import test_cache as cache


pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA or ROCm"),
]

DEVICE = "cuda:0"


# v1 and v2 (partitioned), with GQA, ALiBi and an fp8 KV cache.
@pytest.mark.parametrize(
    "version, dtype, kv_cache_dtype, use_alibi",
    [
        ("v1", torch.half, "auto", False),
        ("v1", torch.bfloat16, "fp8", True),
        ("v2", torch.bfloat16, "auto", True),
        ("v2", torch.half, "fp8", False),
    ],
)
def test_paged_attention_ci(kv_cache_factory, version, dtype, kv_cache_dtype, use_alibi):
    attention.test_paged_attention(
        kv_cache_factory,
        version=version,
        num_seqs=7,
        num_heads=(64, 8),
        head_size=128,
        use_alibi=use_alibi,
        block_size=16,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        seed=0,
        device=DEVICE,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_reshape_and_cache_ci(kv_cache_factory, kv_cache_dtype):
    cache.test_reshape_and_cache(
        kv_cache_factory,
        num_tokens=42,
        num_heads=8,
        head_size=64,
        block_size=16,
        num_blocks=1024,
        dtype=torch.bfloat16,
        seed=0,
        device=DEVICE,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_reshape_and_cache_flash_ci(kv_cache_factory_flashinfer, kv_cache_dtype):
    cache.test_reshape_and_cache_flash(
        kv_cache_factory_flashinfer,
        num_tokens=42,
        num_heads=8,
        head_size=64,
        block_size=16,
        num_blocks=1024,
        dtype=torch.half,
        seed=0,
        device=DEVICE,
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_copy_blocks_ci(kv_cache_factory, kv_cache_dtype):
    cache.test_copy_blocks(
        kv_cache_factory,
        num_mappings=256,
        num_layers=2,
        num_heads=8,
        head_size=64,
        block_size=16,
        num_blocks=1024,
        dtype=torch.half,
        seed=0,
        kv_cache_dtype=kv_cache_dtype,
        device=DEVICE,
    )


# All three copy directions, since each takes a different path in swap_blocks.
@pytest.mark.parametrize("direction", [("gpu", "cpu"), ("gpu", "gpu"), ("cpu", "gpu")])
def test_swap_blocks_ci(kv_cache_factory, direction):
    cache.test_swap_blocks(
        kv_cache_factory,
        direction=direction,
        num_mappings=256,
        num_heads=8,
        head_size=64,
        block_size=16,
        num_blocks=1024,
        dtype=torch.float,
        seed=0,
        device=DEVICE,
        kv_cache_dtype="auto",
    )


def test_fp8_e4m3_conversion_ci():
    cache.test_fp8_e4m3_conversion(
        num_heads=8,
        head_size=64,
        block_size=16,
        num_blocks=1024,
        dtype=torch.half,
        seed=0,
        device=DEVICE,
    )
