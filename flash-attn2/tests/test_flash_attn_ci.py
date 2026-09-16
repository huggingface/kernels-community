# Smoke tests for CI that use a subset of upstream tests. Since the upstream
# tests are parametrized, we call them here with our own (much more limited)
# parametrizations.
#
# The upstream module is imported as a whole rather than importing the test
# functions by name, so that pytest does not collect the full upstream matrix a
# second time from this module.

import pytest
import torch

from . import test_flash_attn as upstream


@pytest.mark.kernels_ci
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_output_ci(mha_type, causal, device):
    upstream.test_flash_attn_output(
        seqlen_q=113,
        seqlen_k=203,
        d=64,
        dropout_p=0.0,
        causal=causal,
        local=False,
        alibi=False,
        deterministic=False,
        mha_type=mha_type,
        dtype=torch.float16,
        kvpacked=False,
        softcap=0.0,
        device=device,
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_output_ci(mha_type, causal, device):
    upstream.test_flash_attn_varlen_output(
        seqlen_q=113,
        seqlen_k=203,
        d=64,
        dropout_p=0.0,
        causal=causal,
        local=False,
        alibi=False,
        deterministic=False,
        mha_type=mha_type,
        dtype=torch.float16,
        kvpacked=False,
        softcap=0.0,
        device=device,
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_splitkv_ci(causal, device):
    upstream.test_flash_attn_splitkv(
        seqlen_q=1,
        seqlen_k=339,
        swap_sq_sk=False,
        d=64,
        causal=causal,
        local=False,
        alibi=False,
        deterministic=False,
        dtype=torch.float16,
        device=device,
    )
