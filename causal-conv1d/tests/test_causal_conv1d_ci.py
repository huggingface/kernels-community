"""Fast smoke tests for CI"""

import pytest
import torch

import test_causal_conv1d as upstream


pytestmark = pytest.mark.kernels_ci


# Forward + backward in both layouts, including initial/final states
# (channel last only), bias and SiLU.
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channel_last", [False, True])
def test_causal_conv1d_ci(itype, channel_last):
    upstream.test_causal_conv1d(
        dim=64,
        seqlen=130,
        width=4,
        has_bias=True,
        silu_activation=True,
        itype=itype,
        channel_last=channel_last,
        has_initial_states=channel_last,
        return_final_states=channel_last,
    )


# Update with cache_seqlens, with and without conv_state_indices.
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_update_ci(itype):
    upstream.test_causal_conv1d_update(
        dim=2048 + 16,
        width=4,
        seqlen=4,
        has_cache_seqlens=True,
        has_bias=True,
        silu_activation=True,
        itype=itype,
    )
    upstream.test_causal_conv1d_update_with_batch_gather(
        dim=2048 + 16,
        width=3,
        seqlen=5,
        has_cache_seqlens=True,
        has_bias=True,
        silu_activation=True,
        itype=itype,
    )


# Forward + backward with seq_idx (channel last).
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_varlen_ci(itype):
    upstream.test_causal_conv1d_varlen(
        dim=64,
        seqlen=151,
        width=3,
        has_bias=True,
        silu_activation=True,
        itype=itype,
    )
