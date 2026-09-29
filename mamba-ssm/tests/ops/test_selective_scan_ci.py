"""Fast smoke tests for CI"""

import pytest
import torch

import test_selective_scan as upstream


pytestmark = pytest.mark.kernels_ci


# Forward + backward of the fused Mamba block, with and without out_proj bias.
@pytest.mark.parametrize("has_out_proj_bias", [False, True])
def test_mamba_inner_fn_ci(has_out_proj_bias):
    upstream.test_mamba_inner_fn(
        is_variable_B=True,
        is_variable_C=True,
        seqlen=128,
        itype=torch.float32,
        wtype=torch.float32,
        has_out_proj_bias=has_out_proj_bias,
    )
