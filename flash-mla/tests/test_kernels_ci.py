"""Fast subset of the upstream tests for the kernels-community CI runner.

The upstream tests are scripts that sweep thousands of configurations from a
``main()``; these call the same test bodies with a handful of small
configurations so ``pytest -m kernels_ci`` stays inexpensive. Dense MLA decoding
needs sm90 and the FMHA kernels need sm100, so each test is skipped elsewhere.
"""

import pytest
import torch

import test_flash_mla_dense_decoding as dense_decoding
import test_fmha_sm100 as fmha_sm100

pytestmark = pytest.mark.kernels_ci

_CAPABILITY_MAJOR = (
    torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
)


@pytest.fixture
def bf16_cuda():
    """What the upstream `main()` functions set up before running the test bodies."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("high")
    try:
        with torch.device("cuda"):
            yield
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(
    _CAPABILITY_MAJOR != 9, reason="dense MLA decoding requires an sm90 device"
)
@pytest.mark.parametrize(
    "b, s_q, s_k, is_varlen, is_causal, h_q, h_kv",
    [
        (1, 1, 20, False, False, 1, 1),
        (2, 2, 140, True, True, 64, 2),
        (6, 1, 4096, True, False, 128, 1),
    ],
)
def test_flash_mla_dense_decoding(
    bf16_cuda, b, s_q, s_k, is_varlen, is_causal, h_q, h_kv
):
    dense_decoding.test_flash_mla(
        dense_decoding.TestParam(
            b,
            s_q,
            s_k,
            is_varlen,
            is_causal,
            test_performance=False,
            h_q=h_q,
            h_kv=h_kv,
        )
    )


@pytest.mark.skipif(
    _CAPABILITY_MAJOR != 10, reason="the FMHA kernels require an sm100 device"
)
@pytest.mark.parametrize(
    "varlen, h, h_k, d, dv, causal, has_bwd",
    [
        (False, 32, 4, 128, 128, False, False),
        (True, 32, 32, 192, 128, True, True),
    ],
)
def test_fmha_sm100(bf16_cuda, varlen, h, h_k, d, dv, causal, has_bwd):
    fmha_sm100.test_flash_attention(
        2, 512, 512, varlen, h, h_k, d, dv, causal, 0, has_bwd
    )
