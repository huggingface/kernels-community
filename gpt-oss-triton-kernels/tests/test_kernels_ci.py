"""Fast subset of the vendored upstream tests for the kernels-community CI runner.

The upstream suites are exhaustive (thousands of parametrizations); these run
the same test bodies with a handful of combinations so ``pytest -m kernels_ci``
stays inexpensive.
"""

import pytest

from . import test_compaction as _compaction
from . import test_mxfp as _mxfp
from . import test_routing as _routing
from . import test_swiglu as _swiglu

pytestmark = pytest.mark.kernels_ci


@pytest.mark.parametrize(
    "n_tokens, n_cols, k, p",
    [(131, 128, 16, 0.6), (496, 128, 16, 0.0)],
)
def test_compaction(n_tokens, n_cols, k, p, device):
    _compaction.test_compaction(n_tokens, n_cols, k, p, device)


@pytest.mark.parametrize("n_tokens_pad, n_tokens_raw", [(371, None), (1152, 911)])
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(128, 32), (1500, 8)])
@pytest.mark.parametrize("sm_first", [True, False])
def test_routing(n_tokens_pad, n_tokens_raw, n_expts_tot, n_expts_act, sm_first, device):
    _routing.test_op(
        n_tokens_pad,
        n_tokens_raw,
        n_expts_tot,
        n_expts_act,
        sm_first,
        False,
        device,
    )


@pytest.mark.parametrize("limit", [1e-2, 10])
def test_swiglu(limit, device):
    _swiglu.test_op(1311, 4352, limit, device)


@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16"])
def test_mxfp4_rounding_cases(dst_dtype):
    _mxfp.test_mxfp4_rounding_cases(dst_dtype)


@pytest.mark.parametrize("src_dtype", ["float4_e2m1", "float8_e5m2"])
def test_mxfp_quant_dequant(src_dtype):
    _mxfp.test_mxfp_quant_dequant(src_dtype, "bfloat16")
