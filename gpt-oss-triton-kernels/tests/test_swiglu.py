import torch
import pytest

import kernels

triton_kernels = kernels.get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)

_private = triton_kernels._private_for_testing

routing_torch = _private.routing.routing_torch
swiglu = _private.swiglu.swiglu
swiglu_torch = _private.swiglu.swiglu_torch
PrecisionConfig = _private.swiglu.PrecisionConfig
assert_close = _private.testing.assert_close

from .test_routing import init_data as init_routing_data

# ---------------
# initialize data
# ---------------


def alloc_rand(shape, device, dtype, requires_grad=True):
    if dtype.itemsize == 1:
        tmp = 2**-(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


# ---------------
# unit tests
# ---------------


@pytest.mark.parametrize("M, N", [(1311, 4352)])
@pytest.mark.parametrize("limit", [1e-2, 10])
def test_op(M, N, limit, device, alpha=0.5):
    torch.manual_seed(2)
    # initialize expert data
    n_expts_tot = 6
    n_expts_act = 2
    logits = init_routing_data(M, n_expts_tot).detach()
    routing_data, _, _ = routing_torch(logits, n_expts_act)
    n_tokens = routing_data.expt_hist.sum()

    # initialize data
    x = alloc_rand([n_tokens, N], device=device, dtype=torch.bfloat16)
    precision_config = PrecisionConfig(limit=limit)
    tri_y = swiglu(x, alpha, precision_config, routing_data)
    ref_y = swiglu_torch(x, alpha, precision_config)
    assert_close(tri_y, ref_y)
