# Fast parity checks for the kernels-community CI runner
# (`nix run .#ci-test`, which runs `pytest -m kernels_ci`).
#
# These compare the auto-selected fused backend for the GPU the CI runs on
# (cutlass-fna, hopper-fna, or blackwell-fna) against NATTEN's reference CUDA
# kernels, forward and backward. The vendored upstream suite (test_fna.py and
# friends) provides exhaustive coverage; this file is a smoke-level subset.

import pytest
import torch

import natten  # noqa: F401
from natten.backends.reference import reference_fna_generic
from natten.functional import na1d, na2d, na3d

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

NA_FUNCS = {1: na1d, 2: na2d, 3: na3d}

CASES = [
    # (input_shape, kernel_size, stride, dilation, is_causal)
    ((64,), (9,), (1,), (1,), (False,)),
    ((64,), (13,), (2,), (2,), (True,)),
    ((20, 20), (5, 5), (1, 1), (1, 1), (False, False)),
    ((20, 20), (7, 5), (2, 1), (2, 2), (False, True)),
    ((8, 10, 12), (3, 5, 5), (1, 1, 1), (1, 1, 1), (False, False, False)),
    ((8, 10, 12), (5, 3, 3), (1, 2, 2), (1, 2, 2), (True, False, True)),
]


@requires_cuda
@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "input_shape,kernel_size,stride,dilation,is_causal", CASES
)
def test_fused_matches_reference(
    input_shape, kernel_size, stride, dilation, is_causal, dtype
):
    torch.manual_seed(42)
    na_dim = len(input_shape)
    batch, heads, head_dim = 2, 2, 64
    device = "cuda"

    q = torch.randn(
        (batch, *input_shape, heads, head_dim),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    d_out = torch.randn_like(q)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = NA_FUNCS[na_dim](
        q,
        k,
        v,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )
    out.backward(d_out)

    out_ref = reference_fna_generic(
        q_ref,
        k_ref,
        v_ref,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )
    out_ref.backward(d_out)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.kernels_ci
def test_ops_registered():
    from natten._ops import ops

    for name in (
        "na1d_forward",
        "na2d_forward",
        "na3d_forward",
        "na1d_backward",
        "na2d_backward",
        "na3d_backward",
        "reference_na2d_forward",
        "hopper_na2d_forward",
        "blackwell_na2d_forward",
        "fmha_forward",
        "token_permute_2d",
        "token_unpermute_2d",
        "compute_delta",
    ):
        assert hasattr(ops, name), f"op {name} is not registered"
