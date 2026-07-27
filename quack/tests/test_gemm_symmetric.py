import torch
import pytest

from .kernel import submodule

gemm_symmetric = submodule("gemm_interface").gemm_symmetric


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("L", [1, 3, 10])
@pytest.mark.parametrize("M", [128, 512, 1024, 4096])
@pytest.mark.parametrize("K", [128, 512, 2048])
@pytest.mark.parametrize(("has_C", "beta"), [(False, 1.0), (True, 1.0), (True, 0.5)])
@pytest.mark.parametrize("alpha", [1.0, 2.5])
# scalar_kind=tensor wraps alpha/beta in 0-d cuda tensors. Regression: the
# wrapper used to silently coerce non-float scalars to 1.0.
@pytest.mark.parametrize("scalar_kind", ["float", "tensor"])
def test_symmetric_gemm(dtype, L, M, K, has_C, alpha, beta, scalar_kind):
    device = "cuda"
    torch.manual_seed(42)

    a = torch.empty_strided((L, M, K), (M * K, 1, M), dtype=dtype, device=device)
    a.uniform_(-2, 2)

    C = None
    if has_C:
        C = torch.randn(L, M, M, dtype=dtype, device=device)
        for l in range(L):
            C[l, :, :] = (C[l, :, :] + C[l, :, :].T) / 2

    a_ref = a.detach().clone().float()
    C_ref = C.detach().clone().float() if C is not None else None
    a_pt = a.detach().clone().to(dtype)
    C_pt = C.detach().clone().to(dtype) if C is not None else None

    if scalar_kind == "tensor":
        alpha_arg = torch.tensor(alpha, device=device, dtype=torch.float32)
        beta_arg = torch.tensor(beta, device=device, dtype=torch.float32)
    else:
        alpha_arg, beta_arg = alpha, beta

    out = gemm_symmetric(a, a.transpose(-2, -1), C=C, alpha=alpha_arg, beta=beta_arg)

    # baddbmm only accepts python scalars; reference always uses the float values.
    out_ref = torch.baddbmm(
        C_ref if C_ref is not None else torch.zeros(L, M, M, dtype=torch.float32, device=device),
        a_ref,
        a_ref.transpose(-2, -1),
        beta=beta if C_ref is not None else 0.0,
        alpha=alpha,
    )
    out_pt = torch.baddbmm(
        C_pt if C_pt is not None else torch.zeros(L, M, M, dtype=dtype, device=device),
        a_pt,
        a_pt.transpose(-2, -1),
        beta=beta if C_pt is not None else 0.0,
        alpha=alpha,
    )

    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-6


# ---- Empty-input tests for gemm_symmetric (out = alpha * A @ A.T + beta * C).
@pytest.mark.parametrize("zero_dim", ["M", "K"])
@pytest.mark.parametrize("has_C", [False, True])
def test_gemm_symmetric_empty(zero_dim, has_C):
    L, M, K = 3, 1024, 1024
    if zero_dim == "M":
        M = 0
    if zero_dim == "K":
        K = 0
    A = torch.empty_strided((L, M, K), (M * K, 1, M), dtype=torch.bfloat16, device="cuda")
    if A.numel() > 0:
        A.uniform_(-2, 2)
    C = torch.randn(L, M, M, dtype=torch.bfloat16, device="cuda") if has_C else None
    if C is not None:
        for l in range(L):
            C[l] = (C[l] + C[l].T) / 2
    out = gemm_symmetric(A, A.transpose(-2, -1), C=C)
    assert out.shape == (L, M, M)
    if K == 0 and has_C:
        # K=0: A @ A.T = 0; out = 0 + 1.0 * C = C.
        assert torch.equal(out, C)


@pytest.mark.parametrize("scalar_kind", ["alpha", "beta"])
def test_gemm_symmetric_tensor_scalar_k0(scalar_kind):
    """Tensor alpha / beta on the K=0 fast path. The K>0 case is folded into
    test_symmetric_gemm via the scalar_kind='tensor' parametrization."""
    torch.manual_seed(0)
    L, M = 3, 1024
    A = torch.empty_strided((L, M, 0), (0, 1, M), dtype=torch.bfloat16, device="cuda")
    C = torch.randn(L, M, M, dtype=torch.bfloat16, device="cuda")
    for l in range(L):
        C[l] = (C[l] + C[l].T) / 2
    val = 2.5 if scalar_kind == "alpha" else 0.5
    val_t = torch.tensor(val, device="cuda", dtype=torch.float32)
    out_tensor = gemm_symmetric(A, A.transpose(-2, -1), C=C.clone(), **{scalar_kind: val_t})
    out_float = gemm_symmetric(A, A.transpose(-2, -1), C=C.clone(), **{scalar_kind: val})
    assert torch.equal(out_tensor, out_float)
