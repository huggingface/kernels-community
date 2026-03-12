import pytest
import torch

import deep_gemm


@pytest.mark.kernels_ci
def test_cublaslt_gemm_nt():
    m, n, k = 256, 1024, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    deep_gemm.cublaslt_gemm_nt(a, b, d)

    ref = a @ b.T
    cos = torch.nn.functional.cosine_similarity(
        d.float().flatten(), ref.float().flatten(), dim=0
    )
    assert cos.item() > 0.99, f"cosine similarity too low: {cos.item()}"
