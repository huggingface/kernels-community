"""Tests for tile shape 256x512 on SM100.

Covers: plain GEMM, GEMM + bias, varlen_m, varlen_k, gather_A.
"""

import math
import pytest
import torch

from .kernel import submodule

get_device_capacity = submodule("cute_dsl_utils").get_device_capacity
quack_gemm = submodule("gemm").gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] not in (10, 11),
    reason="Tile shape 256x512 is SM100 only",
)


TILE_M, TILE_N = 256, 512
CLUSTER_M, CLUSTER_N = 2, 1
DTYPE = torch.bfloat16
ATOL = 3e-2
RTOL = 1e-3


def _run_gemm(A, B, D, **kwargs):
    quack_gemm(
        A,
        B,
        D,
        C=None,
        tile_count_semaphore=None,
        tile_M=TILE_M,
        tile_N=TILE_N,
        cluster_M=CLUSTER_M,
        cluster_N=CLUSTER_N,
        persistent=True,
        **kwargs,
    )


# ── Plain batched GEMM ────────────────────────────────────────────────────────


@pytest.mark.parametrize("l", [1, 4])
@pytest.mark.parametrize("n", [512, 1024, 2048])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("m", [256, 512])
def test_gemm(m, k, n, l):
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    D = torch.empty(l, m, n, dtype=DTYPE, device="cuda")
    _run_gemm(A, B, D)
    ref = torch.bmm(A.float(), B.float().mT).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


# ── GEMM + bias (rowvec) ─────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [512, 2048])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("m", [256, 512])
def test_gemm_bias(m, k, n):
    l = 4
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    bias = torch.randn(l, n, dtype=DTYPE, device="cuda")
    D = torch.empty(l, m, n, dtype=DTYPE, device="cuda")
    _run_gemm(A, B, D, rowvec_bias=bias)
    ref = (torch.bmm(A.float(), B.float().mT) + bias.float().unsqueeze(1)).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


# ── GEMM + C (alpha * A @ B + beta * C) ──────────────────────────────────────


@pytest.mark.parametrize("n", [512, 2048])
@pytest.mark.parametrize("k", [512, 2048])
def test_gemm_add(k, n):
    l, m = 4, 512
    torch.manual_seed(0)
    A = torch.randn(l, m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(l, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    C = torch.randn(l, m, n, dtype=DTYPE, device="cuda")
    D = torch.empty(l, m, n, dtype=DTYPE, device="cuda")
    alpha, beta = 0.5, 0.7
    quack_gemm(
        A,
        B,
        D,
        C=C,
        tile_count_semaphore=None,
        tile_M=TILE_M,
        tile_N=TILE_N,
        cluster_M=CLUSTER_M,
        cluster_N=CLUSTER_N,
        persistent=True,
        alpha=alpha,
        beta=beta,
    )
    ref = (alpha * torch.bmm(A.float(), B.float().mT) + beta * C.float()).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


# ── varlen_m ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("gather_A", [False, True])
@pytest.mark.parametrize("n", [512, 2048])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("num_groups", [3, 8])
def test_gemm_varlen_m(num_groups, k, n, gather_A):
    torch.manual_seed(0)
    m_per_group = 512
    total_m = m_per_group * num_groups
    cu_seqlens_m = torch.arange(0, num_groups + 1, dtype=torch.int32, device="cuda") * m_per_group

    if gather_A:
        A_full = torch.randn(total_m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
        A_idx = torch.randperm(total_m, dtype=torch.int32, device="cuda")
        A = A_full
    else:
        A = torch.randn(total_m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
        A_idx = None

    B = torch.randn(num_groups, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    D = torch.empty(total_m, n, dtype=DTYPE, device="cuda")

    _run_gemm(A, B, D, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

    # Reference
    ref_parts = []
    for i in range(num_groups):
        s, e = cu_seqlens_m[i].item(), cu_seqlens_m[i + 1].item()
        Ai = A[A_idx[s:e]].float() if gather_A else A[s:e].float()
        ref_parts.append(Ai @ B[i].float().T)
    ref = torch.cat(ref_parts).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


# ── varlen_m with bias ───────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [512, 2048])
@pytest.mark.parametrize("k", [512, 2048])
def test_gemm_varlen_m_bias(k, n):
    num_groups = 4
    torch.manual_seed(0)
    m_per_group = 512
    total_m = m_per_group * num_groups
    cu_seqlens_m = torch.arange(0, num_groups + 1, dtype=torch.int32, device="cuda") * m_per_group

    A = torch.randn(total_m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    B = torch.randn(num_groups, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    bias = torch.randn(num_groups, n, dtype=DTYPE, device="cuda")
    D = torch.empty(total_m, n, dtype=DTYPE, device="cuda")

    _run_gemm(A, B, D, cu_seqlens_m=cu_seqlens_m, rowvec_bias=bias)

    ref_parts = []
    for i in range(num_groups):
        s, e = cu_seqlens_m[i].item(), cu_seqlens_m[i + 1].item()
        ref_parts.append((A[s:e].float() @ B[i].float().T + bias[i].float()).to(DTYPE))
    ref = torch.cat(ref_parts)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


# ── varlen_m with variable sequence lengths ──────────────────────────────────


@pytest.mark.parametrize("gather_A", [False, True])
@pytest.mark.parametrize("n", [512, 2048])
@pytest.mark.parametrize("k", [512, 2048])
def test_gemm_varlen_m_variable_seqlens(k, n, gather_A):
    num_groups = 5
    torch.manual_seed(0)
    seq_lens = torch.randint(256, 1024, (num_groups,), device="cuda")
    # Round up to multiple of 256 (tile_M) so tile boundaries align
    total_m = seq_lens.sum().item()
    cu_seqlens_m = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            seq_lens.cumsum(0).to(torch.int32),
        ]
    )

    if gather_A:
        A = torch.randn(total_m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
        A_idx = torch.randperm(total_m, dtype=torch.int32, device="cuda")
    else:
        A = torch.randn(total_m, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
        A_idx = None

    B = torch.randn(num_groups, n, k, dtype=DTYPE, device="cuda") / math.sqrt(k)
    D = torch.empty(total_m, n, dtype=DTYPE, device="cuda")

    _run_gemm(A, B, D, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx)

    ref_parts = []
    for i in range(num_groups):
        s, e = cu_seqlens_m[i].item(), cu_seqlens_m[i + 1].item()
        Ai = A[A_idx[s:e]].float() if gather_A else A[s:e].float()
        ref_parts.append(Ai @ B[i].float().T)
    ref = torch.cat(ref_parts).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)


# ── varlen_k ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [512, 2048])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("num_groups", [3, 8])
def test_gemm_varlen_k(num_groups, k, n):
    torch.manual_seed(0)
    m = 512
    total_k = k * num_groups
    cu_seqlens_k = torch.arange(0, num_groups + 1, dtype=torch.int32, device="cuda") * k

    # m-major A, n-major B for varlen_k
    A = torch.randn(total_k, m, dtype=DTYPE, device="cuda").T / math.sqrt(k)  # (m, total_k) m-major
    B = torch.randn(total_k, n, dtype=DTYPE, device="cuda").T / math.sqrt(k)  # (n, total_k) n-major
    D = torch.empty(num_groups, m, n, dtype=DTYPE, device="cuda")

    _run_gemm(A, B, D, cu_seqlens_k=cu_seqlens_k)

    ref = torch.stack(
        [
            A[:, cu_seqlens_k[i] : cu_seqlens_k[i + 1]].float()
            @ B[:, cu_seqlens_k[i] : cu_seqlens_k[i + 1]].float().T
            for i in range(num_groups)
        ]
    ).to(DTYPE)
    torch.testing.assert_close(D, ref, atol=ATOL, rtol=RTOL)
