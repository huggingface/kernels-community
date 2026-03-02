"""Tests for MoE expert dispatch kernels: batched and grouped."""

import pytest
import torch
import triton

import finegrained_fp8


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn
BLOCK_SIZE = [128, 128]
PROBLEM_SIZES = [
    (8, 4, 256, 512),
    (32, 4, 256, 512),
    (64, 8, 512, 1024),
    (256, 256, 512, 1024),
]


def _make_fp8_weights(shape, block_size, device):
    """Quantize random float32 weights to FP8 with per-block inv-scales.

    Returns (W_fp8, inv_scales) where inv_scales[..., i, j] = max_abs / FP8_MAX
    for block (i, j). Multiplying W_fp8 by inv_scales recovers the original float.
    """
    bo, bi = block_size
    W = torch.randn(shape, dtype=torch.float32, device=device)
    *leading, N, K = W.shape
    rt, ct = triton.cdiv(N, bo), triton.cdiv(K, bi)
    Np, Kp = rt * bo, ct * bi

    Wp = W.new_zeros(*leading, Np, Kp)
    Wp[..., :N, :K] = W
    R = Wp.reshape(*leading, rt, bo, ct, bi)

    max_abs = R.abs().amax(dim=(-3, -1))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scale = FP8_MAX / safe

    Wq = (R * scale.unsqueeze(-1).unsqueeze(-3)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    Wq = Wq.reshape(*leading, Np, Kp)[..., :N, :K].contiguous()
    inv_scales = (1.0 / scale).to(torch.float32)
    return Wq, inv_scales


def _ref(A, B_fp8, Bs, expert_ids, block_size):
    S = A.shape[0]
    N = B_fp8.shape[1]
    bi = block_size[1]
    out = torch.empty(S, N, dtype=torch.float32, device=A.device)
    for i in range(S):
        e = expert_ids[i]
        qA_i, sA_i = finegrained_fp8.fp8_act_quant(A[i : i + 1], bi)
        out[i] = finegrained_fp8.w8a8_block_fp8_matmul(
            qA_i, B_fp8[e], sA_i, Bs[e], block_size
        )
    return out


# ── w8a8_block_fp8_matmul_batched ─────────────────────────────────────────────


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K", PROBLEM_SIZES)
def test_batched_vs_ref(S, E, N, K):
    """Batched output should match the per-token reference (both in float32)."""
    torch.manual_seed(0)
    A = torch.randn(S, K, dtype=torch.float32, device="cuda")
    B_fp8, Bs = _make_fp8_weights((E, N, K), BLOCK_SIZE, "cuda")
    expert_ids = torch.randint(0, E, (S,), device="cuda", dtype=torch.int32)

    out = finegrained_fp8.w8a8_block_fp8_matmul_batched(
        A, B_fp8, Bs, expert_ids, BLOCK_SIZE
    )
    ref = _ref(A, B_fp8, Bs, expert_ids, BLOCK_SIZE)

    torch.testing.assert_close(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K", PROBLEM_SIZES)
def test_batched_output_shape(S, E, N, K):
    A = torch.randn(S, K, dtype=torch.bfloat16, device="cuda")
    B_fp8, Bs = _make_fp8_weights((E, N, K), BLOCK_SIZE, "cuda")
    expert_ids = torch.randint(0, E, (S,), device="cuda", dtype=torch.int32)
    out = finegrained_fp8.w8a8_block_fp8_matmul_batched(
        A, B_fp8, Bs, expert_ids, BLOCK_SIZE
    )
    assert out.shape == (S, N)
    assert out.dtype == torch.bfloat16


# ── w8a8_block_fp8_matmul_grouped ─────────────────────────────────────────────


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K", PROBLEM_SIZES)
def test_grouped_vs_ref(S, E, N, K):
    """Grouped output (on sorted tokens) should match the per-token reference (both in float32)."""
    torch.manual_seed(0)
    A = torch.randn(S, K, dtype=torch.float32, device="cuda")
    B_fp8, Bs = _make_fp8_weights((E, N, K), BLOCK_SIZE, "cuda")

    expert_ids = torch.randint(0, E, (S,), device="cuda")
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]

    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=E, min=0, max=E - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    out = finegrained_fp8.w8a8_block_fp8_matmul_grouped(
        A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE
    )
    ref = _ref(A_sorted, B_fp8, Bs, expert_ids_sorted, BLOCK_SIZE)

    torch.testing.assert_close(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K", PROBLEM_SIZES)
def test_grouped_output_shape(S, E, N, K):
    A = torch.randn(S, K, dtype=torch.bfloat16, device="cuda")
    B_fp8, Bs = _make_fp8_weights((E, N, K), BLOCK_SIZE, "cuda")

    expert_ids = torch.randint(0, E, (S,), device="cuda")
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=E, min=0, max=E - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    out = finegrained_fp8.w8a8_block_fp8_matmul_grouped(
        A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE
    )
    assert out.shape == (S, N)
    assert out.dtype == torch.bfloat16


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K", PROBLEM_SIZES)
def test_batched_and_grouped_agree(S, E, N, K):
    """Batched and grouped kernels should produce identical results for the same inputs."""
    torch.manual_seed(0)
    A = torch.randn(S, K, dtype=torch.bfloat16, device="cuda")
    B_fp8, Bs = _make_fp8_weights((E, N, K), BLOCK_SIZE, "cuda")

    expert_ids = torch.randint(0, E, (S,), device="cuda")
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm].to(torch.int32)

    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=E, min=0, max=E - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    out_batched = finegrained_fp8.w8a8_block_fp8_matmul_batched(
        A_sorted, B_fp8, Bs, expert_ids_sorted, BLOCK_SIZE
    )
    out_grouped = finegrained_fp8.w8a8_block_fp8_matmul_grouped(
        A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE
    )

    torch.testing.assert_close(out_batched, out_grouped)
