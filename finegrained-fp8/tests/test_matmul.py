"""Tests for the base w8a8_fp8_matmul kernel, including non-aligned dimensions."""

import pytest
import torch

import finegrained_fp8  # type: ignore

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_DTYPE = torch.float8_e4m3fn


def _quantize_weights(W, block_n, block_k):
    """Quantize a 2D weight matrix to FP8 with block-wise scales.

    Handles non-aligned N/K by padding to block boundaries before quantizing,
    then trimming back to the original shape.
    For per-tensor quantization, pass block_n=N, block_k=K.
    """
    N, K = W.shape
    pad_n = (-N) % block_n
    pad_k = (-K) % block_k
    W_padded = torch.nn.functional.pad(W, [0, pad_k, 0, pad_n])
    Np, Kp = W_padded.shape
    rt, ct = Np // block_n, Kp // block_k
    R = W_padded.reshape(rt, block_n, ct, block_k)
    max_abs = R.abs().amax(dim=(1, 3))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scale = FP8_MAX / safe
    Wq = (R * scale[:, None, :, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    Wq = Wq.reshape(Np, Kp)[:N, :K].contiguous()
    inv_scales = (1.0 / scale).to(torch.float32)
    return Wq, inv_scales


def _ref_matmul(A_fp8, B_fp8, As, Bs, block_n, block_k, output_dtype=torch.float32):
    """Pure-PyTorch reference: dequant both sides to float32 and matmul.

    Takes the same pre-quantized FP8 inputs as the kernel under test,
    so this only tests the matmul logic, not quantization.
    """
    N, K = B_fp8.shape

    # Dequantize A: A_deq[m, k] = A_fp8[m, k] * As[m, k // block_k]
    A_deq = A_fp8.float()
    for b in range(As.shape[1]):
        start = b * block_k
        end = min(start + block_k, K)
        A_deq[:, start:end] *= As[:, b : b + 1]

    # Dequantize B: B_deq[n, k] = B_fp8[n, k] * Bs[n // block_n, k // block_k]
    B_deq = B_fp8.float()
    for ni in range(Bs.shape[0]):
        n_start = ni * block_n
        n_end = min(n_start + block_n, N)
        for ki in range(Bs.shape[1]):
            k_start = ki * block_k
            k_end = min(k_start + block_k, K)
            B_deq[n_start:n_end, k_start:k_end] *= Bs[ni, ki]

    return (A_deq @ B_deq.T).to(output_dtype)


# (M, N, K, block_size)
# block_size=[128,128] for block-wise, None for per-tensor
CASES = [
    # ── Non-aligned N (MLA kv_a_proj style) ──
    (16, 320, 1024, [128, 128]),
    # ── Aligned block-wise ──
    (16, 1024, 2048, [128, 128]),
    # ── Per-tensor ──
    (16, 512, 1024, None),
]

# CUDA 13+ (shipped with torch 2.11+) has bigger
# numerical diffs vs the dequant+matmul reference.
_cuda_version = torch.version.cuda or "0.0"
_cuda_major = int(_cuda_version.split(".")[0])

if _cuda_major >= 13:
    DTYPE_TO_TOL = {
        torch.bfloat16: (0.2, 0.05),
        torch.float16: (0.2, 0.05),
        torch.float32: (0.2, 0.05),
    }
else:
    DTYPE_TO_TOL = {
        torch.bfloat16: (1e-4, 1e-2),
        torch.float16: (1e-4, 1e-2),
        torch.float32: (1e-4, 1e-4),
    }


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,N,K,block_size", CASES, ids=lambda x: str(x))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_matmul_correctness(M, N, K, block_size, dtype):
    """w8a8_fp8_matmul matches pure-PyTorch dequant+matmul reference."""
    torch.manual_seed(42)
    device = "cuda"

    # Quantize weights
    W = torch.randn(N, K, dtype=torch.float32, device=device)
    if block_size is not None:
        block_n, block_k = block_size
    else:
        block_n, block_k = N, K
    B_fp8, Bs = _quantize_weights(W, block_n, block_k)

    # Quantize activations (always use 128 for block-wise, K for per-tensor)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    act_block = block_k if block_size is not None else K
    qA, sA = finegrained_fp8.fp8_act_quant(A, act_block)

    out = finegrained_fp8.w8a8_fp8_matmul(qA, B_fp8, sA, Bs, block_size, dtype)
    ref = _ref_matmul(qA, B_fp8, sA, Bs, block_n, block_k, dtype)

    assert out.dtype == dtype
    assert out.shape == (M, N)
    torch.testing.assert_close(
        out, ref, atol=DTYPE_TO_TOL[dtype][0], rtol=DTYPE_TO_TOL[dtype][1]
    )
