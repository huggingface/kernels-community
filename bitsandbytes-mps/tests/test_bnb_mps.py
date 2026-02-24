"""Tests for bitsandbytes MPS 4-bit quantization kernels."""

import pytest
import torch

from bitsandbytes_mps import (
    FP4,
    NF4,
    dequantize_4bit,
    gemm_4bit,
    gemv_4bit,
    linear_4bit,
    quantize_4bit,
)

# NF4 codebook values (matching bnb_types.h)
NF4_CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]

FP4_CODEBOOK = [
    0.0, 0.005208333333, 0.66666667, 1.0, 0.33333333, 0.5, 0.16666667, 0.25,
    0.0, -0.005208333333, -0.66666667, -1.0, -0.33333333, -0.5, -0.16666667,
    -0.25,
]

DEVICE = "mps"


def _reference_quantize_nf4(x_flat, blocksize):
    """Reference Python implementation of NF4 blockwise quantization."""
    n = x_flat.numel()
    num_blocks = (n + blocksize - 1) // blocksize
    absmax = torch.zeros(num_blocks, dtype=torch.float32)
    packed = torch.zeros((n + 1) // 2, dtype=torch.uint8)

    codebook = torch.tensor(NF4_CODEBOOK, dtype=torch.float32)

    for b in range(num_blocks):
        start = b * blocksize
        end = min(start + blocksize, n)
        block = x_flat[start:end].float()
        am = block.abs().max().item()
        absmax[b] = am

        if am > 0:
            normalized = (block / am).clamp(-1, 1)
        else:
            normalized = torch.zeros_like(block)

        for i in range(0, end - start, 2):
            v0 = normalized[i].item()
            q0 = (codebook - v0).abs().argmin().item()

            q1 = 0
            if i + 1 < end - start:
                v1 = normalized[i + 1].item()
                q1 = (codebook - v1).abs().argmin().item()

            byte_idx = (start + i) // 2
            packed[byte_idx] = (q0 << 4) | (q1 & 0x0F)

    return packed, absmax


def _reference_dequantize_nf4(packed, absmax, blocksize, numel):
    """Reference Python implementation of NF4 blockwise dequantization."""
    codebook = torch.tensor(NF4_CODEBOOK, dtype=torch.float32)
    output = torch.zeros(numel, dtype=torch.float32)

    for i in range(numel):
        byte_idx = i // 2
        block_idx = i // blocksize
        byte_val = packed[byte_idx].item()

        if i % 2 == 0:
            nibble = (byte_val >> 4) & 0x0F
        else:
            nibble = byte_val & 0x0F

        output[i] = codebook[nibble] * absmax[block_idx].item()

    return output


# ============================================================================
# Quantization / Dequantization Tests
# ============================================================================


@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("quant_type", [NF4, FP4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_dequantize_roundtrip(blocksize, quant_type, dtype):
    """Test that quantize -> dequantize approximately recovers the original."""
    torch.manual_seed(42)
    n = 1024
    x = torch.randn(n, dtype=dtype, device=DEVICE)

    packed, absmax = quantize_4bit(x, blocksize=blocksize, quant_type=quant_type)

    assert packed.shape == (n // 2,)
    assert packed.dtype == torch.uint8
    assert absmax.dtype == torch.float32
    assert absmax.shape == ((n + blocksize - 1) // blocksize,)

    x_deq = dequantize_4bit(
        packed, absmax, blocksize=blocksize, quant_type=quant_type,
        numel=n, output_dtype=dtype,
    )

    assert x_deq.shape == (n,)
    assert x_deq.dtype == dtype

    # 4-bit quantization has significant error; check correlation
    x_cpu = x.float().cpu()
    x_deq_cpu = x_deq.float().cpu()
    cosine_sim = torch.nn.functional.cosine_similarity(
        x_cpu.unsqueeze(0), x_deq_cpu.unsqueeze(0)
    ).item()
    assert cosine_sim > 0.95, f"Cosine similarity too low: {cosine_sim}"


@pytest.mark.parametrize("blocksize", [64, 128])
def test_dequantize_matches_reference(blocksize):
    """Test dequantization matches the Python reference implementation."""
    torch.manual_seed(123)
    n = 256
    x = torch.randn(n, dtype=torch.float16, device=DEVICE)

    packed, absmax = quantize_4bit(x, blocksize=blocksize, quant_type=NF4)

    # GPU dequantize
    x_deq = dequantize_4bit(
        packed, absmax, blocksize=blocksize, quant_type=NF4,
        numel=n, output_dtype=torch.float16,
    )

    # Reference dequantize (on CPU)
    x_ref = _reference_dequantize_nf4(
        packed.cpu(), absmax.cpu(), blocksize, n
    )

    torch.testing.assert_close(
        x_deq.float().cpu(), x_ref, rtol=1e-3, atol=1e-3
    )


# ============================================================================
# GEMV Tests
# ============================================================================


@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("quant_type", [NF4, FP4])
def test_gemv_correctness(blocksize, quant_type):
    """Test fused GEMV against dequantize + matmul reference."""
    torch.manual_seed(42)
    N, K = 256, 256

    # Create weight and quantize
    W = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    W_flat = W.flatten()
    packed, absmax = quantize_4bit(W_flat, blocksize=blocksize, quant_type=quant_type)

    # Reshape for GEMV
    packed_w = packed.view(N, K // 2)
    absmax_w = absmax.view(N, -1)

    # Input vector
    x = torch.randn(K, dtype=torch.float16, device=DEVICE)

    # Fused GEMV
    y = gemv_4bit(x, packed_w, absmax_w, output_features=N,
                  blocksize=blocksize, quant_type=quant_type)

    # Reference: dequantize then matmul
    W_deq = dequantize_4bit(packed, absmax, blocksize=blocksize,
                            quant_type=quant_type, numel=N*K,
                            output_dtype=torch.float16)
    W_deq = W_deq.view(N, K)
    y_ref = W_deq @ x

    # Check relative error
    rel_error = (y.float() - y_ref.float()).abs().mean() / y_ref.float().abs().mean()
    assert rel_error < 0.05, f"GEMV relative error too high: {rel_error}"


# ============================================================================
# GEMM Tests
# ============================================================================


@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("quant_type", [NF4, FP4])
def test_gemm_correctness(blocksize, quant_type):
    """Test fused GEMM against dequantize + matmul reference."""
    torch.manual_seed(42)
    M, N, K = 8, 128, 128

    W = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    W_flat = W.flatten()
    packed, absmax = quantize_4bit(W_flat, blocksize=blocksize, quant_type=quant_type)

    packed_w = packed.view(N, K // 2)
    absmax_w = absmax.view(N, -1)

    X = torch.randn(M, K, dtype=torch.float16, device=DEVICE)

    # Fused GEMM
    Y = gemm_4bit(X, packed_w, absmax_w, output_features=N,
                  blocksize=blocksize, quant_type=quant_type)

    # Reference
    W_deq = dequantize_4bit(packed, absmax, blocksize=blocksize,
                            quant_type=quant_type, numel=N*K,
                            output_dtype=torch.float16)
    W_deq = W_deq.view(N, K)
    Y_ref = X @ W_deq.T

    rel_error = (Y.float() - Y_ref.float()).abs().mean() / Y_ref.float().abs().mean()
    assert rel_error < 0.05, f"GEMM relative error too high: {rel_error}"


# ============================================================================
# Linear layer test
# ============================================================================


def test_linear_4bit_auto_select():
    """Test that linear_4bit auto-selects GEMV vs GEMM."""
    torch.manual_seed(42)
    N, K = 128, 128

    W = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    packed, absmax = quantize_4bit(W.flatten(), blocksize=64, quant_type=NF4)
    packed_w = packed.view(N, K // 2)
    absmax_w = absmax.view(N, -1)

    # Single vector - should use GEMV
    x = torch.randn(K, dtype=torch.float16, device=DEVICE)
    y = linear_4bit(x, packed_w, absmax_w, output_features=N)
    assert y.shape == (N,)

    # Batch - should use GEMM
    X = torch.randn(4, K, dtype=torch.float16, device=DEVICE)
    Y = linear_4bit(X, packed_w, absmax_w, output_features=N)
    assert Y.shape == (4, N)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
