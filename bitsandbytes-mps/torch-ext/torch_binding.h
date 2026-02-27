#pragma once

#include <ATen/ATen.h>
#include <tuple>

// ============================================================================
// Blockwise 4-bit quantization (NF4/FP4)
// ============================================================================

// Quantize and return both packed tensor and absmax
std::tuple<at::Tensor, at::Tensor> bnb_quantize_4bit(
    at::Tensor input,
    int64_t blocksize,
    int64_t quant_type);

// ============================================================================
// Blockwise 4-bit dequantization
// ============================================================================

// Dequantize packed 4-bit tensor back to output_dtype
at::Tensor bnb_dequantize_4bit(
    at::Tensor packed,
    at::Tensor absmax,
    int64_t blocksize,
    int64_t quant_type,
    int64_t numel,
    c10::ScalarType output_dtype);

// ============================================================================
// Fused GEMV: y = dequant(W) @ x
// W: [N, K/2] packed, absmax: [N, K_groups], x: [..., K], y: [..., N]
// ============================================================================

at::Tensor bnb_gemv_4bit(
    at::Tensor x,
    at::Tensor w,
    at::Tensor absmax,
    int64_t blocksize,
    int64_t quant_type,
    int64_t output_features);

// ============================================================================
// Fused GEMM: Y = X @ dequant(W).T
// X: [M, K], W: [N, K/2] packed, absmax: [N, K_groups], Y: [M, N]
// ============================================================================

at::Tensor bnb_gemm_4bit(
    at::Tensor x,
    at::Tensor w,
    at::Tensor absmax,
    int64_t blocksize,
    int64_t quant_type,
    int64_t output_features);
