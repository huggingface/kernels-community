#pragma once

#include <ATen/ATen.h>

// ============================================================================
// FP-quantized (MXFP4) operations — defined in fp_quantized.mm
// ============================================================================

// Matrix-matrix multiply, non-transposed weight: y = x @ dequant(w)
// x: [..., M, K], w: [K_packed, N] (uint32), y: [..., M, N]
at::Tensor mxfp4_qmm_n(
    at::Tensor x,
    at::Tensor w,
    at::Tensor scales,
    int64_t output_features);

// Matrix-vector multiply: y = dequant(w) @ x
// x: [..., K], w: [N, K_packed] (uint32), y: [..., N]
at::Tensor mxfp4_qmv(
    at::Tensor x,
    at::Tensor w,
    at::Tensor scales,
    int64_t output_features);

// ============================================================================
// Affine quantized operations — defined in quantized.mm
// ============================================================================

// Matrix-vector multiply
at::Tensor affine_qmv(
    at::Tensor x,
    at::Tensor w,
    at::Tensor scales,
    at::Tensor biases,
    int64_t group_size,
    int64_t bits,
    int64_t output_features);

// Matrix-matrix multiply, transposed weight: y = x @ dequant(w).T
// x: [..., M, K], w: [N, K_packed], y: [..., M, N]
at::Tensor affine_qmm_t(
    at::Tensor x,
    at::Tensor w,
    at::Tensor scales,
    at::Tensor biases,
    int64_t group_size,
    int64_t bits);

// Matrix-matrix multiply, non-transposed weight: y = x @ dequant(w)
// x: [..., M, K], w: [K_packed, N], y: [..., M, N]
at::Tensor affine_qmm_n(
    at::Tensor x,
    at::Tensor w,
    at::Tensor scales,
    at::Tensor biases,
    int64_t group_size,
    int64_t bits,
    int64_t output_features);

