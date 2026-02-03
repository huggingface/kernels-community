/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU MoE Fallback Implementation
//
// Pure PyTorch implementation for CPUs without AVX512 support.
// This is slower but provides compatibility with older CPUs.

#pragma once

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <optional>
#include <vector>

namespace megablocks {
namespace cpu {
namespace fallback {

// SiLU activation: x * sigmoid(x)
inline at::Tensor silu_activation(const at::Tensor& x) {
  return x * torch::sigmoid(x);
}

// SwigluOAI activation function used in GptOss models
// Formula:
//   gate = clamp(gate, max=limit)
//   up = clamp(up, -limit, limit)
//   glu = gate * sigmoid(gate * alpha)
//   output = (up + 1) * glu
inline at::Tensor swigluoai_activation(const at::Tensor& gate, const at::Tensor& up,
                                       float alpha = 1.702f, float limit = 7.0f) {
  auto gate_clamped = gate.clamp(-std::numeric_limits<float>::infinity(), limit);
  auto up_clamped = up.clamp(-limit, limit);
  auto glu = gate_clamped * torch::sigmoid(gate_clamped * alpha);
  return (up_clamped + 1.0f) * glu;
}

// Fused experts using pure PyTorch operations
// Only supports non-quantized bf16/fp16 weights
//
// Args:
//   hidden_states: [M, K]
//   w1: [E, K, 2N] - gate and up projections (after convert_weight_packed)
//   w2: [E, N, K] - down projection (after convert_weight_packed)
//   topk_weights: [M, topk]
//   topk_ids: [M, topk]
//   w1_bias: optional [E, 2N] - bias for gate and up projections
//   w2_bias: optional [E, K] - bias for down projection
//   alpha: swigluoai alpha parameter (default 1.702)
//   limit: swigluoai limit parameter (default 7.0)
//   inplace: whether to use hidden_states as output
at::Tensor fused_experts(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    float alpha,
    float limit,
    bool inplace);

// Shared expert using pure PyTorch operations
// Only supports non-quantized bf16/fp16 weights
//
// Args:
//   hidden_states: [M, K]
//   w1: [K, 2N] (after convert_weight_packed)
//   w2: [N, K] (after convert_weight_packed)
//   fused_experts_out: [M, K] - output from fused experts
//   routed_scaling_factor: scaling factor for shared expert output
//   inplace: whether to use hidden_states as output
at::Tensor shared_expert(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace);

// Fused experts with MXFP4 quantization using pure PyTorch operations
//
// Args:
//   hidden_states: [M, K]
//   w1: [E, N, K/2] - packed mxfp4 gate and up projections (2 values per byte)
//   w2: [E, K/2, N/2] - packed mxfp4 down projection
//   topk_weights: [M, topk]
//   topk_ids: [M, topk]
//   w1_scale: [E, N*2, K/32] - scales for w1
//   w2_scale: [E, K, N/32] - scales for w2
//   w1_bias: optional [E, 2N] - bias for gate and up projections
//   w2_bias: optional [E, K] - bias for down projection
//   block_size: block size for quantization (typically 32)
//   alpha: swigluoai alpha parameter (default 1.702)
//   limit: swigluoai limit parameter (default 7.0)
//   inplace: whether to use hidden_states as output
at::Tensor fused_experts_mxfp4(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    const at::Tensor& w1_scale,
    const at::Tensor& w2_scale,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    int64_t block_size,
    float alpha,
    float limit,
    bool inplace);

// Dequantize MXFP4 tensor to float
// MXFP4 format: 4-bit E2M1 format with shared 8-bit scale per block
at::Tensor dequantize_mxfp4(
    const at::Tensor& packed_weight,  // [... , N/2] - 2 values per byte
    const at::Tensor& scale,          // [... , N/block_size]
    int64_t block_size);

// Convert weight - for fallback, just transpose for matmul compatibility
at::Tensor convert_weight_packed(at::Tensor& weight);

// Convert scale - for fallback, just return contiguous copy
at::Tensor convert_scale_packed(at::Tensor& scale);

}  // namespace fallback
}  // namespace cpu
}  // namespace megablocks
