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
inline at::Tensor silu(const at::Tensor& x) {
  return x * torch::sigmoid(x);
}

// Fused experts using pure PyTorch operations
// Only supports non-quantized bf16/fp16 weights
//
// Args:
//   hidden_states: [M, K]
//   w1: [E, 2N, K] - gate and up projections
//   w2: [E, K, N] - down projection
//   topk_weights: [M, topk]
//   topk_ids: [M, topk]
//   inplace: whether to use hidden_states as output
at::Tensor fused_experts(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace);

// Shared expert using pure PyTorch operations
// Only supports non-quantized bf16/fp16 weights
//
// Args:
//   hidden_states: [M, K]
//   w1: [2N, K]
//   w2: [K, N]
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

// Convert weight - for fallback, just transpose for matmul compatibility
at::Tensor convert_weight_packed(at::Tensor& weight);

// Convert scale - for fallback, just return contiguous copy
at::Tensor convert_scale_packed(at::Tensor& scale);

}  // namespace fallback
}  // namespace cpu
}  // namespace megablocks
