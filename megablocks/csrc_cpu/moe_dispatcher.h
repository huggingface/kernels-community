/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU MoE Dispatcher Header
//
// This header declares the dispatcher functions that automatically select
// between AVX512-optimized and fallback implementations based on CPU features.

#pragma once

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <optional>
#include <vector>

namespace megablocks {
namespace cpu {
namespace dispatch {

// Dispatched fused_experts - automatically selects implementation based on CPU features
// For AVX512 CPUs: uses optimized BRGEMM-based implementation
// For other CPUs: uses PyTorch fallback (non-quantized only)
at::Tensor fused_experts(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    bool use_mxfp4,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    const std::optional<double>& alpha,
    const std::optional<double>& limit,
    bool is_vnni);

// Dispatched shared_expert - automatically selects implementation based on CPU features
at::Tensor shared_expert(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

// Dispatched convert_weight_packed
// For AVX512: uses VNNI packing
// For other CPUs: returns transposed weight for matmul
at::Tensor convert_weight_packed(at::Tensor& weight);

// Dispatched convert_scale_packed
// For AVX512: uses optimized scale packing for MXFP4
// For other CPUs: returns contiguous copy
at::Tensor convert_scale_packed(at::Tensor& scale);

}  // namespace dispatch
}  // namespace cpu
}  // namespace megablocks
