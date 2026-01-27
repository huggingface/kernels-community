/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU MoE Dispatcher Implementation
//
// This file implements the dispatcher that selects between AVX512-optimized
// and fallback implementations based on runtime CPU feature detection.

#include "moe_dispatcher.h"
#include "moe_ops.h"
#include "moe_fallback.h"
#include "cpu_features.hpp"

namespace megablocks {
namespace cpu {
namespace dispatch {

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
    bool is_vnni) {
  // Runtime CPU feature detection
  if (CPUFeatures::hasAVX512() && CPUFeatures::hasAVX512BF16()) {
    // Use AVX512 optimized implementation
    return megablocks::cpu::fused_experts_cpu(
        hidden_states, w1, w2, topk_weights, topk_ids,
        inplace, use_int8_w8a8, use_fp8_w8a16, use_mxfp4,
        w1_scale, w2_scale, block_size, a1_scale, a2_scale,
        w1_bias, w2_bias, alpha, limit, is_vnni);
  } else {
    // Use fallback implementation
    // Fallback supports non-quantized and MXFP4 weights
    TORCH_CHECK(
        !use_int8_w8a8 && !use_fp8_w8a16,
        "Quantized MoE (int8/fp8) requires AVX512 and AVX512-BF16 support. "
        "Your CPU does not support these features. Please use non-quantized weights or MXFP4.");

    // Get alpha and limit values (default to swigluoai defaults)
    float alpha_val = alpha.has_value() ? static_cast<float>(alpha.value()) : 1.702f;
    float limit_val = limit.has_value() ? static_cast<float>(limit.value()) : 7.0f;

    if (use_mxfp4) {
      // Use MXFP4 fallback implementation
      TORCH_CHECK(w1_scale.has_value() && w2_scale.has_value(),
          "MXFP4 requires w1_scale and w2_scale");
      
      // Default block_size to 32 if not provided (same as AVX512 path)
      int64_t bs = 32;
      if (block_size.has_value() && !block_size->empty()) {
        bs = block_size->front();
      }
      
      return fallback::fused_experts_mxfp4(
          hidden_states, w1, w2, topk_weights, topk_ids,
          w1_scale.value(), w2_scale.value(), w1_bias, w2_bias, bs,
          alpha_val, limit_val, inplace);
    }

    return fallback::fused_experts(
        hidden_states, w1, w2, topk_weights, topk_ids, w1_bias, w2_bias,
        alpha_val, limit_val, inplace);
  }
}

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
    bool is_vnni) {
  // Runtime CPU feature detection
  if (CPUFeatures::hasAVX512() && CPUFeatures::hasAVX512BF16()) {
    // Use AVX512 optimized implementation
    return megablocks::cpu::shared_expert_cpu(
        hidden_states, w1, w2, fused_experts_out,
        routed_scaling_factor, inplace, use_int8_w8a8, use_fp8_w8a16,
        w1_scale, w2_scale, block_size, a1_scale, a2_scale, is_vnni);
  } else {
    // Use fallback implementation
    // Fallback only supports non-quantized weights
    TORCH_CHECK(
        !use_int8_w8a8 && !use_fp8_w8a16,
        "Quantized shared_expert (int8/fp8) requires AVX512 and AVX512-BF16 support. "
        "Your CPU does not support these features. Please use non-quantized weights.");

    return fallback::shared_expert(
        hidden_states, w1, w2, fused_experts_out, routed_scaling_factor, inplace);
  }
}

at::Tensor convert_weight_packed(at::Tensor& weight) {
  // Runtime CPU feature detection
  if (CPUFeatures::hasAVX512() && CPUFeatures::hasAVX512BF16()) {
    // Use AVX512 VNNI packing
    return megablocks::cpu::convert_weight_packed(weight);
  } else {
    // Use fallback (just contiguous copy)
    return fallback::convert_weight_packed(weight);
  }
}

at::Tensor convert_scale_packed(at::Tensor& scale) {
  // Runtime CPU feature detection
  if (CPUFeatures::hasAVX512() && CPUFeatures::hasAVX512BF16()) {
    // Use AVX512 optimized scale packing
    return megablocks::cpu::convert_scale_packed(scale);
  } else {
    // Use fallback (just contiguous copy)
    return fallback::convert_scale_packed(scale);
  }
}

}  // namespace dispatch
}  // namespace cpu
}  // namespace megablocks
