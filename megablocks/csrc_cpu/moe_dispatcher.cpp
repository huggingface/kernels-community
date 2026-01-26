// SPDX-License-Identifier: Apache-2.0
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
    return fused_experts_cpu(
        hidden_states, w1, w2, topk_weights, topk_ids,
        inplace, use_int8_w8a8, use_fp8_w8a16, use_mxfp4,
        w1_scale, w2_scale, block_size, a1_scale, a2_scale,
        w1_bias, w2_bias, alpha, limit, is_vnni);
  } else {
    // Use fallback implementation
    // Fallback only supports non-quantized weights
    TORCH_CHECK(
        !use_int8_w8a8 && !use_fp8_w8a16 && !use_mxfp4,
        "Quantized MoE (int8/fp8/mxfp4) requires AVX512 and AVX512-BF16 support. "
        "Your CPU does not support these features. Please use non-quantized weights.");

    return fallback::fused_experts(
        hidden_states, w1, w2, topk_weights, topk_ids, inplace);
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
    return shared_expert_cpu(
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
