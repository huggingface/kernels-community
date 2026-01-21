// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU MoE Operations
// Based on sglang implementation

#pragma once

#include <torch/torch.h>
#include <string>
#include <optional>
#include <vector>

namespace megablocks {
namespace cpu {

// Convert weight to VNNI packed format for brgemm
// Input:  weight [E, OC, IC] in row-major format
// Output: packed [E, OC, IC] in VNNI format (bf16/fp16)
// Call this once during model loading, then set is_vnni=true in fused_experts_cpu
at::Tensor convert_weight_packed(at::Tensor& weight);

// Convert scale to packed format for MXFP4 quantization
// Input:  scale [E, N, G] where G is number of groups (e.g., K/32)
// Output: packed scale with reordered layout for VNNI access
at::Tensor convert_scale_packed(at::Tensor& scale);

// Fused MoE kernel (sglang compatible interface)
// Supports bf16/fp16 with silu_and_mul or swiglu activation
//
// Args:
//   hidden_states: [M, K]
//   w1: [E, 2N, K] - gate and up projections
//   w2: [E, K, N] - down projection
//   topk_weights: [M, topk] - expert weights
//   topk_ids: [M, topk] - expert indices (int32)
//   inplace: whether to use hidden_states as output
//   use_int8_w8a8: int8 quantization
//   use_fp8_w8a16: fp8 quantization
//   use_mxfp4: mxfp4 quantization
//   w1_scale, w2_scale: quantization scales
//   block_size: block size for fp8
//   a1_scale, a2_scale: activation scales
//   w1_bias, w2_bias: optional biases
//   alpha, limit: parameters for swiglu activation
//   is_vnni: whether weights are pre-packed
at::Tensor fused_experts_cpu(
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

// Shared expert kernel for models with shared expert (e.g., DeepSeek)
at::Tensor shared_expert_cpu(
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

}  // namespace cpu
}  // namespace megablocks
