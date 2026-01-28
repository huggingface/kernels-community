/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU MoE Operations
// Based on sglang implementation

#pragma once

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <string>
#include <optional>
#include <vector>

namespace megablocks {
namespace cpu {

// Activation method for fused experts
enum class CPUAcTMethod : int { silu_and_mul = 0, swiglu = 1 };

// ============================================================================
// Internal kernel implementations for FP8/MXFP4 quantization
// These are called from cpu_moe_kernel.cpp
// ============================================================================

// Fused experts kernel implementation for FP8/MXFP4
// Template parameters:
//   scalar_t: bf16 or fp16
//   packed_t: Float8_e4m3fn (FP8) or uint8_t (MXFP4)
//   param_t: float (FP8) or uint8_t (MXFP4)
//   is_mxfp4: true for MXFP4, false for FP8
template <typename scalar_t, typename packed_t, typename param_t, bool is_mxfp4>
void fused_experts_fp_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const packed_t* __restrict__ packed_w1,
    const packed_t* __restrict__ packed_w2,
    const scalar_t* __restrict__ w1_bias,
    const scalar_t* __restrict__ w2_bias,
    const param_t* __restrict__ w1s,
    const param_t* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad,
    float alpha,
    float limit,
    CPUAcTMethod act_func,
    bool with_bias);

// Fused experts kernel implementation for INT8 W8A8
template <typename scalar_t>
void fused_experts_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad);

// Shared expert kernel implementation for INT8 W8A8
template <typename scalar_t>
void shared_expert_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K);

// Shared expert kernel implementation for FP8
template <typename scalar_t>
void shared_expert_fp8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K);

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
