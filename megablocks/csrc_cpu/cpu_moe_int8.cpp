/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU Fused MoE Implementation for INT8
//
// Stub implementation - INT8 kernels are not yet implemented.

#define CPU_CAPABILITY_AVX512
#include "moe_ops.h"
#include <ATen/ATen.h>

namespace megablocks {
namespace cpu {

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
    int64_t num_tokens_post_pad) {
  TORCH_CHECK(false, "fused_experts_int8_kernel_impl: INT8 W8A8 kernel is not implemented yet!");
}

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
    int64_t K) {
  TORCH_CHECK(false, "shared_expert_int8_kernel_impl: INT8 W8A8 kernel is not implemented yet!");
}

// Explicit template instantiations
template void fused_experts_int8_kernel_impl<at::BFloat16>(
    at::BFloat16* __restrict__ output,
    at::BFloat16* __restrict__ ic1,
    at::BFloat16* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const at::BFloat16* __restrict__ input,
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

template void fused_experts_int8_kernel_impl<at::Half>(
    at::Half* __restrict__ output,
    at::Half* __restrict__ ic1,
    at::Half* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const at::Half* __restrict__ input,
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

template void shared_expert_int8_kernel_impl<at::BFloat16>(
    at::BFloat16* __restrict__ output,
    at::BFloat16* __restrict__ ic1,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const at::BFloat16* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const at::BFloat16* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K);

template void shared_expert_int8_kernel_impl<at::Half>(
    at::Half* __restrict__ output,
    at::Half* __restrict__ ic1,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const at::Half* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const at::Half* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K);

}  // namespace cpu
}  // namespace megablocks
