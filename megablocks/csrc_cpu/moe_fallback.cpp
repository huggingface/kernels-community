/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU MoE Fallback Implementation
//
// Pure PyTorch implementation for CPUs without AVX512 support.
// This is slower but provides compatibility with older CPUs.

#include "moe_fallback.h"
#include <cmath>

namespace megablocks {
namespace cpu {
namespace fallback {

at::Tensor fused_experts(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace) {
  // hidden_states: [M, K]
  // w1: [E, 2N, K] - gate and up projections
  // w2: [E, K, N] - down projection
  // topk_weights: [M, topk]
  // topk_ids: [M, topk]

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t E = w1.size(0);
  int64_t N2 = w1.size(1);  // 2N
  int64_t N = N2 / 2;
  int64_t topk = topk_ids.size(1);

  // Ensure float32 computation for accuracy
  auto hidden_fp32 = hidden_states.to(at::kFloat);
  auto w1_fp32 = w1.to(at::kFloat);
  auto w2_fp32 = w2.to(at::kFloat);
  auto topk_weights_fp32 = topk_weights.to(at::kFloat);

  at::Tensor output_fp32 = at::zeros({M, K}, hidden_fp32.options());

  // Process each token
  for (int64_t m = 0; m < M; ++m) {
    auto token = hidden_fp32[m];  // [K]

    for (int64_t t = 0; t < topk; ++t) {
      int64_t expert_id = topk_ids[m][t].item<int64_t>();
      float weight = topk_weights_fp32[m][t].item<float>();

      if (expert_id < 0 || expert_id >= E) continue;

      // Get expert weights
      auto expert_w1 = w1_fp32[expert_id];  // [2N, K]
      auto expert_w2 = w2_fp32[expert_id];  // [K, N]

      // First linear: [K] @ [K, 2N] -> [2N]
      auto hidden = torch::matmul(token, expert_w1.t());  // [2N]

      // Split into gate and up
      auto gate = hidden.slice(0, 0, N);   // [N]
      auto up = hidden.slice(0, N, N2);    // [N]

      // SiLU and mul
      auto activated = silu_activation(gate) * up;  // [N]

      // Second linear: [N] @ [N, K] -> [K]
      auto expert_out = torch::matmul(activated, expert_w2.t());  // [K]

      // Accumulate with weight
      output_fp32[m] += weight * expert_out;
    }
  }

  // Convert back to original dtype
  at::Tensor output = output_fp32.to(hidden_states.scalar_type());

  if (inplace) {
    hidden_states.copy_(output);
    return hidden_states;
  }
  return output;
}

at::Tensor shared_expert(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace) {
  // hidden_states: [M, K]
  // w1: [2N, K]
  // w2: [K, N]
  // fused_experts_out: [M, K]

  int64_t N2 = w1.size(0);  // 2N
  int64_t N = N2 / 2;

  // Ensure float32 computation for accuracy
  auto hidden_fp32 = hidden_states.to(at::kFloat);
  auto w1_fp32 = w1.to(at::kFloat);
  auto w2_fp32 = w2.to(at::kFloat);
  auto fused_out_fp32 = fused_experts_out.to(at::kFloat);

  // First linear: [M, K] @ [K, 2N] -> [M, 2N]
  auto hidden = torch::matmul(hidden_fp32, w1_fp32.t());

  // Split into gate and up
  auto gate = hidden.slice(1, 0, N);   // [M, N]
  auto up = hidden.slice(1, N, N2);    // [M, N]

  // SiLU and mul
  auto activated = silu_activation(gate) * up;  // [M, N]

  // Second linear: [M, N] @ [N, K] -> [M, K]
  auto expert_out = torch::matmul(activated, w2_fp32.t());

  // Combine with fused_experts_out
  auto output_fp32 = fused_out_fp32 + float(routed_scaling_factor) * expert_out;

  // Convert back to original dtype
  at::Tensor output = output_fp32.to(hidden_states.scalar_type());

  if (inplace) {
    hidden_states.copy_(output);
    return hidden_states;
  }
  return output;
}

// MXFP4 E2M1 lookup table values
// 4-bit format: 1 sign bit, 2 exponent bits, 1 mantissa bit
static const float mxfp4_lut[16] = {
    0.0f,   // 0b0000
    0.5f,   // 0b0001
    1.0f,   // 0b0010
    1.5f,   // 0b0011
    2.0f,   // 0b0100
    3.0f,   // 0b0101
    4.0f,   // 0b0110
    6.0f,   // 0b0111
   -0.0f,   // 0b1000
   -0.5f,   // 0b1001
   -1.0f,   // 0b1010
   -1.5f,   // 0b1011
   -2.0f,   // 0b1100
   -3.0f,   // 0b1101
   -4.0f,   // 0b1110
   -6.0f,   // 0b1111
};

at::Tensor dequantize_mxfp4(
    const at::Tensor& packed_weight,
    const at::Tensor& scale,
    int64_t block_size) {
  // packed_weight: [..., N/2] - 2 x 4-bit values packed per byte
  // scale: [..., N/block_size] - 8-bit scale per block
  // block_size: typically 32
  
  auto packed_flat = packed_weight.contiguous().flatten();
  auto scale_flat = scale.contiguous().to(at::kFloat).flatten();
  
  int64_t num_packed = packed_flat.numel();
  int64_t num_values = num_packed * 2;  // 2 values per byte
  
  // Create output tensor
  auto output = at::zeros({num_values}, at::TensorOptions().dtype(at::kFloat));
  auto output_ptr = output.data_ptr<float>();
  auto packed_ptr = packed_flat.data_ptr<uint8_t>();
  auto scale_ptr = scale_flat.data_ptr<float>();
  
  // Each scale covers block_size elements
  // Scale is stored as 8-bit exponent (bias 127)
  int64_t scale_idx = 0;
  int64_t elem_in_block = 0;
  
  for (int64_t i = 0; i < num_packed; ++i) {
    uint8_t packed = packed_ptr[i];
    uint8_t low = packed & 0x0F;   // low 4 bits
    uint8_t high = (packed >> 4) & 0x0F;  // high 4 bits
    
    // Get scale for current block
    // Scale is 2^(scale_value - 127) in MXFP4
    float s = scale_ptr[scale_idx];
    float scale_factor = std::pow(2.0f, s - 127.0f);
    
    // Dequantize using LUT
    output_ptr[i * 2] = mxfp4_lut[low] * scale_factor;
    output_ptr[i * 2 + 1] = mxfp4_lut[high] * scale_factor;
    
    // Update block tracking
    elem_in_block += 2;
    if (elem_in_block >= block_size) {
      elem_in_block = 0;
      scale_idx++;
    }
  }
  
  return output;
}

at::Tensor fused_experts_mxfp4(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    const at::Tensor& w1_scale,
    const at::Tensor& w2_scale,
    int64_t block_size,
    bool inplace) {
  // hidden_states: [M, K]
  // w1: [E, N*2, K/2] - packed mxfp4 (gate + up projections)
  // w2: [E, K, N/2] - packed mxfp4 (down projection)
  // w1_scale: [E, N*2, K/block_size]
  // w2_scale: [E, K, N/block_size]

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t E = w1.size(0);
  int64_t N2 = w1.size(1);  // 2N (gate + up)
  int64_t N = N2 / 2;
  int64_t topk = topk_ids.size(1);

  // Ensure float32 computation for accuracy
  auto hidden_fp32 = hidden_states.to(at::kFloat);
  auto topk_weights_fp32 = topk_weights.to(at::kFloat);

  at::Tensor output_fp32 = at::zeros({M, K}, hidden_fp32.options());

  // Dequantize all expert weights (this is slow but correct for fallback)
  // In practice, you'd want to cache these
  std::vector<at::Tensor> w1_dequant(E);
  std::vector<at::Tensor> w2_dequant(E);
  
  for (int64_t e = 0; e < E; ++e) {
    // Dequantize w1[e]: [N*2, K/2] -> [N*2, K]
    auto w1_e = w1[e];  // [N*2, K/2]
    auto w1_scale_e = w1_scale[e];  // [N*2, K/block_size]
    
    // Dequantize each row
    std::vector<at::Tensor> w1_rows;
    for (int64_t row = 0; row < N2; ++row) {
      auto row_dequant = dequantize_mxfp4(w1_e[row], w1_scale_e[row], block_size);
      w1_rows.push_back(row_dequant.unsqueeze(0));
    }
    w1_dequant[e] = at::cat(w1_rows, 0);  // [N*2, K]
    
    // Dequantize w2[e]: [K, N/2] -> [K, N]
    auto w2_e = w2[e];  // [K, N/2]
    auto w2_scale_e = w2_scale[e];  // [K, N/block_size]
    
    std::vector<at::Tensor> w2_rows;
    for (int64_t row = 0; row < K; ++row) {
      auto row_dequant = dequantize_mxfp4(w2_e[row], w2_scale_e[row], block_size);
      w2_rows.push_back(row_dequant.unsqueeze(0));
    }
    w2_dequant[e] = at::cat(w2_rows, 0);  // [K, N]
  }

  // Process each token
  for (int64_t m = 0; m < M; ++m) {
    auto token = hidden_fp32[m];  // [K]

    for (int64_t t = 0; t < topk; ++t) {
      int64_t expert_id = topk_ids[m][t].item<int64_t>();
      float weight = topk_weights_fp32[m][t].item<float>();

      if (expert_id < 0 || expert_id >= E) continue;

      // Get dequantized expert weights
      auto expert_w1 = w1_dequant[expert_id];  // [2N, K]
      auto expert_w2 = w2_dequant[expert_id];  // [K, N]

      // First linear: [K] @ [K, 2N] -> [2N]
      auto hidden = torch::matmul(token, expert_w1.t());  // [2N]

      // Split into gate and up
      auto gate = hidden.slice(0, 0, N);   // [N]
      auto up = hidden.slice(0, N, N2);    // [N]

      // SiLU and mul
      auto activated = silu_activation(gate) * up;  // [N]

      // Second linear: [N] @ [N, K] -> [K]
      auto expert_out = torch::matmul(activated, expert_w2.t());  // [K]

      // Accumulate with weight
      output_fp32[m] += weight * expert_out;
    }
  }

  // Convert back to original dtype
  at::Tensor output = output_fp32.to(hidden_states.scalar_type());

  if (inplace) {
    hidden_states.copy_(output);
    return hidden_states;
  }
  return output;
}

at::Tensor convert_weight_packed(at::Tensor& weight) {
  // For fallback, we don't need VNNI packing
  // Just return a contiguous copy of the weight
  // The weight will be used directly with torch::matmul
  return weight.contiguous();
}

at::Tensor convert_scale_packed(at::Tensor& scale) {
  // For fallback, we don't need special scale packing
  // Just return a contiguous copy
  return scale.contiguous();
}

}  // namespace fallback
}  // namespace cpu
}  // namespace megablocks
