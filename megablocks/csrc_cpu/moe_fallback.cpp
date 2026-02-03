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
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    float alpha,
    float limit,
    bool inplace) {
  // hidden_states: [M, K]
  // w1: [E, K, 2N] - gate and up projections (after convert_weight_packed transpose)
  // w2: [E, N, K] - down projection (after convert_weight_packed transpose)
  // topk_weights: [M, topk]
  // topk_ids: [M, topk]
  // w1_bias: optional [E, 2N] - bias for gate and up projections
  // w2_bias: optional [E, K] - bias for down projection

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t E = w1.size(0);
  int64_t N2 = w1.size(2);  // 2N (last dim after transpose)
  int64_t N = N2 / 2;
  int64_t topk = topk_ids.size(1);

  // Initialize output
  auto output = at::zeros({M, K}, hidden_states.options());

  // Process by expert: gather tokens for each expert, do batched matmul
  for (int64_t expert_idx = 0; expert_idx < E; ++expert_idx) {
    // Find tokens assigned to this expert
    // mask: [M, topk] where topk_ids == expert_idx
    auto mask = (topk_ids == expert_idx);
    
    if (!mask.any().item<bool>()) {
      continue;
    }
    
    // Get token indices and topk positions: where mask is true
    auto where_result = at::where(mask);
    auto token_indices = where_result[0];   // [num_selected]
    auto topk_positions = where_result[1];  // [num_selected]
    
    if (token_indices.size(0) == 0) {
      continue;
    }
    
    // Gather input tokens for this expert: [num_selected, K]
    auto current_hidden = hidden_states.index_select(0, token_indices);
    
    // Get weights for this expert (after convert_weight_packed transpose back)
    auto expert_w1 = w1[expert_idx];  // [K, 2N]
    auto expert_w2 = w2[expert_idx];  // [N, K]
    
    // First projection: [num_selected, K] @ [K, 2N] -> [num_selected, 2N]
    auto gate_up = torch::mm(current_hidden, expert_w1);
    
    // Add w1 bias if present
    if (w1_bias.has_value()) {
      gate_up = gate_up + w1_bias.value()[expert_idx];
    }
    
    // Split gate and up (interleaved layout: [g0, u0, g1, u1, ...])
    // This matches GptOss's gate_up_proj layout
    auto gate = gate_up.index({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)});  // [num_selected, N]
    auto up = gate_up.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)});    // [num_selected, N]
    
    // SwigluOAI activation
    auto activated = swigluoai_activation(gate, up, alpha, limit);  // [num_selected, N]
    
    // Second projection: [num_selected, N] @ [N, K] -> [num_selected, K]
    auto expert_out = torch::mm(activated, expert_w2);
    
    // Add w2 bias if present
    if (w2_bias.has_value()) {
      expert_out = expert_out + w2_bias.value()[expert_idx];
    }
    
    // Apply routing weights: [num_selected]
    auto weights = topk_weights.index({token_indices, topk_positions}).unsqueeze(1);
    auto weighted_out = expert_out * weights;
    
    // Accumulate to output using index_add
    output.index_add_(0, token_indices, weighted_out);
  }

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
  // w1: [K, 2N] (after convert_weight_packed transpose)
  // w2: [N, K] (after convert_weight_packed transpose)
  // fused_experts_out: [M, K]

  int64_t N2 = w1.size(1);  // 2N (last dim after transpose)
  int64_t N = N2 / 2;

  // Ensure float32 computation for accuracy
  auto hidden_fp32 = hidden_states.to(at::kFloat);
  auto w1_fp32 = w1.to(at::kFloat);
  auto w2_fp32 = w2.to(at::kFloat);
  auto fused_out_fp32 = fused_experts_out.to(at::kFloat);

  // First linear: [M, K] @ [K, 2N] -> [M, 2N]
  auto hidden = torch::matmul(hidden_fp32, w1_fp32);

  // Split into gate and up (interleaved layout: [g0, u0, g1, u1, ...])
  // This matches GptOss's gate_up_proj layout
  auto gate = hidden.index({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)});  // [M, N]
  auto up = hidden.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)});    // [M, N]

  // SwigluOAI activation (using default alpha=1.702, limit=7.0)
  auto activated = swigluoai_activation(gate, up);  // [M, N]

  // Second linear: [M, N] @ [N, K] -> [M, K]
  auto expert_out = torch::matmul(activated, w2_fp32);

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

at::Tensor dequantize_mxfp4(
    const at::Tensor& packed_weight,
    const at::Tensor& scale,
    int64_t block_size) {
  // packed_weight: [..., N/2] - 2 x 4-bit values packed per byte
  // scale: [..., N/block_size] - 8-bit scale per block
  // block_size: typically 32
  
  // Create LUT tensor for vectorized lookup
  auto lut = at::tensor({0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f},
                        at::kFloat);
  
  auto packed_flat = packed_weight.contiguous().flatten();
  int64_t num_packed = packed_flat.numel();
  
  // Extract low and high nibbles
  auto packed_int = packed_flat.to(at::kInt);
  auto low = packed_int.bitwise_and(0x0F);
  auto high = packed_int.div(16, "trunc").bitwise_and(0x0F);  // >> 4
  
  // Lookup values using index_select
  auto low_f = lut.index_select(0, low.flatten().to(at::kLong));
  auto high_f = lut.index_select(0, high.flatten().to(at::kLong));
  
  // Interleave: [num_packed] -> [num_packed * 2]
  std::vector<at::Tensor> stack_vec = {low_f, high_f};
  auto output = at::stack(stack_vec, 1).flatten();  // [num_packed * 2]
  
  // Apply scales
  auto scale_f = at::pow(2.0f, scale.contiguous().to(at::kFloat).flatten() - 127.0f);
  // Each scale covers block_size elements
  auto scale_expanded = scale_f.unsqueeze(1).expand({-1, block_size}).flatten();
  
  // Truncate or pad scale_expanded to match output size
  int64_t num_values = num_packed * 2;
  if (scale_expanded.size(0) >= num_values) {
    scale_expanded = scale_expanded.slice(0, 0, num_values);
  }
  
  output = output * scale_expanded;
  
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
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    int64_t block_size,
    float alpha,
    float limit,
    bool inplace) {
  // After convert_weight_packed/convert_scale_packed transpose:
  // hidden_states: [M, K]
  // w1: [E, K/2, N*2] - packed mxfp4 (gate + up projections), transposed
  // w2: [E, N/2, K] - packed mxfp4 (down projection), transposed
  // w1_scale: [E, K/block_size, N*2] - scales for w1, transposed
  // w2_scale: [E, N/block_size, K] - scales for w2, transposed
  // w1_bias: optional [E, 2N] - bias for gate and up projections
  // w2_bias: optional [E, K] - bias for down projection

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t E = w1.size(0);
  int64_t N2 = w1.size(2);  // 2N (last dim after transpose)
  int64_t N = N2 / 2;
  int64_t topk = topk_ids.size(1);

  // Create LUT tensor for vectorized lookup
  auto lut = at::tensor({0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f},
                        at::kFloat);

  // Initialize output
  auto output = at::zeros({M, K}, hidden_states.options());

  // Process by expert (same pattern as fused_experts)
  for (int64_t expert_idx = 0; expert_idx < E; ++expert_idx) {
    // Find tokens assigned to this expert
    auto mask = (topk_ids == expert_idx);
    
    if (!mask.any().item<bool>()) {
      continue;
    }
    
    auto where_result = at::where(mask);
    auto token_indices = where_result[0];
    auto topk_positions = where_result[1];
    
    if (token_indices.size(0) == 0) {
      continue;
    }
    
    // Gather input tokens: [num_selected, K]
    auto current_hidden = hidden_states.index_select(0, token_indices).to(at::kFloat);
    
    // Dequantize w1 for this expert only: [K/2, N*2] -> [K, N*2]
    auto expert_w1_packed = w1[expert_idx];  // [K/2, N*2]
    auto expert_w1_scale = w1_scale[expert_idx];  // [K/block_size, N*2]
    
    auto w1_packed_int = expert_w1_packed.to(at::kInt);
    auto w1_low = w1_packed_int.bitwise_and(0x0F);
    auto w1_high = w1_packed_int.div(16, "trunc").bitwise_and(0x0F);
    auto w1_low_f = lut.index_select(0, w1_low.flatten().to(at::kLong)).reshape(expert_w1_packed.sizes());
    auto w1_high_f = lut.index_select(0, w1_high.flatten().to(at::kLong)).reshape(expert_w1_packed.sizes());
    // Stack along dim 0 to get [2, K/2, N*2], then permute and reshape to [K, N*2]
    std::vector<at::Tensor> w1_stack = {w1_low_f, w1_high_f};
    auto expert_w1 = at::stack(w1_stack, 0).permute({1, 0, 2}).reshape({K, N2});  // [K, N*2]
    
    // Apply scale: [K/block_size, N*2] -> expand to [K, N*2]
    auto w1_scale_f = at::pow(2.0f, expert_w1_scale.to(at::kFloat) - 127.0f);
    // Each scale covers block_size elements along K dimension
    auto w1_scale_expanded = w1_scale_f.unsqueeze(1).expand({-1, block_size, N2}).reshape({K, N2});
    expert_w1 = expert_w1 * w1_scale_expanded;
    
    // First projection: [num_selected, K] @ [K, 2N] -> [num_selected, 2N]
    auto gate_up = torch::mm(current_hidden, expert_w1);
    
    // Add w1 bias if present
    if (w1_bias.has_value()) {
      gate_up = gate_up + w1_bias.value()[expert_idx].to(at::kFloat);
    }
    
    // Split and activate (interleaved layout: [g0, u0, g1, u1, ...])
    // This matches GptOss's gate_up_proj layout
    auto gate = gate_up.index({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)});  // [num_selected, N]
    auto up = gate_up.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)});    // [num_selected, N]
    auto activated = swigluoai_activation(gate, up, alpha, limit);  // [num_selected, N]
    
    // Dequantize w2 for this expert: [N/2, K] -> [N, K]
    auto expert_w2_packed = w2[expert_idx];  // [N/2, K]
    auto expert_w2_scale = w2_scale[expert_idx];  // [N/block_size, K]
    
    auto w2_packed_int = expert_w2_packed.to(at::kInt);
    auto w2_low = w2_packed_int.bitwise_and(0x0F);
    auto w2_high = w2_packed_int.div(16, "trunc").bitwise_and(0x0F);
    auto w2_low_f = lut.index_select(0, w2_low.flatten().to(at::kLong)).reshape(expert_w2_packed.sizes());
    auto w2_high_f = lut.index_select(0, w2_high.flatten().to(at::kLong)).reshape(expert_w2_packed.sizes());
    // Stack along dim 0 to get [2, N/2, K], then permute and reshape to [N, K]
    std::vector<at::Tensor> w2_stack = {w2_low_f, w2_high_f};
    auto expert_w2 = at::stack(w2_stack, 0).permute({1, 0, 2}).reshape({N, K});  // [N, K]
    
    // Apply scale: [N/block_size, K] -> expand to [N, K]
    auto w2_scale_f = at::pow(2.0f, expert_w2_scale.to(at::kFloat) - 127.0f);
    auto w2_scale_expanded = w2_scale_f.unsqueeze(1).expand({-1, block_size, K}).reshape({N, K});
    expert_w2 = expert_w2 * w2_scale_expanded;
    
    // Second projection: [num_selected, N] @ [N, K] -> [num_selected, K]
    auto expert_out = torch::mm(activated, expert_w2);
    
    // Add w2 bias if present
    if (w2_bias.has_value()) {
      expert_out = expert_out + w2_bias.value()[expert_idx].to(at::kFloat);
    }
    
    // Apply routing weights and accumulate
    auto weights = topk_weights.index({token_indices, topk_positions}).unsqueeze(1);
    auto weighted_out = expert_out * weights;
    
    output.index_add_(0, token_indices, weighted_out.to(output.scalar_type()));
  }

  if (inplace) {
    hidden_states.copy_(output);
    return hidden_states;
  }
  return output;
}

at::Tensor convert_weight_packed(at::Tensor& weight) {
  // For fallback, we don't need VNNI packing
  // The input weight was transposed in Python from [E, K, 2N] to [E, 2N, K]
  // We transpose it back to [E, K, 2N] so we can use simple matmul without transpose
  // weight: [E, 2N, K] -> [E, K, 2N] (or [E, K, N] -> [E, N, K] for w2)
  return weight.transpose(-1, -2).contiguous();
}

at::Tensor convert_scale_packed(at::Tensor& scale) {
  // For fallback, we don't need special scale packing
  // Just return a contiguous copy
  return scale.transpose(-1, -2).contiguous();
}

}  // namespace fallback
}  // namespace cpu
}  // namespace megablocks
