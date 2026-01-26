// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU MoE Fallback Implementation
//
// Pure PyTorch implementation for CPUs without AVX512 support.
// This is slower but provides compatibility with older CPUs.

#include "moe_fallback.h"

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
      auto activated = silu(gate) * up;  // [N]

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
  auto activated = silu(gate) * up;  // [M, N]

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
