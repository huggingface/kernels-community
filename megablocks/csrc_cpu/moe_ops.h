// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU MoE Operations

#pragma once

#include <torch/torch.h>
#include <string>

namespace megablocks {
namespace cpu {

// Convert weight to VNNI packed format for brgemm
// Input:  weight [E, OC, IC] in row-major format
// Output: packed [E, OC, IC] in VNNI format (bf16/fp16) or column-major (float)
// Call this once during model loading, then set is_vnni=true in fused_moe_cpu
torch::Tensor convert_weight_packed(torch::Tensor weight);

// Optimized CPU fused MoE kernel with brgemm
// Supports both silu_and_mul (standard SwiGLU) and swigluoai (GptOss) activations
// 
// Args:
//   hidden_states: [num_tokens, hidden_size]
//   w1: [num_experts, 2*inter_size, hidden_size] - gate and up projections
//   w2: [num_experts, hidden_size, inter_size] - down projection
//   topk_weights: [num_tokens, topk] - expert weights
//   topk_ids: [num_tokens, topk] - expert indices
//   w1_bias: optional [num_experts, 2*inter_size]
//   w2_bias: optional [num_experts, hidden_size]
//   is_vnni: whether w1 and w2 are already in VNNI packed format
//   activation: "silu" or "swigluoai"
//   alpha, limit: parameters for swigluoai activation
torch::Tensor fused_moe_cpu(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    const c10::optional<torch::Tensor>& w1_bias,
    const c10::optional<torch::Tensor>& w2_bias,
    bool is_vnni,
    const std::string& activation,
    float alpha,
    float limit
);

}  // namespace cpu
}  // namespace megablocks
