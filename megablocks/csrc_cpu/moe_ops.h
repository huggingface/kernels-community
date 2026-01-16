// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU MoE Operations Header
#pragma once

#include <torch/torch.h>
#include <optional>
#include <vector>

namespace megablocks {
namespace cpu {

// ==================== Fused Experts ====================
// Main fused MoE kernel for CPU
// 
// hidden_states: [M, K] - input hidden states
// w1: [E, 2N, K] - gate_up weights (may need transpose from [E, K, 2N])
// w2: [E, K, N] - down weights (may need transpose from [E, N, K])
// topk_weights: [M, topk] - routing weights
// topk_ids: [M, topk] - expert indices
//
// Returns: [M, K] output tensor
torch::Tensor fused_experts(
    torch::Tensor& hidden_states,
    torch::Tensor& w1,
    torch::Tensor& w2,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_ids,
    bool inplace,
    const std::optional<torch::Tensor>& w1_bias,
    const std::optional<torch::Tensor>& w2_bias,
    const std::optional<double>& alpha,    // swigluoai alpha
    const std::optional<double>& limit,    // swigluoai limit
    const std::string& activation          // "silu" or "swigluoai"
);

// ==================== Activation Operations ====================
// SiLU and multiply: out = silu(gate) * up
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

// SwigluOAI: out = (up + 1) * (clamp(gate, max=limit) * sigmoid(gate * alpha))
void swigluoai_and_mul(
    torch::Tensor& out, 
    torch::Tensor& input, 
    double alpha, 
    double limit
);

}  // namespace cpu
}  // namespace megablocks
