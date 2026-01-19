// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU MoE Operations

#pragma once

#include <torch/torch.h>
#include <string>

namespace megablocks {
namespace cpu {

// Optimized CPU fused MoE kernel with brgemm
torch::Tensor fused_moe_cpu(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    const c10::optional<torch::Tensor>& w1_bias,
    const c10::optional<torch::Tensor>& w2_bias,
    const std::string& activation,
    float alpha,
    float limit,
    bool is_interleaved
);

}  // namespace cpu
}  // namespace megablocks
