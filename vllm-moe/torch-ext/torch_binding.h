#pragma once

#include <torch/torch.h>

#include <core/scalar_type.hpp>

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);

void topk_softmax(torch::Tensor &topk_weights, torch::Tensor &topk_indices,
                  torch::Tensor &token_expert_indices,
                  torch::Tensor &gating_output);

void moe_sum(torch::Tensor &input, torch::Tensor &output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

void sgl_moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                              int64_t block_size,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor experts_ids,
                              torch::Tensor num_tokens_post_pad);

void static_scaled_fp8_quant(torch::Tensor &out, torch::Tensor const &input,
                             torch::Tensor const &scale);

void dynamic_scaled_fp8_quant(torch::Tensor &out, torch::Tensor const &input,
                              torch::Tensor &scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor &out, torch::Tensor const &input, torch::Tensor &scale,
    std::optional<torch::Tensor> const &scale_ub);

#ifndef USE_ROCM
torch::Tensor moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit);

torch::Tensor marlin_gemm_moe(
    const torch::Tensor &a, const torch::Tensor &b_q_weights,
    const torch::Tensor &sorted_ids, const torch::Tensor &topk_weights,
    const torch::Tensor &topk_ids, const torch::Tensor &b_scales,
    torch::Tensor &b_zeros, const torch::Tensor &g_idx,
    const torch::Tensor &perm, torch::Tensor &workspace,
    vllm::ScalarTypeId const b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, int64_t num_experts, int64_t topk,
    int64_t moe_block_size, bool replicate_input, bool apply_weights);
#endif
