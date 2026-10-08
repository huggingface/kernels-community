#pragma once

#include "../utils.h"

void moe_sum(torch::stable::Tensor& input, torch::stable::Tensor& output);

void moe_align_block_size(
    torch::stable::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::stable::Tensor sorted_token_ids,
    torch::stable::Tensor experts_ids,
    torch::stable::Tensor num_tokens_post_pad);

void batched_moe_align_block_size(
    int64_t max_tokens_per_batch,
    int64_t block_size,
    torch::stable::Tensor const& expert_num_tokens,
    torch::stable::Tensor sorted_ids,
    torch::stable::Tensor expert_ids,
    torch::stable::Tensor num_tokens_post_pad);

void moe_lora_align_block_size(
    torch::stable::Tensor topk_ids,
    torch::stable::Tensor token_lora_mapping,
    int64_t num_experts,
    int64_t block_size,
    int64_t max_loras,
    int64_t max_num_tokens_padded,
    int64_t max_num_m_blocks,
    torch::stable::Tensor sorted_token_ids,
    torch::stable::Tensor expert_ids,
    torch::stable::Tensor num_tokens_post_pad,
    torch::stable::Tensor adapter_enabled,
    torch::stable::Tensor lora_ids);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> grouped_topk(
    torch::stable::Tensor const& scores,
    torch::stable::Tensor const& scores_with_bias,
    int64_t n_group,
    int64_t topk_group,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fused_grouped_topk(
    const torch::stable::Tensor& hidden_states,
    const torch::stable::Tensor& gating_output,
    const int64_t n_topk,
    const bool renormalize,
    const int64_t n_expert_group,
    const int64_t n_topk_group,
    const std::string& scoring_func,
    const double routed_scaling_factor,
    const std::optional<torch::stable::Tensor>& bias);

void topk_softmax(
    torch::stable::Tensor& topk_weights,
    torch::stable::Tensor& topk_indices,
    torch::stable::Tensor& token_expert_indices,
    torch::stable::Tensor& gating_output,
    const bool renormalize);

void moe_gather(
    torch::stable::Tensor& output,
    const torch::stable::Tensor& moe_output,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& permuted_row_to_unpermuted_row,
    const torch::stable::Tensor& unpermuted_row_to_permuted_row,
    const torch::stable::Tensor& expert_first_token_offset,
    const int64_t num_experts);

void fused_moe_prologue(
    torch::stable::Tensor input,
    torch::stable::Tensor token_selected_experts,
    torch::stable::Tensor token_final_scales,
    torch::stable::Tensor workspace,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t ep_rank,
    int64_t ep_size,
    int64_t num_experts_on_rank);