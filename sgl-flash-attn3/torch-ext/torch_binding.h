#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    std::optional<at::Tensor> k_new_,
    std::optional<at::Tensor> v_new_,
    std::optional<at::Tensor> q_v_,
    std::optional<at::Tensor> out_,
    std::optional<at::Tensor> cu_seqlens_q_,
    std::optional<at::Tensor> cu_seqlens_k_,
    std::optional<at::Tensor> cu_seqlens_k_new_,
    std::optional<at::Tensor> seqused_q_,
    std::optional<at::Tensor> seqused_k_,
    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<at::Tensor> page_table_,
    std::optional<at::Tensor> kv_batch_idx_,
    std::optional<at::Tensor> leftpad_k_,
    std::optional<at::Tensor> rotary_cos_,
    std::optional<at::Tensor> rotary_sin_,
    std::optional<at::Tensor> seqlens_rotary_,
    std::optional<at::Tensor> q_descale_,
    std::optional<at::Tensor> k_descale_,
    std::optional<at::Tensor> v_descale_,
    std::optional<double> softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    double softcap,
    bool is_rotary_interleaved,
    std::optional<at::Tensor> scheduler_metadata_,
    int64_t num_splits,
    std::optional<bool> pack_gqa_,
    int64_t sm_margin,
    std::optional<const at::Tensor>& sinks_);
