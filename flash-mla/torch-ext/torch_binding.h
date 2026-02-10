#pragma once

#include <torch/torch.h>
#include <optional>
#include <tuple>
#include <vector>

// Sparse decode forward
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_decode_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &tile_scheduler_metadata,
    const std::optional<at::Tensor> &num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int64_t d_v,
    double sm_scale
);

// Dense decode forward
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
dense_decode_fwd(
    at::Tensor q,
    const at::Tensor &kcache,
    int64_t head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    double softmax_scale,
    bool is_causal,
    const std::optional<at::Tensor> &tile_scheduler_metadata,
    const std::optional<at::Tensor> &num_splits
);

// Sparse prefill forward
std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
);

// Dense prefill forward (SM100)
void dense_prefill_fwd(
    at::Tensor workspace_buffer,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor cumulative_seqlen_q,
    at::Tensor cumulative_seqlen_kv,
    at::Tensor o,
    at::Tensor lse,
    int64_t mask_mode_code,
    double softmax_scale,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    bool is_varlen
);

// Dense prefill backward (SM100)
void dense_prefill_bwd(
    at::Tensor workspace_buffer,
    at::Tensor d_o,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor cumulative_seqlen_q,
    at::Tensor cumulative_seqlen_kv,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    int64_t mask_mode_code,
    double softmax_scale,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    bool is_varlen
);
