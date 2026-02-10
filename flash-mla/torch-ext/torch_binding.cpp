#include <torch/library.h>
#include <ATen/Tensor.h>

#include "registration.h"
#include "torch_binding.h"

// Forward declarations for the interface functions defined in the API headers
// These are declared here to avoid including headers that pull in pybind11

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> &tile_scheduler_metadata,
    std::optional<at::Tensor> &num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    float sm_scale
);

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
dense_attn_decode_interface(
    at::Tensor &q,
    const at::Tensor &kcache,
    const int head_size_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const float softmax_scale,
    bool is_causal,
    std::optional<at::Tensor> &tile_scheduler_metadata,
    std::optional<at::Tensor> &num_splits
);

std::vector<at::Tensor> sparse_attn_prefill_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
);

// SM100 dense prefill functions - stub implementations for SM90-only build
// These will throw runtime errors if called, as SM100 code is not compiled
static void FMHACutlassSM100FwdRun(
    at::Tensor /*workspace_buffer*/,
    at::Tensor /*q*/,
    at::Tensor /*k*/,
    at::Tensor /*v*/,
    at::Tensor /*cumulative_seqlen_q*/,
    at::Tensor /*cumulative_seqlen_kv*/,
    at::Tensor /*o*/,
    at::Tensor /*lse*/,
    int /*mask_mode_code*/,
    float /*softmax_scale*/,
    int /*max_seqlen_q*/,
    int /*max_seqlen_kv*/,
    bool /*is_varlen*/
) {
    TORCH_CHECK(false, "dense_prefill_fwd requires SM100 (Blackwell) GPU and CUDA 12.9+. This build only supports SM90 (Hopper).");
}

static void FMHACutlassSM100BwdRun(
    at::Tensor /*workspace_buffer*/,
    at::Tensor /*d_o*/,
    at::Tensor /*q*/,
    at::Tensor /*k*/,
    at::Tensor /*v*/,
    at::Tensor /*o*/,
    at::Tensor /*lse*/,
    at::Tensor /*cumulative_seqlen_q*/,
    at::Tensor /*cumulative_seqlen_kv*/,
    at::Tensor /*dq*/,
    at::Tensor /*dk*/,
    at::Tensor /*dv*/,
    int /*mask_mode_code*/,
    float /*softmax_scale*/,
    int /*max_seqlen_q*/,
    int /*max_seqlen_kv*/,
    bool /*is_varlen*/
) {
    TORCH_CHECK(false, "dense_prefill_bwd requires SM100 (Blackwell) GPU and CUDA 12.9+. This build only supports SM90 (Hopper).");
}

// Wrapper functions that adapt the interface for TORCH_LIBRARY

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
) {
    // Create mutable copies for the interface that modifies these
    std::optional<at::Tensor> tile_scheduler_metadata_mut = tile_scheduler_metadata;
    std::optional<at::Tensor> num_splits_mut = num_splits;
    return sparse_attn_decode_interface(
        q, kv, indices, topk_length, attn_sink,
        tile_scheduler_metadata_mut, num_splits_mut,
        extra_kv, extra_indices, extra_topk_length,
        static_cast<int>(d_v), static_cast<float>(sm_scale)
    );
}

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
) {
    // Create mutable copies for the interface that modifies these
    std::optional<at::Tensor> tile_scheduler_metadata_mut = tile_scheduler_metadata;
    std::optional<at::Tensor> num_splits_mut = num_splits;
    return dense_attn_decode_interface(
        q, kcache, static_cast<int>(head_size_v),
        seqlens_k, block_table,
        static_cast<float>(softmax_scale), is_causal,
        tile_scheduler_metadata_mut, num_splits_mut
    );
}

std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    return sparse_attn_prefill_interface(
        q, kv, indices,
        static_cast<float>(sm_scale), static_cast<int>(d_v),
        attn_sink, topk_length
    );
}

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
) {
    FMHACutlassSM100FwdRun(
        workspace_buffer, q, k, v,
        cumulative_seqlen_q, cumulative_seqlen_kv,
        o, lse,
        static_cast<int>(mask_mode_code), static_cast<float>(softmax_scale),
        static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_kv),
        is_varlen
    );
}

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
) {
    FMHACutlassSM100BwdRun(
        workspace_buffer, d_o, q, k, v, o, lse,
        cumulative_seqlen_q, cumulative_seqlen_kv,
        dq, dk, dv,
        static_cast<int>(mask_mode_code), static_cast<float>(softmax_scale),
        static_cast<int>(max_seqlen_q), static_cast<int>(max_seqlen_kv),
        is_varlen
    );
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Sparse decode forward
    ops.def(
        "sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, "
        "Tensor? topk_length, Tensor? attn_sink, "
        "Tensor? tile_scheduler_metadata, Tensor? num_splits, "
        "Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, "
        "int d_v, float sm_scale) -> (Tensor, Tensor, Tensor?, Tensor?)"
    );
    ops.impl("sparse_decode_fwd", torch::kCUDA, &sparse_decode_fwd);

    // Dense decode forward
    ops.def(
        "dense_decode_fwd(Tensor q, Tensor kcache, int head_size_v, "
        "Tensor seqlens_k, Tensor block_table, "
        "float softmax_scale, bool is_causal, "
        "Tensor? tile_scheduler_metadata, Tensor? num_splits) -> (Tensor, Tensor, Tensor?, Tensor?)"
    );
    ops.impl("dense_decode_fwd", torch::kCUDA, &dense_decode_fwd);

    // Sparse prefill forward
    ops.def(
        "sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, "
        "float sm_scale, int d_v, "
        "Tensor? attn_sink, Tensor? topk_length) -> Tensor[]"
    );
    ops.impl("sparse_prefill_fwd", torch::kCUDA, &sparse_prefill_fwd);

    // Dense prefill forward (SM100)
    ops.def(
        "dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, "
        "Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, "
        "Tensor o, Tensor lse, "
        "int mask_mode_code, float softmax_scale, "
        "int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()"
    );
    ops.impl("dense_prefill_fwd", torch::kCUDA, &dense_prefill_fwd);

    // Dense prefill backward (SM100)
    ops.def(
        "dense_prefill_bwd(Tensor workspace_buffer, Tensor d_o, "
        "Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, "
        "Tensor cumulative_seqlen_q, Tensor cumulative_seqlen_kv, "
        "Tensor dq, Tensor dk, Tensor dv, "
        "int mask_mode_code, float softmax_scale, "
        "int max_seqlen_q, int max_seqlen_kv, bool is_varlen) -> ()"
    );
    ops.impl("dense_prefill_bwd", torch::kCUDA, &dense_prefill_bwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
