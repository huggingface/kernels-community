#pragma once

#include <optional>
#include <vector>

#include <torch/torch.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(at::Tensor q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        at::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        at::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        std::optional<at::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<at::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        std::optional<at::Tensor> q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> cu_seqlens_q_,  // b+1
        std::optional<at::Tensor> cu_seqlens_k_,  // b+1
        std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
        std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int64_t> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int64_t> max_seqlen_k_,
        std::optional<at::Tensor> page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<at::Tensor> kv_batch_idx_, // b. indices to index into the KV cache
        std::optional<at::Tensor> leftpad_k_, // b
        std::optional<at::Tensor> rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        std::optional<at::Tensor> rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        std::optional<at::Tensor> seqlens_rotary_, // b
        std::optional<at::Tensor> q_descale_,  // (b, h_k), not (b, h)
        std::optional<at::Tensor> k_descale_,  // (b, h_k)
        std::optional<at::Tensor> v_descale_,  // (b, h_k)
        std::optional<double> softmax_scale_,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        double softcap,
        bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        std::optional<at::Tensor> scheduler_metadata_,  // (b + 1)
        int64_t num_splits,
        std::optional<bool> pack_gqa_,
        int64_t sm_margin
        );

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(
    at::Tensor dout,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    at::Tensor q,     // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    at::Tensor k,     // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    at::Tensor v,     // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
    at::Tensor out,   // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    at::Tensor softmax_lse,    // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    std::optional<at::Tensor> dq_,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    std::optional<at::Tensor> dk_,   // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    std::optional<at::Tensor> dv_,   // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
    std::optional<at::Tensor> cu_seqlens_q_,   // b+1
    std::optional<at::Tensor> cu_seqlens_k_,   // b+1
    std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
    std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<double> softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    bool deterministic,
    int64_t sm_margin
);

std::tuple<at::Tensor, at::Tensor>
mha_combine(at::Tensor out_partial,         // num_splits x batch_size x seqlen x num_heads x head_size
            at::Tensor lse_partial,         // num_splits x batch_size x seqlen x num_heads
            std::optional<at::Tensor> out_,        // batch_size x seqlen x num_heads x head_size
            std::optional<at::ScalarType> out_dtype_
            );

at::Tensor
mha_fwd_get_scheduler_metadata(
        int64_t batch_size,
        int64_t max_seqlen_q,
        int64_t max_seqlen_k,
        int64_t num_heads,
        int64_t num_heads_k,
        int64_t headdim,
        int64_t headdim_v,
        at::ScalarType qkv_dtype,
        at::Tensor seqused_k, // b
        std::optional<at::Tensor> cu_seqlens_q_,  // b+1
        std::optional<at::Tensor> cu_seqlens_k_,  // b+1
        std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
        std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<at::Tensor> leftpad_k_, // b
        std::optional<int64_t> page_size,
        int64_t max_seqlen_k_new,  // 0 means we're not appending new KV
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        bool has_softcap,
        int64_t num_splits,
        std::optional<bool> pack_gqa_,
        int64_t sm_margin
        );

