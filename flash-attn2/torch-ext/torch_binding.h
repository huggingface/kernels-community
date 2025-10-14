#pragma once

#include <torch/torch.h>

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
std::vector<torch::Tensor>
mha_fwd(
    torch::Tensor &q, 
    const torch::Tensor &k, 
    const torch::Tensor &v,
    c10::optional<torch::Tensor> out_,\
    c10::optional<torch::Tensor> alibi_slopes_,
    const double p_dropout, 
    const double softmax_scale, 
    bool is_causal,
    const int64_t window_size_left, 
    const int64_t window_size_right,
    const double softcap, 
    const bool return_softmax,
    c10::optional<at::Generator> gen_);

std::vector<torch::Tensor>
mha_varlen_fwd(
    at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const torch::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const torch::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    c10::optional<torch::Tensor> out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const torch::Tensor &cu_seqlens_q,  // b+1
    const torch::Tensor &cu_seqlens_k,  // b+1
    c10::optional<torch::Tensor> seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    // c10::optional<const at::Tensor> leftpad_k_, // batch_size
    c10::optional<torch::Tensor> leftpad_k_, // batch_size
    c10::optional<torch::Tensor> block_table_, // batch_size x max_num_blocks_per_seq
    c10::optional<torch::Tensor> alibi_slopes_, // num_heads or b x num_heads
    int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const double p_dropout,
    const double softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const double softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_); 

std::vector<torch::Tensor>
mha_bwd(const torch::Tensor &dout,                         // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const torch::Tensor &q,                            // batch_size x seqlen_q x num_heads x head_size
        const torch::Tensor &k,                            // batch_size x seqlen_k x num_heads_k x head_size
        const torch::Tensor &v,                            // batch_size x seqlen_k x num_heads_k x head_size
        const torch::Tensor &out,                          // batch_size x seqlen_q x num_heads x head_size
        const torch::Tensor &softmax_lse,                  // b x h x seqlen_q
        const c10::optional<torch::Tensor> &dq_,           // batch_size x seqlen_q x num_heads x head_size
        const c10::optional<torch::Tensor> &dk_,           // batch_size x seqlen_k x num_heads_k x head_size
        const c10::optional<torch::Tensor> &dv_,           // batch_size x seqlen_k x num_heads_k x head_size
        const c10::optional<torch::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const double p_dropout,                            // probability to drop
        const double softmax_scale,
        const bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        const double softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        const c10::optional<torch::Tensor> &rng_state);


std::vector<torch::Tensor>
mha_varlen_bwd(
        const torch::Tensor &dout,                         // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const torch::Tensor &q,                            // batch_size x seqlen_q x num_heads x head_size
        const torch::Tensor &k,                            // batch_size x seqlen_k x num_heads_k x head_size
        const torch::Tensor &v,                            // batch_size x seqlen_k x num_heads_k x head_size
        const torch::Tensor &out,                          // batch_size x seqlen_q x num_heads x head_size
        const torch::Tensor &softmax_lse,                  // b x h x seqlen_q
        const c10::optional<torch::Tensor> &dq_,           // batch_size x seqlen_q x num_heads x head_size
        const c10::optional<torch::Tensor> &dk_,           // batch_size x seqlen_k x num_heads_k x head_size
        const c10::optional<torch::Tensor> &dv_,           // batch_size x seqlen_k x num_heads_k x head_size
        const torch::Tensor &cu_seqlens_q,                 // batch_size + 1
        const torch::Tensor &cu_seqlens_k,                 // batch_size + 1
        const c10::optional<torch::Tensor> &alibi_slopes_, // num_heads or b x num_heads
        const int64_t max_seqlen_q,
        const int64_t max_seqlen_k,
        const double p_dropout,
        const double softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        const double softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        const c10::optional<torch::Tensor> &rng_state);

std::vector<torch::Tensor>
mha_fwd_kvcache(
        const torch::Tensor &q,                                // batch_size x seqlen_q x num_heads x head_size
        const torch::Tensor &kcache,                           // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        const torch::Tensor &vcache,                           // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
        const c10::optional<torch::Tensor> &k_,                // batch_size x seqlen_knew x num_heads_k x head_size
        const c10::optional<torch::Tensor> &v_,                // batch_size x seqlen_knew x num_heads_k x head_size
        const c10::optional<torch::Tensor> &seqlens_k_,        // batch_size
        const c10::optional<torch::Tensor> &rotary_cos_,       // seqlen_ro x (rotary_dim / 2)
        const c10::optional<torch::Tensor> &rotary_sin_,       // seqlen_ro x (rotary_dim / 2)
        const c10::optional<torch::Tensor> &cache_batch_idx_,  // indices to index into the KV cache
        const c10::optional<torch::Tensor> &leftpad_k_,        // batch_size
        const c10::optional<torch::Tensor> &block_table_,      // batch_size x max_num_blocks_per_seq
        const c10::optional<torch::Tensor> &alibi_slopes_,     // num_heads or batch_size x num_heads
        const c10::optional<torch::Tensor> &out_,              // batch_size x seqlen_q x num_heads x head_size
        const double softmax_scale,
        bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        const double softcap,
        bool is_rotary_interleaved,
        const int64_t num_splits);
