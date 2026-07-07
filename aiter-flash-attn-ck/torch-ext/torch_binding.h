#pragma once

#include <torch/torch.h>

#include <optional>
#include <vector>

// Thin wrappers around aiter's `torch_itfs` Composable-Kernel FlashAttention
// entry points. They expose Torch-library-compatible signatures (no
// `std::optional<const Tensor>`, scalars as double/int64_t) and forward to the
// vendored `aiter::torch_itfs::mha_fwd` / `aiter::torch_itfs::mha_varlen_fwd`.

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,                         // [b, sq, hq, d]
        const at::Tensor &k,                   // [b, sk, hk, d]
        const at::Tensor &v,                   // [b, sk, hk, d]
        double dropout_p,
        double softmax_scale,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t sink_size,
        bool return_softmax_lse,
        bool return_dropout_randval,
        std::optional<at::Tensor> cu_seqlens_q,
        std::optional<at::Tensor> cu_seqlens_kv,
        std::optional<at::Tensor> out,         // [b, sq, hq, d]
        std::optional<at::Tensor> bias,        // [sq, sk]
        std::optional<at::Tensor> alibi_slopes, // [hq] or [b, hq]
        std::optional<at::Tensor> q_descale,   // [1]
        std::optional<at::Tensor> k_descale,   // [1]
        std::optional<at::Tensor> v_descale,   // [1]
        std::optional<at::Tensor> s_aux,    // [hq]
        std::optional<at::Generator> gen);

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,                   // [total_q, hq, d]
               const at::Tensor &k,             // [total_k, hk, d]
               const at::Tensor &v,             // [total_k, hk, d]
               const at::Tensor &cu_seqlens_q,  // [b+1]
               std::optional<at::Tensor> cu_seqlens_k, // [b+1]
               int64_t max_seqlen_q,
               int64_t max_seqlen_k,
               int64_t min_seqlen_q,
               double dropout_p,
               double softmax_scale,
               double logits_soft_cap,
               bool zero_tensors,
               bool is_causal,
               int64_t window_size_left,
               int64_t window_size_right,
               int64_t sink_size,
               bool return_softmax_lse,
               bool return_dropout_randval,
               std::optional<at::Tensor> out,          // [total_q, hq, d]
               std::optional<at::Tensor> block_table,
               std::optional<at::Tensor> bias,
               std::optional<at::Tensor> alibi_slopes,
               std::optional<at::Tensor> q_descale,
               std::optional<at::Tensor> k_descale,
               std::optional<at::Tensor> v_descale,
               std::optional<at::Generator> gen,
               std::optional<at::Tensor> cu_seqlens_q_padded,
               std::optional<at::Tensor> cu_seqlens_k_padded,
               std::optional<at::Tensor> s_aux);
