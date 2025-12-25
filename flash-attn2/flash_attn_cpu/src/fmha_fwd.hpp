#pragma once

#include <ATen/ATen.h>

// Forward declaration of the flash attention varlen implementation
// 
// Parameters:
//   q: [total_q, num_heads, head_size]
//   k: [total_k, num_heads_kv, head_size]  
//   v: [total_k, num_heads_kv, head_size]
//   out: [total_q, num_heads, head_size] - output tensor
//   cu_seqlens_q: [batch_size + 1] - cumulative sequence lengths for queries
//   cu_seqlens_k: [batch_size + 1] - cumulative sequence lengths for keys
//   max_seqlen_q: maximum query sequence length
//   max_seqlen_k: maximum key sequence length
//   softmax_scale: scaling factor for softmax
//   is_causal: whether to apply causal masking
//
void fmha_fwd_varlen_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool is_causal);
