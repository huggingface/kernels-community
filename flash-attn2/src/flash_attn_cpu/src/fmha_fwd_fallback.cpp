/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************************/

// Fallback implementation - no special CPU features required
// Uses PyTorch tensor operations (matmul, softmax) for maximum compatibility

#include <ATen/ATen.h>
#include <limits>

#include "fmha_fwd_fallback.hpp"

namespace flash_attn_cpu {
namespace fallback {

//==============================================================================
// Naive Flash Attention using PyTorch ops
// This works on any CPU but is slower than optimized implementations
//==============================================================================

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
    bool is_causal) {

  // q, k, v: [total_tokens, num_heads, head_dim]
  // cu_seqlens_q, cu_seqlens_k: [batch_size + 1]
  // out: [total_tokens, num_heads, head_dim]

  const int batch_size = cu_seqlens_q.size(0) - 1;
  const int num_heads = q.size(1);
  const int head_dim = q.size(2);
  const int head_dim_v = v.size(2);

  auto cu_seqlens_q_cpu = cu_seqlens_q.to(at::kCPU).contiguous();
  auto cu_seqlens_k_cpu = cu_seqlens_k.to(at::kCPU).contiguous();
  const int32_t* cu_q = cu_seqlens_q_cpu.data_ptr<int32_t>();
  const int32_t* cu_k = cu_seqlens_k_cpu.data_ptr<int32_t>();

  // Process each sequence in the batch
  for (int b = 0; b < batch_size; ++b) {
    int32_t q_start = cu_q[b];
    int32_t q_end = cu_q[b + 1];
    int32_t k_start = cu_k[b];
    int32_t k_end = cu_k[b + 1];

    int32_t seqlen_q = q_end - q_start;
    int32_t seqlen_k = k_end - k_start;

    if (seqlen_q == 0 || seqlen_k == 0) {
      continue;
    }

    // Extract sequences: [seqlen, num_heads, head_dim] -> [num_heads, seqlen, head_dim]
    auto q_seq = q.slice(0, q_start, q_end).transpose(0, 1);  // [num_heads, seqlen_q, head_dim]
    auto k_seq = k.slice(0, k_start, k_end).transpose(0, 1);  // [num_heads, seqlen_k, head_dim]
    auto v_seq = v.slice(0, k_start, k_end).transpose(0, 1);  // [num_heads, seqlen_k, head_dim_v]

    // Compute attention scores: Q @ K^T * scale
    // [num_heads, seqlen_q, head_dim] @ [num_heads, head_dim, seqlen_k] -> [num_heads, seqlen_q, seqlen_k]
    auto scores = at::matmul(q_seq, k_seq.transpose(-2, -1)) * softmax_scale;

    // Apply causal mask if needed
    if (is_causal) {
      // Create causal mask: positions where q_pos > k_pos should be masked
      auto mask = at::ones({seqlen_q, seqlen_k}, scores.options().dtype(at::kBool));
      mask = at::tril(mask, /*diagonal=*/seqlen_k - seqlen_q);
      mask = mask.unsqueeze(0);  // [1, seqlen_q, seqlen_k]

      // Apply mask: set masked positions to -inf
      scores = scores.masked_fill(~mask, -std::numeric_limits<float>::infinity());
    }

    // Softmax over last dimension
    auto attn_weights = at::softmax(scores.to(at::kFloat), -1).to(q.scalar_type());

    // Compute output: attn_weights @ V
    // [num_heads, seqlen_q, seqlen_k] @ [num_heads, seqlen_k, head_dim_v] -> [num_heads, seqlen_q, head_dim_v]
    auto out_seq = at::matmul(attn_weights, v_seq);

    // Transpose back and copy to output: [num_heads, seqlen_q, head_dim_v] -> [seqlen_q, num_heads, head_dim_v]
    out.slice(0, q_start, q_end).copy_(out_seq.transpose(0, 1));
  }
}

}  // namespace fallback
}  // namespace flash_attn_cpu
