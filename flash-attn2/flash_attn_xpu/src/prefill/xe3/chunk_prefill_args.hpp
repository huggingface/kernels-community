/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Host-side interface of the Xe3 chunk-prefill kernel (src/prefill/xe3,
// adapted from vllm-xpu-kernels csrc/xpu/attn/xe_3).
// Built into the CRI image only.
#pragma once

#include <sycl/sycl.hpp>

namespace flash_xpu_prefill::xe3 {

struct chunk_prefill_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  void* q_scale;
  void* k_scale;
  void* v_scale;
  float sm_scale;
  void* sm_sink;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_paged = false;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  // softmax_lse output (nullptr when not requested)
  float* softmax_lse = nullptr;
  int lse_stride = 0;  // stride along head dim (= total_seqlen_q)
  // Q/O strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int q_stride_seq = 0;
  int q_stride_heads = 0;
  int q_stride_batch = 0;
  // K strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int k_stride_seq = 0;
  int k_stride_heads = 0;
  int k_stride_batch = 0;
  // V strides in CUTLASS order: (head_size=1, seq, heads, batch)
  int v_stride_seq = 0;
  int v_stride_heads = 0;
  int v_stride_batch = 0;
  // O strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int o_stride_seq = 0;
  int o_stride_heads = 0;
  int o_stride_batch = 0;
  // per-batch mask: true = prefill, false = decode; nullptr = process all
  void* is_prefill = nullptr;
  int page_stride_elements = 0;
  // kc: skip batches with fewer queries (left to the split-KV decode kernel).
  int min_seqlen_q = 0;
};

// Launches the varlen chunk-prefill kernel for head size HD (fp16 when !Bf16),
// paged or contiguous K/V. Returns false without launching when the flag /
// page-size combination is not instantiated; the caller then uses its own
// kernel. One instantiation per flash_chunk_prefill_hdim*_*_cri.cpp file.
template <int HD, bool Bf16, bool Paged>
bool chunk_prefill_impl(sycl::queue& queue, const chunk_prefill_args_t& args);

template <int HD, bool Bf16>
bool chunk_prefill(sycl::queue& queue, const chunk_prefill_args_t& args) {
  return args.is_paged ? chunk_prefill_impl<HD, Bf16, true>(queue, args)
                       : chunk_prefill_impl<HD, Bf16, false>(queue, args);
}

}  // namespace flash_xpu_prefill::xe3
