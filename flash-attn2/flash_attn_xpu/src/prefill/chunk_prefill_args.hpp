/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Host-side interface of the chunk-prefill kernel (src/prefill, adapted from
// vllm-xpu-kernels csrc/xpu/attn/xe_2 and xe_3).
#pragma once

#include <sycl/sycl.hpp>

namespace flash_xpu_prefill {

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
  void* q_scale;  // Xe35 only
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

// Launches the varlen chunk-prefill kernel of architecture Arch (20: Xe2,
// 35: Xe35, CRI image) for head size HD (fp16 when !Bf16), paged or
// contiguous K/V. Returns false without launching when the flag / page-size
// combination is not instantiated; the caller then uses its own kernel. One
// instantiation per file under src/instantiations/<arch>/chunk_prefill/.
template <int Arch, int HD, bool Bf16, bool Paged>
bool chunk_prefill_impl(sycl::queue& queue, const chunk_prefill_args_t& args);

// Paged K/V for every head size; contiguous K/V for head size 128, and for 64
// and 256 on Xe35.
template <int Arch, int HD, bool Bf16>
bool chunk_prefill_hd(sycl::queue& queue, const chunk_prefill_args_t& args) {
  if (args.is_paged)
    return chunk_prefill_impl<Arch, HD, Bf16, true>(queue, args);
  if constexpr (HD == 128 || (Arch >= 35 && (HD == 64 || HD == 256)))
    return chunk_prefill_impl<Arch, HD, Bf16, false>(queue, args);
  return false;
}

template <int Arch>
bool chunk_prefill(sycl::queue& queue, const chunk_prefill_args_t& args,
                   bool bf16) {
  switch (args.head_size) {
#define FLASH_XPU_CHUNK_PREFILL_CASE(HD)                     \
  case HD:                                                   \
    return bf16 ? chunk_prefill_hd<Arch, HD, true>(queue, args) \
                : chunk_prefill_hd<Arch, HD, false>(queue, args);
    FLASH_XPU_CHUNK_PREFILL_CASE(64)
    FLASH_XPU_CHUNK_PREFILL_CASE(96)
    FLASH_XPU_CHUNK_PREFILL_CASE(128)
    FLASH_XPU_CHUNK_PREFILL_CASE(192)
    FLASH_XPU_CHUNK_PREFILL_CASE(256)
    FLASH_XPU_CHUNK_PREFILL_CASE(512)
#undef FLASH_XPU_CHUNK_PREFILL_CASE
    default:
      return false;
  }
}

}  // namespace flash_xpu_prefill
