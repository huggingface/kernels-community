/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Host-visible interface of the split-KV decode path (seqlen_q == 1).
// Kept free of CUTLASS/CuTe so flash_api.cpp can include it.

#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

namespace flash_xpu_splitkv {

// Work-group shape of the decode kernel (kv_tile 64, 4 subgroups of 16).
constexpr int kKVTile = 64;
constexpr int kMaxSplits = 64;

struct decode_args_t {
  const void* q = nullptr;
  const void* k = nullptr;
  const void* v = nullptr;
  void* out = nullptr;            // [rows, Hq, Dv], contiguous, q dtype
  float* softmax_lse = nullptr;   // final LSE, see lse_*_stride
  float* oaccum = nullptr;        // [rows, splits, Hq, Dv] (num_splits > 1)
  float* lse_accum = nullptr;     // [rows, Hq, splits]     (num_splits > 1)

  // Query row of batch b: cu_seqlens_q ? cu_seqlens_q[b] : b. A batch whose
  // cu_seqlens_q range is empty is skipped. Only one query per batch is read.
  const int* cu_seqlens_q = nullptr;
  // KV length of batch b:
  //   seqused_k   ? seqused_k[b] + seqlen_k_add
  //   cu_seqlens_k? cu_seqlens_k[b + 1] - cu_seqlens_k[b]
  //   otherwise     seqlen_k
  // Batches with length <= 0 are skipped (out / softmax_lse untouched).
  // With seqused_k the length is clamped to seqlen_k (the cache capacity).
  const int* seqused_k = nullptr;
  int seqlen_k_add = 0;
  const int* cu_seqlens_k = nullptr;  // non-paged only: first KV row of b
  int seqlen_k = 0;

  // Paged KV: block_table != nullptr. k/v point at the whole cache.
  const int* block_table = nullptr;
  int64_t block_table_batch_stride = 0;
  int page_size = 0;
  int max_pages_per_seq = 0;
  int page_stride_rows = 0;  // k_batch_stride / k_seq_stride
  int64_t total_rows_kv = 0; // num_pages * page_stride_rows

  int batch = 0;
  int num_heads_q = 0;
  int num_heads_kv = 0;
  int head_size = 0;
  int head_size_v = 0;

  // Strides in elements. "batch" is the page stride for paged KV.
  int64_t q_row_stride = 0, q_head_stride = 0;
  int64_t k_batch_stride = 0, k_seq_stride = 0, k_head_stride = 0;
  int64_t v_batch_stride = 0, v_seq_stride = 0, v_head_stride = 0;

  // softmax_lse[hq * lse_head_stride + row * lse_row_stride]
  int64_t lse_head_stride = 0, lse_row_stride = 0;

  float softmax_scale = 1.f;
  int window_size_left = -1;  // < 0: no local mask
  // Upper bound on KV splits. A batch whose windowed KV has fewer than
  // num_splits * min_tiles_per_split tiles uses fewer splits; a batch that
  // ends up with one split writes out / softmax_lse directly.
  int num_splits = 1;
  int min_tiles_per_split = 1;
};

// Head-dim buckets with a compiled kernel. The runtime head size may be any
// multiple of 8 up to the bucket.
template <int HeadDim>
void splitkv_decode_fp16(sycl::queue& queue, const decode_args_t& args);
template <int HeadDim>
void splitkv_decode_bf16(sycl::queue& queue, const decode_args_t& args);

#define FLASH_XPU_SPLITKV_DECLARE(HD)                                        \
  template <>                                                                \
  void splitkv_decode_fp16<HD>(sycl::queue&, const decode_args_t&);          \
  template <>                                                                \
  void splitkv_decode_bf16<HD>(sycl::queue&, const decode_args_t&);
FLASH_XPU_SPLITKV_DECLARE(64)
FLASH_XPU_SPLITKV_DECLARE(96)
FLASH_XPU_SPLITKV_DECLARE(128)
FLASH_XPU_SPLITKV_DECLARE(192)
FLASH_XPU_SPLITKV_DECLARE(256)
FLASH_XPU_SPLITKV_DECLARE(512)
#undef FLASH_XPU_SPLITKV_DECLARE

// Scatter k_new/v_new [b, s_new, Hkv, D] into the cache at positions
// cache_seqlens[b] + i, like CUDA FlashAttention's fused append. Rows whose
// total length (cache_seqlens[b] + s_new) is <= 0 are skipped, as are
// positions outside the cache or mapped to a negative page.
struct append_args_t {
  const void* k_new = nullptr;
  const void* v_new = nullptr;
  void* k_cache = nullptr;
  void* v_cache = nullptr;
  const int* cache_seqlens = nullptr;
  const int* block_table = nullptr;  // nullptr: non-paged
  int64_t block_table_batch_stride = 0;
  int page_size = 0;
  int max_pages_per_seq = 0;
  int batch = 0, seqlen_new = 0, num_heads_kv = 0, head_size = 0;
  int seqlen_cache = 0;  // non-paged capacity
  int64_t kn_batch_stride = 0, kn_seq_stride = 0, kn_head_stride = 0;
  int64_t vn_batch_stride = 0, vn_seq_stride = 0, vn_head_stride = 0;
  int64_t kc_batch_stride = 0, kc_seq_stride = 0, kc_head_stride = 0;
  int64_t vc_batch_stride = 0, vc_seq_stride = 0, vc_head_stride = 0;
};

void splitkv_append_kv(sycl::queue& queue, const append_args_t& args);

}  // namespace flash_xpu_splitkv
