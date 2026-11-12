/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Host-visible interface of the split-KV decode path (short queries).
// Kept free of CUTLASS/CuTe so flash_api.cpp can include it.

#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

// The device headers are compiled once per architecture (see
// splitkv_decode_fp16) and sycl-tla expands them differently each time;
// separate inline namespaces keep the linker from merging symbols that share
// a name but not a body.
// Test the value, not defined(): <sycl/sycl.hpp> defines every
// __SYCL_TARGET_* macro it does not see as 0 (as sycl-tla's cutlass.h does).
#if __SYCL_TARGET_INTEL_GPU_CRI__
#define FLASH_XPU_SPLITKV_TARGET_NS xe35
#define FLASH_XPU_SPLITKV_TARGET_ARCH 35
#define FLASH_XPU_VLLM_NS vllm_xe35
// vllm-xpu-kernels builds its XE3 kernels with 512 GRF (add_xe3_kernel_library).
#ifndef VLLM_GRF_SIZE
#define VLLM_GRF_SIZE 512
#endif
#else
#define FLASH_XPU_SPLITKV_TARGET_NS xe20
#define FLASH_XPU_SPLITKV_TARGET_ARCH 20
#define FLASH_XPU_VLLM_NS vllm_xe20
#endif

namespace flash_xpu_splitkv {

// Work-group shape of the decode kernel (kv_tile 64, 4 subgroups of 16).
constexpr int kKVTile = 64;
constexpr int kQTile = 8;  // Q rows per work-group
constexpr int kMaxSplits = 64;

struct decode_args_t {
  const void* q = nullptr;
  const void* k = nullptr;
  const void* v = nullptr;
  void* out = nullptr;            // [rows, Hq, Dv], contiguous, q dtype
  float* softmax_lse = nullptr;   // final LSE, see lse_*_stride
  float* oaccum = nullptr;        // [rows, splits, Hq, Dv] (num_splits > 1)
  float* lse_accum = nullptr;     // [rows, Hq, splits]     (num_splits > 1)

  // Every batch b has up to q_per_batch queries (row = cu_seqlens_q ?
  // cu_seqlens_q[b] + j : b * q_per_batch + j). With cu_seqlens_q, batches
  // with more than q_per_batch queries are skipped (left to the generic
  // kernel), as are positions past the batch's query count.
  // A work-group handles kQTile consecutive rows of the (query, Q head of
  // the GQA group) rows of one (batch, KV head), head fastest, like
  // PackGQA of FlashAttention 3: KV is read once per kQTile rows.
  const int* cu_seqlens_q = nullptr;
  int q_per_batch = 1;
  // Bottom-right causal: query j of a batch with n queries sees the first
  // seq_len - (n - 1 - j) keys. Without it every query sees all keys.
  bool causal = false;
  // KV length of batch b:
  //   seqused_k   ? seqused_k[b] + seqlen_k_add
  //   cu_seqlens_k? cu_seqlens_k[b + 1] - cu_seqlens_k[b]
  //   otherwise     seqlen_k
  // Rows without a visible key (batches with length <= 0, leading causal
  // queries) get output 0 and LSE empty_lse.
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

  // softmax_lse[b * lse_batch_stride + hq * lse_head_stride
  //             + row * lse_row_stride]
  int64_t lse_batch_stride = 0, lse_head_stride = 0, lse_row_stride = 0;

  float softmax_scale = 1.f;
  float empty_lse = 0.f;      // LSE of a row without a visible key
  int window_size_left = -1;  // < 0: no local mask
  // Upper bound on KV splits. A batch whose windowed KV has fewer than
  // num_splits * min_tiles_per_split tiles uses fewer splits; a batch that
  // ends up with one split writes out / softmax_lse directly.
  int num_splits = 1;
  int min_tiles_per_split = 1;
};

// Head-dim buckets with a compiled kernel. The runtime head size may be any
// multiple of 8 up to the bucket.
// The kernels are built twice: Arch 20 for pvc/bmg, and Arch 35 from the
// CRI-only translation units (-D__SYCL_TARGET_INTEL_GPU_CRI__), where
// sycl-tla emits the Xe35 block loads (cache hints, wider transposed loads).
template <int Arch, int HeadDim>
void splitkv_decode_fp16(sycl::queue& queue, const decode_args_t& args);
template <int Arch, int HeadDim>
void splitkv_decode_bf16(sycl::queue& queue, const decode_args_t& args);

#define FLASH_XPU_SPLITKV_DECLARE(ARCH, HD)                                  \
  template <>                                                                \
  void splitkv_decode_fp16<ARCH, HD>(sycl::queue&, const decode_args_t&);    \
  template <>                                                                \
  void splitkv_decode_bf16<ARCH, HD>(sycl::queue&, const decode_args_t&);
#define FLASH_XPU_SPLITKV_DECLARE_ARCH(ARCH)                                 \
  FLASH_XPU_SPLITKV_DECLARE(ARCH, 64)                                        \
  FLASH_XPU_SPLITKV_DECLARE(ARCH, 96)                                        \
  FLASH_XPU_SPLITKV_DECLARE(ARCH, 128)                                       \
  FLASH_XPU_SPLITKV_DECLARE(ARCH, 192)                                       \
  FLASH_XPU_SPLITKV_DECLARE(ARCH, 256)                                       \
  FLASH_XPU_SPLITKV_DECLARE(ARCH, 512)
FLASH_XPU_SPLITKV_DECLARE_ARCH(20)
FLASH_XPU_SPLITKV_DECLARE_ARCH(35)
#undef FLASH_XPU_SPLITKV_DECLARE_ARCH
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

// Paged single-query decode of vllm-xpu-kernels (csrc/xpu/attn/xe_3,
// src/decode/vllm), fp16/bf16, page size a multiple of 64. Same fields as
// vllm's paged_decode_args_t, plus lse_row_stride. Built for Xe35 (CRI) only.
struct vllm_decode_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* tem_out;
  void* softmax_lse_accum;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;  // per-batch KV lengths (seqused_k), or cumulative
  bool kv_cumulative = false;  // cu_seqlens_k is cumulative (b + 1 entries)
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  void* q_scale = nullptr;
  void* k_scale = nullptr;
  void* v_scale = nullptr;
  float sm_scale;
  void* sm_sink = nullptr;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int v_head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = true;
  bool is_paged = true;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  int num_kv_splits = 1;
  const int* splits_per_seq = nullptr;
  const void* work_list = nullptr;
  int total_wgs = 0;
  int64_t k_stride_page = 0;
  int64_t k_stride_seq = 0;
  int64_t k_stride_heads = 0;
  int64_t v_stride_page = 0;
  int64_t v_stride_seq = 0;
  int64_t v_stride_heads = 0;
  void* is_prefill = nullptr;
  int64_t q_stride_seq = 0;
  int64_t q_stride_heads = 0;
  int64_t q_stride_batch = 0;
  int page_stride_elements = 0;
  float* softmax_lse = nullptr;
  int lse_stride = 0;      // between heads
  int lse_row_stride = 1;  // between query tokens
  // Packed speculative decode: num_heads_q holds spec_q_len query positions
  // per KV head group, position-major (vllm's spec_q_len).
  int spec_q_len = 1;
};

template <int Arch, int HeadDim>
void vllm_paged_decode_fp16(sycl::queue& queue, const vllm_decode_args_t& args);
template <int Arch, int HeadDim>
void vllm_paged_decode_bf16(sycl::queue& queue, const vllm_decode_args_t& args);

#define FLASH_XPU_VLLM_DECLARE(HD)                                            \
  template <>                                                                \
  void vllm_paged_decode_fp16<35, HD>(sycl::queue&, const vllm_decode_args_t&); \
  template <>                                                                \
  void vllm_paged_decode_bf16<35, HD>(sycl::queue&, const vllm_decode_args_t&);
FLASH_XPU_VLLM_DECLARE(64)
FLASH_XPU_VLLM_DECLARE(96)
FLASH_XPU_VLLM_DECLARE(128)
FLASH_XPU_VLLM_DECLARE(192)
FLASH_XPU_VLLM_DECLARE(256)
FLASH_XPU_VLLM_DECLARE(512)
#undef FLASH_XPU_VLLM_DECLARE

}  // namespace flash_xpu_splitkv
