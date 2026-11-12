/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#include <sycl/sycl.hpp>

#include "decode/splitkv_decode.hpp"

namespace flash_xpu_splitkv {

void splitkv_append_kv(sycl::queue& queue, const append_args_t& a) {
  size_t total = size_t(a.batch) * a.seqlen_new * a.num_heads_kv * a.head_size;
  if (total == 0) return;
  const append_args_t p = a;
  queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
    size_t i = id[0];
    int d = int(i % p.head_size);
    i /= p.head_size;
    int h = int(i % p.num_heads_kv);
    i /= p.num_heads_kv;
    int s = int(i % p.seqlen_new);
    int b = int(i / p.seqlen_new);

    int base = p.cache_seqlens[b];
    if (base + p.seqlen_new <= 0) return;
    int pos = base + s;
    if (pos < 0) return;

    int64_t kc_off, vc_off;
    if (p.block_table != nullptr) {
      if (pos >= p.max_pages_per_seq * p.page_size) return;
      int page = p.block_table[int64_t(b) * p.block_table_batch_stride +
                               pos / p.page_size];
      if (page < 0) return;
      int r = pos % p.page_size;
      kc_off = int64_t(page) * p.kc_batch_stride + int64_t(r) * p.kc_seq_stride;
      vc_off = int64_t(page) * p.vc_batch_stride + int64_t(r) * p.vc_seq_stride;
    } else {
      if (pos >= p.seqlen_cache) return;
      kc_off = int64_t(b) * p.kc_batch_stride + int64_t(pos) * p.kc_seq_stride;
      vc_off = int64_t(b) * p.vc_batch_stride + int64_t(pos) * p.vc_seq_stride;
    }
    kc_off += int64_t(h) * p.kc_head_stride + d;
    vc_off += int64_t(h) * p.vc_head_stride + d;
    int64_t kn_off = int64_t(b) * p.kn_batch_stride +
        int64_t(s) * p.kn_seq_stride + int64_t(h) * p.kn_head_stride + d;
    int64_t vn_off = int64_t(b) * p.vn_batch_stride +
        int64_t(s) * p.vn_seq_stride + int64_t(h) * p.vn_head_stride + d;

    // 16-bit elements (fp16 / bf16): copy the raw bits.
    static_cast<uint16_t*>(p.k_cache)[kc_off] =
        static_cast<const uint16_t*>(p.k_new)[kn_off];
    static_cast<uint16_t*>(p.v_cache)[vc_off] =
        static_cast<const uint16_t*>(p.v_new)[vn_off];
  });
}

}  // namespace flash_xpu_splitkv
