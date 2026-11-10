/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Split-KV decode kernels: one work-group per (V tile, Q tile of the GQA
// group, batch x KV head x split) plus a combine kernel that merges the
// per-split partial outputs. The work decomposition follows vllm-xpu-kernels
// csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp.

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "splitkv_decode.hpp"
#include "splitkv_mainloop.hpp"
#include "splitkv_epilogue.hpp"

namespace flash_xpu_splitkv {

using namespace cute;

// Per-query work split, shared by the attention and combine kernels so both
// agree on which splits exist.
struct BatchSplit {
  int b;          // batch
  int row;        // query row
  int q_len;      // # queries of the batch
  int seq_len;    // KV length visible to the query
  int kv_len;     // KV length visible to the work-group (last query)
  int k_block0;   // first KV tile inside the window
  int windowed;   // # KV tiles inside the window
  int splits;     // # non-empty splits (>= 1)
  int nbps;       // KV tiles per split
  bool valid;     // the work-group has KV to process
};

// Query j of batch b. With pack_queries the KV range and its splits cover
// all queries of the batch, so every query of a work-group (and the combine
// kernel) agrees on them; a query may then see no key of a split, or none at
// all (seq_len <= 0).
CUTLASS_DEVICE BatchSplit get_batch_split(decode_args_t const& a, int b, int j) {
  BatchSplit s{};
  s.valid = false;
  s.b = b;
  int q_len;
  if (a.cu_seqlens_q != nullptr) {
    s.row = a.cu_seqlens_q[b] + j;
    q_len = a.cu_seqlens_q[b + 1] - a.cu_seqlens_q[b];
  } else {
    s.row = b * a.q_per_batch + j;
    q_len = a.q_per_batch;
  }
  s.q_len = q_len;
  if (j >= q_len || q_len > a.q_per_batch) return s;
  int full;
  if (a.seqused_k != nullptr) {
    full = a.seqused_k[b] + a.seqlen_k_add;
    full = full < a.seqlen_k ? full : a.seqlen_k;
  } else if (a.cu_seqlens_k != nullptr) {
    full = a.cu_seqlens_k[b + 1] - a.cu_seqlens_k[b];
  } else {
    full = a.seqlen_k;
  }
  s.seq_len = a.causal ? full - (q_len - 1 - j) : full;
  // [lo_len - 1 - window, kv_len): keys of the shortest / longest query.
  s.kv_len = a.pack_queries ? full : s.seq_len;
  int lo_len = a.pack_queries && a.causal ? full - (q_len - 1) : s.kv_len;
  if (s.kv_len <= 0) return s;

  int k_blocks = (s.kv_len + kKVTile - 1) / kKVTile;
  s.k_block0 = 0;
  if (a.window_size_left >= 0) {
    int first_col = lo_len - 1 - a.window_size_left;
    s.k_block0 = (first_col > 0 ? first_col : 0) / kKVTile;
  }
  s.windowed = k_blocks - s.k_block0;

  int min_tiles = a.min_tiles_per_split > 1 ? a.min_tiles_per_split : 1;
  int splits = s.windowed / min_tiles;
  splits = splits < a.num_splits ? splits : a.num_splits;
  splits = splits > 1 ? splits : 1;
  s.nbps = (s.windowed + splits - 1) / splits;
  s.splits = (s.windowed + s.nbps - 1) / s.nbps;
  s.valid = true;
  return s;
}

template <class T>
using Tensor2D = decltype(make_tensor(
    make_gmem_ptr(static_cast<T*>(nullptr)),
    make_layout(make_shape(int{}, int{}), make_stride(int64_t{}, _1{}))));

template <class T>
using Tensor2DT = decltype(make_tensor(
    make_gmem_ptr(static_cast<T*>(nullptr)),
    make_layout(make_shape(int{}, int{}), make_stride(_1{}, int64_t{}))));

template <class Mainloop_, class Epilogue_, class ElementO_>
struct SplitKVDecodeKernel {
  using Mainloop = Mainloop_;
  using Epilogue = Epilogue_;
  using ElementQ = typename Mainloop::ElementQ;
  using ElementK = typename Mainloop::ElementK;
  using ElementO = ElementO_;
  using TileShapeQK = typename Mainloop::TileShapeQK;
  using TileShapeO = typename Epilogue::TileShapeO;
  using SGPerWG = typename Mainloop::SGPerWG;
  using FragA = typename Mainloop::FragA;
  using FragARow = typename Mainloop::FragARow;
  static constexpr bool PagedKV = Mainloop::PagedKV;

  static_assert(get<1>(TileShapeQK{}) == kKVTile, "KV tile mismatch");

  using MainloopSharedStorage = typename Mainloop::SharedStorage;
  using EpilogueSharedStorage = typename Epilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };
  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  struct Params {
    decode_args_t args;
    typename Mainloop::Params mainloop;
  };

  static dim3 get_grid_shape(decode_args_t const& a) {
    int group = a.num_heads_q / a.num_heads_kv;
    int tile_q = get<0>(TileShapeO{});
    int tile_v = get<1>(TileShapeO{});
    int rows = a.pack_queries ? a.q_per_batch : group;
    int units = a.pack_queries ? a.batch * a.num_heads_q
                               : a.batch * a.q_per_batch * a.num_heads_kv;
    return dim3(
        (a.head_size_v + tile_v - 1) / tile_v,
        (rows + tile_q - 1) / tile_q,
        units * a.num_splits);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    decode_args_t const& a = params.args;

    int thr_id = int(ThreadIdxX());
    int blk_v = int(BlockIdxX());
    int blk_q = int(BlockIdxY());
    int flat = int(BlockIdxZ());
    const bool pack = a.pack_queries;
    int group = a.num_heads_q / a.num_heads_kv;
    // Work-group units: (batch, Q head) when packing queries, else
    // (batch, query, KV head).
    int heads = pack ? a.num_heads_q : a.num_heads_kv;
    int bh = (pack ? a.batch : a.batch * a.q_per_batch) * heads;
    int idx_split = flat / bh;
    int rem = flat - idx_split * bh;
    int idx_u = rem / heads;
    int h = rem - idx_u * heads;

    BatchSplit s = pack
        ? get_batch_split(a, idx_u, 0)
        : get_batch_split(a, idx_u / a.q_per_batch, idx_u % a.q_per_batch);
    if (!s.valid || idx_split >= s.splits) return;
    if (pack && blk_q * get<0>(TileShapeO{}) >= s.q_len) return;
    int idx_b = s.b;

    int head = pack ? h / group : h;      // KV head
    int head_q0 = pack ? h : h * group;   // Q head of tile row 0
    int rows = pack ? s.q_len : group;
    // Pack + causal: query j sees kv_len - (q_len - 1 - j) keys.
    int causal_rows = pack && a.causal ? s.q_len : 0;
    // Leading queries that see no key at all keep their LSE untouched.
    int first_row = causal_rows > s.kv_len ? causal_rows - s.kv_len : 0;
    int blk_k0 = s.k_block0 + idx_split * s.nbps;
    int blk_k1 = blk_k0 + s.nbps;
    int blk_end = s.k_block0 + s.windowed;
    blk_k1 = blk_k1 < blk_end ? blk_k1 : blk_end;

    auto q_ptr = const_cast<ElementQ*>(static_cast<ElementQ const*>(a.q)) +
        int64_t(s.row) * a.q_row_stride + int64_t(head_q0) * a.q_head_stride;
    Tensor Q_2D = make_tensor(
        make_gmem_ptr(q_ptr),
        make_layout(
            make_shape(rows, a.head_size),
            make_stride(pack ? a.q_row_stride : a.q_head_stride, _1{})));

    int64_t k_off, v_off;
    int kv_rows;
    if constexpr (PagedKV) {
      k_off = 0;
      v_off = 0;
      kv_rows = int(a.total_rows_kv);
    } else {
      if (a.cu_seqlens_k != nullptr) {
        k_off = int64_t(a.cu_seqlens_k[idx_b]) * a.k_seq_stride;
        v_off = int64_t(a.cu_seqlens_k[idx_b]) * a.v_seq_stride;
      } else {
        k_off = int64_t(idx_b) * a.k_batch_stride;
        v_off = int64_t(idx_b) * a.v_batch_stride;
      }
      kv_rows = s.kv_len;
    }
    auto k_ptr = const_cast<ElementK*>(static_cast<ElementK const*>(a.k)) +
        k_off + int64_t(head) * a.k_head_stride;
    auto v_ptr = const_cast<ElementK*>(static_cast<ElementK const*>(a.v)) +
        v_off + int64_t(head) * a.v_head_stride;
    Tensor K_2D = make_tensor(
        make_gmem_ptr(k_ptr),
        make_layout(
            make_shape(kv_rows, a.head_size), make_stride(a.k_seq_stride, _1{})));
    Tensor V_2D = make_tensor(
        make_gmem_ptr(v_ptr),
        make_layout(
            make_shape(a.head_size_v, kv_rows),
            make_stride(_1{}, a.v_seq_stride)));

    FragA tArA;
    FragARow tA_max, tA_sum;
    auto blk_qv = make_coord(blk_q, blk_v);

    Mainloop mainloop(params.mainloop, shared.mainloop);
    bool has_large_surface = false;
    if constexpr (PagedKV) {
      has_large_surface = a.total_rows_kv > (int64_t(1) << 24);
    }
    if (has_large_surface) {
      if constexpr (PagedKV) {
        mainloop.template operator()<true>(
            Q_2D, K_2D, V_2D, tArA, tA_max, tA_sum, blk_qv, idx_b, blk_k0,
            blk_k1, thr_id, s.kv_len, causal_rows);
      }
    } else {
      mainloop.template operator()<false>(
          Q_2D, K_2D, V_2D, tArA, tA_max, tA_sum, blk_qv, idx_b, blk_k0,
          blk_k1, thr_id, s.kv_len, causal_rows);
    }

    if constexpr (
        !is_empty_v<MainloopSharedStorage> &&
        !is_empty_v<EpilogueSharedStorage>) {
      sycl::group_barrier(
          sycl::ext::oneapi::this_work_item::get_work_group<3>());
    }

    Epilogue epilogue{{}, shared.epilogue};
    int Hq = a.num_heads_q;
    int Dv = a.head_size_v;
    // Tile rows are Q heads (stride 1 head) or queries (stride 1 row).
    if (s.splits == 1) {
      // Single split: write the final output and LSE directly.
      auto o_ptr = static_cast<ElementO*>(a.out) +
          (int64_t(s.row) * Hq + head_q0) * Dv;
      Tensor O_2D = make_tensor(
          make_gmem_ptr(o_ptr),
          make_layout(
              make_shape(rows, Dv),
              make_stride(int64_t(pack ? Hq : 1) * Dv, _1{})));
      float* lse_ptr = a.softmax_lse + int64_t(idx_b) * a.lse_batch_stride +
          int64_t(head_q0) * a.lse_head_stride +
          int64_t(s.row) * a.lse_row_stride;
      epilogue(
          O_2D, tArA, tA_max, tA_sum, blk_qv, thr_id, rows, first_row,
          lse_ptr, pack ? a.lse_row_stride : a.lse_head_stride);
    } else {
      int S = a.num_splits;
      float* o_ptr = a.oaccum +
          ((int64_t(s.row) * S + idx_split) * Hq + head_q0) * Dv;
      Tensor O_2D = make_tensor(
          make_gmem_ptr(o_ptr),
          make_layout(
              make_shape(rows, Dv),
              make_stride(int64_t(pack ? S * Hq : 1) * Dv, _1{})));
      float* lse_acc =
          a.lse_accum + (int64_t(s.row) * Hq + head_q0) * S + idx_split;
      epilogue(
          O_2D, tArA, tA_max, tA_sum, blk_qv, thr_id, rows, first_row,
          lse_acc, int64_t(pack ? Hq : 1) * S);
    }
  }
};

// Merges the per-split partial outputs of queries that used more than one
// split. Grid (1, Hq, batch * q_per_batch).
template <class ElementO_>
struct SplitKVCombineKernel {
  using ElementO = ElementO_;
  static constexpr int kThreads = 64;
  static_assert(kThreads >= kMaxSplits, "one thread per split");

  struct SharedStorage {
    float lse[kMaxSplits];
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Params {
    decode_args_t args;
  };

  static dim3 get_grid_shape(decode_args_t const& a) {
    return dim3(1, a.num_heads_q, a.batch * a.q_per_batch);
  }
  static dim3 get_block_shape() { return dim3(kThreads, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    decode_args_t const& a = params.args;
    int t = int(ThreadIdxX());
    int hq = int(BlockIdxY());
    int pb = int(BlockIdxZ());

    BatchSplit s = get_batch_split(a, pb / a.q_per_batch, pb % a.q_per_batch);
    if (!s.valid || s.splits <= 1 || s.seq_len <= 0) return;

    int S = a.num_splits;
    int Hq = a.num_heads_q;
    int Dv = a.head_size_v;
    if (t < s.splits) {
      shared.lse[t] = a.lse_accum[(int64_t(s.row) * Hq + hq) * S + t];
    }
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());

    float lse_max = -INFINITY;
    for (int i = 0; i < s.splits; i++) {
      lse_max = sycl::fmax(lse_max, shared.lse[i]);
    }
    if (lse_max == -INFINITY) lse_max = 0.f;
    float sum = 0.f;
    for (int i = 0; i < s.splits; i++) {
      sum += sycl::exp(shared.lse[i] - lse_max);
    }
    float inv_sum = 1.f / sum;

    const float* oacc = a.oaccum + (int64_t(s.row) * S * Hq + hq) * Dv;
    int64_t split_stride = int64_t(Hq) * Dv;
    ElementO* o = static_cast<ElementO*>(a.out) + (int64_t(s.row) * Hq + hq) * Dv;
    for (int d = t; d < Dv; d += kThreads) {
      float acc = 0.f;
      for (int i = 0; i < s.splits; i++) {
        float w = sycl::exp(shared.lse[i] - lse_max) * inv_sum;
        acc += w * oacc[i * split_stride + d];
      }
      o[d] = ElementO(acc);
    }
    if (t == 0) {
      a.softmax_lse[int64_t(s.b) * a.lse_batch_stride +
                    int64_t(hq) * a.lse_head_stride +
                    int64_t(s.row) * a.lse_row_stride] =
          lse_max + sycl::log(sum);
    }
  }
};

}  // namespace flash_xpu_splitkv
