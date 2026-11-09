/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Split-KV decode mainloop (seqlen_q == 1, GQA head group packed into the Q
// tile). Adapted from vllm-xpu-kernels
// csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp (DecodeFwdMainloop).
// Lives in its own namespace so its symbols can never bind to the identically
// named vLLM or flash-attn prefill templates loaded in the same process.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"

#include "../xe_barrier_compat.hpp"
#include "../collective/copy_block_slm.hpp"

namespace flash_xpu_splitkv {

using namespace cute;

CUTLASS_DEVICE void wg_barrier() {
  asm volatile("lsc_fence.ugm.none.group\n");
  asm volatile("barrier\n");
}

// The per-head block-2D surfaces start wherever the head lands in memory. The
// Xe 2D block messages need a 64B-aligned base, so round the base down and
// widen the surface by the misalignment; the coordinate tensors are shifted by
// the same amount (make_shifted_identity_tensor) so the head is still
// addressed correctly. An aligned base makes both a no-op.
template <int XMode, class Engine, class Layout>
CUTLASS_DEVICE auto
align_block_2d_base(cute::Tensor<Engine, Layout> const& t, int& x_shift) {
  using T = cute::remove_cvref_t<typename Engine::value_type>;
  constexpr uintptr_t kAlign = 64;
  auto raw = reinterpret_cast<uintptr_t>(cute::raw_pointer_cast(t.data()));
  uintptr_t misalign = raw & (kAlign - 1);
  x_shift = static_cast<int>(misalign / sizeof(T));
  auto shape =
      cute::replace<XMode>(t.shape(), cute::get<XMode>(t.shape()) + x_shift);
  return cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<T*>(raw - misalign)),
      cute::make_layout(shape, t.stride()));
}

template <int XMode, class Shape>
CUTLASS_DEVICE auto
make_shifted_identity_tensor(Shape const& shape, int x_shift) {
  static_assert(cute::rank_v<Shape> == 2, "expected a rank-2 shape");
  if constexpr (XMode == 0) {
    return cute::domain_offset(
        cute::make_coord(x_shift, 0), cute::make_identity_tensor(shape));
  } else {
    return cute::domain_offset(
        cute::make_coord(0, x_shift), cute::make_identity_tensor(shape));
  }
}

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(
      get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool PagedKV_,
    bool LocalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ2D_,   // Per-head 2D Q/K/V views
    class TensorK2D_,
    class TensorV2D_>
struct DecodeMainloop {
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ2D = TensorQ2D_;
  using TensorK2D = TensorK2D_;
  using TensorV2D = TensorV2D_;

  using ElementQ = typename TensorQ2D::engine_type::value_type;
  using ElementK = typename TensorK2D::engine_type::value_type;

  using TiledCopyQ = decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{}));
  using TiledCopyK = decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{}));
  using TiledCopyV = decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{}));

  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool LocalMask = LocalMask_;

  struct Params {
    ElementS scale;  // softmax scale * log2(e)
    int const* ptr_page_table;
    int64_t page_table_batch_stride;
    int page_size;
    int page_stride_rows;  // physical rows between pages (>= page_size)
    int window_size_left;
  };

  struct SharedStorage {};

  Params params;

  CUTLASS_DEVICE
  DecodeMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  // Q_2D:   (q, d)  packed GQA rows of one KV head
  // K_2D:   (k, d)  one KV head; whole cache when paged, one sequence else
  // V_2D:   (v, k)
  // [blk_k0, blk_k1): logical KV tiles handled by this work-group.
  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,
      TensorK2D const& K_2D,
      TensorV2D const& V_2D,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int const& idx_b,
      int blk_k0,
      int blk_k1,
      int thr_id,
      int seq_len) {
    using namespace sycl::ext::oneapi::this_work_item;

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    int x_shift_q, x_shift_k, x_shift_v;
    auto Q_2D_al = align_block_2d_base<1>(Q_2D, x_shift_q);
    auto K_2D_al = align_block_2d_base<1>(K_2D, x_shift_k);
    auto V_2D_al = align_block_2d_base<0>(V_2D, x_shift_v);

    Tensor cQ = make_shifted_identity_tensor<1>(Q_2D.shape(), x_shift_q);
    Tensor cK = make_shifted_identity_tensor<1>(K_2D.shape(), x_shift_k);
    Tensor cV = make_shifted_identity_tensor<0>(V_2D.shape(), x_shift_v);
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));

    auto K_2D_0 = K_2D_al;
    auto V_2D_0 = V_2D_al;

    // The Xe 2D block-load descriptor stores the surface height in a 24-bit
    // field. A paged cache can have more rows than that, so split the K/V
    // surfaces into page-aligned sub-surfaces of at most 2^24 rows and route
    // each physical tile to the sub-surface containing it. Only paged KV can
    // get here; a non-paged surface is one sequence (checked on the host).
    constexpr int tile_k = get<1>(TileShapeQK{});
    [[maybe_unused]] int tiles_per_surface = 0x3fffffff;
    [[maybe_unused]] int rows_per_surface = 0;
    [[maybe_unused]] int k_rows_total = get<0>(K_2D.shape());
    [[maybe_unused]] int v_cols_total = get<1>(V_2D.shape());
    [[maybe_unused]] int64_t k_surf_shift = 0;
    [[maybe_unused]] int64_t v_surf_shift = 0;
    [[maybe_unused]] int k_pre_surf = -1;

    [[maybe_unused]] auto make_k_surface = [&](int surf) {
      int64_t rem = int64_t(k_rows_total) - int64_t(surf) * rows_per_surface;
      int rows = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (rows < 1) rows = 1;
      return make_tensor(
          K_2D_al.data() + int64_t(surf) * k_surf_shift,
          make_layout(
              make_shape(rows, get<1>(K_2D_al.shape())), K_2D_al.stride()));
    };
    [[maybe_unused]] auto make_v_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) cols = 1;
      return make_tensor(
          V_2D_al.data() + int64_t(surf) * v_surf_shift,
          make_layout(
              make_shape(get<0>(V_2D_al.shape()), cols), V_2D_al.stride()));
    };

    if constexpr (has_large_surface) {
      static_assert(PagedKV, "large surfaces are only expected for paged KV");
      constexpr int kXeMaxSurfaceRows = 1 << 24;
      int page_stride_tiles = params.page_stride_rows / tile_k;
      int blocks_per_surface = kXeMaxSurfaceRows / params.page_stride_rows;
      if (blocks_per_surface < 1) blocks_per_surface = 1;
      tiles_per_surface = blocks_per_surface * page_stride_tiles;
      rows_per_surface = tiles_per_surface * tile_k;

      int k_rows0 =
          k_rows_total < rows_per_surface ? k_rows_total : rows_per_surface;
      k_surf_shift = int64_t(rows_per_surface) * get<0>(K_2D.stride());
      int v_cols0 =
          v_cols_total < rows_per_surface ? v_cols_total : rows_per_surface;
      v_surf_shift = int64_t(rows_per_surface) * get<1>(V_2D.stride());

      K_2D_0 = make_tensor(
          K_2D_al.data(),
          make_layout(
              make_shape(k_rows0, get<1>(K_2D_al.shape())), K_2D_al.stride()));
      V_2D_0 = make_tensor(
          V_2D_al.data(),
          make_layout(
              make_shape(get<0>(V_2D_al.shape()), v_cols0), V_2D_al.stride()));

      cK = make_shifted_identity_tensor<1>(
          make_shape(rows_per_surface, get<1>(K_2D.shape())), x_shift_k);
      cV = make_shifted_identity_tensor<0>(
          make_shape(get<0>(V_2D.shape()), rows_per_surface), x_shift_v);
    }

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    TiledCopyQ copy_q{Q_2D_al};
    TiledCopyK copy_k{K_2D_0};
    TiledCopyV copy_v{V_2D_0};

    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);

    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // Logical KV tile -> physical tile of the K/V surfaces.
    [[maybe_unused]] int tiles_per_page = params.page_size / tile_k;
    [[maybe_unused]] int page_stride_tiles = params.page_stride_rows / tile_k;
    [[maybe_unused]] int64_t b_offset =
        int64_t(idx_b) * params.page_table_batch_stride;
    auto physical_tile = [&](int logical) {
      if constexpr (PagedKV) {
        int page_local_idx = logical / tiles_per_page;
        int block_idx = params.ptr_page_table[b_offset + page_local_idx];
        block_idx = block_idx < 0 ? 0 : block_idx;
        return block_idx * page_stride_tiles + logical % tiles_per_page;
      } else {
        return logical;
      }
    };

    int tile_idx = physical_tile(blk_k0);

    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    [[maybe_unused]] auto update_surface_info = [&](int& next_page_idx) {
      int k_surf = next_page_idx / tiles_per_surface;
      next_page_idx = next_page_idx - k_surf * tiles_per_surface;

      if (k_pre_surf != k_surf) {
        k_pre_surf = k_surf;

        auto K_2D_n = make_k_surface(k_surf);
        auto V_2D_n = make_v_surface(k_surf);

        copy_k = TiledCopyK{K_2D_n};
        copy_v = TiledCopyV{V_2D_n};

        auto thr_copy_k_n = copy_k.get_slice(thr_id);
        auto thr_copy_v_n = copy_v.get_slice(thr_id);

        prefetch_k = make_block_2d_prefetch(copy_k);
        prefetch_v =
            make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_n);

        Tensor cK_n = make_shifted_identity_tensor<1>(
            make_shape(get<0>(K_2D_n.shape()), get<1>(K_2D.shape())),
            x_shift_k);
        Tensor cV_n = make_shifted_identity_tensor<0>(
            make_shape(get<0>(V_2D.shape()), get<1>(V_2D_n.shape())),
            x_shift_v);

        Tensor gK_n = local_tile(
            cK_n, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor gV_n =
            local_tile(cV_n, tile_shape_v, make_coord(get<1>(blk_qv), _));
        Tensor gV_split_n = local_tile(
            gV_n, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});

        tKgK = thr_copy_k_n.partition_S(gK_n);
        tVgV = thr_copy_v_n.partition_S(gV_split_n);

        pKgK = prefetch_k.get_slice(thr_id).partition_S(gK_n);
        pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_n);
      }
    };

    if constexpr (has_large_surface) {
      update_surface_info(tile_idx);
    }

    for (int D = 0; D < size<4>(pKgK); D++) {
      prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % tile_k != 0);

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      auto tKgK_cache = tKgK(_, _, _, tile_idx, _);
      auto tVgV_cache = tVgV(_, _, _, _, tile_idx);

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);

        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, tile_idx));

      /* Sliding window: the query sits at KV position seq_len - 1. */
      if constexpr (LocalMask) {
        int first_col = seq_len - 1 - params.window_size_left;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          if (get<1>(cS_thread(i)) < first_col) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for the remainder tile */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        // Logical KV position: the coordinate tensors are indexed by the
        // logical tile K, independent of the physical page.
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      softmax(params.scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      wg_barrier();

      // Only look up tiles this work-group will process: page-table entries
      // past the sequence may be padding (e.g. -1).
      if (K + 1 < blk_k1) {
        tile_idx = physical_tile(K + 1);
        if constexpr (has_large_surface) {
          update_surface_info(tile_idx);
        }
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
        }
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void softmax(
      ElementS scale,
      bool first_block,
      FragS& tS,
      FragSRow& tS_max,
      FragSRow& tS_sum,
      FragA& tA) {
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), scale * tS_bmax(i));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(scale * tS(i) - broadcast<0>(tS_max, tS, i));

    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

}  // namespace flash_xpu_splitkv
