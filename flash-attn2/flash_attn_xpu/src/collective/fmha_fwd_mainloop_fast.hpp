/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * "Fast" FMHA forward mainloop -- minimal functionality matching the
 * torch-xpu-ops SDPA flash-attention backend.  Supports:
 *   - Forward pass only
 *   - Optional causal masking
 *   - Contiguous (non-paged, non-varlen) Q/K/V
 *   - No dropout, no local (sliding-window) mask
 *
 * This kernel is forked from the full FA2 mainloop in order to keep the
 * compiled binary small enough to avoid IGC register spill on BMG.
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "./fmha_fusion.hpp"

namespace cutlass::fmha {

// Dispatch tag for the fast path. Distinct from XeDefault to avoid
// conflicting partial specialization with the full-featured mainloop.
template <int Stages>
class XeFast {};

}  // namespace cutlass::fmha

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,
    class TiledCopyK_ = void,
    class TiledCopyV_ = void>
struct FMHAFwdMainloopFast {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a fast mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_>
struct FMHAFwdMainloopFast<
    XeFast<Stages>,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  //
  // Accumulator types
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
  };

  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloopFast(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params to_underlying_arguments(
      Arguments const& args,
      void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{val};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,
      int thr_id,
      int seq_len,
      int seq_len_qo,
      int seq_len_kv,
      int l_coord,
      int& tile_row_idx,
      const int& rows_of_maxima) {
    using namespace sycl::ext::oneapi::this_work_item;

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor cK = make_identity_tensor(K_2D.shape());
    Tensor cV = make_identity_tensor(V_2D.shape());
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));

    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});
    Tensor gK = local_tile(
        cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));
    Tensor gV_split = local_tile(
        gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});

    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tKgK = thr_copy_k.partition_S(gK);
    auto tVgV = thr_copy_v.partition_S(gV_split);

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
    auto prefetch_v = make_block_2d_prefetch(copy_v);

    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);

    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }
    int prefetch_k_stages = (total_blk < Stages ? total_blk : Stages);
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = blk_k0; K < blk_k0 + prefetch_k_stages; K++) {
        prefetch(prefetch_k, pKgK(_, _, _, K, D));
      }
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    Tensor cPgP = make_identity_tensor(make_shape(seq_len_qo, seq_len_kv));
    Tensor gP_all = local_tile(
        cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), _));

    for (int K = blk_k0; K < blk_k1; K++) {
      /* GEMM 1: S = Q * K^T */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK(_, _, _, K, D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v, pVgV(_, _, _, VV, K));
      }

      auto cS_thread = thr_mma_qk.partition_C(gP_all(_, _, K));

      if (check_remainder_k && K == blk_k1 - 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i));
          if (col_idx >= seq_len_kv) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      if constexpr (CausalMask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int row_idx = get<0>(cS_thread(i));
          int col_idx = get<1>(cS_thread(i));

          if (seq_len_qo == seq_len_kv) {
            if (col_idx > row_idx) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
          if (seq_len_kv > seq_len_qo) {
            int first_masked_col_index = seq_len_kv - (seq_len_qo - 1) + row_idx;
            if (col_idx >= first_masked_col_index) {
              tSrS(i) = ElementS{-INFINITY};
            }
          }
          if (seq_len_qo > seq_len_kv) {
            int first_non_masked_sequence = seq_len_qo - seq_len_kv;
            if (row_idx < first_non_masked_sequence ||
                col_idx > row_idx - first_non_masked_sequence) {
              tSrS(i) = ElementS{-INFINITY};
            }
          }
        }
      }

      softmax(K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV(_, _, _, VV, K), tVrV);
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      int K_next = K + Stages;
      if (K_next < blk_k1) {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, K_next, D));
        }
      }
    }

    get_LSE_metadata(
        thr_id, TileShapePV{}, thr_mma_pv, rows_of_maxima, tile_row_idx);
  }

  template <class Shape, class ThrMMA>
  CUTLASS_DEVICE void get_LSE_metadata(
      const int& thr_id,
      const Shape& tile_shape_PV,
      const ThrMMA& thr_mma_pv,
      const int& rows_of_maxima,
      int& tile_row_idx) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int lane_id = static_cast<int>(sg.get_local_linear_id());
    auto coord_tensor = make_identity_tensor(tile_shape_PV);
    auto thr_mma = thr_mma_pv.get_slice(thr_id);
    auto tC_coords = thr_mma.partition_C(coord_tensor);

    tile_row_idx = -1;
    if (lane_id < rows_of_maxima) {
      auto coord = tC_coords(lane_id);
      tile_row_idx = get<0>(coord);
    }
  }

  CUTLASS_DEVICE
  void softmax(
      bool first_block,
      FragS& tS,
      FragSRow& tS_max,
      FragSRow& tS_sum,
      FragA& tA) {
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(
          params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    if (!first_block) {
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

}  // namespace cutlass::fmha::collective
