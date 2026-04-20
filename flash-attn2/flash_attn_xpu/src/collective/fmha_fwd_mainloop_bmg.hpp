/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * BMG-path FMHA forward mainloop -- minimal functionality matching the
 * torch-xpu-ops SDPA flash-attention backend.  Supports:
 *   - Forward pass only
 *   - Optional causal masking
 *   - Contiguous (non-paged, non-varlen) Q/K/V
 *   - No dropout, no local (sliding-window) mask
 *
 * This kernel is forked from the full FA2 mainloop in order to keep the
 * compiled binary small enough to avoid IGC register spill on BMG.
 *
 * Common type aliases live in fmha_fwd_common.hpp (FMHAFwdMainloopTraits).
 **************************************************************************************************/

#pragma once

#include "./fmha_fwd_common.hpp"

namespace cutlass::fmha {

// Dispatch tag for the BMG path. Distinct from XeDefault to avoid
// conflicting partial specialization with the full-featured mainloop.
template <int Stages>
class XeBmg {};

}  // namespace cutlass::fmha

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,
    class TiledCopyK_ = void,
    class TiledCopyV_ = void>
struct FMHAFwdMainloopBmg {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a BMG mainloop specialization.");
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
struct FMHAFwdMainloopBmg<
    XeBmg<Stages>,
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

  // Pull in common type aliases from the shared traits.
  using Traits = FMHAFwdMainloopTraits<
      TiledMMAQK_, TiledMMAPV_, VTiles_,
      TensorQ_, TensorK_, TensorV_,
      TiledCopyQ_, TiledCopyK_, TiledCopyV_>;

  using TiledMMAQK = typename Traits::TiledMMAQK;
  using TiledMMAPV = typename Traits::TiledMMAPV;
  using TileShapeQK = typename Traits::TileShapeQK;
  using TileShapePV = typename Traits::TileShapePV;
  static constexpr int VTiles = Traits::VTiles;
  using SubgroupLayoutQK = typename Traits::SubgroupLayoutQK;
  using SGPerWG = typename Traits::SGPerWG;

  using TensorQ = typename Traits::TensorQ;
  using TensorK = typename Traits::TensorK;
  using TensorV = typename Traits::TensorV;
  using TensorQ2D = typename Traits::TensorQ2D;
  using TensorK2D = typename Traits::TensorK2D;
  using TensorV2D = typename Traits::TensorV2D;
  using TiledCopyQ = typename Traits::TiledCopyQ;
  using TiledCopyK = typename Traits::TiledCopyK;
  using TiledCopyV = typename Traits::TiledCopyV;

  using FragS = typename Traits::FragS;
  using FragSRow = typename Traits::FragSRow;
  using ElementS = typename Traits::ElementS;
  using SingleFragA = typename Traits::SingleFragA;
  using FragA = typename Traits::FragA;
  using FragARow = typename Traits::FragARow;
  using ElementA = typename Traits::ElementA;

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

  FMHAFwdMainloopBmg(Params const& params_, SharedStorage&) : params(params_) {}

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
      TensorQ2D const& Q_2D,
      TensorK2D const& K_2D,
      TensorV2D const& V_2D,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int blk_k0,
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

    // Use shared get_LSE_metadata from fmha_fwd_common.hpp
    cutlass::fmha::collective::get_LSE_metadata(
        thr_id, TileShapePV{}, mma_pv, rows_of_maxima, tile_row_idx);
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
