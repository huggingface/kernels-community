/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * BMG-path FMHA forward epilogue -- minimal functionality matching the
 * torch-xpu-ops SDPA flash-attention backend epilogue.
 *
 * Common type aliases and the ReduceK>1 reduce body live in
 * fmha_fwd_common.hpp (FMHAFwdEpilogueTraits).
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include "./fmha_fwd_common.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template <
    class CollectiveMainloop,
    class TileShapeO_,
    class TensorO_,
    class TiledCopyO_ = void>
class FMHAFwdEpilogueBmg {
 public:
  using ETraits = FMHAFwdEpilogueTraits<
      CollectiveMainloop, TileShapeO_, TensorO_, TiledCopyO_>;

  // Re-export types from traits for external consumers.
  using TiledMMAPV   = typename ETraits::TiledMMAPV;
  using TileShapePV  = typename ETraits::TileShapePV;
  using TileShapeO   = typename ETraits::TileShapeO;
  using SGPerWG      = typename ETraits::SGPerWG;
  using TensorO      = typename ETraits::TensorO;
  using TensorO2D    = typename ETraits::TensorO2D;
  using ElementO     = typename ETraits::ElementO;
  using FragA        = typename ETraits::FragA;
  using FragARow     = typename ETraits::FragARow;
  using ElementA     = typename ETraits::ElementA;
  using ReduceK      = typename ETraits::ReduceK;
  using TiledCopyO   = typename ETraits::TiledCopyO;
  using SharedStorage = typename ETraits::SharedStorage;

  // Stateless design -- no arguments or parameters.
  struct Arguments {};
  struct Params {};

 private:
  SharedStorage& shared;

 public:
  static constexpr Params to_underlying_arguments(
      Arguments const&, void*) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FMHAFwdEpilogueBmg(Params const&, SharedStorage& shared_) : shared(shared_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int thr_id,
      float* pLSE,
      const std::tuple<int, int, int, int, int, int, int, int>& metadata_for_lse) {
    using namespace cute;
    using ElementA_ = typename FragA::element_type;

    auto [rA, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    if (!active)
      return;

    auto non_reciprocal_rAsum = rA_sum(0);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      rA_sum(i) = ElementA_(1) / rA_sum(i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      rA(i) *= broadcast<0>(rA_sum, rA, i);
      if (std::isnan(rA(i))) {
        rA(i) = 0;
      }
    }

    // Write output using shared helper.
    ETraits::write_output(O, rA, blk_qv, thr_id);

    /* Calculate the LSE */
    if (pLSE == nullptr)
      return;

    auto
        [blk_q,
         lse_stride_head,
         lse_stride_batch,
         seq_len_qo,
         batch_idx,
         q_head_idx,
         tile_row_idx,
         rows_of_maxima] = metadata_for_lse;
    int blk_q_coord = get<0>(blk_qv);
    size_t lse_offset =
        static_cast<size_t>(batch_idx) * lse_stride_batch +
        static_cast<size_t>(q_head_idx) * lse_stride_head +
        static_cast<size_t>(blk_q_coord) * blk_q;
    size_t seq_coord =
        static_cast<size_t>(blk_q_coord) * blk_q + tile_row_idx;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    int lane_id = static_cast<int>(sg.get_local_linear_id());

    if (tile_row_idx != -1 && seq_coord < static_cast<size_t>(seq_len_qo) &&
        (tile_row_idx % rows_of_maxima) == lane_id) {
      constexpr double kLog2e = 1.4426950408889634074;
      float max_val = static_cast<float>(tA_max[0]) / static_cast<float>(kLog2e);
      float lse_val = max_val + logf(non_reciprocal_rAsum);
      pLSE[lse_offset + tile_row_idx] = lse_val == -INFINITY ? 0 : lse_val;
    }
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  template <typename FragA_, typename FragARow_>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA_& tArA,
      FragARow_& tA_max,
      FragARow_& tA_sum,
      int thr_id) {
    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, true);
    } else {
      auto [rA, rA_sum, rA_max, active] =
          ETraits::reduce_A_multi_k(tArA, tA_max, tA_sum, thr_id, shared);
      // BMG epilogue doesn't need rA_max; drop it.
      return std::make_tuple(rA, rA_sum, active);
    }
  }
};

}  // namespace cutlass::fmha::collective
