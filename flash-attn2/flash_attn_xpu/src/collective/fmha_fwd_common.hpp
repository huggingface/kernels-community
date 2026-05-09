/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Common type traits and device functions shared by the Xe2
 * mainloop/epilogue.
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "./fmha_fusion.hpp"
#include "./copy_block_slm.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// FMHAFwdMainloopTraits: common type aliases derived from TiledMMA / VTiles.
// Used by FMHAFwdMainloopXe2 (Xe2 / BMG path).
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,
    class TiledCopyK_ = void,
    class TiledCopyV_ = void>
struct FMHAFwdMainloopTraits {
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

  // Accumulator fragment types
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// get_LSE_metadata: computes per-lane tile_row_idx for LSE storage.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class TileShapePV, class ThrMMA>
CUTLASS_DEVICE void get_LSE_metadata(
    const int& thr_id,
    const TileShapePV& tile_shape_PV,
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

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// get_sg_layout_pv: derive the PV subgroup layout from the QK one.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(
      get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// FMHAFwdEpilogueTraits: common type aliases and the ReduceK>1 reduce_A
// body used by FMHAFwdEpilogueXe2.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class CollectiveMainloop, class TileShapeO_, class TensorO_, class TiledCopyO_ = void>
struct FMHAFwdEpilogueTraits {
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D =
      decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<
        (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg)
                                  : v_total_sg>{};
  }

  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout =
      decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));
  using SGTileShapeO =
      decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
      make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(
          TiledMMAPV{},
          ReduceFragA{}.tv_layout(),
          ReduceSGLayout{},
          TensorO2D{});
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO =
      conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  using AlignedSGTileA_Q =
      C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) *
        intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data,
        a_max_data;
  };

  using SharedStorage = conditional_t<
      (ReduceK{} > _1{}),
      SharedStorageReduceK,
      SharedStorageNone>;

  //
  // reduce_A_impl: the ReduceK>1 body. Returns (rA, rA_sum, rA_max, active).
  // Both epilogues call this and then pick which tuple elements they need.
  //
  template <typename FragA_, typename FragARow_>
  CUTLASS_DEVICE static auto reduce_A_multi_k(
      FragA_& tArA,
      FragARow_& tA_max,
      FragARow_& tA_sum,
      int thr_id,
      SharedStorage& shared) {
    using namespace sycl::ext::oneapi::this_work_item;

    auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                       .get_flat_coord(assert_uniform(thr_id));
    auto a_tile = get<1>(thr_vak);
    auto k_blk = get<2>(thr_vak);

    auto shape_A =
        append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
    auto shape_A_row = make_shape(
        get<0>(SGTileShapeO{}),
        shape(ReduceSGLayout{}),
        ReduceK{},
        SGPerWG{} / ReduceK{});

    auto sA_layout = group<2, 4>(flat_divide(
        make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}),
        SGTileShapeO{}));
    auto sA_row_stride = make_stride(
        _1{},
        make_stride(get<0>(shape_A_row), _0{}),
        AlignedSGTileA_Q{},
        AlignedSGTileA_Q{} * ReduceK{});
    auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

    auto basis2 = make_basis_like(SGTileShapeO{});
    auto sA_coords = make_layout(
        append(SGTileShapeO{}, shape(ReduceSGLayout{})),
        append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

    auto sA = make_tensor(
        make_smem_ptr<ElementA>(&shared.a_data), sA_layout);
    auto sA_max = make_tensor(
        make_smem_ptr<ElementA>(&shared.a_max_data), sA_row_layout);
    auto sA_sum = make_tensor(
        make_smem_ptr<ElementA>(&shared.a_sum_data), sA_row_layout);

    copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
    barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
    copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
    copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

    bool active = (k_blk < size(ReduceSGLayout{})) ||
                  (ReduceK{} == size(ReduceSGLayout{}));

    barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
    barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

    ReduceFragA rA;
    ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

    if (active) {
      CUTLASS_PRAGMA_UNROLL
      for (int kr = 0; kr < ReduceK{}; kr++) {
        copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
      }

      rA_max = rA_kmax[0];
      for (int kr = 1; kr < ReduceK{}; kr++)
        cute::transform(rA_max, rA_kmax[kr], rA_max, cute::max_fn{});

      for (int kr = 0; kr < ReduceK{}; kr++) {
        cute::transform(
            rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) {
              return sycl::native::exp2(kmax - gmax);
            });
      }
    }

    barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

    if (active) {
      clear(rA_sum);

      CUTLASS_PRAGMA_UNROLL
      for (int kr = 0; kr < ReduceK{}; kr++) {
        ReduceFragARow rA_sum_read;
        copy_block_s2r(sA_sum(_, k_blk, kr, a_tile), rA_sum_read);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < rA_sum_read.size(); i++) {
          rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
        }
      }

      clear(rA);

      CUTLASS_PRAGMA_UNROLL
      for (int kr = 0; kr < ReduceK{}; kr++) {
        ReduceFragA rA_read;
        copy_block_s2r(
            sA(_, _, k_blk, kr, a_tile), sA_coords(_, _, 0), rA_read);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < rA_read.size(); i++) {
          rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
        }
      }
    }
    return std::make_tuple(rA, rA_sum, rA_max, active);
  }

  //
  // write_output: tile O and write the output tensor.
  //
  template <typename FragA_, typename QVCoord>
  CUTLASS_DEVICE static void write_output(
      TensorO2D const& O,
      FragA_& rA,
      QVCoord blk_qv,
      int thr_id) {
    Tensor cO = make_identity_tensor(O.shape());
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);

    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }
};

}  // namespace cutlass::fmha::collective
