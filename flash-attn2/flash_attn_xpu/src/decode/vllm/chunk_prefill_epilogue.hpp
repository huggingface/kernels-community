/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#pragma once

#include <sycl/sycl.hpp>
#include "../splitkv_decode.hpp"  // FLASH_XPU_VLLM_NS
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"

#include "../../collective/copy_block_slm.hpp"

namespace cutlass::fmha::collective {
// Arch-tagged inline namespace: gives these definitions a mangled name
// distinct from the other Xe architecture's identically named copies,
// while leaving name lookup (cutlass::fmha::...) unchanged.
inline namespace FLASH_XPU_VLLM_NS {

using namespace cute;

template <
    bool Sink_,
    class CollectiveMainloop,  // Attention mainloop
    class TileShapeO_,         // Shape of output tile, may be larger than P*V
                               // GEMM
    class TensorO_,            // 2D slice of global output tensor
    class TiledCopyO_ = void>  // Optional TiledCopy for loading O
class FMHAFwdEpilogue {
 public:
  //
  // Type Aliases
  //
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

  // softmax sink, same dtype
  static constexpr bool Sink = Sink_;
  using ElementSink = typename CollectiveMainloop::TensorQ::element_type;

  // Split k-reduced tiles between participating subgroups.
  // Assumption: the A tile is contiguous.
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<
        (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg)
                                  : v_total_sg>{};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout =
      decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

  using SGTileShapeO =
      decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
      make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_base_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(
          TiledMMAPV{},
          ReduceFragA{}.tv_layout(),
          ReduceSGLayout{},
          TensorO2D{});
  }

  static auto default_tiled_copy_O_helper() {
    auto default_copy = default_tiled_copy_O_base_helper();
    // For 16-bit outputs, pack two values into each d32 block2D(d32.16x8nn).
    // This increases bytes per message and avoids the slower d16
    // store(d16.16x8nn).
    using DefaultStoreOp = typename decltype(default_copy)::CopyOp;
    if constexpr (
        sizeof_bits_v<ElementO> == 16 && DefaultStoreOp::AtomWidth <= 16) {
      static_assert(
          DefaultStoreOp::CopyBits == 16 && DefaultStoreOp::AtomWidth % 2 == 0);
      using D32StoreOp = XE_STORE_2D<
          32,
          DefaultStoreOp::AtomHeight,
          DefaultStoreOp::AtomWidth>;
      if constexpr (ReduceK{} == _1{}) {
        return make_block_2d_copy_CD(D32StoreOp{}, TiledMMAPV{}, TensorO2D{});
      } else {
        return make_block_2d_copy_CD_subtiled<ElementO>(
                   D32StoreOp{},
                   TiledMMAPV{},
                   atuple_coshape(ReduceFragA{}.tv_layout()),
                   ReduceSGLayout{},
                   TensorO2D{}.stride())
            .with(TensorO2D{});
      }
    } else {
      return default_copy;
    }
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO =
      conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  // Stateless design -- no arguments or parameters.
  struct Arguments {};
  struct Params {};

  // Shared memory storage
  // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe
  // block load infrastructure.
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

 private:
  SharedStorage& shared;

 public:
  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FMHAFwdEpilogue(Params const&, SharedStorage& shared_) : shared(shared_) {}

  template <bool SumIsReduced = false, typename QVCoord, typename FragSPRow>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,        // Global O tensor: (q,v)
      FragA& tArA,               // O accumulator:   (q,v)
      FragARow& tA_max,          // Softmax row-wise max accumulator
      FragSPRow& tA_sum,         // Softmax row-wise partial sum (per-lane)
      QVCoord blk_qv,            // WG tile indices: (q,v)
      ElementSink const& tSink,  // Sink for current head
      int thr_id) {              // Work-item ID

    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Collapse the deferred per-lane partial row sums into the full row sum,
    // unless the caller already performed the horizontal reduction.
    auto tA_sum_full =
        reduce<0, ReduceMode::Horizontal>(tA_sum, sycl::plus<void>{});

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum_full, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    /* Complete softmax, dividing out sums. */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++) {
      if constexpr (Sink) {
        constexpr double kLog2e = 1.4426950408889634074;
        rA_sum(i) += sycl::native::exp2(
            static_cast<ElementA>(tSink * kLog2e) - tA_max(i));
      }
      // Fully-masked rows have a zero row sum; write 0 instead of 1/0 == inf
      // so O == 0 instead of NaN (mirrors DecodeFwdEpilogue).
      rA_sum(i) =
          rA_sum(i) > ElementA(0) ? ElementA(1) / rA_sum(i) : ElementA(0);
    }

    // CUTLASS_PRAGMA_UNROLL
    // for (int i = 0; i < rA.size(); i++)
    //   rA(i) *= broadcast<0>(rA_sum, rA, i);

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape());       // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);
    cute::reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  // Note that each k block has its own scale factor based on A_max,
  //   so A/A_sum contributions need to be rescaled to match.
  template <typename FragA, typename FragARow, typename FragSPRow>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA& tArA,        // O accumulator:   (q,v)
      FragARow& tA_max,   // Softmax row-wise max accumulator
      FragSPRow& tA_sum,  // Softmax row-wise sum accumulator (reduced)
      int thr_id) {       // Work-item ID

    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, true);
    } else {
      /* Identify A tile ID and k block for this subgroup. */
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      /* Set up SLM tensors and partition A tiles among participating subgroups
       */
      auto shape_A =
          append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
      auto shape_A_row = make_shape(
          get<0>(SGTileShapeO{}),
          shape(ReduceSGLayout{}),
          ReduceK{},
          SGPerWG{} / ReduceK{});

      /* Physical layouts, with subtile modes broken out */
      auto sA_layout = group<2, 4>(flat_divide(
          make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}),
          SGTileShapeO{}));
      auto sA_row_stride = make_stride(
          _1{},
          make_stride(get<0>(shape_A_row), _0{}),
          AlignedSGTileA_Q{},
          AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      /* Coordinate layouts, with subtile modes broken out */
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(
          append(SGTileShapeO{}, shape(ReduceSGLayout{})),
          append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_data),
          sA_layout);  // (q,v,rblk_dst,rblk_src,a_tile)
      auto sA_max = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_max_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
      auto sA_sum = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_sum_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) ||
                    (ReduceK{} == size(ReduceSGLayout{}));  // help compiler out

      /* Wait for maxima to be available, signal other data available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        /* Read A_max back from SLM and reduce. */
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); i++) {
            rA_max(i) = cute::max(rA_max(i), rA_kmax[kr](i));
          }
        }

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); i++) {
            rA_kmax[kr](i) = sycl::native::exp2(rA_kmax[kr](i) - rA_max(i));
          }
        }
      }

      /* Wait for A/A_sum data to be available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce.
         */
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
      return std::make_tuple(rA, rA_sum, active);
    }
  }
};

template <
    class CollectiveMainloop,  // Attention mainloop
    class TileShapeO_,  // Shape of output tile, may be larger than P*V GEMM
    class TensorO_,     // 2D slice of global output tensor
    class TensorLSE_ = void,   // Optional tensor for storing intermediate exp
                               // sums and max logits
    class TiledCopyO_ = void,  // Optional TiledCopy for loading O
    bool Sink_ = false>        // Whether to sink softmax into epilogue
class DecodeFwdEpilogue {
 public:
  //
  // Type Aliases
  //
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D =
      decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  using TensorLSE = TensorLSE_;
  using TensorLSE2D = conditional_t<
      is_void_v<TensorLSE_>,
      void,
      decltype(TensorLSE_{}(append<rank_v<TensorLSE_>>(make_coord(_, _), 0)))>;
  using ElementLSE = conditional_t<
      is_void_v<TensorLSE_>,
      void,
      typename TensorLSE_::value_type>;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

  // softmax sink, same dtype
  static constexpr bool Sink = Sink_;
  using ElementSink = typename CollectiveMainloop::TensorQ::element_type;
  // Full-fp8: the mainloop biases the softmax exponent by +8 (see softmax()),
  // scaling both the P*V numerator and the row-sum denominator by 2^8. The sink
  // term is added to the denominator here, so it must carry the same bias.
  static constexpr bool Fp8Q =
      cute::is_same_v<ElementSink, cutlass::float_e5m2_t> ||
      cute::is_same_v<ElementSink, cutlass::float_e4m3_t>;
  static constexpr int kFp8SinkOffset = Fp8Q ? 8 : 0;

  // Split k-reduced tiles between participating subgroups.
  // Assumption: the A tile is contiguous.
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<
        (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg)
                                  : v_total_sg>{};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
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

  // Stateless design -- no arguments or parameters.
  struct Arguments {};
  struct Params {};

  // Shared memory storage
  // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe
  // block load infrastructure.
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

 private:
  SharedStorage& shared;

 public:
  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  DecodeFwdEpilogue(Params const&, SharedStorage& shared_) : shared(shared_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,  // Global O tensor: (q,v)
      FragA& tArA,         // O accumulator:   (q,v)
      FragARow& tA_max,    // Softmax row-wise max accumulator
      FragARow& tA_sum,    // Softmax row-wise sum accumulator
      QVCoord blk_qv,      // WG tile indices: (q,v)
      int thr_id) {        // Work-item ID

    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_max_unused, rA_sum, active] =
        reduce_A(tArA, tA_max, tA_sum, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    /* Complete softmax, dividing out sums. */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      // Fully-masked rows have a zero row sum; write 0 instead of 1/0 == inf
      // so O == 0 instead of NaN (mirrors DecodeFwdEpilogue).
      rA_sum(i) =
          rA_sum(i) > ElementA(0) ? ElementA(1) / rA_sum(i) : ElementA(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape());       // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // splitK version
  template <typename QVCoord, class TensorSink>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,                    // Global O tensor: (q,v)
      FragA& tArA,                           // O accumulator:   (q,v)
      FragARow& tA_max,                      // Softmax row-wise max accumulator
      FragARow& tA_sum,                      // Softmax row-wise sum accumulator
      QVCoord blk_qv,                        // WG tile indices: (q,v)
      int thr_id,                            // Work-item ID
      const TensorLSE2D& softmax_lse_accum,  // Per-split natural-log LSE
      int idx_kv_split,
      int head_group_q,
      TensorSink& tSink,  // Sink for current head
      int num_kv_splits,
      ElementLSE* ptr_lse = nullptr,  // softmax_lse row base (null = disabled)
      int lse_stride = 0) {           // stride between packed GQA rows
    using namespace cute;
    using ElementA = typename FragA::element_type;
    constexpr float kLn2 = 0.6931471805599453f;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    // Decode tiles head_group_q across the grid's Q dimension (blk_qv[0]).
    // Within the tile, subgroups are laid out (a_tile, k_blk): a_tile picks
    // a block of sg_rows packed-Q rows, k_blk a KV slice. Row statistics of
    // a_tile are owned by its k_blk == 0 subgroup, row = lane (both before
    // and after reduce_A). With one a_tile this is thr_id < q_tile_rows.
    constexpr int q_tile_rows = cute::size<0>(TileShapeO{});
    constexpr int sg_rows = cute::size<0>(SGTileShapeA{});
    auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                       .get_flat_coord(assert_uniform(thr_id));
    const int a_tile = get<1>(thr_vak);
    const int k_blk = get<2>(thr_vak);
    const int lane = thr_id % intel::sg_size;
    int q_row = get<0>(blk_qv) * q_tile_rows + a_tile * sg_rows + lane;
    bool row_valid = (k_blk == 0) && (lane < sg_rows) &&
                     (a_tile * sg_rows + lane < q_tile_rows) &&
                     (q_row < head_group_q);
    if constexpr (Sink) {
      constexpr double kLog2e = 1.4426950408889634074;
      if (idx_kv_split == 0 && row_valid) {
        tA_sum(0) += sycl::native::exp2(
            static_cast<ElementA>(tSink(q_row) * kLog2e) - tA_max(0) +
            ElementA(kFp8SinkOffset));
      }
    }

    auto [rA, rA_max, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    // Store one natural-log LSE for this KV split. ReduceSplitK uses it exactly
    // like CUDA FlashAttention: weight_i = exp(LSE_i - logsumexp(LSE)).
    // The sink (split 0 only) is already folded into tA_sum above, so rA_sum(0)
    // is the full softmax denominator for q_row. rA_max is in log2 units.
    // Full fp8 scales the row sum by 2^kFp8SinkOffset (the mainloop's +8
    // exponent bias); remove it so the reported LSE is unbiased.
    // Assume seq_len_qo == 1.
    if (row_valid && (num_kv_splits > 1 || ptr_lse != nullptr)) {
      ElementLSE row_lse = static_cast<ElementLSE>(
          (static_cast<float>(rA_max(0)) - float(kFp8SinkOffset)) * kLn2 +
          sycl::log(static_cast<float>(rA_sum(0))));
      if (num_kv_splits > 1) {
        softmax_lse_accum(q_row, idx_kv_split) = row_lse;
      }

      // Without a ReduceSplitK pass this work-item holds the complete
      // softmax statistics for its query row, so write softmax_lse here.
      if (ptr_lse != nullptr) {
        ptr_lse[q_row * lse_stride] = row_lse;
      }
    }

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active) return;

    // Normalize every split independently. ReduceSplitK combines normalized
    // partial outputs with exp(LSE_i - LSE_global), matching CUDA
    // FlashAttention's split-K contract. A split whose keys are all masked
    // (e.g. packed spec decode) has a zero row sum; write O = 0 instead of NaN.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++) {
      rA_sum(i) =
          rA_sum(i) > ElementA(0) ? ElementA(1) / rA_sum(i) : ElementA(0);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      rA(i) *= broadcast<0>(rA_sum, rA, i);
    }

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape());       // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);  // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  // Note that each k block has its own scale factor based on A_max,
  //   so A/A_sum contributions need to be rescaled to match.
  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA& tArA,       // O accumulator:   (q,v)
      FragARow& tA_max,  // Softmax row-wise max accumulator
      FragARow& tA_sum,  // Softmax row-wise sum accumulator
      int thr_id) {      // Work-item ID

    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_max, tA_sum, true);
    } else {
      /* Identify A tile ID and k block for this subgroup. */
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      /* Set up SLM tensors and partition A tiles among participating subgroups
       */
      auto shape_A =
          append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
      auto shape_A_row = make_shape(
          get<0>(SGTileShapeO{}),
          shape(ReduceSGLayout{}),
          ReduceK{},
          SGPerWG{} / ReduceK{});

      /* Physical layouts, with subtile modes broken out */
      auto sA_layout = group<2, 4>(flat_divide(
          make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}),
          SGTileShapeO{}));
      auto sA_row_stride = make_stride(
          _1{},
          make_stride(get<0>(shape_A_row), _0{}),
          AlignedSGTileA_Q{},
          AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      /* Coordinate layouts, with subtile modes broken out */
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(
          append(SGTileShapeO{}, shape(ReduceSGLayout{})),
          append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_data),
          sA_layout);  // (q,v,rblk_dst,rblk_src,a_tile)
      auto sA_max = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_max_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)
      auto sA_sum = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_sum_data),
          sA_row_layout);  // (q,rblk_dst,rblk_src,a_tile)

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) ||
                    (ReduceK{} == size(ReduceSGLayout{}));  // help compiler out

      /* Wait for maxima to be available, signal other data available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        /* Read A_max back from SLM and reduce. */
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); i++) {
            rA_max(i) = cute::max(rA_max(i), rA_kmax[kr](i));
          }
        }

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_max.size(); i++) {
            rA_kmax[kr](i) = sycl::native::exp2(rA_kmax[kr](i) - rA_max(i));
          }
        }
      }

      /* Wait for A/A_sum data to be available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce.
         */
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
      return std::make_tuple(rA, rA_max, rA_sum, active);
    }
  }
};

}  // namespace vllm_xpu_xe3

}  // namespace cutlass::fmha::collective
