/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Xe2 (BMG / Arc Pro B60) FMHA forward mainloop. Supports the full FA2
 * feature set:
 *   - Forward pass only
 *   - Optional causal masking
 *   - Optional local (sliding-window) mask
 *   - Optional dropout
 *   - Optional paged KV cache
 *   - Contiguous or variable-length Q/K/V
 *
 * Common type aliases live in fmha_fwd_common.hpp (FMHAFwdMainloopTraits).
 **************************************************************************************************/

#pragma once

#include "../philox.hpp"
#include "./fmha_fwd_common.hpp"
#include "cutlass/gemm/collective/xe_common_blockscaled_mxfp.hpp"

namespace cutlass::fmha {

// Dispatch tag for the Xe2 path (BMG / Arc Pro B60).
template <int Stages>
class Xe2 {};

}  // namespace cutlass::fmha

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool LocalMask_,
    bool HasDropout_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,
    class TiledCopyK_ = void,
    class TiledCopyV_ = void,
    bool HasRotary_ = false,
    bool UseScale_ = false,       // MXFP: block-scaled MMA
    bool F8kvF16mma_ = false,     // MXFP: F8 KV with F16 MMA (scalar scale)
    class TensorScaleQ_ = void,   // MXFP: optional scale tensors
    class TensorScaleK_ = void,
    class TensorScaleV_ = void>
struct FMHAFwdMainloopXe2 {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find an Xe2 mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    bool LocalMask_,
    bool HasDropout_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    bool HasRotary_,
    bool UseScale_,
    bool F8kvF16mma_,
    class TensorScaleQ_,
    class TensorScaleK_,
    class TensorScaleV_>
struct FMHAFwdMainloopXe2<
    Xe2<Stages>,
    CausalMask_,
    LocalMask_,
    HasDropout_,
    PagedKV_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    HasRotary_,
    UseScale_,
    F8kvF16mma_,
    TensorScaleQ_,
    TensorScaleK_,
    TensorScaleV_> {

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
  static constexpr bool LocalMask  = LocalMask_;
  static constexpr bool HasDropout = HasDropout_;
  static constexpr bool PagedKV   = PagedKV_;
  static constexpr bool HasRotary = HasRotary_;

  // --- MXFP (block-scaled) support ---------------------------------------
  using ElementQ = typename TensorQ::element_type;
  static constexpr bool FP4Input =
      cute::is_same_v<ElementQ, cutlass::float_e2m1_t>;
  static constexpr bool UseScale = UseScale_;
  static constexpr bool F8kvF16mma = F8kvF16mma_;

  using TensorScaleQ = TensorScaleQ_;
  using TensorScaleK = TensorScaleK_;
  using TensorScaleV = TensorScaleV_;
  using TensorScaleQ2D = conditional_t<is_void_v<TensorScaleQ_>, void,
      decltype(TensorScaleQ_{}(append<rank_v<TensorScaleQ_>>(make_coord(_, _), 0)))>;
  using TensorScaleK2D = conditional_t<is_void_v<TensorScaleK_>, void,
      decltype(TensorScaleK_{}(append<rank_v<TensorScaleK_>>(make_coord(_, _), 0)))>;
  using TensorScaleV2D = conditional_t<is_void_v<TensorScaleV_>, void,
      decltype(TensorScaleV_{}(append<rank_v<TensorScaleV_>>(make_coord(_, _), 0)))>;

  using ElementScaleQ = conditional_t<is_void_v<TensorScaleQ_>, float,
      typename TensorScaleQ::element_type>;
  using ElementScaleK = conditional_t<is_void_v<TensorScaleK_>, float,
      typename TensorScaleK::element_type>;
  using ElementScaleV = conditional_t<is_void_v<TensorScaleV_>, float,
      typename TensorScaleV::element_type>;

  // Block and MMA dimensions used for block-scale addressing.
  static constexpr int BLK_Q = get<0>(TileShapeQK{});
  static constexpr int BLK_K = get<1>(TileShapeQK{});
  static constexpr int BLK_QK_D = get<2>(TileShapeQK{});
  static constexpr int BLK_P = get<0>(TileShapePV{});
  static constexpr int BLK_V = get<1>(TileShapePV{});
  static constexpr int BLK_PV_D = get<2>(TileShapePV{});

  static constexpr int ATOM_Q = get<1>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_K = get<2>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_QK_D = get<3>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());
  static constexpr int MMA_QK_D = get<2>(typename TiledMMAQK::Shape_MNK{});
  static constexpr int ATOM_P = get<1>(typename TiledMMAPV::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_V = get<2>(typename TiledMMAPV::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_PV_D = get<3>(typename TiledMMAPV::ThrLayoutVMNK{}.shape());
  static constexpr int MMA_PV_D = get<2>(typename TiledMMAPV::Shape_MNK{});

  static constexpr int SG_Q = ceil_div(BLK_Q, ATOM_Q);
  static constexpr int SG_K = ceil_div(BLK_K, ATOM_K);
  static constexpr int SG_QK_D = ceil_div(BLK_QK_D, ATOM_QK_D);
  static constexpr int SG_P = ceil_div(BLK_P, ATOM_P);
  static constexpr int SG_V = ceil_div(BLK_V, ATOM_V);
  static constexpr int SG_PV_D = ceil_div(BLK_PV_D, ATOM_PV_D);

  static constexpr auto GROUP_K = 32;

  using DefScaleType = cutlass::float_ue8m0_t;
  using ElementScaleP = cute::conditional_t<UseScale, ElementScaleV, DefScaleType>;
  // -----------------------------------------------------------------------


  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    // Local Mask (sliding window). Only consumed when LocalMask is true.
    int local_left  = 0;
    int local_right = 0;
    // Dropout. Only consumed when HasDropout is true.
    float p_dropout = 0.0f;
    uint64_t philox_seed = 0;
    uint64_t philox_offset = 0;
    void* s_dmask_ptr = nullptr;
    int seqlen_q_rounded = 0;
    int seqlen_k_rounded = 0;
    // Paged KV. Only consumed when PagedKV is true.
    int* ptr_page_table = nullptr;
    int page_size = 0;
    int max_pages_per_seq = 0;
    int total_seqlen_kv = 0;
    const typename TensorQ::element_type* rotary_cos = nullptr;
    const typename TensorQ::element_type* rotary_sin = nullptr;
    int rotary_dim = 0;
    bool is_rotary_interleaved = true;
    // MXFP: scalar scales for F8 KV with F16 MMA path.
    float scale_k_scalar = 1.0f;
    float scale_v_scalar = 1.0f;
  };

  struct LocalMaskFields {
    int local_left, local_right;
  };
  struct EmptyLocal {};

  struct DropoutFields {
    cutlass::fmha::Dropout dropout;
    void* s_dmask_ptr;
    int seqlen_q_rounded;
    int seqlen_k_rounded;
  };
  struct EmptyDropout {};

  struct PagedKVFields {
    int* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
  };
  struct EmptyPaged {};

  struct RotaryFields {
    const typename TensorQ::element_type* rotary_cos = nullptr;
    const typename TensorQ::element_type* rotary_sin = nullptr;
    int rotary_dim = 0;
    bool is_rotary_interleaved = true;
  };
  struct EmptyRotary {};

  // Kernel-facing parameters
  struct Params {
    ElementS scale;
    [[no_unique_address]] conditional_t<LocalMask, LocalMaskFields, EmptyLocal>
        local;
    [[no_unique_address]] conditional_t<HasDropout, DropoutFields, EmptyDropout>
        dropout_fields;
    [[no_unique_address]] conditional_t<PagedKV, PagedKVFields, EmptyPaged>
        paged;
    [[no_unique_address]] conditional_t<HasRotary, RotaryFields, EmptyRotary>
      rotary;
    // MXFP: scalar scales for F8 KV with F16 MMA path.
    float scale_k_scalar = 1.0f;
    float scale_v_scalar = 1.0f;
  };

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloopXe2(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params to_underlying_arguments(
      Arguments const& args,
      void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    Params p{};
    p.scale = val;
    if constexpr (LocalMask) {
      p.local = {args.local_left, args.local_right};
    }
    if constexpr (HasDropout) {
      p.dropout_fields = {
          cutlass::fmha::Dropout(
              args.philox_seed, args.philox_offset, args.p_dropout),
          args.s_dmask_ptr,
          args.seqlen_q_rounded,
          args.seqlen_k_rounded};
    }
    if constexpr (PagedKV) {
      p.paged = {args.ptr_page_table, args.page_size,
                 args.max_pages_per_seq, args.total_seqlen_kv};
    }
    if constexpr (HasRotary) {
      p.rotary = {args.rotary_cos, args.rotary_sin,
                  args.rotary_dim, args.is_rotary_interleaved};
    }
    p.scale_k_scalar = args.scale_k_scalar;
    p.scale_v_scalar = args.scale_v_scalar;
    return p;
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
      int idx_b,
      int& tile_row_idx,
      const int& rows_of_maxima,
      int head_q,
      int num_heads,
      int q_offset_sg,
      int rotary_base,
      int l_coord = 0,
      TensorScaleQ2D const& scaleQ_2D = TensorScaleQ2D{},
      TensorScaleK2D const& scaleK_2D = TensorScaleK2D{},
      TensorScaleV2D const& scaleV_2D = TensorScaleV2D{}) {
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
    auto prefetch_v_paged =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);

    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pVgV_paged = prefetch_v_paged.get_slice(thr_id).partition_S(gV);

    // PagedKV: translate logical K index to physical page-tile index.
    int tiles_per_page = 0;
    int b_offset = 0;
    int page_idx = 0, next_page_idx = blk_k0;
    if constexpr (PagedKV) {
      tiles_per_page = params.paged.page_size / get<1>(TileShapeQK{});
      b_offset = idx_b * params.paged.max_pages_per_seq;
      int page_local_idx =
          blk_k0 * get<1>(TileShapeQK{}) / params.paged.page_size;
      next_page_idx =
          params.paged.ptr_page_table[b_offset + page_local_idx] *
              tiles_per_page +
          blk_k0 % tiles_per_page;
    }

    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }
    if constexpr (PagedKV) {
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
      }
    } else {
      int prefetch_k_stages = (total_blk < Stages ? total_blk : Stages);
      for (int D = 0; D < size<4>(pKgK); D++) {
        CUTLASS_PRAGMA_UNROLL
        for (int K = blk_k0; K < blk_k0 + prefetch_k_stages; K++) {
          int pk;
          if constexpr (PagedKV) {
            int ploc = K * get<1>(TileShapeQK{}) / params.paged.page_size;
            pk = params.paged.ptr_page_table[b_offset + ploc] *
                     tiles_per_page +
                 K % tiles_per_page;
          } else {
            pk = K;
          }
          prefetch(prefetch_k, pKgK(_, _, _, pk, D));
        }
      }
    }
    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    Tensor cPgP = make_identity_tensor(make_shape(seq_len_qo, seq_len_kv));
    Tensor gP_all = local_tile(
        cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), _));

    for (int K = blk_k0; K < blk_k1; K++) {
      // PagedKV: advance page index (current = next computed last iter).
      if constexpr (PagedKV) {
        page_idx = next_page_idx;
        int next_logical = K + 1;
        int next_page_local_idx =
            next_logical * get<1>(TileShapeQK{}) / params.paged.page_size;
        bool valid_page =
            next_page_local_idx < params.paged.max_pages_per_seq;
        if (valid_page) {
          next_page_idx =
              params.paged.ptr_page_table[b_offset + next_page_local_idx] *
                  tiles_per_page +
              next_logical % tiles_per_page;
        } else {
          next_page_idx = params.paged.max_pages_per_seq * tiles_per_page - 1;
        }
      }

      auto tKgK_cache =
          PagedKV ? tKgK(_, _, _, page_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache =
          PagedKV ? tVgV(_, _, _, _, page_idx) : tVgV(_, _, _, _, K);

      // Paged path uses the old whole-V-tile prefetch pattern; split-V
      // prefetch-after-GEMM can hang on BMG for some paged cases.
      if constexpr (PagedKV) {
        prefetch(prefetch_v_paged, pVgV_paged(_, _, _, page_idx));
      } else if constexpr (!PagedKV) {
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prefetch(prefetch_v, pVgV(_, _, _, VV, K));
        }
      }

      /* GEMM 1: S = Q * K^T */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        if constexpr (HasRotary) {
          if (params.rotary.rotary_dim > 0 &&
              params.rotary.rotary_cos != nullptr &&
              params.rotary.rotary_sin != nullptr) {
            auto tQrQ_coords = tQrQ.tv_layout();
            int lane_id = static_cast<int>(get_sub_group().get_local_linear_id());
            int q_tile_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) + q_offset_sg;
            int dim_tile_base = D * get<2>(TileShapeQK{});
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tQrQ.size(); ++i) {
              auto value_coord = idx2crd(
                  i, make_shape(
                         get<1>(shape(tQrQ_coords)),
                         get<2>(shape(tQrQ_coords))));
              auto coord = tQrQ_coords(
                  make_coord(lane_id, get<0>(value_coord), get<1>(value_coord)));
              int row = q_tile_base + get<0>(coord);
              int dim = dim_tile_base + get<1>(coord);
                if (row < seq_len_qo && dim < params.rotary.rotary_dim) {
                int pair_dim = rotary_pair_dim(
                  dim, params.rotary.rotary_dim,
                  params.rotary.is_rotary_interleaved);
                int position = rotary_base + ((CausalMask || LocalMask) ? row : 0);
                tQrQ(i) = apply_rotary_scalar(
                  tQrQ(i), Q_2D(row, pair_dim), params.rotary.rotary_cos,
                  params.rotary.rotary_sin, position, dim,
                  params.rotary.rotary_dim,
                  params.rotary.is_rotary_interleaved);
              }
            }
          }
        }
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);
        if constexpr (FP4Input) {
          copy(tQrQ, tSrQ);
          copy(tKrK, tSrK);
        } else {
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
        }

        if constexpr (UseScale) {
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
          // Block-scaled MMA with scale factors for Q*K. The block-scaled
          // (BDPAS) atom and helpers only exist on CRI (SYCL_INTEL_TARGET==35).
          // On BMG this kernel is instantiated (host references it) but never
          // executed, so the body is replaced by an invalid-control-path stub.
          const int q_coord = get<0>(blk_qv) * BLK_Q +
              ((thr_id / intel::sg_size) / ATOM_K) * SG_Q;
          const int k_coord = K * BLK_K +
              ((thr_id % intel::sg_size) / ATOM_K) * SG_K;

          auto [tiled_copy_scaleQ, copy_iter_scaleQ, fragment_scaleQ] =
              gemm::collective::make_scaled_copy<void, ElementScaleQ,
                  SG_Q, SG_QK_D, GROUP_K>(scaleQ_2D, q_coord, l_coord, size<4>(tKgK));
          auto [tiled_copy_scaleK, copy_iter_scaleK, fragment_scaleK] =
              gemm::collective::make_scaled_copy<void, ElementScaleK,
                  SG_K, SG_QK_D, GROUP_K>(scaleK_2D, k_coord, l_coord, size<4>(tKgK));
          auto [gemm_qm_offsets, gemm_kn_offsets, gemm_qk_offsets, gemm_kk_offsets] =
              gemm::collective::make_scaled_offsets<
                  decltype(size<1>(tSrQ.shape()))::value,
                  decltype(size<1>(tSrK.shape()))::value,
                  decltype(size<2>(tSrK.shape()))::value,
                  MMA_QK_D,
                  GROUP_K,
                  typename decltype(tiled_copy_scaleQ)::BlockShape,
                  typename decltype(tiled_copy_scaleK)::BlockShape>();

          copy(tiled_copy_scaleQ, copy_iter_scaleQ(_, _, _, D), fragment_scaleQ);
          copy(tiled_copy_scaleK, copy_iter_scaleK(_, _, _, D), fragment_scaleK);

          using scaleQSize = decltype(size(fragment_scaleQ));
          using scaleKSize = decltype(size(fragment_scaleK));

          Tensor scaleQ_view = make_tensor(
              recast<intel::vector_t<ElementScaleQ, scaleQSize::value>>(fragment_scaleQ).data(),
              make_layout(Shape<_1, decltype(size<1>(tSrQ.shape())), _1>{}, Stride<_1, _0, _0>{}));
          Tensor scaleK_view = make_tensor(
              recast<intel::vector_t<ElementScaleK, scaleKSize::value>>(fragment_scaleK).data(),
              make_layout(Shape<_1, decltype(size<1>(tSrK.shape())), _1>{}, Stride<_1, _0, _0>{}));

          auto zipped_q = make_zip_tensor(tSrQ, scaleQ_view, gemm_qm_offsets, gemm_qk_offsets);
          auto zipped_k = make_zip_tensor(tSrK, scaleK_view, gemm_kn_offsets, gemm_kk_offsets);

          cute::gemm(mma_qk, zipped_q, zipped_k, tSrS);
#else
          CUTE_INVALID_CONTROL_PATH(
              "Block-scaled (MXFP) Q*K path is only available on CRI "
              "(SYCL_INTEL_TARGET==35).");
#endif
        } else {
          if constexpr (F8kvF16mma) {
            // F8 KV with F16 MMA path - apply scalar scale to K.
            for (int i = 0; i < tSrK.size(); i++)
              tSrK(i) = static_cast<typename TiledMMAQK::ValTypeB>(
                  params.scale_k_scalar * static_cast<float>(tSrK(i)));
          }
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
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
        if constexpr (!PagedKV) {
          // Optimized: separate loops per case, avoid redundant branches.
          if (seq_len_qo == seq_len_kv) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
              if (get<1>(cS_thread(i)) > get<0>(cS_thread(i))) {
                tSrS(i) = ElementS(-INFINITY);
              }
            }
          } else if (seq_len_kv > seq_len_qo) {
            int base = seq_len_kv - (seq_len_qo - 1);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
              if (get<1>(cS_thread(i)) >= base + get<0>(cS_thread(i))) {
                tSrS(i) = ElementS{-INFINITY};
              }
            }
          } else {
            int first_non_masked = seq_len_qo - seq_len_kv;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
              int row_idx = get<0>(cS_thread(i));
              if (row_idx < first_non_masked ||
                  get<1>(cS_thread(i)) > row_idx - first_non_masked) {
                tSrS(i) = ElementS{-INFINITY};
              }
            }
          }
        } else {
          // Paged path: single loop to avoid IGC codegen hang on BMG.
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
      }

      /* Local masking (sliding window) */
      if constexpr (LocalMask) {
        int full_tile_offset = seq_len_kv - seq_len_qo;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int row_idx = get<0>(cS_thread(i));
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask  = col_idx < row_idx - params.local.local_left;
          bool right_mask = col_idx > row_idx + params.local.local_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      softmax(K == blk_k0, tSrS, tA_max, tA_sum, tArA);

      /* Apply dropout to attention probabilities (P) */
      if constexpr (HasDropout) {
        uint32_t batch_head =
            static_cast<uint32_t>(idx_b * num_heads + head_q);
        if (params.dropout_fields.s_dmask_ptr != nullptr) {
          using ElementInput = typename TensorQ::element_type;
          auto* s_dmask_base = reinterpret_cast<ElementInput*>(
              params.dropout_fields.s_dmask_ptr);
          int64_t bh_offset = int64_t(idx_b * num_heads + head_q) *
              int64_t(params.dropout_fields.seqlen_q_rounded) *
              params.dropout_fields.seqlen_k_rounded;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            bool keep = params.dropout_fields.dropout.should_keep(
                batch_head, row_idx, col_idx);
            ElementInput val = static_cast<ElementInput>(tSrS(i));
            s_dmask_base[bh_offset +
                         int64_t(row_idx) *
                             params.dropout_fields.seqlen_k_rounded +
                         col_idx] = keep ? val : -val;
            tSrS(i) = keep
                ? tSrS(i) * params.dropout_fields.dropout.get_scale()
                : ElementS(0);
          }
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            tSrS(i) = params.dropout_fields.dropout.apply(
                tSrS(i), batch_head, row_idx, col_idx);
          }
        }
      }

      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        reorder(tVrV, tArV);
        if constexpr (UseScale && !FP4Input) {
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
          // Block-scaled MMA with scale factors for P*V. CRI-only (BDPAS);
          // on BMG this branch is instantiated but never runs.
          const int v_coord = get<1>(blk_qv) * VTiles * BLK_V + VV * BLK_V +
              ((thr_id % intel::sg_size) / ATOM_V) * SG_V;

          // P is a dummy scale, the same shape as V.
          auto [tiled_copy_scaleP, copy_iter_scaleP, fragment_scaleP] =
              gemm::collective::make_scaled_copy<void, ElementScaleP,
                  SG_P, SG_PV_D, GROUP_K>(scaleV_2D);
          auto [tiled_copy_scaleV, copy_iter_scaleV, fragment_scaleV] =
              gemm::collective::make_scaled_copy<void, ElementScaleV,
                  SG_V, SG_PV_D, GROUP_K>(scaleV_2D, v_coord, l_coord, blk_k1);
          auto [gemm_p_offsets, gemm_v_offsets, gemm_pk_offsets, gemm_vk_offsets] =
              gemm::collective::make_scaled_offsets<
                  decltype(size<1>(tArP.shape()))::value,
                  decltype(size<1>(tArV.shape()))::value,
                  decltype(size<2>(tArV.shape()))::value,
                  MMA_PV_D,
                  GROUP_K,
                  typename decltype(tiled_copy_scaleP)::BlockShape,
                  typename decltype(tiled_copy_scaleV)::BlockShape>();

          fill(fragment_scaleP, ElementScaleV(1));
          copy(tiled_copy_scaleV, copy_iter_scaleV(_, _, _, K), fragment_scaleV);

          using scalePSize = decltype(size(fragment_scaleP));
          using scaleVSize = decltype(size(fragment_scaleV));

          Tensor scaleV_view = make_tensor(
              recast<intel::vector_t<ElementScaleV, scaleVSize::value>>(fragment_scaleV).data(),
              make_layout(Shape<_1, decltype(size<1>(tArV.shape())), _1>{}, Stride<_1, _0, _0>{}));
          Tensor scaleP_view = make_tensor(
              recast<intel::vector_t<ElementScaleV, scalePSize::value>>(fragment_scaleP).data(),
              make_layout(Shape<_1, decltype(size<1>(tArP.shape())), _1>{}, Stride<_1, _0, _0>{}));

          auto zipped_p = make_zip_tensor(tArP, scaleP_view, gemm_p_offsets, gemm_pk_offsets);
          auto zipped_v = make_zip_tensor(tArV, scaleV_view, gemm_v_offsets, gemm_vk_offsets);

          cute::gemm(mma_pv, zipped_p, zipped_v, tArA(_, _, _, VV));
#else
          CUTE_INVALID_CONTROL_PATH(
              "Block-scaled (MXFP) P*V path is only available on CRI "
              "(SYCL_INTEL_TARGET==35).");
#endif
        } else {
          if constexpr (F8kvF16mma) {
            // F8 KV with F16 MMA path - apply scalar scale to V.
            for (int i = 0; i < tArV.size(); i++)
              tArV(i) = static_cast<typename TiledMMAQK::ValTypeB>(
                  params.scale_v_scalar * static_cast<float>(tArV(i)));
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        }
      }

      if constexpr (PagedKV) {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
        }
      } else {
        int K_next = K + Stages;
        if (K_next < blk_k1) {
          if constexpr (PagedKV) {
            int next_page_local_idx =
                K_next * get<1>(TileShapeQK{}) / params.paged.page_size;
            int pk_next;
            if (next_page_local_idx < params.paged.max_pages_per_seq) {
              pk_next =
                  params.paged.ptr_page_table[b_offset + next_page_local_idx] *
                      tiles_per_page +
                  K_next % tiles_per_page;
            } else {
              pk_next = params.paged.max_pages_per_seq * tiles_per_page - 1;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int D = 0; D < size<4>(pKgK); D++) {
              prefetch(prefetch_k, pKgK(_, _, _, pk_next, D));
            }
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int D = 0; D < size<4>(pKgK); D++) {
              prefetch(prefetch_k, pKgK(_, _, _, K_next, D));
            }
          }
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
