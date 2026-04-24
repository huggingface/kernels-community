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
    class TiledCopyV_ = void>
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
    class TiledCopyV_>
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
  static constexpr bool LocalMask  = LocalMask_;
  static constexpr bool HasDropout = HasDropout_;
  static constexpr bool PagedKV   = PagedKV_;

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

  // Kernel-facing parameters
  struct Params {
    ElementS scale;
    [[no_unique_address]] conditional_t<LocalMask, LocalMaskFields, EmptyLocal>
        local;
    [[no_unique_address]] conditional_t<HasDropout, DropoutFields, EmptyDropout>
        dropout_fields;
    [[no_unique_address]] conditional_t<PagedKV, PagedKVFields, EmptyPaged>
        paged;
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
      int num_heads) {
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
    int prefetch_k_stages = (total_blk < Stages ? total_blk : Stages);
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = blk_k0; K < blk_k0 + prefetch_k_stages; K++) {
        int pk = PagedKV ? next_page_idx : K;
        prefetch(prefetch_k, pKgK(_, _, _, pk, D));
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

      // Prefetch V early to overlap with GEMM1 computation
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        int pk = PagedKV ? page_idx : K;
        prefetch(prefetch_v, pVgV(_, _, _, VV, pk));
      }

      /* GEMM 1: S = Q * K^T */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
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
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

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
