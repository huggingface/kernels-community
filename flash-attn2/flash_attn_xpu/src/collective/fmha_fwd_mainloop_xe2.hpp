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
    class TiledCopyV_ = void,
    bool HasRotary_ = false>
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
    bool HasRotary_>
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
    HasRotary_> {

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
  using FragSCol = typename Traits::FragSCol;
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
  };

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloopXe2(Params const& params_, SharedStorage&) : params(params_) {}

  static Params to_underlying_arguments(
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
    return p;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Physical K tile holding logical K block `K` of batch `idx_b` in the
  // flat paged cache surface (same as vllm-xpu-kernels' chunk_prefill mainloop).
  CUTLASS_DEVICE int get_paged_idx(int K, int idx_b) {
    constexpr int tile_k = get<1>(TileShapeQK{});
    int tiles_per_page = params.paged.page_size / tile_k;
    int b_offset = idx_b * params.paged.max_pages_per_seq;
    int page_local_idx = K * tile_k / params.paged.page_size;
    if (page_local_idx >= params.paged.max_pages_per_seq) {
      page_local_idx = params.paged.max_pages_per_seq - 1;
    }
    int block_idx = params.paged.ptr_page_table[b_offset + page_local_idx];
    return block_idx * tiles_per_page + K % tiles_per_page;
  }

  // K/V block range [blk_k0, blk_k1) must be the same for every sub-group of
  // the work-group (the loop runs a work-group split barrier per block).
  // Causal masking is applied to blocks K >= blk_k1_causal.
  template <bool has_large_surface, typename QVCoord>
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
      int blk_k1_causal,
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
      int rotary_base) {
    using namespace sycl::ext::oneapi::this_work_item;

    constexpr int tile_k = get<1>(TileShapeQK{});
    // The lane-based masks assume each sub-group owns whole rows of the
    // K tile (sub-groups split only along Q), as in all prefill policies.
    constexpr bool kLaneMask = size<1>(SubgroupLayoutQK{}) == 1;

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor cK = make_identity_tensor(K_2D.shape());
    Tensor cV = make_identity_tensor(V_2D.shape());
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));

    auto K_2D_0 = K_2D;
    auto V_2D_0 = V_2D;

    // The Xe 2D block descriptor stores the surface height in 24 bits. A
    // paged cache taller than that is split into page-aligned sub-surfaces
    // and every tile is routed to the one holding it (as in vllm-xpu-kernels).
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
      if (rows < 1) {
        rows = 1;
      }
      return make_tensor(
          K_2D.data() + int64_t(surf) * k_surf_shift,
          make_layout(make_shape(rows, get<1>(K_2D.shape())), K_2D.stride()));
    };
    [[maybe_unused]] auto make_v_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) {
        cols = 1;
      }
      return make_tensor(
          V_2D.data() + int64_t(surf) * v_surf_shift,
          make_layout(make_shape(get<0>(V_2D.shape()), cols), V_2D.stride()));
    };

    if constexpr (has_large_surface) {
      constexpr int kXeMaxSurfaceRows = 1 << 24;
      int page_tiles = params.paged.page_size / tile_k;
      int blocks_per_surface = kXeMaxSurfaceRows / params.paged.page_size;
      if (blocks_per_surface < 1) {
        blocks_per_surface = 1;
      }
      tiles_per_surface = blocks_per_surface * page_tiles;
      rows_per_surface = tiles_per_surface * tile_k;

      int k_rows0 =
          k_rows_total < rows_per_surface ? k_rows_total : rows_per_surface;
      k_surf_shift = int64_t(rows_per_surface) * get<0>(K_2D.stride());
      int v_cols0 =
          v_cols_total < rows_per_surface ? v_cols_total : rows_per_surface;
      v_surf_shift = int64_t(rows_per_surface) * get<1>(V_2D.stride());

      K_2D_0 = make_tensor(
          K_2D.data(),
          make_layout(make_shape(k_rows0, get<1>(K_2D.shape())), K_2D.stride()));
      V_2D_0 = make_tensor(
          V_2D.data(),
          make_layout(make_shape(get<0>(V_2D.shape()), v_cols0), V_2D.stride()));
      cK = make_identity_tensor(
          make_shape(rows_per_surface, get<1>(K_2D.shape())));
      cV = make_identity_tensor(
          make_shape(get<0>(V_2D.shape()), rows_per_surface));
    }

    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});
    Tensor gK = local_tile(
        cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));
    Tensor gV_split = local_tile(
        gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});

    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D_0};
    TiledCopyV copy_v{V_2D_0};

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
    // Whole V tile prefetched cooperatively by the work-group.
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);

    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    /* Masking scalars */
    int lane_id = thr_id % intel::sg_size;
    int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) + q_offset_sg;
    int full_tile_offset = seq_len_kv - seq_len_qo;

    // Coordinates of this thread's scores in K block `K` (slow path masks).
    auto make_cS = [&](int K) {
      Tensor cPgP = make_identity_tensor(make_shape(seq_len_qo, seq_len_kv));
      Tensor gP = local_tile(
          cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
      return thr_mma_qk.partition_C(gP);
    };

    [[maybe_unused]] int page_idx = blk_k0;
    [[maybe_unused]] int next_page_idx = blk_k0;
    if constexpr (PagedKV) {
      next_page_idx = get_paged_idx(blk_k0, idx_b);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % tile_k != 0);

    [[maybe_unused]] auto route_current = [&](int cur) -> int {
      int k_surf = cur / tiles_per_surface;
      int local = cur - k_surf * tiles_per_surface;
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

        Tensor cK_n = make_identity_tensor(
            make_shape(get<0>(K_2D_n.shape()), get<1>(K_2D.shape())));
        Tensor cV_n = make_identity_tensor(
            make_shape(get<0>(V_2D.shape()), get<1>(V_2D_n.shape())));

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
      return local;
    };

    if constexpr (!has_large_surface) {
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
      }
    }

    for (int K = blk_k0; K < blk_k1; K++) {
      // Split barrier to keep the sub-groups of a work-group together, as in
      // sycl-tla's xe_fmha_fwd_mainloop and vllm-xpu-kernels' chunk_prefill mainloop.
      // Without it, two work-groups sharing one BMG Xe-core produce wrong
      // results or hang the GPU.
      barrier_arrive(ScopeWorkgroup);

      int load_idx;
      if constexpr (has_large_surface) {
        load_idx = route_current(get_paged_idx(K, idx_b));
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, load_idx, D));
        }
      } else {
        page_idx = next_page_idx;
        next_page_idx = K + 1;
        if constexpr (PagedKV) {
          next_page_idx = get_paged_idx(next_page_idx, idx_b);
        }
        load_idx = PagedKV ? page_idx : K;
      }

      auto tKgK_cache = tKgK(_, _, _, load_idx, _);
      auto tVgV_cache = tVgV(_, _, _, _, load_idx);

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, load_idx));

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
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* Causal masking: score (row, col) is masked iff
       * col - row > seq_len_kv - seq_len_qo. */
      if constexpr (CausalMask) {
        if (K >= blk_k1_causal) {
          if constexpr (kLaneMask) {
            constexpr int n_reps = tile_k / intel::sg_size;
            constexpr int elems_per_n = tSrS.size() / n_reps;
            int k_base = K * tile_k;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < n_reps; n++) {
              int col = k_base + n * intel::sg_size + lane_id;
              int causal_bound = col - full_tile_offset - row_base;
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < elems_per_n; j++) {
                if (j < causal_bound) {
                  tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
                }
              }
            }
          } else {
            auto cS_thread = make_cS(K);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrS.size(); ++i) {
              if (get<1>(cS_thread(i)) - get<0>(cS_thread(i)) >
                  full_tile_offset) {
                tSrS(i) = ElementS(-INFINITY);
              }
            }
          }
        }
      }

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        if constexpr (kLaneMask) {
          FragSCol k_rem_mask;
          int k = K * tile_k + lane_id;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) =
                (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
        } else {
          auto cS_thread = make_cS(K);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            if (get<1>(cS_thread(i)) >= seq_len) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }

      /* Local masking (sliding window) */
      if constexpr (LocalMask) {
        auto cS_thread = make_cS(K);
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

      auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);

      /* Apply dropout to attention probabilities (P) */
      if constexpr (HasDropout) {
        auto cS_thread = make_cS(K);
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
            // The Q tile may be taller than the S_dmask row rounding.
            if (row_idx < params.dropout_fields.seqlen_q_rounded &&
                col_idx < params.dropout_fields.seqlen_k_rounded) {
              s_dmask_base[bh_offset +
                           int64_t(row_idx) *
                               params.dropout_fields.seqlen_k_rounded +
                           col_idx] = keep ? val : -val;
            }
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
        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++)
            tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      if constexpr (!has_large_surface) {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
        }
      }
      barrier_wait(ScopeWorkgroup);
    }

    // Use shared get_LSE_metadata from fmha_fwd_common.hpp
    cutlass::fmha::collective::get_LSE_metadata(
        thr_id, TileShapePV{}, mma_pv, rows_of_maxima, tile_row_idx);
  }

  // Single step of blocked softmax. Returns the rescale factor of the
  // previous accumulator rows; the caller applies it to A.
  CUTLASS_DEVICE
  FragSRow softmax(
      bool first_block,
      FragS& tS,
      FragSRow& tS_max,
      FragSRow& tS_sum) {
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
    }

    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
    return rescale;
  }
};

}  // namespace cutlass::fmha::collective
