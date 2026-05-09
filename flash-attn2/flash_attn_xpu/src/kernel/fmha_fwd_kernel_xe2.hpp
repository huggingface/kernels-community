/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Xe2 (BMG / Arc Pro B60) FMHA forward outer kernel -- supports the full FA2
 * feature set (causal, local, dropout, paged, varlen) via compile-time flags.
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "../collective/fmha_fusion.hpp"
#include "../collective/fmha_fwd_mainloop_xe2.hpp"
#include "../collective/fmha_fwd_epilogue_xe2.hpp"
#include "./fmha_problem_shape.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class XeFMHAFwdKernelXe2 {
 public:
  using ProblemShape = ProblemShape_;
  static constexpr bool is_var_len = cutlass::fmha::collective::
      is_variable_length_v<typename ProblemShape::SeqLenType>;

  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;

  static constexpr bool CausalMask = CollectiveMainloop::CausalMask;
  static constexpr bool LocalMask  = CollectiveMainloop::LocalMask;
  static constexpr bool HasDropout = CollectiveMainloop::HasDropout;
  static constexpr bool PagedKV    = CollectiveMainloop::PagedKV;

  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    int64_t q_batch_stride;
    int64_t q_head_stride;
    int64_t q_row_stride;
    const ElementK* K;
    int64_t k_batch_stride;
    int64_t k_head_stride;
    int64_t k_row_stride;
    const ElementV* V;
    int64_t v_batch_stride;
    int64_t v_head_stride;
    int64_t v_row_stride;
    ElementO* O;
    int64_t o_batch_stride;
    int64_t o_head_stride;
    int64_t o_row_stride;
    float* pLSE;
    int lse_stride_head;
    int lse_stride_batch;
    // KV Cache: per-batch effective KV length (nullptr for non-kvcache paths)
    int* cache_seqlens = nullptr;
    int* cache_batch_idx = nullptr;
    int* cache_leftpad = nullptr;
    // Fused KV cache append
    const ElementK* Knew = nullptr;
    int64_t knew_batch_stride = 0;
    int64_t knew_head_stride = 0;
    int64_t knew_row_stride = 0;
    const ElementV* Vnew = nullptr;
    int64_t vnew_batch_stride = 0;
    int64_t vnew_head_stride = 0;
    int64_t vnew_row_stride = 0;
    int seqlen_knew = 0;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) &&
        CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const&) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const&, void* = nullptr,
      cudaStream_t = nullptr, CudaHostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(
          Shape<
              cutlass::fmha::collective::VariableLength,
              cutlass::fmha::collective::VariableLength>{
              problem_shape.seq_len_qo, problem_shape.seq_len_kv},
          batch);
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  CUTLASS_DEVICE
  int calculate_longest_non_masked_length(
      const int& seq_len_kv, const int& seq_len_qo,
      const int& last_seq_coord, const int& first_non_masked_sequence) {
    int longest_non_masked_length = 0;
    if (seq_len_kv > seq_len_qo) {
      int elements_in_first_line = seq_len_kv - (seq_len_qo - 1);
      longest_non_masked_length = elements_in_first_line + last_seq_coord;
    } else {
      longest_non_masked_length = cute::min(
          seq_len_kv,
          cute::max(0, last_seq_coord - first_non_masked_sequence + 1));
    }
    longest_non_masked_length = cute::min(seq_len_kv, longest_non_masked_length);
    return longest_non_masked_length;
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(
        sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] = tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = head_q / head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo)
        continue;

      // KV Cache: override seq_len_kv with per-batch effective length
      int effective_seq_kv = seq_len_kv;
      int leftpad_k = 0;
      // bidx maps the logical request to a slot in the physical KV cache.
      // - paged path: cache_batch_idx is forbidden, K/V are flat blocks
      // - non-paged path: bidx selects the per-batch KV slice
      int bidx = (!PagedKV && p.cache_batch_idx) ? p.cache_batch_idx[idx_b] : idx_b;
      if (p.cache_seqlens) {
        int orig_cache_seqlens = p.cache_seqlens[idx_b];
        if (p.cache_leftpad) {
          leftpad_k = p.cache_leftpad[idx_b];
        }

        // Fused cache update: copy knew/vnew into kcache/vcache
        if (p.Knew != nullptr && p.seqlen_knew > 0) {
          constexpr int num_threads = SGPerWG::value * cute::intel::sg_size;
          auto* k_src = p.Knew
              + idx_b * p.knew_batch_stride + head * p.knew_head_stride;
          auto* v_src = p.Vnew
              + idx_b * p.vnew_batch_stride + head * p.vnew_head_stride;
          if constexpr (PagedKV) {
            // Paged scatter: per-token compute (block, page_offset) from
            // block_table, then write to the corresponding page slot.
            // Each "block" in the paged K/V tensor spans `page_size` rows,
            // so the per-block byte stride is `page_size * row_stride`,
            // not `k_batch_stride` (which is sized for the *whole* logical
            // KV layout, not per-page).
            const int page_size = params.mainloop.paged.page_size;
            const int max_pages_per_seq =
                params.mainloop.paged.max_pages_per_seq;
            const int* page_table = params.mainloop.paged.ptr_page_table
                + idx_b * max_pages_per_seq;
            const int64_t k_block_stride =
                static_cast<int64_t>(page_size) * p.k_row_stride;
            const int64_t v_block_stride =
                static_cast<int64_t>(page_size) * p.v_row_stride;
            for (int si = 0; si < p.seqlen_knew; si++) {
              int global_pos = orig_cache_seqlens + si;
              int page_idx = global_pos / page_size;
              int page_off = global_pos % page_size;
              int block = page_table[page_idx];
              auto* k_dst = const_cast<ElementK*>(p.K)
                  + static_cast<int64_t>(block) * k_block_stride
                  + head * p.k_head_stride
                  + static_cast<int64_t>(page_off) * p.k_row_stride;
              auto* v_dst = const_cast<ElementV*>(p.V)
                  + static_cast<int64_t>(block) * v_block_stride
                  + head * p.v_head_stride
                  + static_cast<int64_t>(page_off) * p.v_row_stride;
              for (int d = thr_id; d < s.head_size_qk; d += num_threads) {
                auto k_value = k_src[si * p.knew_row_stride + d];
                if constexpr (CollectiveMainloop::HasRotary) {
                  if (params.mainloop.rotary.rotary_dim > 0 &&
                      params.mainloop.rotary.rotary_cos != nullptr &&
                      params.mainloop.rotary.rotary_sin != nullptr &&
                      d < params.mainloop.rotary.rotary_dim) {
                    int pair_dim = cutlass::fmha::collective::rotary_pair_dim(
                        d, params.mainloop.rotary.rotary_dim,
                        params.mainloop.rotary.is_rotary_interleaved);
                    if (pair_dim < s.head_size_qk) {
                      k_value = cutlass::fmha::collective::apply_rotary_scalar(
                          k_value, k_src[si * p.knew_row_stride + pair_dim],
                          params.mainloop.rotary.rotary_cos,
                          params.mainloop.rotary.rotary_sin,
                          global_pos, d, params.mainloop.rotary.rotary_dim,
                          params.mainloop.rotary.is_rotary_interleaved);
                    }
                  }
                }
                k_dst[d] = k_value;
              }
              for (int d = thr_id; d < s.head_size_vo; d += num_threads) {
                v_dst[d] = v_src[si * p.vnew_row_stride + d];
              }
            }
          } else {
            auto* k_dst = const_cast<ElementK*>(p.K)
                + bidx * p.k_batch_stride + head * p.k_head_stride
                + static_cast<int64_t>(orig_cache_seqlens) * p.k_row_stride;
            auto* v_dst = const_cast<ElementV*>(p.V)
                + bidx * p.v_batch_stride + head * p.v_head_stride
                + static_cast<int64_t>(orig_cache_seqlens) * p.v_row_stride;
            for (int si = 0; si < p.seqlen_knew; si++) {
              for (int d = thr_id; d < s.head_size_qk; d += num_threads) {
                auto k_value = k_src[si * p.knew_row_stride + d];
                if constexpr (CollectiveMainloop::HasRotary) {
                  if (params.mainloop.rotary.rotary_dim > 0 &&
                      params.mainloop.rotary.rotary_cos != nullptr &&
                      params.mainloop.rotary.rotary_sin != nullptr &&
                      d < params.mainloop.rotary.rotary_dim) {
                    int pair_dim = cutlass::fmha::collective::rotary_pair_dim(
                        d, params.mainloop.rotary.rotary_dim,
                        params.mainloop.rotary.is_rotary_interleaved);
                    if (pair_dim < s.head_size_qk) {
                      k_value = cutlass::fmha::collective::apply_rotary_scalar(
                          k_value, k_src[si * p.knew_row_stride + pair_dim],
                          params.mainloop.rotary.rotary_cos,
                          params.mainloop.rotary.rotary_sin,
                          orig_cache_seqlens + si, d,
                          params.mainloop.rotary.rotary_dim,
                          params.mainloop.rotary.is_rotary_interleaved);
                    }
                  }
                }
                k_dst[si * p.k_row_stride + d] = k_value;
              }
            }
            for (int si = 0; si < p.seqlen_knew; si++) {
              for (int d = thr_id; d < s.head_size_vo; d += num_threads) {
                v_dst[si * p.v_row_stride + d] = v_src[si * p.vnew_row_stride + d];
              }
            }
          }
          sycl::group_barrier(get_work_group<3>());
          effective_seq_kv = (orig_cache_seqlens + p.seqlen_knew) - leftpad_k;
        } else {
          effective_seq_kv = orig_cache_seqlens - leftpad_k;
        }
      }
      if (effective_seq_kv <= 0) continue;

      auto full_tile_offset = effective_seq_kv - seq_len_qo;
      int seq_coord = cute::min(
          seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));
      int last_seq_coord = seq_coord + q_sg_tile - 1;
      int first_non_masked_sequence = seq_len_qo - effective_seq_kv;

      // Causal-only early-exit: skip SGs that are fully masked. With
      // LocalMask we can't easily do this here, so let the loop body mask.
      if (CausalMask && !LocalMask &&
          first_non_masked_sequence > last_seq_coord) {
        continue;
      }

      // Per-SG effective KV length.
      int seq_len;
      if constexpr (CausalMask && LocalMask) {
        seq_len = cute::min(
            effective_seq_kv,
            full_tile_offset + seq_coord + q_sg_tile +
                params.mainloop.local.local_right);
      } else if constexpr (CausalMask) {
        seq_len = calculate_longest_non_masked_length(
            effective_seq_kv, seq_len_qo, last_seq_coord,
            first_non_masked_sequence);
      } else {
        seq_len = effective_seq_kv;
      }
      if (seq_len < 0) seq_len = 0;

      int k_block0;
      if constexpr (LocalMask) {
        k_block0 = cute::max(
            seq_coord + full_tile_offset - params.mainloop.local.local_left,
            0) / get<1>(TileShapeQK{});
      } else {
        k_block0 = 0;
      }
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
      const int total_blk = k_blocks - k_block0;

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q  * s.head_size_qk * qo_cumulative[idx_b];
        offset_k = PagedKV
            ? 0
            : s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
        offset_v = PagedKV
            ? 0
            : s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];
        offset_o = s.num_heads_q  * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto batch_dim_q = is_var_len ? 1 : s.batch;
      // Paged KV is laid out as (num_blocks * page_size, head, num_heads_kv)
      // with no batch dimension; treat it like the varlen K/V layout.
      auto batch_dim_kv = (is_var_len || PagedKV) ? 1 : s.batch;
      int total_seqlen_kv;
      if constexpr (PagedKV) {
        total_seqlen_kv = params.mainloop.paged.total_seqlen_kv;
      } else {
        total_seqlen_kv = seq_len_kv;
      }
      // When leftpad is applied, the K/V base pointer is shifted forward.
      // Reduce the surface height so it accurately describes the data
      // reachable from the shifted base, preventing 2D block loads from
      // extending past the per-batch allocation.
      int kv_surface_len = total_seqlen_kv - leftpad_k;
      auto shape_Q =
          make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim_q);
      auto shape_K = make_shape(
          kv_surface_len, s.head_size_qk, s.num_heads_kv, batch_dim_kv);
      auto shape_V = make_shape(
          s.head_size_vo, kv_surface_len, s.num_heads_kv, batch_dim_kv);
      auto shape_O =
          make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim_q);

      auto stride_q = cutlass::make_stride(
          static_cast<int>(p.q_row_stride), Int<1>{},
          static_cast<int>(p.q_head_stride),
          static_cast<int>(p.q_batch_stride));
      auto stride_k = cutlass::make_stride(
          static_cast<int>(p.k_row_stride), Int<1>{},
          static_cast<int>(p.k_head_stride),
          static_cast<int>(p.k_batch_stride));
      auto stride_v = cutlass::make_stride(
          Int<1>{}, static_cast<int>(p.v_row_stride),
          static_cast<int>(p.v_head_stride),
          static_cast<int>(p.v_batch_stride));
      auto stride_o = cutlass::make_stride(
          static_cast<int>(p.o_row_stride), Int<1>{},
          static_cast<int>(p.o_head_stride),
          static_cast<int>(p.o_batch_stride));

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto ptrO = p.O + offset_o;

      // Offset K/V by leftpad to skip left-padding tokens in the cache
      if (leftpad_k > 0) {
        dcK += leftpad_k * static_cast<int>(p.k_row_stride);
        dcV += leftpad_k * static_cast<int>(p.v_row_stride);
      }

      auto layout_q = is_var_len
          ? make_ordered_layout(shape_Q, Step<_2, _0, _1, _3>{})
          : make_layout(shape_Q, stride_q);
      auto layout_k = is_var_len
          ? make_ordered_layout(shape_K, Step<_2, _0, _1, _3>{})
          : make_layout(shape_K, stride_k);
      auto layout_v = is_var_len
          ? make_ordered_layout(shape_V, Step<_0, _2, _1, _3>{})
          : make_layout(shape_V, stride_v);
      auto layout_o = is_var_len
          ? make_ordered_layout(shape_O, Step<_2, _0, _1, _3>{})
          : make_layout(shape_O, stride_o);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), layout_q);
      Tensor K = make_tensor(make_gmem_ptr(dcK), layout_k);
      Tensor V = make_tensor(make_gmem_ptr(dcV), layout_v);
      Tensor O = make_tensor(make_gmem_ptr(ptrO), layout_o);

      FragA tArA;
      FragARow tA_max, tA_sum;
      int tile_row_idx = -1;
      int rows_of_maxima =
          get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

      // For non-paged KV, reuse cache_batch_idx remap (bidx) so that the
      // KV slice matches the per-request seqlen. Q/O always use idx_b.
      int l_coord_q  = is_var_len ? 0 : idx_b;
      int l_coord_kv = is_var_len ? 0
                       : (PagedKV ? 0 : bidx);
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q(_, _, head_q, l_coord_q),
          K(_, _, head, l_coord_kv),
          V(_, _, head, l_coord_kv),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          k_block0,
          k_blocks,
          total_blk,
          thr_id,
          seq_len,
          seq_len_qo,
          effective_seq_kv,
          idx_b,
          tile_row_idx,
          rows_of_maxima,
          head_q,
          s.num_heads_q,
          q_offset_sg,
          p.cache_seqlens ? p.cache_seqlens[idx_b] : 0);

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      auto metadata_for_lse = std::make_tuple(
          get<0>(TileShapePV{}),
          p.lse_stride_head,
          p.lse_stride_batch,
          seq_len_qo,
          idx_b,
          head_q,
          tile_row_idx,
          rows_of_maxima);
      epilogue(
          O(_, _, head_q, l_coord_q),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id,
          p.pLSE,
          metadata_for_lse);
    }
  }
};

}  // namespace cutlass::fmha::kernel
