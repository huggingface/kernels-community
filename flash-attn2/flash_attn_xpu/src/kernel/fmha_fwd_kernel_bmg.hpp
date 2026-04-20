/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * BMG-path FMHA forward outer kernel -- minimal functionality matching the
 * torch-xpu-ops SDPA flash-attention backend kernel, plus optional varlen.
 *
 *   - Non-paged, non-local, non-dropout.
 *   - Optional causal mask.
 *   - Optional variable-length sequences (selected via ProblemShape::SeqLenType).
 *   - For non-varlen: contiguous int64_t strides (row_stride, head_stride,
 *     batch_stride) rather than cute::Stride types, matching SDPA.
 *   - Optional LSE output pointer.
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "../collective/fmha_fusion.hpp"
#include "../collective/fmha_fwd_mainloop_bmg.hpp"
#include "../collective/fmha_fwd_epilogue_bmg.hpp"
#include "./fmha_fwd_kernel.hpp"  // for FMHAProblemShape

namespace cutlass::fmha::kernel {

using namespace cute;

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class XeFMHAFwdKernelBmg {
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

      int seq_coord = cute::min(
          seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));
      int first_non_masked_sequence = seq_len_qo - seq_len_kv;
      int last_seq_coord = seq_coord + q_sg_tile - 1;

      if (CausalMask && first_non_masked_sequence > last_seq_coord) {
        continue;
      }

      const int seq_len = CausalMask
          ? calculate_longest_non_masked_length(
                seq_len_kv, seq_len_qo, last_seq_coord,
                first_non_masked_sequence)
          : seq_len_kv;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q  * s.head_size_qk * qo_cumulative[idx_b];
        offset_k = s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
        offset_v = s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];
        offset_o = s.num_heads_q  * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q =
          make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K =
          make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V =
          make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O =
          make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim);

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

      int l_coord = is_var_len ? 0 : idx_b;
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q(_, _, head_q, l_coord),
          K(_, _, head, l_coord),
          V(_, _, head, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          0,
          k_blocks,
          k_blocks,
          thr_id,
          seq_len,
          seq_len_qo,
          seq_len_kv,
          idx_b,
          tile_row_idx,
          rows_of_maxima);

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
          O(_, _, head_q, l_coord),
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
