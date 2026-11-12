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

#include "../splitkv_decode.hpp"  // FLASH_XPU_VLLM_NS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "../../collective/fmha_fusion.hpp"
#include "chunk_prefill_mainloop.hpp"
#include "chunk_prefill_epilogue.hpp"
#include "paged_decode_tuning.h"

namespace cutlass::fmha::kernel {
// Arch-tagged inline namespace: gives these definitions a mangled name
// distinct from the other Xe architecture's identically named copies,
// while leaving name lookup (cutlass::fmha::...) unchanged.
inline namespace FLASH_XPU_VLLM_NS {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct DecodeProblemShape {
  using SeqLenType = cute::
      conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv;
  int head_size_qk, head_size_vo;
  // flash-attn2: seq_len_kv.cumulative_length holds cumulative offsets
  // (cu_seqlens_k) instead of per-batch lengths (seqused_k).
  bool kv_cumulative = false;
};

///////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_,
    bool LargeSurface_ = false>
class XeFMHAFwdSplitKVKernel {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
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

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;
  static constexpr int dpas_max_repeat_count = 8;

  // Paged decode tuned for XE3P (csrc/xpu/attn/paged_decode_tuning.h): the
  // launcher caps work-groups per Xe core and the host may split short
  // sequences. Otherwise sequences shorter than the legacy minimum KV tile
  // count ignore the requested split count and run as a single split.
  using TuningPrecision = flash_xpu_splitkv::paged_decode_tuning::Precision;
  static constexpr int kHeadDimBucket = get<1>(TileShapeO{});
  static constexpr bool kTunedDecode = flash_xpu_splitkv::paged_decode_tuning::is_tuned(
      kHeadDimBucket, get<1>(TileShapeQK{}));
  static constexpr TuningPrecision kPrecision =
      CollectiveMainloop::Fp8Q    ? TuningPrecision::kFullFp8
      : CollectiveMainloop::Fp8KV ? TuningPrecision::kFp8KV
                                  : TuningPrecision::k16Bit;
  static constexpr int kMinBlocksForSplit =
      kTunedDecode ? flash_xpu_splitkv::paged_decode_tuning::kMinTilesForSplit
                   : ((get<1>(TileShapeQK{}) <= 64) ? 32 : 128);
  static constexpr bool Sink = CollectiveEpilogue::Sink;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  // Set for paged K/V pools taller than one Xe 2D block-load surface (see
  // collective::kXeMaxSurfaceRows); the mainloop then splits K/V into
  // sub-surfaces. A separate instantiation leaves the common path unchanged.
  static constexpr bool LargeSurface = LargeSurface_;
  static_assert(
      !LargeSurface || CollectiveMainloop::PagedKV,
      "LargeSurface only applies to paged K/V");

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K;
    StrideK dK;
    const ElementV* V;
    StrideV dV;
    ElementO* Oaccum;
    StrideO dOaccum;
    ElementLSE* softmax_lse_accum;
    StrideO dSoftmax_lse_accum;

    const ElementSink* sm_sink;

    // per-batch mask: true = prefill, false = decode; nullptr = process all
    const bool* is_prefill;
    const int* splits_per_seq =
        nullptr;  // per-seq split counts; null => use global num_kv_splits
    // softmax_lse output [num_heads_q, total_seqlen_q]; nullptr = disabled.
    // Only written here when num_kv_splits <= 1 (no ReduceSplitK pass).
    float* softmax_lse = nullptr;
    int lse_stride = 0;
    // flash-attn2: stride between query tokens in softmax_lse.
    int lse_row_stride = 1;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;  // no split by default
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    if (!is_var_len && args.kernel.shape.seq_len_qo != 1) {
      // decode only
      return false;
    }

    if (args.num_kv_splits > max_num_kv_splits) {
      return false;
    }

    return CollectiveMainloop::can_implement(args.mainloop) &&
           CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
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
      auto q_len = cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength>{problem_shape.seq_len_qo}, batch);
      const int* kv_len = problem_shape.seq_len_kv.cumulative_length;
      return Shape<int, int>{
          get<0>(q_len),
          problem_shape.kv_cumulative ? kv_len[batch + 1] - kv_len[batch]
                                      : kv_len[batch]};
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(
        sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits_;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto
          [blk_q,
           blk_v,
           head,
           idx_b,
           idx_kv_split,
           wl_tile_start,
           wl_tile_count] =
              tile_scheduler
                  .get_block_coord();  // (Q,V,h,b,split,tile_start,tile_count)

      // Skip prefill batches when is_prefill mask is provided
      if (p.is_prefill != nullptr && p.is_prefill[idx_b]) continue;

      auto blk_qv = make_coord(blk_q, blk_v);
      int head_q_start = head * head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;
      // Decode packs the GQA head-group into the Q/row dimension (seq_len_qo
      // is always 1). blk_q tiles head_group_q in steps of the packed-Q tile
      // size, so skip work-groups whose first head-group row is out of range.
      if (blk_q * get<0>(TileShapeQK{}) >= head_group_q) continue;

      auto offset = cute::min(seq_len_qo, seq_len_kv);
      auto discard_seq_coord = seq_len_qo - offset;
      auto full_tile_offset = seq_len_kv - offset;
      int seq_coord =
          cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      if (CollectiveMainloop::CausalMask && seq_coord < discard_seq_coord)
        continue;
      // For decode window_size_right doesn't have effect
      const int seq_len = seq_len_kv;
      // For decode, all packed GQA heads are at position seq_len_kv - 1.
      // Use seq_len - 1 (= seq_len_kv - 1) as the decode position for
      // k_block0 to match ReduceSplitK's computation.
      const int k_block0 =
          CollectiveMainloop::LocalMask
              ? cute::max(seq_len - 1 - params.mainloop.window_size_left, 0) /
                    get<1>(TileShapeQK{})
              : 0;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
      const int windowed_k_blocks = k_blocks - k_block0;

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_softmax_lse_accum = 0;
      int offset_lse = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;

        // Use Q's actual per-token stride (get<0>(dQ)) instead of assuming
        // the packed value num_heads_q * head_size_qk so non-contiguous Q
        // tensors (e.g. strided slices, permuted/transposed views, slices
        // of a wider buffer) are read correctly.
        offset_q = get<0>(p.dQ) * qo_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * num_kv_splits *
                   qo_cumulative[idx_b];
        offset_softmax_lse_accum =
            s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        // softmax_lse is (num_heads_q, total_seqlen_q); decode has one query
        // token per sequence, so the column is the sequence's cumulative
        // query offset.
        offset_lse = qo_cumulative[idx_b] * p.lse_row_stride;

        // flash-attn2: a batch without queries owns no output row.
        if (seq_len_qo <= 0) continue;
        // for gqa packing, seq_len_qo must be 1
        seq_len_qo = 1;
      } else {
        // Non-varlen decode has exactly one query token per batch entry.
        offset_lse = idx_b * p.lse_row_stride;
      }

      // neglect seq_len_qo since it's always 1 for decode
      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q =
          make_shape(head_group_q, s.head_size_qk, s.num_heads_kv, batch_dim);
      // shape
      auto total_seqlen_kv = params.mainloop.total_seqlen_kv;
      auto shape_K = make_shape(
          total_seqlen_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(
          s.head_size_vo, total_seqlen_kv, s.num_heads_kv, batch_dim);

      auto shape_O = make_shape(
          head_group_q,
          s.head_size_vo,
          s.num_heads_kv,
          num_kv_splits,
          batch_dim);
      auto shape_softmax_lse_accum =
          make_shape(head_group_q, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_sink = make_shape(s.num_heads_kv, head_group_q);

      int kv_split_offset;
      int num_effective_kv_blocks;
      int seq_num_kv_splits;
      bool is_single_split;

      if (wl_tile_start >= 0) {
        // Compact grid: use pre-computed tile range from work_list.
        // Python (build_decode_split_plan) has already folded the
        // single-split heuristic and balanced assignment into the plan.
        kv_split_offset = k_block0 + wl_tile_start;
        num_effective_kv_blocks = wl_tile_count;
        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : num_kv_splits;
        is_single_split = (seq_num_kv_splits <= 1);
      } else {
        // Legacy path: compute split range on the fly
        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : num_kv_splits;

        if (idx_kv_split >= seq_num_kv_splits) {
          continue;
        }

        int num_blocks_per_split =
            cute::ceil_div(windowed_k_blocks, seq_num_kv_splits);

        is_single_split =
            (seq_num_kv_splits > 1) && (windowed_k_blocks < kMinBlocksForSplit);

        if (is_single_split) {
          if (idx_kv_split > 0) {
            continue;
          }
          kv_split_offset = k_block0;
          num_effective_kv_blocks = windowed_k_blocks;
        } else {
          kv_split_offset = k_block0 + idx_kv_split * num_blocks_per_split;
          num_effective_kv_blocks = cute::min(
              windowed_k_blocks - idx_kv_split * num_blocks_per_split,
              num_blocks_per_split);
        }
      }

      if (num_effective_kv_blocks <= 0) {
        // flash-attn2: a sequence without keys gets O = 0 and LSE = -inf.
        // With several splits ReduceSplitK writes them; with one split the
        // output is O itself, written here.
        if (seq_len <= 0 && num_kv_splits <= 1) {
          const int q0 = blk_q * get<0>(TileShapeO{});
          const int v0 = blk_v * get<1>(TileShapeO{});
          const int rows = cute::min(int(get<0>(TileShapeO{})), head_group_q - q0);
          const int cols = cute::min(int(get<1>(TileShapeO{})), s.head_size_vo - v0);
          ElementO* o_tile = p.Oaccum + offset_o +
              (head_q_start + q0) * s.head_size_vo + v0;
          for (int i = thr_id; i < rows * cols; i += SGPerWG::value * intel::sg_size)
            o_tile[(i / cols) * s.head_size_vo + i % cols] = ElementO(0);
          if (p.softmax_lse != nullptr && blk_v == 0) {
            for (int r = thr_id; r < rows; r += SGPerWG::value * intel::sg_size)
              p.softmax_lse[(head_q_start + q0 + r) * p.lse_stride + offset_lse] =
                  -cutlass::platform::numeric_limits<float>::infinity();
          }
        }
        continue;
      }

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);
      auto ptrO = p.Oaccum + offset_o;
      auto ptrSoftmax_lse_accum =
          p.softmax_lse_accum + offset_softmax_lse_accum;

      // softmax_lse row base for this (kv head, sequence). Rows of the packed
      // GQA head-group are lse_stride apart. Only written here when no
      // ReduceSplitK pass follows; otherwise ReduceSplitK owns the write
      // because it holds the cross-split statistics.
      ElementLSE* ptrLSE =
          (p.softmax_lse != nullptr && num_kv_splits <= 1)
              ? p.softmax_lse + head_q_start * p.lse_stride + offset_lse
              : nullptr;

      // Q layout uses the tensor's actual head stride (get<2>(dQ)) so
      // non-contiguous Q (e.g. permuted heads or sliced from a wider
      // head-dim buffer) is read correctly. The GQA grouping splits the
      // num_heads_q dim into (num_heads_kv outer, head_group_q inner) which
      // assumes heads are regularly strided by q_stride_heads.
      auto layout_q = make_layout(
          shape_Q,
          make_stride(
              get<2>(p.dQ),
              _1{},
              static_cast<int>(head_group_q) * get<2>(p.dQ),
              get<3>(p.dQ)));
      auto layout_k = make_layout(
          shape_K, make_stride(get<0>(p.dK), _1{}, get<2>(p.dK), get<3>(p.dK)));
      auto layout_v = make_layout(
          shape_V, make_stride(_1{}, get<1>(p.dV), get<2>(p.dV), get<3>(p.dV)));

      auto layout_o = make_ordered_layout(shape_O, Step<_1, _0, _2, _3, _4>{});
      auto layout_softmax_lse_accum =
          make_ordered_layout(shape_softmax_lse_accum, Step<_1, _0, _2, _3>{});
      auto layout_sink = make_ordered_layout(shape_sink, Step<_1, _0>{});

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), layout_q);
      Tensor K = make_tensor(make_gmem_ptr(dcK), layout_k);
      Tensor V = make_tensor(make_gmem_ptr(dcV), layout_v);
      Tensor O = make_tensor(make_gmem_ptr(ptrO), layout_o);
      Tensor softmax_lse_accum = make_tensor(
          make_gmem_ptr(ptrSoftmax_lse_accum), layout_softmax_lse_accum);
      Tensor sinks = make_tensor(
          make_gmem_ptr(const_cast<ElementSink*>(p.sm_sink)), layout_sink);

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Main loop
      int l_coord = is_var_len ? 0 : idx_b;

      int start_blk = kv_split_offset;
      int end_blk = kv_split_offset + num_effective_kv_blocks;

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      // KV-head pair prefetch (full fp8, token-major caches) pays off unless
      // the work-group streams a single KV tile, or a large grid streams long
      // sequences (>= 2048 work-groups of >= 128 KV tiles each).
      const int wg_tiles = end_blk - start_blk;
      const int64_t grid_wgs =
          int64_t{GridDimX()} * int64_t{GridDimY()} * int64_t{GridDimZ()};
      const bool pair_kv_heads =
          wg_tiles >= 2 && !(grid_wgs >= 2048 && wg_tiles >= 128);

      mainloop.template operator()<LargeSurface>(
          Q(_, _, head, l_coord),
          K(_, _, head, l_coord),
          V(_, _, head, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          idx_b,
          pair_kv_heads ? head : -1,
          start_blk,
          end_blk,
          k_blocks,
          thr_id,
          seq_len,
          full_tile_offset,
          discard_seq_coord);

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      if constexpr (Sink) {
        auto sinks_per_kv = sinks(head, _);
        epilogue(
            O(_, _, head, idx_kv_split, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            thr_id,
            softmax_lse_accum(_, _, head, l_coord),
            idx_kv_split,
            head_group_q,
            sinks_per_kv,
            num_kv_splits,
            ptrLSE,
            p.lse_stride);
      } else {
        epilogue(
            O(_, _, head, idx_kv_split, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            thr_id,
            softmax_lse_accum(_, _, head, l_coord),
            idx_kv_split,
            head_group_q,
            sinks,
            num_kv_splits,
            ptrLSE,
            p.lse_stride);
      }
    }
  }
};

template <class ProblemShape_, class TileScheduler_, class FMHAKernel_>
class ReduceSplitK {
 public:
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
  using TileScheduler = TileScheduler_;
  static_assert(
      is_same_v<
          TileScheduler,
          cutlass::fmha::kernel::XeReduceSplitKTileScheduler>,
      "ReduceSplitK kernel requires XeReduceSplitKTileScheduler");
  using TileSchedulerParams = typename TileScheduler::Params;

  using ElementO = typename FMHAKernel_::ElementO;
  using StrideO = typename FMHAKernel_::StrideO;
  using TileShapeO = typename FMHAKernel_::TileShapeO;
  using TileShapeQK = typename FMHAKernel_::TileShapeQK;

  using ElementLSE = typename FMHAKernel_::ElementLSE;

  using SGPerWG = typename FMHAKernel_::SGPerWG;
  // Subgroups per reduce work-group. Tuned decode kernels use at least
  // head_dim / 64, so each work-item combines at most four outputs on the
  // vectorized path below (full fp8 decode work-groups have only 2 subgroups).
  static constexpr int kReduceSGs =
      FMHAKernel_::kTunedDecode
          ? cute::max(int(SGPerWG::value), int(get<1>(TileShapeO{})) / 64)
          : int(SGPerWG::value);

  // num values (head_dim) processed by each thread
  constexpr static int num_vals_per_thread =
      int(get<1>(TileShapeO{}) / (kReduceSGs * intel::sg_size));

  //
  // Types
  //

  struct KernelArguments {
    ProblemShape shape;
    // outputs:
    ElementO* O;
    StrideO dO;
    // below are inputs
    // TODO: whether same dtype as output or accum?
    const ElementO* Oaccum;
    StrideO dOaccum;
    const ElementLSE* softmax_lse_accum;
    StrideO dSoftmax_lse_accum;
    int window_size_left = -1;

    // per-batch mask: true = prefill, false = decode; nullptr = process all
    const bool* is_prefill;
    const int* splits_per_seq = nullptr;  // per-seq split counts
    // softmax_lse output [num_heads_q, total_seqlen_q]; nullptr = disabled.
    float* softmax_lse = nullptr;
    int lse_stride = 0;
    // flash-attn2: stride between query tokens in softmax_lse.
    int lse_row_stride = 1;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;  // no split by default
  };

  /// Params structure
  struct Params {
    KernelParams kernel;
    TileSchedulerParams scheduler;
  };

  struct SharedStorage {
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits>
        softmax_lse_slm_array;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

 public:
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    // only support decode
    if (!is_var_len && args.kernel.shape.seq_len_qo > 1) {
      return false;
    }

    if (args.num_kv_splits > FMHAKernel_::max_num_kv_splits) {
      return false;
    }
    return true;
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(kReduceSGs * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      auto q_len = cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength>{problem_shape.seq_len_qo}, batch);
      const int* kv_len = problem_shape.seq_len_kv.cumulative_length;
      return Shape<int, int>{
          get<0>(q_len),
          problem_shape.kv_cumulative ? kv_len[batch + 1] - kv_len[batch]
                                      : kv_len[batch]};
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(ThreadIdxX());
    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits;

    auto batch_dim = is_var_len ? 1 : s.batch;
    auto num_heads_q = s.num_heads_q;
    auto head_size_vo = s.head_size_vo;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [seq_idx, head_q, idx_b] = tile_scheduler.get_block_coord();

      // Skip prefill batches when is_prefill mask is provided
      if (p.is_prefill != nullptr && p.is_prefill[idx_b]) continue;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;

      // when varlen enabled, use largest seq_len_qo to decide work group num
      if (seq_idx >= seq_len_qo) continue;

      const int k_blocks = cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));
      // Sliding window: skip blocks before the window
      constexpr bool LocalMask = FMHAKernel_::CollectiveMainloop::LocalMask;
      const int k_block0 =
          LocalMask ? cute::max(seq_len_kv - 1 - p.window_size_left, 0) /
                          get<1>(TileShapeQK{})
                    : 0;
      const int windowed_k_blocks = k_blocks - k_block0;
      // Per-sequence adaptive split count
      int seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                  ? p.splits_per_seq[idx_b]
                                  : num_kv_splits;

      int num_blocks_per_split =
          cute::ceil_div(windowed_k_blocks, seq_num_kv_splits);

      // is_single_split is a heuristic owned by the FMHA kernel; when the
      // host provides splits_per_seq, the host has already applied the same
      // policy (see build_decode_split_plan in flash_attn_interface.py), so
      // trust it directly. Otherwise (legacy path), mirror the FMHA kernel.
      //
      // plan_driven also selects how empty splits are detected. In the legacy
      // path the FMHA kernel derives each split's tile range from the same
      // ceil_div(windowed_k_blocks, splits) formula, so
      // "i * num_blocks_per_split >= windowed_k_blocks" identifies exactly the
      // splits it skipped. The host plan instead partitions the tiles evenly
      // (base / base+1) and guarantees every emitted split owns at least one
      // tile, and that partition is *finer* than num_blocks_per_split; using
      // the tile-derived guard there can therefore drop a split that really
      // did produce output (e.g. 33 tiles over 8 splits).
      const bool plan_driven = (p.splits_per_seq != nullptr);
      int effective_splits;
      if (plan_driven) {
        effective_splits = seq_num_kv_splits;
      } else {
        bool is_single_split =
            (seq_num_kv_splits > 1) &&
            (windowed_k_blocks < FMHAKernel_::kMinBlocksForSplit);
        effective_splits = is_single_split ? 1 : seq_num_kv_splits;
      }

      int offset_o = 0, offset_o_accum = 0;
      int offset_softmax_lse_accum = 0;

      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;

        offset_o_accum = s.num_heads_q * s.head_size_vo * num_kv_splits *
                         qo_cumulative[idx_b];
        offset_softmax_lse_accum =
            s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto shape_O =
          make_shape(seq_len_qo, head_size_vo, num_heads_q, batch_dim);
      auto shape_Oaccum = is_var_len ? make_shape(
                                           seq_len_qo,
                                           head_size_vo,
                                           num_heads_q * num_kv_splits,
                                           batch_dim)
                                     : make_shape(
                                           seq_len_qo,
                                           head_size_vo,
                                           num_heads_q * num_kv_splits,
                                           batch_dim);

      auto shape_softmax_lse_accum =
          make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);

      auto dcOaccum = const_cast<ElementO*>(p.Oaccum + offset_o_accum);
      auto ptrSoftmax_lse_accum = const_cast<ElementLSE*>(
          p.softmax_lse_accum + offset_softmax_lse_accum);
      auto ptrO = p.O + offset_o;

      auto stride_o = is_var_len
                          ? cutlass::make_cute_packed_stride(StrideO{}, shape_O)
                          : p.dO;
      auto stride_o_accum =
          is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_Oaccum)
                     : p.dOaccum;
      auto stride_softmax_lse_accum =
          is_var_len ? cutlass::make_cute_packed_stride(
                           StrideO{}, shape_softmax_lse_accum)
                     : p.dSoftmax_lse_accum;

      Tensor Oaccum = make_tensor(
          make_gmem_ptr(dcOaccum), make_layout(shape_Oaccum, stride_o_accum));
      Tensor O =
          make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      Tensor softmax_lse_accum = make_tensor(
          make_gmem_ptr(ptrSoftmax_lse_accum),
          make_layout(shape_softmax_lse_accum, stride_softmax_lse_accum));

      int l_coord = is_var_len ? 0 : idx_b;

      // Splits [0, num_valid_splits) produced output. A legacy-path sequence
      // without KV blocks has no valid splits.
      const int num_valid_splits =
          plan_driven ? effective_splits
          : num_blocks_per_split > 0
              ? cute::min(
                    effective_splits,
                    cute::ceil_div(windowed_k_blocks, num_blocks_per_split))
              : 0;
      const bool split_thread_active = thr_id < num_valid_splits;

      // Load the per-split natural-log LSE values into SLM and find their
      // maximum. This matches CUDA FlashAttention's split-K combine contract.
      ElementLSE local_lse{
          cutlass::platform::numeric_limits<ElementLSE>::lowest()};
      ElementLSE global_max_lse{
          cutlass::platform::numeric_limits<ElementLSE>::lowest()};
      if (split_thread_active) {
        local_lse = softmax_lse_accum(seq_idx, thr_id, head_q, l_coord);
        global_max_lse = local_lse;
        shared_storage.softmax_lse_slm_array[thr_id] = local_lse;
      }

      sycl::group_barrier(get_work_group<3>());

      global_max_lse = reduce_over_group(
          get_work_group<1>(), global_max_lse, sycl::maximum<>());
      global_max_lse =
          sycl::group_broadcast(get_work_group<1>(), global_max_lse, 0);

      constexpr float kLog2e = 1.4426950408889634074f;
      ElementLSE local_weight{0};
      // Splits with every key masked report LSE = -inf; give them zero weight.
      if (split_thread_active && local_lse != ElementLSE(-INFINITY)) {
        local_weight = sycl::native::exp2(
            (local_lse - global_max_lse) * ElementLSE(kLog2e));
      }
      ElementLSE lse_exp_sum =
          reduce_over_group(get_work_group<1>(), local_weight, sycl::plus<>());
      lse_exp_sum = sycl::group_broadcast(get_work_group<1>(), lse_exp_sum, 0);
      // If every active split is fully masked (all local_weight == 0), the sum
      // is zero; guard the reciprocal so we write 0 instead of 1/0 == +inf,
      // which would otherwise yield 0 * inf == NaN below. Mirrors the rA_sum
      // guard in chunk_prefill_epilogue.hpp's DecodeFwdEpilogue.
      ElementLSE inv_lse_exp_sum = lse_exp_sum > ElementLSE(0)
                                       ? ElementLSE(1) / lse_exp_sum
                                       : ElementLSE(0);

      // Convert LSE values to normalized weights once per split. All output
      // elements reuse the SLM weights instead of recomputing exponentials.
      if (split_thread_active) {
        shared_storage.softmax_lse_slm_array[thr_id] =
            local_weight * inv_lse_exp_sum;
      }

      if (p.softmax_lse != nullptr && thr_id == 0) {
        int global_q = seq_idx;
        if constexpr (is_var_len) {
          global_q += s.seq_len_qo.cumulative_length[idx_b];
        } else {
          global_q += idx_b * seq_len_qo;
        }
        p.softmax_lse[head_q * p.lse_stride + global_q * p.lse_row_stride] =
            static_cast<float>(global_max_lse) +
            sycl::log(static_cast<float>(lse_exp_sum));
      }

      sycl::group_barrier(get_work_group<3>());

      // Combine the normalized partial outputs with the normalized split
      // weights. Loads for kSplitChunk splits and kVals output elements are
      // issued together so their latencies overlap (a serial loop over splits
      // made this kernel latency bound).
      constexpr int kWgThreads = kReduceSGs * intel::sg_size;
      constexpr int kVals = cute::max(
          1, (int(get<1>(TileShapeO{})) + kWgThreads - 1) / kWgThreads);
      constexpr int kSplitChunk = 8;
      bool combined = false;
      if constexpr (
          FMHAKernel_::kTunedDecode && (kVals == 2 || kVals == 4) &&
          sizeof(ElementO) == 2) {
        // kVals 16-bit outputs per work-item = kWords 32-bit words.
        constexpr int kWords = kVals / 2;
        const bool vec_ok =
            (reinterpret_cast<uintptr_t>(&Oaccum(seq_idx, 0, head_q, l_coord)) %
                 (4 * kWords) ==
             0) &&
            (reinterpret_cast<uintptr_t>(&O(seq_idx, 0, head_q, l_coord)) %
                 (4 * kWords) ==
             0) &&
            (get<2>(Oaccum.stride()) % kVals == 0) &&
            (s.head_size_vo % kVals == 0) &&
            // One pass of the work-group must cover the whole head: D512
            // tiles O 256 wide (decode_shapeout_v), so kVals * kWgThreads is
            // only half of head_size_vo there.
            (s.head_size_vo <= kVals * kWgThreads);
        if (vec_ok) {
          combined = true;
          // Each work-item owns kVals contiguous outputs and loads them with
          // kWords 32-bit loads per split. Index clamping plus zero weights
          // keep the unrolled split loop branch-free so all loads are in flight
          // at once.
          using VecT = uint64_t;
          const int idx0 = thr_id * kVals;
          if (idx0 < s.head_size_vo) {
            ElementLSE acc[kVals];
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kVals; ++v)
              acc[v] = ElementLSE(0);
            const ElementO* base = &Oaccum(seq_idx, idx0, head_q, l_coord);
            const int64_t split_stride =
                int64_t(num_heads_q) * get<2>(Oaccum.stride());
            for (int i0 = 0; i0 < num_valid_splits; i0 += kSplitChunk) {
              VecT raw[kSplitChunk];
              ElementLSE w[kSplitChunk];
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < kSplitChunk; ++j) {
                const int i = cute::min(i0 + j, num_valid_splits - 1);
                const uint32_t* p32 =
                    reinterpret_cast<const uint32_t*>(base + i * split_stride);
                raw[j] = VecT(p32[0]);
                if constexpr (kWords > 1) raw[j] |= VecT(p32[1]) << 32;
                w[j] = (i0 + j < num_valid_splits)
                           ? shared_storage.softmax_lse_slm_array[i0 + j]
                           : ElementLSE(0);
              }
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < kSplitChunk; ++j) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kVals; ++v) {
                  ElementLSE x = static_cast<ElementLSE>(ElementO::bitcast(
                      static_cast<uint16_t>(raw[j] >> (16 * v))));
                  // Empty splits may hold uninitialized data: select, not *0.
                  acc[v] += w[j] > ElementLSE(0) ? x * w[j] : ElementLSE(0);
                }
              }
            }
            VecT out_raw = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kVals; ++v)
              out_raw |= VecT(static_cast<ElementO>(acc[v]).raw()) << (16 * v);
            uint32_t* o32 =
                reinterpret_cast<uint32_t*>(&O(seq_idx, idx0, head_q, l_coord));
            o32[0] = static_cast<uint32_t>(out_raw);
            if constexpr (kWords > 1)
              o32[1] = static_cast<uint32_t>(out_raw >> 32);
          }
        }
      }
      if (!combined) {
        for (int idx0 = thr_id; idx0 < s.head_size_vo;
             idx0 += kVals * kWgThreads) {
          ElementLSE acc[kVals];
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVals; ++v)
            acc[v] = ElementLSE(0);

          for (int i0 = 0; i0 < num_valid_splits; i0 += kSplitChunk) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kSplitChunk; ++j) {
              const int i = i0 + j;
              if (i < num_valid_splits) {
                const ElementLSE weight =
                    shared_storage.softmax_lse_slm_array[i];
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kVals; ++v) {
                  const int idx = idx0 + v * kWgThreads;
                  if (idx < s.head_size_vo) {
                    ElementLSE o_accum_val = static_cast<ElementLSE>(Oaccum(
                        seq_idx, idx, i * num_heads_q + head_q, l_coord));
                    acc[v] += weight > ElementLSE(0) ? o_accum_val * weight
                                                     : ElementLSE(0);
                  }
                }
              }
            }
          }

          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVals; ++v) {
            const int idx = idx0 + v * kWgThreads;
            if (idx < s.head_size_vo) {
              O(seq_idx, idx, head_q, l_coord) = static_cast<ElementO>(acc[v]);
            }
          }
        }
      }
    }
  }
};

}  // namespace vllm_xpu_xe3

}  // namespace cutlass::fmha::kernel
