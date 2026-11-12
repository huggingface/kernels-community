// Paged decode launcher of vllm-xpu-kernels (internal main efc398e):
//   csrc/xpu/attn/xe_3/paged_decode.hpp  (DecodeKernelLauncher, PagedDecodeConfig)
//   csrc/xpu/attn/fmha_utils.hpp         (decode_policy_qpacked_head)
// fp16/bf16 only. Changes: the zero-size workspace is a null pointer
// (FMHAKernel/ReduceSplitK need none), softmax_lse takes a row stride, KV
// lengths may be cumulative (cu_seqlens_k), and paged_decode_args_t is
// flash_xpu_splitkv::vllm_decode_args_t.

#pragma once

#include <iostream>
#include <limits>

#include <sycl/sycl.hpp>
#include <c10/util/Exception.h>

#include "cutlass/cutlass.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

#include "../splitkv_decode.hpp"
#include "chunk_prefill_scheduler.hpp"
#include "chunk_prefill_epilogue.hpp"
#include "paged_decode_kernel.hpp"
#include "paged_decode_tuning.h"
#include "grf_mode.hpp"

#ifndef VLLM_GRF_SIZE
  #define VLLM_GRF_SIZE 256
#endif

namespace flash_xpu_splitkv::vllm {
inline namespace FLASH_XPU_VLLM_NS {
using namespace cute;

using paged_decode_args_t = flash_xpu_splitkv::vllm_decode_args_t;

// ---------------------------------------------------------------------------
// fmha_utils.hpp: decode policies
// ---------------------------------------------------------------------------
static constexpr int kDecodeMaxShapeOutV = 256;
static constexpr int kDecodeAccBudget = 2048;

template <typename q_packed, typename head_dim>
constexpr int decode_shapeout_v_value() {
  if constexpr (head_dim::value <= kDecodeMaxShapeOutV) {
    return head_dim::value;
  } else {
    constexpr int budgeted = kDecodeAccBudget / q_packed::value;
    return budgeted < kDecodeMaxShapeOutV ? budgeted : kDecodeMaxShapeOutV;
  }
}

template <typename q_packed, typename head_dim>
using decode_shapeout_v =
    cute::Int<decode_shapeout_v_value<q_packed, head_dim>()>;

template <typename q_packed, typename head_dim, typename kv_tile>
struct decode_policy_qpacked_head {
  static_assert(
      cute::is_same_v<kv_tile, _16> || cute::is_same_v<kv_tile, _32> ||
          cute::is_same_v<kv_tile, _64> || cute::is_same_v<kv_tile, _128>,
      "Unsupported kv_tile for decode_policy_qpacked_head "
      "(supported: _16, _32, _64, _128)");
};

// kv_tile == _64; any page size that is a positive multiple of 64.
template <typename q_packed, typename head_dim>
struct decode_policy_qpacked_head<q_packed, head_dim, _64> {
  using HeadDim = head_dim;
  using ShapeQK = Shape<q_packed, _64, _64>;
  using ShapePV = Shape<q_packed, _32, _64>;
  using ShapeOut = Shape<q_packed, decode_shapeout_v<q_packed, head_dim>>;
  using SubgroupLayoutQK = Layout<Shape<_1, _4, _1>>;
};

// head_dim 256, GQA ratio 9..16: one 16-row work-group reads each K/V tile
// once for all 16 query heads, where two 8-row work-groups would each stream
// it. Subgroups are 2 (Q rows) x 4 (KV slices), so each keeps the 8 x 256
// accumulator of the 8-row policy.
struct decode_policy_q16_h256_p64 : decode_policy_qpacked_head<_16, _256, _64> {
  using SubgroupLayoutQK = Layout<Shape<_2, _4, _1>>;
};

// ---------------------------------------------------------------------------
// paged_decode.hpp
// ---------------------------------------------------------------------------
template <class FMHAKernel, class ReductionSplitKernel, bool isVarLen>
struct DecodeKernelLauncher {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;
  using ElementLSE = typename FMHAKernel::ElementLSE;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::DecodeProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideO stride_Oaccum;
  StrideO stride_softmax_lse_accum;

  int num_kv_splits;

  ProblemShapeType initialize(const paged_decode_args_t& args) {
    ProblemShapeType shape;
    ProblemShapeTypeInit shape_init;
    auto batch = shape.batch = shape_init.batch = args.batch_size;
    auto num_heads_q = shape.num_heads_q = shape_init.num_heads_q =
        args.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv = shape_init.num_heads_kv =
        args.num_heads_k;
    auto head_size_qk = shape.head_size_qk = shape_init.head_size_qk =
        args.head_size;
    auto head_size_vo = shape.head_size_vo = shape_init.head_size_vo =
        args.v_head_size;

    if constexpr (isVarLen) {
      batch = shape_init.batch = 1;
      shape_init.seq_len_qo = args.total_seqlen_q;
      shape_init.seq_len_kv = args.total_seqlen_k;

      shape.seq_len_qo =
          cutlass::fmha::collective::VariableLength{args.max_queries};
      shape.seq_len_qo.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_q);
      shape.seq_len_kv =
          cutlass::fmha::collective::VariableLength{args.max_keys};
      shape.seq_len_kv.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_k);
      shape.kv_cumulative = args.kv_cumulative;
    } else {
      shape.seq_len_qo = shape_init.seq_len_qo = args.max_queries;
      shape.seq_len_kv = shape_init.seq_len_kv = args.max_keys;
    }

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

    num_kv_splits = args.num_kv_splits;

    if (args.q_stride_seq > 0 || args.q_stride_heads > 0 ||
        args.q_stride_batch > 0) {
      constexpr int64_t kIntMax =
          static_cast<int64_t>(std::numeric_limits<int>::max());
      TORCH_CHECK(
          args.q_stride_seq <= kIntMax && args.q_stride_heads <= kIntMax &&
              args.q_stride_batch <= kIntMax,
          "Q stride exceeds int32 max (",
          kIntMax,
          "): q_stride_seq=",
          args.q_stride_seq,
          " q_stride_heads=",
          args.q_stride_heads,
          " q_stride_batch=",
          args.q_stride_batch);
      stride_Q = StrideQ{
          static_cast<int>(args.q_stride_seq),
          _1{},
          static_cast<int>(args.q_stride_heads),
          static_cast<int>(args.q_stride_batch)};
    } else {
      stride_Q = cutlass::make_cute_packed_stride(
          StrideQ{},
          cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch));
    }
    if (args.k_stride_seq > 0) {
      constexpr int64_t kIntMax =
          static_cast<int64_t>(std::numeric_limits<int>::max());
      TORCH_CHECK(
          args.k_stride_seq <= kIntMax && args.k_stride_heads <= kIntMax &&
              args.k_stride_page <= kIntMax && args.v_stride_seq <= kIntMax &&
              args.v_stride_heads <= kIntMax && args.v_stride_page <= kIntMax,
          "KV cache stride exceeds int32 max (",
          kIntMax,
          "): k_stride_seq=",
          args.k_stride_seq,
          " k_stride_heads=",
          args.k_stride_heads,
          " k_stride_page=",
          args.k_stride_page,
          " v_stride_seq=",
          args.v_stride_seq,
          " v_stride_heads=",
          args.v_stride_heads,
          " v_stride_page=",
          args.v_stride_page);
      stride_K = StrideK{
          static_cast<int>(args.k_stride_seq),
          _1{},
          static_cast<int>(args.k_stride_heads),
          static_cast<int>(args.k_stride_page)};
      stride_V = StrideV{
          _1{},
          static_cast<int>(args.v_stride_seq),
          static_cast<int>(args.v_stride_heads),
          static_cast<int>(args.v_stride_page)};
    } else {
      stride_K = cutlass::make_cute_packed_stride(
          StrideK{},
          cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
      stride_V = cutlass::make_cute_packed_stride(
          StrideV{},
          cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
    }
    stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));
    stride_Oaccum = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(
            seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch));

    stride_softmax_lse_accum = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    return shape;
  }

  cutlass::Status
  run(sycl::queue& queue,
      const paged_decode_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            reinterpret_cast<ElementQ*>(args.query),
            stride_Q,
            reinterpret_cast<ElementK*>(args.key),
            stride_K,
            reinterpret_cast<ElementV*>(args.value),
            stride_V,
            reinterpret_cast<ElementO*>(args.tem_out),
            stride_Oaccum,
            reinterpret_cast<ElementLSE*>(args.softmax_lse_accum),
            stride_softmax_lse_accum,
            reinterpret_cast<ElementQ*>(args.sm_sink),
            static_cast<const bool*>(args.is_prefill),
            args.splits_per_seq,
            args.softmax_lse,
            args.lse_stride,
            args.lse_row_stride,
        },
        {args.sm_scale,
         args.q_scale,
         args.k_scale,
         args.v_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right,
         args.page_stride_elements,
         args.spec_q_len,
         args.num_heads_q / args.num_heads_k / args.spec_q_len},
        {},
        hw_info,
        args.num_kv_splits};

    typename ReductionSplitKernel::Arguments reduce_arg{
        {shape,
         reinterpret_cast<ElementO*>(args.out),
         stride_O,
         reinterpret_cast<ElementO*>(args.tem_out),
         stride_Oaccum,
         reinterpret_cast<ElementLSE*>(args.softmax_lse_accum),
         stride_softmax_lse_accum,
         args.window_size_left,
         static_cast<const bool*>(args.is_prefill),
         args.splits_per_seq,
         args.softmax_lse,
         args.lse_stride,
         args.lse_row_stride},
        hw_info,
        args.num_kv_splits};

    // Neither kernel needs a workspace (get_workspace_size() == 0).
    uint8_t* workspace = nullptr;

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << args.batch_size << 'x'
                << args.num_heads_q << 'x' << args.max_queries << 'x'
                << args.max_keys << 'x' << args.head_size << 'x'
                << args.head_size << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    FMHAKernel::initialize_workspace(arguments, workspace);

    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace);
    auto reduce_params =
        ReductionSplitKernel::to_underlying_arguments(reduce_arg, workspace);

    if (args.work_list != nullptr && args.total_wgs > 0) {
      params.scheduler.work_list =
          reinterpret_cast<const cutlass::fmha::kernel::DecodeWorkItem*>(
              args.work_list);
      params.scheduler.total_wgs = args.total_wgs;
      params.scheduler.grid.z = args.total_wgs * args.num_heads_k;
    }

    ReductionSplitKernel::initialize_workspace(reduce_arg, workspace);
    run(queue, params, reduce_params, args.num_kv_splits > 1);

    return cutlass::Status::kSuccess;
  }

  static void
  run(sycl::queue& queue,
      typename FMHAKernel::Params params,
      typename ReductionSplitKernel::Params reduce_params,
      bool need_reduce) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    int smem_size = FMHAKernel::SharedStorageSize;
    if constexpr (FMHAKernel::kTunedDecode) {
      // Cap work-groups per Xe core by reserving SLM (paged_decode_tuning.h).
      // grid.y is the number of q-tiles per KV head.
      const auto device = queue.get_device();
      const int local_mem_size = static_cast<int>(
          device.get_info<sycl::info::device::local_mem_size>());
      const int num_xe_cores =
          device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
          device.get_info<
              sycl::ext::intel::info::device::gpu_subslices_per_slice>();
      constexpr int kv_tile = get<1>(typename FMHAKernel::TileShapeQK{});
      const int max_seqlen_kv = params.kernel.shape.seq_len_kv;
      const int kv_tiles =
          static_cast<int>((int64_t{max_seqlen_kv} + kv_tile - 1) / kv_tile);
      const int wgs_per_core =
          flash_xpu_splitkv::paged_decode_tuning::occupancy_cap(
              FMHAKernel::kHeadDimBucket,
              FMHAKernel::kPrecision,
              cute::ceil_div(
                  kv_tiles, std::max(params.scheduler.num_kv_splits_, 1)),
              int64_t{grid.x} * int64_t{grid.y} * int64_t{grid.z},
              num_xe_cores,
              grid.y,
              FMHAKernel::SGPerWG::value);
      if (wgs_per_core > 0) {
        smem_size =
            std::max(smem_size, (local_mem_size / wgs_per_core) & ~1023);
      }
    }

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        VLLM_GRF_PROP(VLLM_GRF_SIZE)};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
        policy, queue, params);

    if (need_reduce) {
      dim3 const reduce_grid =
          ReductionSplitKernel::get_grid_shape(reduce_params);
      int reduce_smem_size = ReductionSplitKernel::SharedStorageSize;
      const auto reduce_block = ReductionSplitKernel::get_block_shape();
      const auto reduce_sycl_block =
          compat::dim3(reduce_block.x, reduce_block.y, reduce_block.z);
      const auto reduce_sycl_grid =
          compat::dim3(reduce_grid.x, reduce_grid.y, reduce_grid.z);
      compat::experimental::launch_properties launch_props_reduce{
          syclex::work_group_scratch_size(reduce_smem_size),
      };
      compat::experimental::launch_policy reduce_policy{
          reduce_sycl_grid,
          reduce_sycl_block,
          launch_props_reduce,
          kernel_props};

      compat::experimental::launch<
          cutlass::device_kernel<ReductionSplitKernel>>(
          reduce_policy, queue, reduce_params);
    }
  }
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_, /* void -> default */
    int PipelineStages,
    bool Causal = false,
    bool Local = false,
    bool Sink = false,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = float,
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename StrideOaccum = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct PagedDecodeConfig {
  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>,
      MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<
      is_void_v<SubgroupLayoutPV_>,
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
      SubgroupLayoutPV_>;

  template <class Scheduler>
  static void run(sycl::queue& queue, const paged_decode_args_t& args) {
    constexpr bool VarLen = true;
    constexpr bool Paged = true;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<VarLen>;

    using TiledMMAQK = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapeQK>,
        SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapePV>,
        SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(
          make_gmem_ptr(&val),
          make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideOaccum{}));
    using TensorLSE = decltype(make_dummy_tensor(float{}, StrideO{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::DecodeFwdMainloop<
        MainloopDispatchPolicy,
        Paged,
        Causal,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV,
        Local>;

    using CollectiveEpilogue = cutlass::fmha::collective::DecodeFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        TensorLSE,
        void,
        Sink>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    using ReduceSplitKernel = cutlass::fmha::kernel::ReduceSplitK<
        ProblemShapeType,
        cutlass::fmha::kernel::XeReduceSplitKTileScheduler,
        FMHAKernel>;

    // A K/V pool taller than one Xe 2D block-load surface needs the kernel
    // variant that splits it into sub-surfaces. The split-K reduction does not
    // read K/V, so both variants share it.
    if (args.total_seqlen_k > cutlass::fmha::collective::kXeMaxSurfaceRows) {
      using LargeSurfaceKernel = cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<
          ProblemShapeType,
          CollectiveMainloop,
          CollectiveEpilogue,
          Scheduler,
          true>;
      DecodeKernelLauncher<LargeSurfaceKernel, ReduceSplitKernel, VarLen>
          launcher;
      launcher.run(queue, args, hw_info);
      return;
    }

    DecodeKernelLauncher<FMHAKernel, ReduceSplitKernel, VarLen> launcher;

    launcher.run(queue, args, hw_info);
  }

  static void
  kernel_dispatch(sycl::queue& queue, const paged_decode_args_t& args) {
    return run<cutlass::fmha::kernel::DecodeTileScheduler>(queue, args);
  }
};

// decode_policy_dispatch_impl / dispatch_by_page_size /
// cutlass_paged_decode_impl, for Q = K = V = O = Element, no causal mask (as
// vllm passes for decode), no sink, page size a multiple of 64.
template <typename decode_policy, bool Local, typename Element>
void decode_policy_dispatch_impl(
    sycl::queue& queue, const paged_decode_args_t& args) {
  const int PipelineStages = 1;
  PagedDecodeConfig<
      typename decode_policy::ShapeQK,
      typename decode_policy::ShapePV,
      typename decode_policy::ShapeOut,
      typename decode_policy::SubgroupLayoutQK,
      void,
      PipelineStages,
      false,
      Local,
      false,
      Element,
      Element,
      Element,
      Element>::kernel_dispatch(queue, args);
}

template <typename QGroup, typename Element, int HeadDim>
void dispatch_local(sycl::queue& queue, const paged_decode_args_t& args) {
  using policy = cute::conditional_t<
      HeadDim == 256 && cute::is_same_v<QGroup, _16>,
      decode_policy_q16_h256_p64,
      decode_policy_qpacked_head<QGroup, Int<HeadDim>, _64>>;
  if (args.is_local) {
    decode_policy_dispatch_impl<policy, true, Element>(queue, args);
  } else {
    decode_policy_dispatch_impl<policy, false, Element>(queue, args);
  }
}

template <typename Element, int HeadDim>
void paged_decode(sycl::queue& queue, const paged_decode_args_t& args) {
  TORCH_CHECK(
      args.block_size > 0 && args.block_size % 64 == 0,
      "paged decode: page size must be a multiple of 64");
  // paged_decode_xe3.cpp: head sizes above 128 use the 8-row tile (the
  // 16-row reduction needs 64 KB SLM at head_dim 256 and hangs), except
  // head_dim 256 with GQA ratio 9..16, which uses decode_policy_q16_h256_p64.
  const int group = args.num_heads_q / args.num_heads_k;
  if constexpr (HeadDim == 256) {
    if (group > 8 && group <= 16) {
      dispatch_local<_16, Element, HeadDim>(queue, args);
      return;
    }
  }
  if (group <= 8 || HeadDim > 128) {
    dispatch_local<_8, Element, HeadDim>(queue, args);
  } else {
    if constexpr (HeadDim <= 128) {
      dispatch_local<_16, Element, HeadDim>(queue, args);
    }
  }
}

}  // namespace FLASH_XPU_VLLM_NS
}  // namespace flash_xpu_splitkv::vllm

#define FLASH_XPU_VLLM_INSTANTIATE(HD, NAME, ELEMENT)                         \
  namespace flash_xpu_splitkv {                                              \
  template <>                                                                \
  void vllm_paged_decode_##NAME<FLASH_XPU_SPLITKV_TARGET_ARCH, HD>(          \
      sycl::queue & q, const vllm_decode_args_t& a) {                        \
    vllm::paged_decode<ELEMENT, HD>(q, a);                                   \
  }                                                                          \
  }
