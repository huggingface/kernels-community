#pragma once

#include "fmha_fwd_types.hpp"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "./collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>

#include "cutlass/util/device_memory.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "./collective/fmha_fwd_scheduler.hpp"
#include "./collective/fmha_fwd_epilogue.hpp"
#include "./kernel/fmha_fwd_kernel.hpp"

// Fast-path (SDPA-equivalent) kernel for non-varlen / non-paged / non-local /
// no-dropout configurations. Forked to keep binary small enough to avoid IGC
// register spill on BMG.
#include "./collective/fmha_fwd_mainloop_fast.hpp"
#include "./collective/fmha_fwd_epilogue_fast.hpp"
#include "./kernel/fmha_fwd_kernel_fast.hpp"

#include "fmha_utils.hpp"

using namespace cute;

template <class FMHAKernel, bool isVarLen>
struct KernelLauncher {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::FMHAProblemShape<false>;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  ProblemShapeType initialize(const fmha_fwd_args_t& args) {
    ProblemShapeType shape;
    ProblemShapeTypeInit shape_init;

    // Common dimensions shared by both shape types
    const auto num_heads_q  = args.num_heads_q;
    const auto num_heads_kv = args.num_heads_k;
    const auto head_size_qk = args.head_size;
    const auto head_size_vo = args.head_size;

    shape.batch       = shape_init.batch       = args.batch_size;
    shape.num_heads_q = shape_init.num_heads_q = num_heads_q;
    shape.num_heads_kv = shape_init.num_heads_kv = num_heads_kv;
    shape.head_size_qk = shape_init.head_size_qk = head_size_qk;
    shape.head_size_vo = shape_init.head_size_vo = head_size_vo;

    auto batch = shape.batch;

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
    } else {
      shape.seq_len_qo = shape_init.seq_len_qo = args.max_queries;
      shape.seq_len_kv = shape_init.seq_len_kv = args.max_keys;
    }

    const auto seq_len_qo = shape_init.seq_len_qo;
    const auto seq_len_kv = shape_init.seq_len_kv;

    if constexpr (isVarLen) {
      stride_Q = cutlass::make_cute_packed_stride(
          StrideQ{},
          cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch));
      stride_K = cutlass::make_cute_packed_stride(
          StrideK{},
          cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
      stride_V = cutlass::make_cute_packed_stride(
          StrideV{},
          cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
      stride_O = cutlass::make_cute_packed_stride(
          StrideO{},
          cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));
    } else {
      stride_Q = StrideQ{num_heads_q * head_size_qk, _1{}, head_size_qk, seq_len_qo * num_heads_q * head_size_qk};
      stride_K = StrideK{num_heads_kv * head_size_qk, _1{}, head_size_qk, seq_len_kv * num_heads_kv * head_size_qk};
      stride_V = StrideV{_1{}, num_heads_kv * head_size_vo, head_size_vo, seq_len_kv * num_heads_kv * head_size_vo};
      stride_O = StrideO{num_heads_q * head_size_vo, _1{}, head_size_vo, seq_len_qo * num_heads_q * head_size_vo};
    }

    return shape;
  }

  cutlass::Status
  run(sycl::queue& queue, 
      const fmha_fwd_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    // Calculate LSE strides: LSE layout is (batch, num_heads, seqlen_q)
    // For both varlen and non-varlen, use max_queries as the stride
    int lse_stride_head = args.max_queries;
    int lse_stride_batch = args.num_heads_q * lse_stride_head;

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<ElementQ*>(args.query),
         stride_Q,
         reinterpret_cast<ElementK*>(args.key),
         stride_K,
         reinterpret_cast<ElementV*>(args.value),
         stride_V,
         reinterpret_cast<ElementO*>(args.out),
         stride_O},
        {args.sm_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right,
         args.p_dropout,
         args.philox_seed,
         args.philox_offset,
         args.s_dmask,
         args.seqlen_q_rounded,
         args.seqlen_k_rounded},
        {reinterpret_cast<float*>(args.softmax_lse),
         lse_stride_head,
         lse_stride_batch},
        hw_info};

    // Workspace is always 0 for forward kernels — skip allocation
    auto params =
        FMHAKernel::to_underlying_arguments(arguments, nullptr);

    run(queue, params);

    return cutlass::Status::kSuccess;
  }

  static void run(sycl::queue& queue, typename FMHAKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
        policy, queue, params);
  }
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_,
    int PipelineStages,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = bfloat16_t,
    typename MMAOperation_ = void,
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void,
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct FMHAConfig {
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

  template <
      class Scheduler,
      bool VarLen,
      bool Paged,
      bool Causal,
      bool Local,
      bool Dropout = false>
  static void run(sycl::queue& queue, const fmha_fwd_args_t& args) {
    cutlass::KernelHardwareInfo hw_info;

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<VarLen>;

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
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        Local,
        Paged,
        Dropout,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV>;

    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    KernelLauncher<FMHAKernel, VarLen> launcher;

    launcher.run(queue, args, hw_info);
  }

  //
  // Fast path: SDPA-equivalent, forked kernel. Only used when
  //   !is_varlen && !is_paged && !is_local && p_dropout == 0.
  //
  template <class Scheduler, bool Causal>
  static void run_fast(sycl::queue& queue, const fmha_fwd_args_t& args) {
    cutlass::KernelHardwareInfo hw_info;

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<false>;

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
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeFast<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloopFast<
        MainloopDispatchPolicy,
        Causal,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV>;

    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogueFast<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernelFast<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    using ElementS = typename CollectiveMainloop::ElementS;

    // Problem shape
    ProblemShapeType shape;
    shape.batch = args.batch_size;
    shape.num_heads_q = args.num_heads_q;
    shape.num_heads_kv = args.num_heads_k;
    shape.head_size_qk = args.head_size;
    shape.head_size_vo = args.head_size;
    shape.seq_len_qo = args.max_queries;
    shape.seq_len_kv = args.max_keys;

    // Contiguous (B,S,H,D) layout strides.
    int64_t head_size = args.head_size;
    int64_t nh_q = args.num_heads_q;
    int64_t nh_kv = args.num_heads_k;
    int64_t sq = args.max_queries;
    int64_t sk = args.max_keys;

    int64_t q_row_stride = nh_q * head_size;
    int64_t q_head_stride = head_size;
    int64_t q_batch_stride = sq * nh_q * head_size;

    int64_t k_row_stride = nh_kv * head_size;
    int64_t k_head_stride = head_size;
    int64_t k_batch_stride = sk * nh_kv * head_size;

    int64_t v_row_stride = nh_kv * head_size;
    int64_t v_head_stride = head_size;
    int64_t v_batch_stride = sk * nh_kv * head_size;

    int64_t o_row_stride = nh_q * head_size;
    int64_t o_head_stride = head_size;
    int64_t o_batch_stride = sq * nh_q * head_size;

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<const ElementQ*>(args.query),
         q_batch_stride, q_head_stride, q_row_stride,
         reinterpret_cast<const ElementK*>(args.key),
         k_batch_stride, k_head_stride, k_row_stride,
         reinterpret_cast<const ElementV*>(args.value),
         v_batch_stride, v_head_stride, v_row_stride,
         reinterpret_cast<ElementO*>(args.out),
         o_batch_stride, o_head_stride, o_row_stride,
         reinterpret_cast<float*>(args.softmax_lse)},
        {static_cast<ElementS>(args.sm_scale)},
        {},
        hw_info};

    auto params = FMHAKernel::to_underlying_arguments(arguments, nullptr);

    // Launch (same pattern as KernelLauncher::run).
    {
      namespace syclex = sycl::ext::oneapi::experimental;
      namespace intelex = sycl::ext::intel::experimental;

      dim3 const block = FMHAKernel::get_block_shape();
      dim3 const grid = FMHAKernel::get_grid_shape(params);

      int smem_size = FMHAKernel::SharedStorageSize;

      const auto sycl_block = compat::dim3(block.x, block.y, block.z);
      const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

      compat::experimental::launch_properties launch_props{
          syclex::work_group_scratch_size(smem_size),
      };
      compat::experimental::kernel_properties kernel_props{
          syclex::sub_group_size<cute::intel::sg_size>,
          intelex::grf_size<256>};
      compat::experimental::launch_policy policy{
          sycl_grid, sycl_block, launch_props, kernel_props};
      compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
          policy, queue, params);
    }
  }

  // Returns true if the current call matches the fast-path feature set.
  static bool can_use_fast_path(const fmha_fwd_args_t& args) {
    return !args.is_varlen && !args.is_paged && !args.is_local &&
        args.p_dropout == 0.0f;
  }

  // Dispatch the fast path by expanding the causal boolean at compile time.
  template <class Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>
  static void
  fast_dispatch(sycl::queue& queue, const fmha_fwd_args_t& args) {
    if (args.is_causal) {
      run_fast<Scheduler, true>(queue, args);
    } else {
      run_fast<Scheduler, false>(queue, args);
    }
  }

  template <bool... Bs>
  static void
  kernel_dispatch(sycl::queue& queue, const fmha_fwd_args_t& args) {
    return run<cutlass::fmha::kernel::XeFHMAIndividualTileScheduler, Bs...>(queue, args);
  }

  template <bool... Bs, typename... Ts>
  static void kernel_dispatch(sycl::queue& queue, const fmha_fwd_args_t& args, bool b, Ts... ts) {
    if (b) {
      kernel_dispatch<Bs..., true>(queue, args, ts...);
    } else {
      kernel_dispatch<Bs..., false>(queue, args, ts...);
    }
  }
};

template <typename chunk_policy, int PipelineStages, int IsVarLen = -1, int IsPaged = -1>
void policy_dispatch(sycl::queue& queue, CutlassType cuType, const fmha_fwd_args_t& args) {
  auto dispatch = [&]<typename ElemT>(ElemT) {
    using Config = FMHAConfig<
        typename chunk_policy::ShapeQK,
        typename chunk_policy::ShapePV,
        typename chunk_policy::ShapeOut,
        typename chunk_policy::SubgroupLayoutQK,
        void,
        PipelineStages,
        ElemT, ElemT, ElemT, ElemT>;

    // Fast path: minimal SDPA-equivalent kernel for the common case.
    // Only considered when the caller has not statically forced varlen/paged.
    if constexpr (
        (IsVarLen == -1 || IsVarLen == 0) &&
        (IsPaged == -1 || IsPaged == 0)) {
      if (Config::can_use_fast_path(args)) {
        return Config::fast_dispatch(queue, args);
      }
    }

    if constexpr (IsVarLen != -1 && IsPaged != -1) {
      return Config::template kernel_dispatch<IsVarLen, IsPaged>(
          queue, args, args.is_causal, args.is_local);
    } else {
      return Config::kernel_dispatch(
          queue, args, args.is_varlen, args.is_paged, args.is_causal, args.is_local);
    }
  };

  if (cuType == CutlassType::half) {
    dispatch(half_t{});
  } else {
    dispatch(bfloat16_t{});
  }
}
