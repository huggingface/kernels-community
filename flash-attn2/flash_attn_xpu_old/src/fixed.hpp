#pragma once

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/device_memory.h"
#include <cute/tensor.hpp>

#include "./compat_wrapper.hpp"
#include "./kernel/fixed_scheduler.hpp"
#include "./kernel/fixed_kernel.hpp"
#include "./collective/fmha_fusion.hpp"
#include "./collective/fixed_epilogue.hpp"
#include "./collective/fixed_softmax_epilogue.hpp"

#include "fmha_utils.hpp"

namespace cutlass::flash_attention::fixed {

using namespace cute;

// Fixed length specific arguments
struct prefill_args_fixed_t {
  void* query;
  void* key;
  void* value;
  void* out;
  float softmax_scale;
  int num_heads_q;
  int num_heads_kv;
  int head_size;
  bool is_causal;
  int batch_size;
  int seq_len_q;
  int seq_len_k;
  int window_size_left;
  int window_size_right;
  bool is_local;
};

template <class FMHAPrefillKernel>
struct KernelLauncher {
  using StrideQ = typename FMHAPrefillKernel::StrideQ;
  using StrideK = typename FMHAPrefillKernel::StrideK;
  using StrideV = typename FMHAPrefillKernel::StrideV;
  using StrideO = typename FMHAPrefillKernel::StrideO;

  using ElementQ = typename FMHAPrefillKernel::ElementQ;
  using ElementK = typename FMHAPrefillKernel::ElementK;
  using ElementV = typename FMHAPrefillKernel::ElementV;

  using CollectiveEpilogue = typename FMHAPrefillKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;

  using ProblemShapeType = typename FMHAPrefillKernel::ProblemShape;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  ProblemShapeType initialize(const prefill_args_fixed_t& args) {
    auto problem_shape = cute::make_tuple(
        args.batch_size, args.num_heads_q, args.num_heads_kv, 
        args.seq_len_q, args.seq_len_k, args.head_size, args.head_size);

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, 
        cute::make_shape(args.seq_len_q, args.head_size, args.batch_size * args.num_heads_q));
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, 
        cute::make_shape(args.seq_len_k, args.head_size, args.batch_size * args.num_heads_kv));
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, 
        cute::make_shape(args.head_size, args.seq_len_k, args.batch_size * args.num_heads_kv));
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, 
        cute::make_shape(args.seq_len_q, args.head_size, args.batch_size * args.num_heads_q));

    return problem_shape;
  }

  cutlass::Status run(const prefill_args_fixed_t& args, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = initialize(args);

    typename FMHAPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {reinterpret_cast<ElementQ*>(args.query), stride_Q,
         reinterpret_cast<ElementK*>(args.key), stride_K,
         reinterpret_cast<ElementV*>(args.value), stride_V,
         args.window_size_left, args.window_size_right,},
        {args.softmax_scale},
        {reinterpret_cast<ElementOutput*>(args.out), stride_O},
        hw_info};

    return run_kernel(arguments);
  }

private:
  cutlass::Status run_kernel(typename FMHAPrefillKernel::Arguments& arguments) {
    // Define device-global scratch memory
    size_t workspace_size = FMHAPrefillKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAPrefillKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHAPrefillKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto params = FMHAPrefillKernel::to_underlying_arguments(arguments, workspace.get());

    // Run the Flash Attention implementation.
    run_device_kernel(params);

    return cutlass::Status::kSuccess;
  }

public:
  static void run_device_kernel(typename FMHAPrefillKernel::Params params) {
    dim3 const block = FMHAPrefillKernel::get_block_shape();
    dim3 const grid = FMHAPrefillKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAPrefillKernel::SharedStorageSize;

    const auto sycl_block = COMPAT::dim3(block.x, block.y, block.z);
    const auto sycl_grid = COMPAT::dim3(grid.x, grid.y, grid.z);

// Launch parameters depend on whether SYCL compiler supports work-group scratch memory extension
#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace COMPAT::experimental;
    auto event = launch<cutlass::device_kernel<FMHAPrefillKernel>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<FMHAPrefillKernel::DispatchPolicy::SubgroupSize>}},
        params);
#else
    COMPAT::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    COMPAT::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<
            FMHAPrefillKernel::DispatchPolicy::SubgroupSize>};
    COMPAT::experimental::launch_policy policy{sycl_grid, sycl_block,
                                                   launch_props, kernel_props};
#if defined(OLD_API)
    auto event = COMPAT::experimental::launch<cutlass::device_kernel<FMHAPrefillKernel>>(policy, params);
#else
    auto event = COMPAT::experimental::launch<cutlass::device_kernel<FMHAPrefillKernel>, FMHAPrefillKernel>(policy, params);
#endif
#endif

    EventManager::getInstance().addEvent(event);
  }
};

template <typename TileShapeQK, typename TileShapePV, typename TileShapeOutput,
          typename SubgroupLayout, int PipelineStages,
          typename ElementInputQ = bfloat16_t,
          typename ElementInputKV = bfloat16_t, 
          typename MMAOperation = XE_8x16x16_F32BF16BF16F32_TT,
          typename ElementOutput = bfloat16_t,
          typename GmemTiledCopyStore = XE_2D_U16x8x16_ST_N,
          typename GmemTiledCopyQ = XE_2D_U16x8x32_LD_N,
          typename GmemTiledCopyK = XE_2D_U16x16x16_LD_T,
          typename GmemTiledCopyV = XE_2D_U16x16x32_LD_V,
          typename ElementAccumulator = float,
          typename ElementComputeEpilogue = float>
struct FMHAKernel {
  template <bool Causal, bool Local, class Scheduler>
  static void run_impl(const prefill_args_fixed_t& args) {
    cutlass::KernelHardwareInfo hw_info;

    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    
    using CollectiveEpilogue =
        cutlass::flash_attention::collective::FlashPrefillEpilogue<
            EpilogueDispatchPolicy, MMAOperation, TileShapeOutput,
            SubgroupLayout, ElementComputeEpilogue, ElementOutput,
            cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput,
            GmemTiledCopyStore>;
    
    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<
            Causal, Local, EpilogueDispatchPolicy, ElementAccumulator>;

    using ProblemShape = cute::tuple<int, int, int, int, int, int, int>;

    using CollectiveMainloop =
        cutlass::flash_attention::collective::FlashPrefillMma<
            GEMMDispatchPolicy, ProblemShape, ElementInputQ,
            cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation, TileShapeQK,
            TileShapePV, SubgroupLayout, GmemTiledCopyQ, GmemTiledCopyK, 
            GmemTiledCopyV, Causal, Local>;

    using FMHAPrefillKernel = cutlass::flash_attention::kernel::FMHAPrefill<
        ProblemShape, CollectiveMainloop, CollectiveSoftmaxEpilogue,
        CollectiveEpilogue, Scheduler>;

    KernelLauncher<FMHAPrefillKernel> launcher;
    launcher.run(args, hw_info);
  }

  static void dispatch(const prefill_args_fixed_t& args) {
    if (args.is_causal) {
      if (args.is_local) {
        run_impl<true, true, cutlass::flash_attention::kernel::fixed::IndividualScheduler>(args);
      } else {
        run_impl<true, false, cutlass::flash_attention::kernel::fixed::IndividualScheduler>(args);
      }
    } else {
      if (args.is_local) {
        run_impl<false, true, cutlass::flash_attention::kernel::fixed::IndividualScheduler>(args);
      } else {
        run_impl<false, false, cutlass::flash_attention::kernel::fixed::IndividualScheduler>(args);
      }
    }
  }
};

template <typename prefill_policy>
void policy_dispatch(CutlassType cuType, const prefill_args_fixed_t& args) {
  constexpr int PipelineStages = 2;
  
  if (cuType == CutlassType::half) {
    FMHAKernel<typename prefill_policy::ShapeQK, 
               typename prefill_policy::ShapePV,
               typename prefill_policy::ShapeOutPut,
               typename prefill_policy::SubgroupLayout, 
               PipelineStages,
               cutlass::half_t, cutlass::half_t, 
               XE_8x16x16_F32F16F16F32_TT, cutlass::half_t>::dispatch(args);
  } else {
    FMHAKernel<typename prefill_policy::ShapeQK, 
               typename prefill_policy::ShapePV,
               typename prefill_policy::ShapeOutPut,
               typename prefill_policy::SubgroupLayout,
               PipelineStages>::dispatch(args);
  }
}

void dispatch_by_head_size(CutlassType cuType, const prefill_args_fixed_t& args) {
  const int h = args.head_size;
  if (h <= 32) {
    policy_dispatch<prefill_policy_head32>(cuType, args);
  }
  else if (h <= 64) {
    policy_dispatch<prefill_policy_head64>(cuType, args);
  }
  else if (h <= 96) {
    policy_dispatch<prefill_policy_head96>(cuType, args);
  }
  else if (h <= 128) {
    policy_dispatch<prefill_policy_head128>(cuType, args);
  }
  else if (h <= 160) {
    policy_dispatch<prefill_policy_head160>(cuType, args);
  }
  else if (h <= 192) {
    policy_dispatch<prefill_policy_head192>(cuType, args);
  }
  else if (h <= 256) {
    policy_dispatch<prefill_policy_head256>(cuType, args);
  }
  else {
    throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
  }
}

// Fixed length implementation
void cutlass_fixed_impl(
    const at::Tensor& query,      // [batch, seq_q, heads, head_size]  B S H D
    const at::Tensor& key,        // [batch, seq_k, heads, head_size]
    const at::Tensor& value,      // [batch, seq_k, heads, head_size]
    at::Tensor& out,              // [batch, seq_q, heads, head_size]
    double softmax_scale, int window_size_left = -1, int  window_size_right = -1,
    bool is_causal = false, bool is_local = false) {
  
  int batch_size = query.size(0);
  int seq_len_q = query.size(1);
  int num_heads_q = query.size(2);
  int head_size = query.size(3);
  int seq_len_k = key.size(1);
  int num_heads_kv = key.size(2);

  // [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]  B, H, S, D
  auto q_reshaped = query.transpose(1, 2).contiguous();
  auto k_reshaped = key.transpose(1, 2).contiguous();
  auto v_reshaped = value.transpose(1, 2).contiguous();
  auto out_temp = torch::zeros_like(q_reshaped);

  if (is_local) {
    window_size_left = window_size_left == -1 ? seq_len_q : window_size_left;
    window_size_right = window_size_right == -1 ? seq_len_k : window_size_right;
  }

  // Prepare arguments
  prefill_args_fixed_t args{
    q_reshaped.data_ptr(), k_reshaped.data_ptr(), v_reshaped.data_ptr(), 
    out_temp.data_ptr(), static_cast<float>(softmax_scale), 
    num_heads_q, num_heads_kv, head_size, is_causal, batch_size,
    seq_len_q, seq_len_k, window_size_left, window_size_right, is_local
  };
  
  dispatch_by_head_size(aten_to_Cutlass_dtype(query), args);
  out.copy_(out_temp.transpose(1, 2));
}

}  // namespace cutlass::flash_attention::fixed