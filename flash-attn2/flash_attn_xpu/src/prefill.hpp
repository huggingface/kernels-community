#pragma once

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/device_memory.h"
#include <cute/tensor.hpp>

#include "./compat_wrapper.hpp"
#include "./kernel/tile_scheduler.hpp"
#include "./kernel/xe_flash_attn_prefill.hpp"
#include "./collective/fmha_fusion.hpp"
#include "./collective/xe_flash_attn_prefill_epilogue.hpp"
#include "./collective/xe_flash_attn_prefill_softmax_epilogue.hpp"

#include "fmha_utils.hpp"

using namespace cute;

// Base structure for common arguments
struct prefill_args_base_t {
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
};

// Variable length specific arguments
struct prefill_args_varlen_t : public prefill_args_base_t {
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_seqlen_q;
  int max_seqlen_k;
  int total_seqlen_q;
  int total_seqlen_k;
};

// Fixed length (non-varlen) specific arguments
struct prefill_args_fixed_t : public prefill_args_base_t {
  int seq_len_q;
  int seq_len_k;
};

template <class FMHAPrefillKernel, bool isVarLen>
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

  // Specialization for variable length
  template<bool isVL = isVarLen>
  typename std::enable_if_t<isVL, ProblemShapeType>
  initialize(const prefill_args_varlen_t& args) {
    auto problem_shape_out = cute::make_tuple(
        args.batch_size, args.num_heads_q, args.num_heads_kv,
        cutlass::fmha::collective::VariableLength{args.max_seqlen_q, nullptr},  // cu_q
        cutlass::fmha::collective::VariableLength{args.max_seqlen_k, nullptr},  // cu_kv
        args.head_size, args.head_size);

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, 
        cute::make_shape(args.total_seqlen_q, args.head_size, args.num_heads_q));
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, 
        cute::make_shape(args.total_seqlen_k, args.head_size, args.num_heads_kv));
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, 
        cute::make_shape(args.head_size, args.total_seqlen_k, args.num_heads_kv));
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, 
        cute::make_shape(args.total_seqlen_q, args.head_size, args.num_heads_q));

    cute::get<3>(problem_shape_out).cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_q);
    cute::get<4>(problem_shape_out).cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_k);

    return problem_shape_out;
  }

  // Specialization for fixed length
  template<bool isVL = isVarLen>
  typename std::enable_if_t<!isVL, ProblemShapeType>
  initialize(const prefill_args_fixed_t& args) {
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

  // Run function for variable length
  template<bool isVL = isVarLen>
  typename std::enable_if_t<isVL, cutlass::Status>
  run(const prefill_args_varlen_t& args, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = initialize(args);

    typename FMHAPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {reinterpret_cast<ElementQ*>(args.query), stride_Q,
         reinterpret_cast<ElementK*>(args.key), stride_K,
         reinterpret_cast<ElementV*>(args.value), stride_V,
        },  // window_left, window_right for local mask (not supported currently)
        {args.softmax_scale},
        {reinterpret_cast<ElementOutput*>(args.out), stride_O},
        hw_info};

    return run_kernel(arguments);
  }

  // Run function for fixed length
  template<bool isVL = isVarLen>
  typename std::enable_if_t<!isVL, cutlass::Status>
  run(const prefill_args_fixed_t& args, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = initialize(args);

    typename FMHAPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {reinterpret_cast<ElementQ*>(args.query), stride_Q,
         reinterpret_cast<ElementK*>(args.key), stride_K,
         reinterpret_cast<ElementV*>(args.value), stride_V,
        },  // window_left, window_right for local mask (not supported currently)
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
  template <bool Causal, class Scheduler, typename ArgsType>
  static void run_impl(const ArgsType& args) {
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
            Causal, EpilogueDispatchPolicy, ElementAccumulator>;

    using ProblemShape = typename std::conditional<
        std::is_same<ArgsType, prefill_args_varlen_t>::value,
        cute::tuple<int, int, int, cutlass::fmha::collective::VariableLength, 
                    cutlass::fmha::collective::VariableLength, int, int>,
        cute::tuple<int, int, int, int, int, int, int>
    >::type;

    using CollectiveMainloop =
        cutlass::flash_attention::collective::FlashPrefillMma<
            GEMMDispatchPolicy, ProblemShape, ElementInputQ,
            cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation, TileShapeQK,
            TileShapePV, SubgroupLayout, GmemTiledCopyQ, GmemTiledCopyK, 
            GmemTiledCopyV, Causal>;

    using FMHAPrefillKernel = cutlass::flash_attention::kernel::FMHAPrefill<
        ProblemShape, CollectiveMainloop, CollectiveSoftmaxEpilogue,
        CollectiveEpilogue, Scheduler>;

    constexpr bool isVarLen = std::is_same<ArgsType, prefill_args_varlen_t>::value;
    KernelLauncher<FMHAPrefillKernel, isVarLen> launcher;
    launcher.run(args, hw_info);
  }

  template <typename ArgsType>
  static void dispatch(const ArgsType& args) {
    if (args.is_causal) {
      run_impl<true, cutlass::flash_attention::IndividualScheduler>(args);
    } else {
      run_impl<false, cutlass::flash_attention::IndividualScheduler>(args);
    }
  }
};

template <typename prefill_policy, typename ArgsType>
void policy_dispatch(CutlassType cuType, const ArgsType& args) {
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

class TensorRearranger {
public:
  // [total_seq, heads, head_size] -> [total_seq * heads, head_size]
  static void to_block_layout(
      const at::Tensor& input, at::Tensor& output,
      const at::Tensor& cu_seqlens, int batch_size, int num_heads) {
    
    int offset = 0;
    for (int b = 0; b < batch_size; ++b) {
      const int start = cu_seqlens[b].item<int>();
      const int end = cu_seqlens[b + 1].item<int>();
      const int seq_len = end - start;
      
      for (int h = 0; h < num_heads; ++h) {
        output.slice(0, offset, offset + seq_len).copy_(
            input.slice(0, start, end).select(1, h));
        offset += seq_len;
      }
    }
  }

  // [total_seq * heads, head_size] -> [total_seq, heads, head_size]
  static void from_block_layout(
      const at::Tensor& input, at::Tensor& output,
      const at::Tensor& cu_seqlens, int batch_size, int num_heads) {
    
    int offset = 0;
    for (int b = 0; b < batch_size; ++b) {
      const int start = cu_seqlens[b].item<int>();
      const int end = cu_seqlens[b + 1].item<int>();
      const int seq_len = end - start;
      
      for (int h = 0; h < num_heads; ++h) {
        output.slice(0, start, end).select(1, h).copy_(
            input.slice(0, offset, offset + seq_len));
        offset += seq_len;
      }
    }
  }
};

template <typename ArgsType>
void dispatch_by_head_size(CutlassType cuType, const ArgsType& args) {
  const int h = args.head_size;
  if (h <= 64) {
    policy_dispatch<prefill_policy_head64>(cuType, args);
  } 
  else if (h <= 96) {
    policy_dispatch<prefill_policy_head96>(cuType, args);
  } 
  else if (h <= 128) {
    policy_dispatch<prefill_policy_head128>(cuType, args);
  } 
  else if (h <= 192) {
    policy_dispatch<prefill_policy_head192>(cuType, args);
  } 
  else {
    throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 192");
  }
}

// Variable length implementation
void cutlass_prefill_varlen_impl(
    const at::Tensor& query,      // [total_seq_q, heads, head_size]  B*S, H, D
    const at::Tensor& key,        // [total_seq_k, heads, head_size]
    const at::Tensor& value,      // [total_seq_k, heads, head_size]
    at::Tensor& out,              // [total_seq_q, heads, head_size]
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q, int max_seqlen_k,
    double softmax_scale, bool is_causal) {

  int num_heads_q = query.size(1);
  int num_heads_kv = key.size(1);
  int head_size = query.size(2);
  int batch_size = cu_seqlens_q.numel() - 1;
  int total_seqlen_q = query.size(0);
  int total_seqlen_k = key.size(0);

  auto cu_q = cu_seqlens_q.to(torch::kInt32);
  auto cu_k = cu_seqlens_k.to(torch::kInt32);

  // Create block layouts
  auto q_block = torch::empty({total_seqlen_q * num_heads_q, head_size}, query.options());
  auto k_block = torch::empty({total_seqlen_k * num_heads_kv, head_size}, key.options());
  auto v_block = torch::empty({total_seqlen_k * num_heads_kv, head_size}, value.options());
  auto out_block = torch::empty({total_seqlen_q * num_heads_q, head_size}, query.options());

  // Rearrange tensors
  TensorRearranger::to_block_layout(query, q_block, cu_q, batch_size, num_heads_q);
  TensorRearranger::to_block_layout(key, k_block, cu_k, batch_size, num_heads_kv);
  TensorRearranger::to_block_layout(value, v_block, cu_k, batch_size, num_heads_kv);
  
  // Prepare arguments
  prefill_args_varlen_t args{
    {q_block.data_ptr(), k_block.data_ptr(), v_block.data_ptr(), out_block.data_ptr(),
     static_cast<float>(softmax_scale), num_heads_q, num_heads_kv, head_size, is_causal, batch_size},
    cu_seqlens_q.data_ptr(), cu_seqlens_k.data_ptr(),
    max_seqlen_q, max_seqlen_k, total_seqlen_q, total_seqlen_k
  };
  
  dispatch_by_head_size(aten_to_Cutlass_dtype(query), args);
  TensorRearranger::from_block_layout(out_block, out, cu_q, batch_size, num_heads_q);
}

// Fixed length implementation
void cutlass_prefill_fixed_impl(
    const at::Tensor& query,      // [batch, seq_q, heads, head_size]  B S H D
    const at::Tensor& key,        // [batch, seq_k, heads, head_size]
    const at::Tensor& value,      // [batch, seq_k, heads, head_size]
    at::Tensor& out,              // [batch, seq_q, heads, head_size]
    double softmax_scale, bool is_causal) {
  
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

  // Prepare arguments
  prefill_args_fixed_t args{
    {q_reshaped.data_ptr(), k_reshaped.data_ptr(), v_reshaped.data_ptr(), 
     out_temp.data_ptr(), static_cast<float>(softmax_scale), 
     num_heads_q, num_heads_kv, head_size, is_causal, batch_size},
    seq_len_q, seq_len_k
  };
  
  dispatch_by_head_size(aten_to_Cutlass_dtype(query), args);
  out.copy_(out_temp.transpose(1, 2));
}
