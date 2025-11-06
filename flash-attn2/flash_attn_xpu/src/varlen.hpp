#pragma once

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/device_memory.h"
#include <cute/tensor.hpp>

#include "./compat_wrapper.hpp"
#include "./kernel/varlen_scheduler.hpp"
#include "./kernel/varlen_kernel.hpp"
#include "./collective/fmha_fusion.hpp"
#include "./collective/varlen_epilogue.hpp"
#include "./collective/varlen_softmax_epilogue.hpp"

#include "fmha_utils.hpp"

namespace cutlass::flash_attention::varlen {

using namespace cute;

struct chunk_prefill_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  float sm_scale;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int max_blocks_per_seq;
  int block_size;
  bool is_causal;
  bool use_paged_kv;
};

template <class FMHAChunkPrefillKernel, bool isVarLen>
struct KernelLauncher {
  using StrideQ = typename FMHAChunkPrefillKernel::StrideQ;
  using StrideK = typename FMHAChunkPrefillKernel::StrideK;
  using StrideV = typename FMHAChunkPrefillKernel::StrideV;
  using StrideO = typename FMHAChunkPrefillKernel::StrideO;

  using ElementQ = typename FMHAChunkPrefillKernel::ElementQ;
  using ElementK = typename FMHAChunkPrefillKernel::ElementK;
  using ElementV = typename FMHAChunkPrefillKernel::ElementV;
  using ElementAcc = typename FMHAChunkPrefillKernel::ElementAccumulator;

  using CollectiveEpilogue =
      typename FMHAChunkPrefillKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename FMHAChunkPrefillKernel::ProblemShape;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K_cache;
  StrideV stride_V_cache;
  StrideO stride_O;
  uint64_t seed = 0;

  ProblemShapeType initialize(const chunk_prefill_args_t& args) {
    auto problem_shape = cute::make_tuple(
        1, args.num_heads_q, args.num_heads_k, args.total_seqlen_q,
        args.total_seqlen_k, args.head_size, args.head_size);
    auto problem_shape_out = cute::make_tuple(
        args.batch_size, args.num_heads_q, args.num_heads_k,
        cutlass::fmha::collective::VariableLength{args.max_queries},  // cu_q
        cutlass::fmha::collective::VariableLength{
            args.max_keys},  // cu_kv_cache
        args.head_size, args.head_size);
    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv_cache,
          head_size_qk, head_size_vo] = problem_shape;
    auto group_q_size = num_heads_q / num_heads_kv;
    auto group_q_num = num_heads_q / group_q_size;

    stride_Q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(seq_len_qo, num_heads_q * head_size_qk, batch));
    stride_K_cache = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(seq_len_kv_cache, num_heads_kv * head_size_qk, batch));
    stride_V_cache = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(head_size_vo * num_heads_kv, seq_len_kv_cache, batch));

    stride_O = cutlass::make_cute_packed_stride(
        StrideO{}, cute::make_shape(seq_len_qo * group_q_size,
                                    group_q_num * head_size_vo, batch));

    get<3>(problem_shape_out).cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_q);
    get<4>(problem_shape_out).cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_k);

    return problem_shape_out;
  }

  cutlass::Status run(const chunk_prefill_args_t& args,
                      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = initialize(args);

    typename FMHAChunkPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {reinterpret_cast<ElementQ*>(args.query), stride_Q,
         reinterpret_cast<ElementK*>(args.key), stride_K_cache,
         reinterpret_cast<ElementV*>(args.value), stride_V_cache,
         static_cast<int*>(args.block_table), args.block_size,
         args.max_blocks_per_seq, args.total_seqlen_k, -1, -1},
        {args.sm_scale},
        {reinterpret_cast<ElementOutput*>(args.out), stride_O},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size =
        FMHAChunkPrefillKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAChunkPrefillKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHAChunkPrefillKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params = FMHAChunkPrefillKernel::to_underlying_arguments(
        arguments, workspace.get());

    // Run the Flash Attention implementation.
    run(params);

    return cutlass::Status::kSuccess;
  }

  static void run(typename FMHAChunkPrefillKernel::Params params) {
    dim3 const block = FMHAChunkPrefillKernel::get_block_shape();
    dim3 const grid = FMHAChunkPrefillKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAChunkPrefillKernel::SharedStorageSize;

    const auto sycl_block = COMPAT::dim3(block.x, block.y, block.z);
    const auto sycl_grid = COMPAT::dim3(grid.x, grid.y, grid.z);

    COMPAT::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    COMPAT::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<
            FMHAChunkPrefillKernel::DispatchPolicy::SubgroupSize>};
    COMPAT::experimental::launch_policy policy{sycl_grid, sycl_block,
                                                   launch_props, kernel_props};
#if defined(OLD_API)
    auto event = COMPAT::experimental::launch<cutlass::device_kernel<FMHAChunkPrefillKernel>>(policy, params);
#else
    auto event = COMPAT::experimental::launch<cutlass::device_kernel<FMHAChunkPrefillKernel>, FMHAChunkPrefillKernel>(policy, params);
#endif
    // EventManager::getInstance().addEvent(event);
  }
};

template <typename TileShapeQK, typename TileShapePV, typename TileShapeOutput,
          typename SubgroupLayout, int PipelineStages,
          typename ElementInputQ = bfloat16_t,
          typename MMAOperation = XE_8x16x16_F32BF16BF16F32_TT,
          typename GmemTiledCopyQ = XE_2D_U16x8x32_LD_N,
          typename GmemTiledCopyK = XE_2D_U16x16x16_LD_T,
          typename GmemTiledCopyV = XE_2D_U16x16x32_LD_V,
          typename ElementAccumulator = float,
          typename ElementComputeEpilogue = float,
          typename GmemTiledCopyStore = XE_2D_U16x8x16_ST_N>
struct FMHAKernel {
  template <bool isVarLen, bool Causal, bool PagedKV, bool Local,
            class Scheduler>
  static void run(const chunk_prefill_args_t& args) {
    cutlass::KernelHardwareInfo hw_info;

    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using ElementInputKV = ElementInputQ;
    using ElementOutput = ElementInputQ;

    using GEMMDispatchPolicy =
        cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    using CollectiveEpilogue =
        cutlass::flash_attention::collective::FlashChunkPrefillEpilogue<
            EpilogueDispatchPolicy, MMAOperation, TileShapeOutput,
            SubgroupLayout, ElementComputeEpilogue, ElementOutput,
            cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput,
            GmemTiledCopyStore>;
    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashChunkPrefillSoftmaxEpilogue<
            Causal, Local, EpilogueDispatchPolicy, ElementAccumulator>;

    using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;
    using namespace cutlass::fmha::collective;
    using ProblemShapeVarlen =
        cute::tuple<int, int, int, VariableLength, VariableLength, int, int>;
    using ProblemShapeType =
        std::conditional_t<isVarLen, ProblemShapeVarlen, ProblemShapeRegular>;

    // Mainloop
    using CollectiveMainloop =
        cutlass::flash_attention::collective::FlashChunkPrefillMma<
            GEMMDispatchPolicy, ProblemShapeType, ElementInputQ,
            cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation, TileShapeQK,
            TileShapePV, SubgroupLayout,
            GmemTiledCopyQ,  // Q
            GmemTiledCopyK,  // K
            GmemTiledCopyV,  // V,
            Causal, Local, PagedKV>;

    using FMHAChunkPrefillKernel =
        cutlass::flash_attention::kernel::FMHAPrefillChunk<
            ProblemShapeType, CollectiveMainloop, CollectiveSoftmaxEpilogue,
            CollectiveEpilogue, Scheduler>;

    KernelLauncher<FMHAChunkPrefillKernel, isVarLen> launcher;

    launcher.run(args, hw_info);
  }

  static void dispatch(const chunk_prefill_args_t& args) {
    if (args.use_paged_kv) {
      if (args.is_causal) {
        run<true, true, true, false,
            cutlass::flash_attention::kernel::varlen::IndividualScheduler>(args);
      } else {
        run<true, false, true, false,
            cutlass::flash_attention::kernel::varlen::IndividualScheduler>(args);
      }
    } else {
      if (args.is_causal) {
        run<true, true, false, false,
            cutlass::flash_attention::kernel::varlen::IndividualScheduler>(args);
      } else {
        run<true, false, false, false,
            cutlass::flash_attention::kernel::varlen::IndividualScheduler>(args);
      }
    }
  }
};

template <typename chunk_policy>
void policy_dispatch(CutlassType cuType, const chunk_prefill_args_t& args) {
  const int PipelineStages = 2;

  if (cuType == CutlassType::half) {
    FMHAKernel<typename chunk_policy::ShapeQK, typename chunk_policy::ShapePV,
               typename chunk_policy::ShapeOutPut,
               typename chunk_policy::SubgroupLayout, PipelineStages,
               cutlass::half_t, XE_8x16x16_F32F16F16F32_TT>::dispatch(args);
  } else {
    FMHAKernel<typename chunk_policy::ShapeQK, typename chunk_policy::ShapePV,
               typename chunk_policy::ShapeOutPut,
               typename chunk_policy::SubgroupLayout,
               PipelineStages>::dispatch(args);
  }
}

template <typename ArgsType>
void dispatch_by_head_size(CutlassType cuType, const ArgsType& args) {
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

void cutlass_varlen_impl(
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache, at::Tensor& out,
    const std::optional<at::Tensor>& block_table, const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k, int max_seqlen_q, int max_seqlen_k,
    double sm_scale, bool is_causal) {
  int num_heads_q = query.size(1);
  int head_size = query.size(2);
  int batch_size = cu_seqlens_q.numel() - 1;
  int total_seqlen_q = query.size(0);
  bool use_paged_kv = block_table.has_value() && block_table->defined();
  int num_block, block_size, num_heads_kv, max_blocks_per_seq, total_seqlen_k;

  if (use_paged_kv) {
    num_block = key_cache.size(0);
    block_size = key_cache.size(1);
    num_heads_kv = key_cache.size(2);
    max_blocks_per_seq = block_table->size(1);
    total_seqlen_k = num_block * block_size;
  } else {
    // [total_seqlen_k, heads, head_size]
    num_block = 0;
    block_size = 0;
    max_blocks_per_seq = 0;
    num_heads_kv = key_cache.size(1);
    total_seqlen_k = key_cache.size(0);
  }

  chunk_prefill_args_t args = {query.data_ptr(),
                               key_cache.data_ptr(),
                               value_cache.data_ptr(),
                               out.data_ptr(),
                               block_table.has_value() ? block_table->data_ptr() : nullptr,
                               cu_seqlens_q.data_ptr(),
                               cu_seqlens_k.data_ptr(),
                               max_seqlen_q,
                               max_seqlen_k,
                               total_seqlen_q,
                               total_seqlen_k,
                               static_cast<float>(sm_scale),
                               batch_size,
                               num_heads_q,
                               num_heads_kv,
                               head_size,
                               max_blocks_per_seq,
                               block_size,
                               is_causal,
                               use_paged_kv};

  CutlassType cuType = aten_to_Cutlass_dtype(query);

  dispatch_by_head_size(cuType, args);
}

}  // namespace cutlass::flash_attention::varlen