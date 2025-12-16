#pragma once

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "./collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>

#include "cutlass/util/device_memory.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "./compat_wrapper.hpp"
#include "./collective/fmha_fwd_scheduler.hpp"
#include "./collective/fmha_fwd_epilogue.hpp"
#include "./kernel/fmha_fwd_kernel.hpp"

#include "fmha_utils.hpp"

using namespace cute;

struct fmha_fwd_args_t {
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
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_paged = false;
  bool is_causal = false;
  bool is_local = false;
};

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

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  ProblemShapeType initialize(const fmha_fwd_args_t& args) {
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
        args.head_size;

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

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

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
         args.window_size_right},
        {},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Initialize the workspace
    FMHAKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params =
        FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    run(queue, params);

    return cutlass::Status::kSuccess;
  }

  static void run(sycl::queue& queue, typename FMHAKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = COMPAT::dim3(block.x, block.y, block.z);
    const auto sycl_grid = COMPAT::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    COMPAT::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    COMPAT::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    COMPAT::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    auto event =
        COMPAT::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
            policy, queue, params);

    EventManager::getInstance().addEvent(event);
  }
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_, /* void -> default */
    int PipelineStages,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = bfloat16_t,
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
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
      bool Local>
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

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        Local,
        Paged,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV>;

    // Epilogue
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

template <typename chunk_policy, int PipelineStages>
void policy_dispatch(sycl::queue& queue, CutlassType cuType, const fmha_fwd_args_t& args) {
  if (cuType == CutlassType::half) {
    return FMHAConfig<
        typename chunk_policy::ShapeQK,
        typename chunk_policy::ShapePV,
        typename chunk_policy::ShapeOut,
        typename chunk_policy::SubgroupLayoutQK,
        void,
        PipelineStages,
        half_t,
        half_t,
        half_t,
        half_t>::
        kernel_dispatch(
            queue,
            args,
            args.is_varlen,
            args.is_paged,
            args.is_causal,
            args.is_local);
  } else {
    return FMHAConfig<
        typename chunk_policy::ShapeQK,
        typename chunk_policy::ShapePV,
        typename chunk_policy::ShapeOut,
        typename chunk_policy::SubgroupLayoutQK,
        void,
        PipelineStages>::
        kernel_dispatch(
            queue,
            args,
            args.is_varlen,
            args.is_paged,
            args.is_causal,
            args.is_local);
  }
}

void cutlass_fmha_fwd_varlen_impl(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size] or [total_seqlen_k, heads, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    const std::optional<at::Tensor>& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local) {
  // general params
  int batch_size, num_heads_q, num_heads_kv, head_size;
  // additional params
  int total_seqlen_q, total_seqlen_k;
  int num_blocks, block_size, max_blocks_per_seq;
  if (is_varlen) {
    // query: [total_seq, num_heads, head_size]
    batch_size = cu_seqlens_q.numel() - 1;
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(2);
    total_seqlen_q = query.size(0);
    total_seqlen_k = key_cache.size(0);
  } else {
    // query: [batch, num_heads, seq, head_size]
    batch_size = query.size(0);
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(3);
    max_seqlen_q = query.size(2);
    max_seqlen_k = key_cache.size(2);
  }
  if (is_paged) {
    num_blocks = key_cache.size(0);
    block_size = key_cache.size(1);
    num_heads_kv = key_cache.size(2);
    max_blocks_per_seq = block_table->size(1);
    total_seqlen_k = num_blocks * block_size;
  } else {
    // [total_seqlen_k, heads, head_size]
    num_blocks = 0;
    block_size = 0;
    num_heads_kv = key_cache.size(1);
    max_blocks_per_seq = 0;
    total_seqlen_k = key_cache.size(0);
  }

  if (is_local) {
    window_size_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
    window_size_right = window_size_right == -1 ? max_seqlen_k : window_size_right;
    if (is_causal) {
      window_size_right = 0;
      is_causal = false;
    }
  }

  fmha_fwd_args_t args = {
      query.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      out.data_ptr(),
      is_paged && block_table.has_value() ? block_table->data_ptr() : nullptr,
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
      window_size_left,
      window_size_right,
      is_varlen,  // varlen
      is_paged,   // paged
      is_causal,
      is_local};

  CutlassType cuType = aten_to_Cutlass_dtype(query);

  const int h = args.head_size;
  if (h <= 32) {
    policy_dispatch<prefill_policy_head32, PipelineStages_Prefill>(queue, cuType, args);
  }
  // else if (h <= 64) {
  //   policy_dispatch<prefill_policy_head64, PipelineStages_Prefill>(queue, cuType, args);
  // }
  // else if (h <= 96) {
  //   policy_dispatch<prefill_policy_head96, PipelineStages_Prefill>(queue, cuType, args);
  // }
  // else if (h <= 128) {
  //   policy_dispatch<prefill_policy_head128, PipelineStages_Prefill>(queue, cuType, args);
  // }
  // else if (h <= 160) {
  //   policy_dispatch<prefill_policy_head160, PipelineStages_Prefill>(queue, cuType, args);
  // }
  // else if (h <= 192) {
  //   policy_dispatch<prefill_policy_head192, PipelineStages_Prefill>(queue, cuType, args);
  // }
  // else if (h <= 256) {
  //   policy_dispatch<prefill_policy_head256, PipelineStages_Prefill>(queue, cuType, args);
  // }
  else {
    throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
  }
}

void cutlass_fmha_fwd_fix_impl(
    sycl::queue& queue,
    const at::Tensor& query,      // [batch, seq_q, num_heads, head_size]
    const at::Tensor& key,        // [batch, seq_k, num_heads_k, head_size]
    const at::Tensor& value,      // [batch, seq_k, num_heads_k, head_size]
    at::Tensor& out,              // [batch, seq_q, num_heads, head_size]
    float sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local) {
  // query: [batch, seq, num_heads, head_size]
  int batch_size = query.size(0);
  int max_seqlen_q = query.size(1);
  int num_heads_q = query.size(2);
  int head_size = query.size(3);

  int max_seqlen_k = key.size(1);
  int num_heads_kv = key.size(2);

  int total_seqlen_q = batch_size * max_seqlen_q;
  int total_seqlen_k = batch_size * max_seqlen_k;

  if (is_local) {
    window_size_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
    window_size_right = window_size_right == -1 ? max_seqlen_k : window_size_right;
    if (is_causal) {
      window_size_right = 0;
      is_causal = false;
    }
  }

  fmha_fwd_args_t args = {
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      out.data_ptr(),
      nullptr, // block_table
      nullptr, // cu_seqlens_q
      nullptr, // cu_seqlens_k
      max_seqlen_q,
      max_seqlen_k,
      total_seqlen_q,
      total_seqlen_k,
      sm_scale,
      batch_size,
      num_heads_q,
      num_heads_kv,
      head_size,
      0, // max_blocks_per_seq
      0, // block_size
      window_size_left,
      window_size_right,
      false, // is_varlen
      false, // is_paged
      is_causal,
      is_local};

  CutlassType cuType = aten_to_Cutlass_dtype(query);
  const int h = args.head_size;

  if (max_seqlen_q == 1) {
    if (h <= 32) {
      policy_dispatch<decode_policy_head32, PipelineStages_Decode>(queue, cuType, args);
    }
    // else if (h <= 64) {
    //   policy_dispatch<decode_policy_head64, PipelineStages_Decode>(queue, cuType, args);
    // }
    // else if (h <= 96) {
    //   policy_dispatch<decode_policy_head96, PipelineStages_Decode>(queue, cuType, args);
    // }
    // else if (h <= 128) {
    //   policy_dispatch<decode_policy_head128, PipelineStages_Decode>(queue, cuType, args);
    // }
    // else if (h <= 160) {
    //   policy_dispatch<decode_policy_head160, PipelineStages_Decode>(queue, cuType, args);
    // }
    // else if (h <= 192) {
    //   policy_dispatch<decode_policy_head192, PipelineStages_Decode>(queue, cuType, args);
    // }
    // else if (h <= 256) {
    //   policy_dispatch<decode_policy_head256, PipelineStages_Decode>(queue, cuType, args);
    // }
    else {
      throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
    }
  } else {
    if (h <= 32) {
      policy_dispatch<prefill_policy_head32, PipelineStages_Prefill>(queue, cuType, args);
    }
    // else if (h <= 64) {
    //   policy_dispatch<prefill_policy_head64, PipelineStages_Prefill>(queue, cuType, args);
    // }
    // else if (h <= 96) {
    //   policy_dispatch<prefill_policy_head96, PipelineStages_Prefill>(queue, cuType, args);
    // }
    // else if (h <= 128) {
    //   policy_dispatch<prefill_policy_head128, PipelineStages_Prefill>(queue, cuType, args);
    // }
    // else if (h <= 160) {
    //   policy_dispatch<prefill_policy_head160, PipelineStages_Prefill>(queue, cuType, args);
    // }
    // else if (h <= 192) {
    //   policy_dispatch<prefill_policy_head192, PipelineStages_Prefill>(queue, cuType, args);
    // }
    // else if (h <= 256) {
    //   policy_dispatch<prefill_policy_head256, PipelineStages_Prefill>(queue, cuType, args);
    // }
    else {
      throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
    }
  }
}
