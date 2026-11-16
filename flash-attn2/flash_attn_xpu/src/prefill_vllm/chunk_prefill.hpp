#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "../collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

#include "cutlass/util/device_memory.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#ifndef VLLM_GRF_SIZE
  #define VLLM_GRF_SIZE 256
#endif

#include "chunk_prefill_scheduler.hpp"
#include "chunk_prefill_epilogue.hpp"
#include "chunk_prefill_kernel.hpp"

#include "fmha_utils.hpp"
#include "chunk_prefill_args.hpp"

// Everything below belongs to this architecture alone.
//
// XE2 and XE3 are built into separate shared libraries but instantiate kernel
// templates with the same names from divergent sources (for example
// chunk_prefill_args_t differs between the two). At global namespace those
// instantiations mangle identically and are emitted as weak, default-visibility
// symbols, so the dynamic loader binds every reference to whichever library it
// resolves first -- an XE3 caller then runs XE2 code and reads the argument
// struct with the wrong layout, silently corrupting the KV cache strides.
//
// Keeping each architecture in its own namespace makes the mangled names
// disjoint, so the two libraries can never cross-bind. Do not move any of these
// declarations back to global namespace.
namespace kc_vllm_prefill::xe2 {
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

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  ProblemShapeType initialize(const chunk_prefill_args_t& args) {
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

    // Use actual tensor strides instead of packed strides
    stride_Q = StrideQ{};
    get<0>(stride_Q) = args.q_stride_seq;
    get<2>(stride_Q) = args.q_stride_heads;
    get<3>(stride_Q) = args.q_stride_batch;

    stride_K = StrideK{};
    get<0>(stride_K) = args.k_stride_seq;
    get<2>(stride_K) = args.k_stride_heads;
    get<3>(stride_K) = args.k_stride_batch;

    stride_V = StrideV{};
    get<1>(stride_V) = args.v_stride_seq;
    get<2>(stride_V) = args.v_stride_heads;
    get<3>(stride_V) = args.v_stride_batch;

    stride_O = StrideO{};
    get<0>(stride_O) = args.o_stride_seq;
    get<2>(stride_O) = args.o_stride_heads;
    get<3>(stride_O) = args.o_stride_batch;

    return shape;
  }

  cutlass::Status
  run(sycl::queue& queue,
      const chunk_prefill_args_t& args,
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
         stride_O,
         reinterpret_cast<ElementQ*>(args.sm_sink),
         args.softmax_lse,
         args.lse_stride,
         static_cast<const bool*>(args.is_prefill),
         args.dynamic_causal,
         args.min_seqlen_q},
        {args.sm_scale,
         args.k_scale,
         args.v_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right,
         args.page_stride_elements},
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

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        intelex::grf_size<VLLM_GRF_SIZE>};
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
    typename SubgroupLayoutPV_, /* void -> default */
    int PipelineStages,
    bool Paged = false,
    bool Causal = false,
    bool Local = false,
    bool Sink = false,
    bool SoftmaxLSE = false,
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

  template <class Scheduler>
  static void run(sycl::queue& queue, const chunk_prefill_args_t& args) {
    constexpr bool VarLen = true;
    // constexpr bool Paged = true;
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
    constexpr int VTiles =
        ceil_div(get<1>(TileShapeOutput{}), get<1>(TileShapePV{}));

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
        Sink,
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler,
        SoftmaxLSE>;

    KernelLauncher<FMHAKernel, VarLen> launcher;

    launcher.run(queue, args, hw_info);
  }

  static void
  kernel_dispatch(sycl::queue& queue, const chunk_prefill_args_t& args) {
    return run<cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(
        queue, args);
  }
};

// Hidden visibility prevents these per-arch template instantiations from being
// exported and interposed across the XE2/XE3 kernel shared libraries (which
// both define symbols with identical mangled names). Without this, the XE3
// wrapper would bind to the XE2 instantiation at load time (XE2 is linked
// first) and launch a kernel built for the wrong architecture.
template <
    typename chunk_policy,
    bool Paged,
    bool Causal,
    bool Local,
    bool Sink,
    bool SoftmaxLSE>
__attribute__((visibility("hidden"))) void policy_dispatch_impl(
    sycl::queue& queue,
    CutlassQKType& cuQKType,
    const chunk_prefill_args_t& args) {
  const int PipelineStages = 2;
  if (cuQKType.q_type == CutlassDType::half) {
    if (cuQKType.k_type == CutlassDType::half) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          Paged,
          Causal,
          Local,
          Sink,
          SoftmaxLSE,
          half_t,
          half_t,
          half_t,
          half_t>::kernel_dispatch(queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e4m3) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          Paged,
          Causal,
          Local,
          Sink,
          SoftmaxLSE,
          half_t,
          float_e4m3_t,
          float_e4m3_t,
          half_t>::kernel_dispatch(queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e5m2) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          Paged,
          Causal,
          Local,
          Sink,
          SoftmaxLSE,
          half_t,
          float_e5m2_t,
          float_e5m2_t,
          half_t>::kernel_dispatch(queue, args);
    }
  } else {
    if (cuQKType.k_type == CutlassDType::bfloat16) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          Paged,
          Causal,
          Local,
          Sink,
          SoftmaxLSE,
          bfloat16_t,
          bfloat16_t,
          bfloat16_t,
          bfloat16_t>::kernel_dispatch(queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e4m3) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          Paged,
          Causal,
          Local,
          Sink,
          SoftmaxLSE,
          bfloat16_t,
          float_e4m3_t,
          float_e4m3_t,
          bfloat16_t>::kernel_dispatch(queue, args);
    } else if (cuQKType.k_type == CutlassDType::float8_e5m2) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          Paged,
          Causal,
          Local,
          Sink,
          SoftmaxLSE,
          bfloat16_t,
          float_e5m2_t,
          float_e5m2_t,
          bfloat16_t>::kernel_dispatch(queue, args);
    }
  }
}

// kc entry: half/bf16 only (no fp8 K/V), Sink off. Mirrors vllm's
// cutlass_chunk_prefill_impl policy choice: *_b16 tiles for paged block sizes
// that are neither 32 nor a multiple of 64.
template <typename Policy, typename T, bool Paged, bool Causal, bool Local,
          bool LSE>
void kc_launch(sycl::queue& queue, const chunk_prefill_args_t& args) {
  FMHAConfig<
      typename Policy::ShapeQK,
      typename Policy::ShapePV,
      typename Policy::ShapeOut,
      typename Policy::SubgroupLayoutQK,
      void,
      2,
      Paged,
      Causal,
      Local,
      false,
      LSE,
      T,
      T,
      T,
      T>::kernel_dispatch(queue, args);
}

template <typename Policy, typename T, bool Paged>
void kc_dispatch_flags(sycl::queue& queue, const chunk_prefill_args_t& args) {
  const bool lse = args.softmax_lse != nullptr;
  if (args.is_local) {
    // vllm builds Local without LSE only.
    if (args.is_causal) {
      kc_launch<Policy, T, Paged, true, true, false>(queue, args);
    } else {
      kc_launch<Policy, T, Paged, false, true, false>(queue, args);
    }
  } else if (args.is_causal) {
    if (lse) kc_launch<Policy, T, Paged, true, false, true>(queue, args);
    else     kc_launch<Policy, T, Paged, true, false, false>(queue, args);
  } else {
    if (lse) kc_launch<Policy, T, Paged, false, false, true>(queue, args);
    else     kc_launch<Policy, T, Paged, false, false, false>(queue, args);
  }
}

template <typename Policy, typename PolicyB16, typename T, bool Paged>
bool kc_chunk_prefill(sycl::queue& queue, const chunk_prefill_args_t& args) {
  if (args.is_local && args.softmax_lse != nullptr) return false;
  if constexpr (Paged) {
    if (args.block_size <= 0 || args.block_size % 16 != 0) return false;
    if (args.block_size != 32 && args.block_size % 64 != 0) {
      kc_dispatch_flags<PolicyB16, T, true>(queue, args);
    } else {
      kc_dispatch_flags<Policy, T, true>(queue, args);
    }
  } else {
    kc_dispatch_flags<Policy, T, false>(queue, args);
  }
  return true;
}

template <int HD> struct kc_policy;
#define KC_VLLM_POLICY(HD)                                \
  template <> struct kc_policy<HD> {                      \
    using std_t = chunk_policy_head##HD;                  \
    using b16_t = chunk_policy_head##HD##_b16;            \
  };
KC_VLLM_POLICY(64)
KC_VLLM_POLICY(96)
KC_VLLM_POLICY(128)
KC_VLLM_POLICY(192)
KC_VLLM_POLICY(256)
KC_VLLM_POLICY(512)
#undef KC_VLLM_POLICY

template <int HD, bool Bf16, bool Paged>
bool chunk_prefill_impl(sycl::queue& queue, const chunk_prefill_args_t& args) {
  using T = cute::conditional_t<Bf16, bfloat16_t, half_t>;
  return kc_chunk_prefill<typename kc_policy<HD>::std_t,
                          typename kc_policy<HD>::b16_t, T, Paged>(queue, args);
}

}  // namespace kc_vllm_prefill::xe2
