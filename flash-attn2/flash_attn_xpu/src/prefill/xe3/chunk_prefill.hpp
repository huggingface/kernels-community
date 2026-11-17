/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Adapted from vllm-xpu-kernels csrc/xpu/attn/xe_3/chunk_prefill.hpp
// (KernelLauncher, FMHAConfig) with the Xe3 tile policies of
// csrc/xpu/attn/fmha_utils.hpp and chunk_policy_dispatch_shapes of
// xe_3/chunk_prefill_common.hpp. CRI image only; the collectives in this
// directory are shared with the paged decode kernel (src/decode).
#pragma once

#include "../../decode/splitkv_decode.hpp"
#include "chunk_prefill_args.hpp"

#if !FLASH_XPU_HAS_XE3_KERNELS
// No 512-GRF Xe35 build: report "not handled" so the caller falls back.
namespace flash_xpu_prefill::xe3 {
template <int HD, bool Bf16, bool Paged>
bool chunk_prefill_impl(sycl::queue&, const chunk_prefill_args_t&) {
  return false;
}
}  // namespace flash_xpu_prefill::xe3
#else

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "../../collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

#include "cutlass/util/device_memory.h"

#include "../../decode/grf_mode.hpp"

#ifndef FLASH_XPU_GRF_SIZE
  #define FLASH_XPU_GRF_SIZE 256
#endif

#include "chunk_prefill_scheduler.hpp"
#include "chunk_prefill_epilogue.hpp"
#include "chunk_prefill_kernel.hpp"

namespace flash_xpu_prefill::xe3 {
using namespace cute;

// fmha_utils.hpp, VLLM_XPU_ENABLE_XE3 variants.
struct chunk_policy_head64 {
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head64_b16 {
  using ShapeQK = Shape<_128, _16, _32>;
  using ShapePV = Shape<_128, _32, _16>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head128 {
  using ShapeQK = Shape<_512, _64, _64>;
  using ShapePV = Shape<_512, _64, _64>;
  using ShapeOut = Shape<_512, _128>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;

  using ShapeQK_Causal = Shape<_256, _64, _64>;
  using ShapePV_Causal = Shape<_256, _64, _64>;
  using ShapeOut_Causal = Shape<_256, _128>;
  using SubgroupLayoutQK_Causal = Layout<Shape<_16, _1, _1>>;
};

struct chunk_policy_head128_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
};

struct chunk_policy_head256 {
  using ShapeQK = Shape<_512, _16, _64>;
  using ShapePV = Shape<_512, _64, _16>;
  using ShapeOut = Shape<_512, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;

  using ShapeQK_Causal = Shape<_128, _16, _64>;
  using ShapePV_Causal = Shape<_128, _64, _16>;
  using ShapeOut_Causal = Shape<_128, _256>;
  using SubgroupLayoutQK_Causal = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head256_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct chunk_policy_head96 {
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head96_b16 {
  using ShapeQK = Shape<_128, _16, _32>;
  using ShapePV = Shape<_128, _32, _16>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct chunk_policy_head192 {
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct chunk_policy_head192_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct chunk_policy_head512 {
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct chunk_policy_head512_b16 {
  using ShapeQK = Shape<_256, _16, _32>;
  using ShapePV = Shape<_256, _32, _16>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

// chunk_prefill_common.hpp
template <typename Policy, bool Causal, typename = void>
struct chunk_policy_dispatch_shapes {
  using ShapeQK = typename Policy::ShapeQK;
  using ShapePV = typename Policy::ShapePV;
  using ShapeOut = typename Policy::ShapeOut;
  using SubgroupLayoutQK = typename Policy::SubgroupLayoutQK;
};

template <typename Policy>
struct chunk_policy_dispatch_shapes<
    Policy,
    true,
    cute::void_t<typename Policy::ShapeQK_Causal>> {
  using ShapeQK = typename Policy::ShapeQK_Causal;
  using ShapePV = typename Policy::ShapePV_Causal;
  using ShapeOut = typename Policy::ShapeOut_Causal;
  using SubgroupLayoutQK = typename Policy::SubgroupLayoutQK_Causal;
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
         args.min_seqlen_q},
        {args.sm_scale,
         args.k_scale,
         args.v_scale,
         args.q_scale,
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
        FLASH_XPU_GRF_PROP(FLASH_XPU_GRF_SIZE)};
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

    // K/V taller than one 2D block-load surface needs the sub-surface variant.
    if constexpr (Paged) {
      if (args.total_seqlen_k > cutlass::fmha::collective::kXeMaxSurfaceRows) {
        using LargeSurfaceKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
            ProblemShapeType,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler,
            SoftmaxLSE,
            true>;
        KernelLauncher<LargeSurfaceKernel, VarLen> launcher;
        launcher.run(queue, args, hw_info);
        return;
      }
    }

    KernelLauncher<FMHAKernel, VarLen> launcher;

    launcher.run(queue, args, hw_info);
  }

  static void
  kernel_dispatch(sycl::queue& queue, const chunk_prefill_args_t& args) {
    return run<cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<Causal>>(
        queue, args);
  }
};

// kc entry: half/bf16 only (no fp8 K/V), Sink off. Mirrors vllm-xpu-kernels'
// policy_dispatch_impl (causal-specific tile shapes) and
// cutlass_chunk_prefill_impl, except that the policy is picked by whether its
// KV tile divides block_size. vllm-xpu-kernels sends every block_size != 16 to
// the standard policy, but on Xe3 head128 that tile is 64 rows: with
// block_size 32, get_paged_idx gets tiles_per_page = page_stride_tiles = 0
// and reads the wrong pages.
template <typename Policy, typename T, bool Paged, bool Causal, bool Local,
          bool LSE>
void kc_launch(sycl::queue& queue, const chunk_prefill_args_t& args) {
  using S = chunk_policy_dispatch_shapes<Policy, Causal>;
  FMHAConfig<
      typename S::ShapeQK,
      typename S::ShapePV,
      typename S::ShapeOut,
      typename S::SubgroupLayoutQK,
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
    // vllm-xpu-kernels builds Local without LSE only.
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
    auto kv_tile = [&](auto policy) {
      using P = decltype(policy);
      return args.is_causal
          ? int(get<1>(typename chunk_policy_dispatch_shapes<P, true>::ShapeQK{}))
          : int(get<1>(typename chunk_policy_dispatch_shapes<P, false>::ShapeQK{}));
    };
    if (args.block_size <= 0) return false;
    if (args.block_size % kv_tile(Policy{}) == 0) {
      kc_dispatch_flags<Policy, T, true>(queue, args);
    } else if (args.block_size % kv_tile(PolicyB16{}) == 0) {
      kc_dispatch_flags<PolicyB16, T, true>(queue, args);
    } else {
      return false;
    }
  } else {
    kc_dispatch_flags<Policy, T, false>(queue, args);
  }
  return true;
}

template <int HD> struct kc_policy;
template <> struct kc_policy<64> {
  using std_t = chunk_policy_head64;
  using b16_t = chunk_policy_head64_b16;
};
template <> struct kc_policy<128> {
  using std_t = chunk_policy_head128;
  using b16_t = chunk_policy_head128_b16;
};
template <> struct kc_policy<256> {
  using std_t = chunk_policy_head256;
  using b16_t = chunk_policy_head256_b16;
};
template <> struct kc_policy<96> {
  using std_t = chunk_policy_head96;
  using b16_t = chunk_policy_head96_b16;
};
template <> struct kc_policy<192> {
  using std_t = chunk_policy_head192;
  using b16_t = chunk_policy_head192_b16;
};
template <> struct kc_policy<512> {
  using std_t = chunk_policy_head512;
  using b16_t = chunk_policy_head512_b16;
};

template <int HD, bool Bf16, bool Paged>
bool chunk_prefill_impl(sycl::queue& queue, const chunk_prefill_args_t& args) {
  using T = cute::conditional_t<Bf16, bfloat16_t, half_t>;
  return kc_chunk_prefill<typename kc_policy<HD>::std_t,
                          typename kc_policy<HD>::b16_t, T, Paged>(queue, args);
}

}  // namespace flash_xpu_prefill::xe3

#endif  // FLASH_XPU_HAS_XE3_KERNELS
