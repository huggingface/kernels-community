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
#include "./collective/fmha_fwd_mainloop_xe2.hpp"
#include "./collective/fmha_fwd_epilogue_xe2.hpp"
#include "./kernel/fmha_fwd_kernel_xe2.hpp"

#include "fmha_utils.hpp"

using namespace cute;

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
      bool Causal,
      bool Local,
      bool Dropout,
      bool Paged,
      bool VarLen,
      bool HasRotary>
  static void run_xe2(sycl::queue& queue, const fmha_fwd_args_t& args) {
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

    using MainloopDispatchPolicy = cutlass::fmha::Xe2<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloopXe2<
        MainloopDispatchPolicy,
        Causal,
        Local,
        Dropout,
        Paged,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV,
        HasRotary>;

    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogueXe2<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernelXe2<
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
    if constexpr (VarLen) {
      shape.seq_len_qo =
          cutlass::fmha::collective::VariableLength{args.max_queries};
      shape.seq_len_qo.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_q);
      shape.seq_len_kv =
          cutlass::fmha::collective::VariableLength{args.max_keys};
      shape.seq_len_kv.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_k);
    } else {
      shape.seq_len_qo = args.max_queries;
      shape.seq_len_kv = args.max_keys;
    }

    // Contiguous (B,S,H,D) layout strides (used only for non-varlen; for
    // varlen the kernel constructs ordered layouts and ignores them).
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

    // LSE strides: (batch, num_heads, max_queries) -- same for varlen.
    int lse_stride_head = args.max_queries;
    int lse_stride_batch = args.num_heads_q * lse_stride_head;

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
         reinterpret_cast<float*>(args.softmax_lse),
         lse_stride_head,
         lse_stride_batch,
         args.cache_seqlens,
         args.cache_batch_idx,
         args.cache_leftpad,
         reinterpret_cast<const ElementK*>(args.knew),
         args.knew_batch_stride, args.knew_head_stride, args.knew_row_stride,
         reinterpret_cast<const ElementV*>(args.vnew),
         args.vnew_batch_stride, args.vnew_head_stride, args.vnew_row_stride,
         args.seqlen_knew},
        {static_cast<ElementS>(args.sm_scale),
         args.window_size_left, args.window_size_right,
         args.p_dropout,
         args.philox_seed, args.philox_offset,
         args.s_dmask,
         args.seqlen_q_rounded, args.seqlen_k_rounded,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         reinterpret_cast<const ElementQ*>(args.rotary_cos),
         reinterpret_cast<const ElementQ*>(args.rotary_sin),
         args.rotary_dim,
         args.is_rotary_interleaved},
        {},
        hw_info};

    auto params = FMHAKernel::to_underlying_arguments(arguments, nullptr);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid  = FMHAKernel::get_grid_shape(params);

    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid  = compat::dim3(grid.x,  grid.y,  grid.z);

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

  // Dispatch the Xe2 path by expanding (causal, local, dropout) at compile
  // time. (varlen, paged) are gated by the IsVarLen / IsPaged template
  // parameters: each per-head TU instantiates only one (varlen, paged) combo
  // in order to keep per-TU code-gen and IGC time bounded.
  template <
      int IsVarLen,
      int IsPaged,
      bool HasRotary = false,
      class Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>
  static void
  xe2_dispatch(sycl::queue& queue, const fmha_fwd_args_t& args) {
    static_assert(IsVarLen == 0 || IsVarLen == 1,
        "policy_dispatch must pin IsVarLen to 0 or 1");
    static_assert(IsPaged == 0 || IsPaged == 1,
        "policy_dispatch must pin IsPaged to 0 or 1");
    constexpr bool kVarLen = (IsVarLen == 1);
    constexpr bool kPaged  = (IsPaged  == 1);

    bool has_dropout = args.p_dropout > 0.0f;
#define XE2_CASE(C, L, D)                                                      \
  if (args.is_causal == C && args.is_local == L && has_dropout == D) {         \
        run_xe2<Scheduler, C, L, D, kPaged, kVarLen, HasRotary>(queue, args);      \
    return;                                                                    \
  }
    XE2_CASE(false, false, false)
    XE2_CASE(true,  false, false)
    XE2_CASE(false, true,  false)
    XE2_CASE(true,  true,  false)
    XE2_CASE(false, false, true )
    XE2_CASE(true,  false, true )
    XE2_CASE(false, true,  true )
    XE2_CASE(true,  true,  true )
#undef XE2_CASE
  }
};

// Single-dtype dispatch: only instantiates one dtype path per TU to reduce
// per-file IGC memory usage from ~40 GB to ~20 GB.
template <typename chunk_policy, int PipelineStages, int IsVarLen, int IsPaged,
          bool HasRotary = false>
void policy_dispatch_fp16(sycl::queue& queue, const fmha_fwd_args_t& args) {
  using Config = FMHAConfig<
      typename chunk_policy::ShapeQK,
      typename chunk_policy::ShapePV,
      typename chunk_policy::ShapeOut,
      typename chunk_policy::SubgroupLayoutQK,
      void,
      PipelineStages,
      half_t, half_t, half_t, half_t>;
    Config::template xe2_dispatch<IsVarLen, IsPaged, HasRotary>(queue, args);
}

template <typename chunk_policy, int PipelineStages, int IsVarLen, int IsPaged,
                    bool HasRotary = false>
void policy_dispatch_bf16(sycl::queue& queue, const fmha_fwd_args_t& args) {
  using Config = FMHAConfig<
      typename chunk_policy::ShapeQK,
      typename chunk_policy::ShapePV,
      typename chunk_policy::ShapeOut,
      typename chunk_policy::SubgroupLayoutQK,
      void,
      PipelineStages,
      bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>;
    Config::template xe2_dispatch<IsVarLen, IsPaged, HasRotary>(queue, args);
}

// Combined policy_dispatch is now defined inline in fmha_fwd.hpp
