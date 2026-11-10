/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Compile-time configuration and launch of the split-KV decode kernels.
// Included only by the per-head-dim translation units.

#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cute/tensor.hpp"

#include "splitkv_kernel.hpp"

namespace flash_xpu_splitkv {

using namespace cute;

template <class Kernel>
void launch_kernel(
    sycl::queue& queue,
    typename Kernel::Params const& params,
    dim3 grid,
    dim3 block) {
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(Kernel::SharedStorageSize),
  };
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
  compat::experimental::launch_policy policy{
      compat::dim3(grid.x, grid.y, grid.z),
      compat::dim3(block.x, block.y, block.z),
      launch_props,
      kernel_props};
  compat::experimental::launch<cutlass::device_kernel<Kernel>>(
      policy, queue, params);
}

// HeadDimV: largest V head size handled by this instantiation. Q/K head size
// is looped over in chunks of 64 at runtime.
template <class Element, int HeadDimV>
struct SplitKVConfig {
  static constexpr int kTileV = HeadDimV < 256 ? HeadDimV : 256;
  using ShapeQK = Shape<_8, Int<kKVTile>, _64>;
  using ShapePV = Shape<_8, _32, Int<kKVTile>>;
  using ShapeOut = Shape<_8, Int<kTileV>>;
  using SubgroupLayoutQK = Layout<Shape<_1, _4, _1>>;
  using SubgroupLayoutPV = decltype(get_sg_layout_pv(SubgroupLayoutQK{}));
  static constexpr int VTiles = (kTileV + 31) / 32;

  using MMAOperation = XE_DPAS_TT<8, float, Element>;
  using TiledMMAQK = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>,
      Layout<ShapeQK>,
      SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>,
      Layout<ShapePV>,
      SubgroupLayoutPV>::TiledMMA;

  template <bool Paged, bool Local, bool LargeSurface, bool MultiQ>
  static void run(sycl::queue& queue, decode_args_t const& args) {
    using Mainloop = DecodeMainloop<
        Paged,
        Local,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        Tensor2D<Element>,
        Tensor2D<Element>,
        Tensor2DT<Element>>;
    using Epilogue = DecodeEpilogue<Mainloop, ShapeOut>;
    using Kernel = SplitKVDecodeKernel<
        Mainloop, Epilogue, Element, LargeSurface, MultiQ>;

    typename Kernel::Params params{
        args,
        {args.softmax_scale * 1.4426950408889634f,  // log2(e)
         args.block_table,
         args.block_table_batch_stride,
         args.page_size,
         args.page_stride_rows,
         args.window_size_left}};
    launch_kernel<Kernel>(
        queue, params, Kernel::get_grid_shape(args), Kernel::get_block_shape());

    if (args.num_splits > 1) {
      using Combine = SplitKVCombineKernel<Element>;
      typename Combine::Params cparams{args};
      launch_kernel<Combine>(
          queue,
          cparams,
          Combine::get_grid_shape(args),
          Combine::get_block_shape());
    }
  }

  template <bool Paged, bool Local>
  static void run_mode(sycl::queue& queue, decode_args_t const& args) {
    // A tile of tile_q (query, GQA head) rows spans several queries only
    // with several queries per batch and a group size tile_q doesn't divide.
    constexpr int tile_q = get<0>(ShapeOut{});
    int group = args.num_heads_q / args.num_heads_kv;
    bool multi_q = args.q_per_batch > 1 && group > 1 && group % tile_q != 0;
    if (Paged && args.total_rows_kv > (int64_t(1) << 24)) {
      if constexpr (Paged) run<Paged, Local, true, true>(queue, args);
    } else if (multi_q) {
      run<Paged, Local, false, true>(queue, args);
    } else {
      run<Paged, Local, false, false>(queue, args);
    }
  }

  static void dispatch(sycl::queue& queue, decode_args_t const& args) {
    bool paged = args.block_table != nullptr;
    bool local = args.window_size_left >= 0;
    if (paged) {
      if (local) run_mode<true, true>(queue, args);
      else       run_mode<true, false>(queue, args);
    } else {
      if (local) run_mode<false, true>(queue, args);
      else       run_mode<false, false>(queue, args);
    }
  }
};

}  // namespace flash_xpu_splitkv

// One dtype per translation unit keeps per-TU compile memory bounded.
#define FLASH_XPU_SPLITKV_INSTANTIATE(HD, NAME, ELEMENT)                      \
  namespace flash_xpu_splitkv {                                               \
  template <>                                                                 \
  void splitkv_decode_##NAME<HD>(sycl::queue & q, const decode_args_t& a) {   \
    SplitKVConfig<ELEMENT, HD>::dispatch(q, a);                               \
  }                                                                           \
  }
