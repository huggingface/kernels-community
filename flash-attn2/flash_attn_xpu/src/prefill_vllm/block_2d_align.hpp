/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {
// Arch-tagged inline namespace: gives these definitions a mangled name
// distinct from the other Xe architecture's identically named copies,
// while leaving name lookup (cutlass::fmha::...) unchanged.
inline namespace kc_vllm_prefill_xe2 {

// Xe LSC block-2D loads/stores require a 64-byte aligned surface base
// address. The attention kernels hand each work-group a per-head 2D slice of
// Q/K/V/O whose base is head_idx * head_stride bytes into the allocation, so
// any head stride that is not a multiple of 64 bytes (head size 72/80 in
// half/bf16, 72 in fp8, ...) yields a misaligned base on some heads. Xe2 (BMG)
// tolerates this; PVC addresses from the rounded-down base and reads/writes
// the neighbouring head instead.
//
// Round the base down to 64 bytes and fold the remainder into the X
// (contiguous) coordinate: the surface is widened by the same amount so its
// right-hand bound still ends at this head, and coordinate tensors built on
// the slice are shifted by `x_shift` elements (make_shifted_identity_tensor).
// An already aligned base makes both a no-op (x_shift == 0).
template <int XMode, class Engine, class Layout>
CUTLASS_DEVICE auto
align_block_2d_base(cute::Tensor<Engine, Layout> const& t, int& x_shift) {
  using T = cute::remove_cvref_t<typename Engine::value_type>;
  constexpr uintptr_t kAlign = 64;
  auto raw = reinterpret_cast<uintptr_t>(cute::raw_pointer_cast(t.data()));
  uintptr_t misalign = raw & (kAlign - 1);
  x_shift = static_cast<int>(misalign / sizeof(T));
  auto shape =
      cute::replace<XMode>(t.shape(), cute::get<XMode>(t.shape()) + x_shift);
  return cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<T*>(raw - misalign)),
      cute::make_layout(shape, t.stride()));
}

// Rank-2 identity (coordinate) tensor over `shape`, shifted by `x_shift` in
// XMode so that copies built on an align_block_2d_base() surface still address
// the original head.
template <int XMode, class Shape>
CUTLASS_DEVICE auto
make_shifted_identity_tensor(Shape const& shape, int x_shift) {
  static_assert(cute::rank_v<Shape> == 2, "expected a rank-2 shape");
  if constexpr (XMode == 0) {
    return cute::domain_offset(
        cute::make_coord(x_shift, 0), cute::make_identity_tensor(shape));
  } else {
    return cute::domain_offset(
        cute::make_coord(0, x_shift), cute::make_identity_tensor(shape));
  }
}

}  // namespace kc_vllm_prefill_xe2
}  // namespace cutlass::fmha::collective
