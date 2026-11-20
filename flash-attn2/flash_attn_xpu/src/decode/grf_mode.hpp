/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace flash_xpu_grf {

// Kernel properties with a GRF size of N.
//
// N == 0 or 512 sets no GRF property. The CRI image is built with
// [general.xpu] cri-grf-size = 512 (ocloc -cl-intel-512-GRF-per-thread),
// which gives 512 GRF to every kernel that does not set a size; other
// devices pick the size automatically.
//
// 128 is not allowed: with the 512 option, IGC gives a grf_size<128> kernel
// 32 registers on CRI (it spills, and some kernels end up with no CRI code).
template <int N, typename... Props>
constexpr auto with_grf(Props... props) {
  static_assert(N == 0 || N == 256 || N == 512, "GRF size: 0, 256 or 512");
  if constexpr (N == 0 || N == 512) {
    return sycl::ext::oneapi::experimental::properties{props...};
  } else {
    return sycl::ext::oneapi::experimental::properties{
        props..., sycl::ext::intel::experimental::grf_size<N>};
  }
}

}  // namespace flash_xpu_grf
