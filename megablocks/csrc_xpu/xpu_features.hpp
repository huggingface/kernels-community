/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// XPU Device Feature Detection for MegaBlocks

#pragma once

#include <ATen/xpu/XPUContext.h>
#include <cstdlib>

namespace megablocks {
namespace xpu {

// XPU device feature detection, keyed on the device IP version reported by the
// driver (pvc 12.x, bmg 20.x, CRI 35.x).
class XPUFeatures {
 public:
  // True when the device is Xe35 (CRI) or newer, which selects the kernels
  // built from the CRI-only translation unit.
  static bool isXe35(c10::DeviceIndex device) {
    return ipVersionMajor(device) >= 35;
  }

  // Major component of the device IP version, e.g. 20 for "20.1.0".
  static int ipVersionMajor(c10::DeviceIndex device) {
    return std::atoi(at::xpu::getDeviceProperties(device)->version.c_str());
  }
};

}  // namespace xpu
}  // namespace megablocks
