#pragma once

#include <c10/xpu/XPUFunctions.h>

namespace kernels::xpu {
namespace dpcpp {

using DeviceId = at::DeviceIndex;

bool dpcppGetDeviceHasXMX(DeviceId device_id = 0) noexcept;

bool dpcppGetDeviceHas2DBlock(DeviceId device_id = 0) noexcept;

} // namespace dpcpp
} // namespace xpu
