#pragma once

#include "DPCPP.h"

using namespace kernels::xpu::dpcpp;

namespace at {
namespace AtenTypeXPU {

sycl::event dpcpp_q_barrier(sycl::queue& q);
sycl::event dpcpp_q_barrier(sycl::queue& q, std::vector<sycl::event>& events);

} // namespace AtenTypeXPU
} // namespace at
