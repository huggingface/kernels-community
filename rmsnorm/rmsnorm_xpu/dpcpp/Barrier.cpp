#include "Utils.h"
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include "DPCPP.h"

using namespace kernels::xpu::dpcpp;

namespace at {
namespace AtenTypeXPU {

struct dpcpp_q_barrier_functor {
  void operator()() const {}
};

sycl::event dpcpp_q_barrier(sycl::queue& q) {
  return q.ext_oneapi_submit_barrier();
}

sycl::event dpcpp_q_barrier(sycl::queue& q, std::vector<sycl::event>& events) {
  return q.ext_oneapi_submit_barrier(events);
}
} // namespace AtenTypeXPU
} // namespace at
