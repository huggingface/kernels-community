/******************************************************************************
 * Shared plumbing for the SYCL/XPU backend: tuning constants, stride bundles,
 * launch helpers and the type/width dispatch macros.
 ******************************************************************************/

#pragma once

#include <torch/all.h>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <mutex>
#include <unordered_map>

#define CHECK_SHAPE(x, ...)                                                    \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                  \
              #x " must have shape (" #__VA_ARGS__ ")")

namespace causal_conv1d_xpu {

// The CUDA backend only supports widths 2..4; keep the same limit so that the
// unrolled register buffers below have a compile-time bound.
constexpr int kMaxWidth = 4;
// Number of output positions handled by a single work item in the forward and
// backward sequence kernels.
constexpr int kSeqChunk = 64;
constexpr int kWorkGroupSize = 128;
// Work groups per compute unit that the sequence kernels aim for. One group per
// unit would only saturate the machine if nothing ever stalled; these kernels
// are memory bound, so the unit needs several groups resident to have something
// to switch to while a load is outstanding.
//
// Swept on BMG-G21 (160 compute units, so this yields 81920 work items). The
// resulting threshold has a plateau on one side and a cliff on the other:
// 65536 and 131072 were indistinguishable, but 262144 shortened the chunk of
// short-sequence shapes all the way to 8, where re-priming the sliding window
// dominates and the worst shape fell to 0.50x of stock PyTorch. Disabling the
// mechanism entirely cost as much again. So pick a value inside the plateau and
// away from its upper edge rather than hunting for an optimum.
constexpr int kGroupsPerComputeUnit = 4;
// Positions each lane of the element-wise backward kernel owns. Neighbouring
// positions share most of their sliding window, so batching them cuts the work
// per element; that kernel turned out to be issue-bound rather than bandwidth-
// bound, which is what this trades against.
constexpr int kBwdItemsPerLane = 4;
// Same idea for the forward element-wise kernel.
constexpr int kFwdItemsPerLane = 4;

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int64_t round_up(int64_t a, int64_t b) { return ceil_div(a, b) * b; }

// Work items below which the device is not saturated and it pays to shorten the
// chunk, even though re-priming the sliding window costs a few extra loads.
// Derived from the device rather than fixed because the same kernel ships to
// everything from integrated graphics to datacentre parts, and a work item count
// tuned on one of them means something quite different on another.
inline int64_t min_work_items(const sycl::device &dev) {
  // Querying the device is a host-side call and this sits on the launch path of
  // kernels that can themselves be a few microseconds long, so memoise it.
  static std::mutex mu;
  static std::unordered_map<sycl::device, int64_t> cache;
  const std::lock_guard<std::mutex> guard(mu);
  auto it = cache.find(dev);
  if (it == cache.end()) {
    const int64_t compute_units =
        dev.get_info<sycl::info::device::max_compute_units>();
    it = cache
             .emplace(dev, compute_units * kWorkGroupSize *
                               kGroupsPerComputeUnit)
             .first;
  }
  return it->second;
}

// Shortens the chunk until there is enough work to fill the device.
inline int pick_seq_chunk(int64_t batch_dim, int seqlen, int64_t min_items) {
  int seq_chunk = kSeqChunk;
  while (seq_chunk > 8 &&
         batch_dim * ceil_div(seqlen, seq_chunk) < min_items) {
    seq_chunk /= 2;
  }
  return seq_chunk;
}

inline float silu(float x) { return x / (1.f + sycl::exp(-x)); }

// Derivative of silu w.r.t. its pre-activation input.
inline float dsilu(float x) {
  const float s = 1.f / (1.f + sycl::exp(-x));
  return s * (1.f + x * (1.f - s));
}

struct ConvStrides {
  int batch;
  int dim;
  int seqlen;
  bool silu_activation;

  int64_t x_batch_stride;
  int64_t x_c_stride;
  int64_t x_l_stride;
  int64_t weight_c_stride;
  int64_t weight_width_stride;
  int64_t out_batch_stride;
  int64_t out_c_stride;
  int64_t out_l_stride;
};

struct StateStrides {
  int64_t batch_stride;
  int64_t c_stride;
  int64_t l_stride;
};

////////////////////////////////////////////////////////////////////////////////
// Dispatch helpers
////////////////////////////////////////////////////////////////////////////////

inline StateStrides make_state_strides(const at::Tensor &t) {
  return StateStrides{t.stride(0), t.stride(1), t.stride(2)};
}

template <typename Kernel>
void launch_1d(sycl::queue &q, int64_t n_items, Kernel kernel) {
  if (n_items <= 0) {
    return;
  }
  const int64_t global = round_up(n_items, kWorkGroupSize);
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global),
                                       sycl::range<1>(kWorkGroupSize)),
                     kernel);
  });
}

// Like launch_1d, but the caller counts work groups instead of work items. Used
// by kernels that do a collective reduction, which needs full groups, and lets
// the caller pick the group size since it doubles as the tile width.
template <typename Kernel>
void launch_groups(sycl::queue &q, int64_t n_groups, int wg_size,
                   Kernel kernel) {
  if (n_groups <= 0) {
    return;
  }
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_groups * wg_size),
                                       sycl::range<1>(wg_size)),
                     kernel);
  });
}

#define DISPATCH_ITYPE(ITYPE, NAME, ...)                                       \
  if (ITYPE == at::ScalarType::Half) {                                         \
    using input_t = at::Half;                                                  \
    __VA_ARGS__();                                                             \
  } else if (ITYPE == at::ScalarType::BFloat16) {                              \
    using input_t = at::BFloat16;                                              \
    __VA_ARGS__();                                                             \
  } else if (ITYPE == at::ScalarType::Float) {                                 \
    using input_t = float;                                                     \
    __VA_ARGS__();                                                             \
  } else {                                                                     \
    AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
  }

#define DISPATCH_WTYPE(WTYPE, NAME, ...)                                       \
  if (WTYPE == at::ScalarType::Half) {                                         \
    using weight_t = at::Half;                                                 \
    __VA_ARGS__();                                                             \
  } else if (WTYPE == at::ScalarType::BFloat16) {                              \
    using weight_t = at::BFloat16;                                             \
    __VA_ARGS__();                                                             \
  } else if (WTYPE == at::ScalarType::Float) {                                 \
    using weight_t = float;                                                    \
    __VA_ARGS__();                                                             \
  } else {                                                                     \
    AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE),     \
             "'");                                                             \
  }

// Turning the filter width into a compile-time constant lets the compiler fully
// unroll the tap loops and keep the sliding window in registers.
#define DISPATCH_WIDTH(WIDTH, NAME, ...)                                       \
  if ((WIDTH) == 2) {                                                          \
    constexpr int kWidth = 2;                                                  \
    __VA_ARGS__();                                                             \
  } else if ((WIDTH) == 3) {                                                   \
    constexpr int kWidth = 3;                                                  \
    __VA_ARGS__();                                                             \
  } else if ((WIDTH) == 4) {                                                   \
    constexpr int kWidth = 4;                                                  \
    __VA_ARGS__();                                                             \
  } else {                                                                     \
    AT_ERROR(#NAME, " not implemented for width ", (WIDTH));                   \
  }

inline ConvStrides make_conv_strides(const at::Tensor &x, const at::Tensor &weight,
                              const at::Tensor &out, int batch, int dim,
                              int seqlen, bool silu_activation) {
  ConvStrides p{};
  p.batch = batch;
  p.dim = dim;
  p.seqlen = seqlen;
  p.silu_activation = silu_activation;
  p.x_batch_stride = x.stride(0);
  p.x_c_stride = x.stride(1);
  p.x_l_stride = x.stride(-1);
  p.weight_c_stride = weight.stride(0);
  p.weight_width_stride = weight.stride(1);
  p.out_batch_stride = out.stride(0);
  p.out_c_stride = out.stride(1);
  p.out_l_stride = out.stride(-1);
  return p;
}

} // namespace causal_conv1d_xpu
