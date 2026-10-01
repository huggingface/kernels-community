// Self-contained Philox RNG state + unpack helper.
//
// This replaces ATen's `at::PhiloxCudaState` and `at::cuda::philox::unpack`
// (formerly pulled in via <ATen/cuda/CUDAGeneratorImpl.h> and
// <ATen/cuda/detail/UnpackRaw.cuh>). Those ATen headers cannot be included in
// stable-ABI builds, where they hard-error under TORCH_STABLE_ONLY. The layout
// and semantics here mirror ATen's so the kernels are unchanged.

#pragma once

#include <cstdint>
#include <tuple>

#include "namespace_config.h"

namespace FLASH_NAMESPACE {

struct PhiloxCudaState {
  PhiloxCudaState() = default;

  // Non-captured (eager) state: literal seed and offset values.
  PhiloxCudaState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }

  // Captured (CUDA graph) state: pointers resolved at kernel launch time.
  PhiloxCudaState(int64_t *seed, int64_t *offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  union Payload {
    uint64_t val;
    int64_t *ptr;
  };

  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

// Namespace deliberately not named `philox`: FLASH_NAMESPACE already declares a
// `philox(...)` function in philox.cuh, which would collide.
namespace philox_compat {

__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t>
unpack(PhiloxCudaState arg) {
  if (arg.captured_) {
    // offset_intragraph_ counts thread-local subsequence usage within a graph.
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*arg.offset_.ptr) + arg.offset_intragraph_);
  } else {
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
  }
}

} // namespace philox_compat

} // namespace FLASH_NAMESPACE
