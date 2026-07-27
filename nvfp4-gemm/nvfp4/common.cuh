#pragma once

// Classic-torch replacements for the vLLM `libtorch_stable` helper headers
// (torch_utils.h, cutlass_extensions/common.hpp, core/math.hpp) used by the
// vendored NVFP4 kernels.
// Helper signatures mirror vLLM (https://github.com/vllm-project/vllm, Apache-2.0).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <climits>
#include <cstdint>

#define CUTLASS_CHECK(status)                                 \
  {                                                           \
    cutlass::Status error = status;                           \
    TORCH_CHECK(error == cutlass::Status::kSuccess,           \
                cutlassGetStatusString(error));               \
  }

inline int32_t get_sm_version_num() {
  int device;
  cudaGetDevice(&device);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  return major * 10 + minor;
}

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}


namespace vllm {
// vLLM's batch-invariant mode is not exposed here; always use the fast path.
inline bool vllm_is_batch_invariant() { return false; }
}  // namespace vllm

inline int get_device_attribute(cudaDeviceAttr attr, int device) {
  if (device < 0) {
    cudaGetDevice(&device);
  }
  int value = 0;
  cudaDeviceGetAttribute(&value, attr, device);
  return value;
}

// Classic-ATen replacement for vLLM's stable dispatch macro (defines
// `scalar_t` for Half and BFloat16 cases).
#define VLLM_STABLE_DISPATCH_HALF_TYPES(TYPE, NAME, ...)         \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                 \
                     AT_DISPATCH_CASE(at::ScalarType::Half,      \
                                      __VA_ARGS__)               \
                     AT_DISPATCH_CASE(at::ScalarType::BFloat16,  \
                                      __VA_ARGS__))
