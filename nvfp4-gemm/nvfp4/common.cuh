#pragma once

// Stable-ABI replacements for the vLLM `libtorch_stable` helper headers
// (torch_utils.h, cutlass_extensions/common.hpp, core/math.hpp) used by the
// vendored NVFP4 kernels.
// Helper signatures mirror vLLM (https://github.com/vllm-project/vllm, Apache-2.0).

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/shim_utils.h>

// The shim's stream accessor is guarded by USE_CUDA, so declare it here.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream);

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

using torch::headeronly::ScalarType;

// Not `using torch::stable::Tensor`: the CUTLASS translation units pull in
// `using namespace cute`, where an unqualified `Tensor` is ambiguous.
using nvfp4_tensor = torch::stable::Tensor;

namespace tsa = torch::stable::accelerator;

#define CUTLASS_CHECK(status)                              \
  {                                                        \
    cutlass::Status error = status;                         \
    STD_TORCH_CHECK(error == cutlass::Status::kSuccess,     \
                    cutlassGetStatusString(error));         \
  }

// Stable-ABI equivalent of `at::cuda::getCurrentCUDAStream()`: the current
// stream of the active device. Call sites hold a DeviceGuard for the operand's
// device, so this is that tensor's stream.
inline cudaStream_t get_current_cuda_stream() {
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      static_cast<int32_t>(tsa::getCurrentDeviceIndex()), &stream));
  return static_cast<cudaStream_t>(stream);
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

// Dispatch over the half types without ATen's AT_DISPATCH machinery (defines
// `scalar_t` for the Half and BFloat16 cases).
#define VLLM_STABLE_DISPATCH_HALF_TYPES(TYPE, NAME, ...)                     \
  [&] {                                                                      \
    const ScalarType _st = TYPE;                                             \
    switch (_st) {                                                           \
      case ScalarType::Half: {                                               \
        using scalar_t = c10::Half;                                          \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case ScalarType::BFloat16: {                                           \
        using scalar_t = c10::BFloat16;                                      \
        return __VA_ARGS__();                                                \
      }                                                                      \
      default:                                                               \
        STD_TORCH_CHECK(false, NAME, " not implemented for dtype ", _st);    \
    }                                                                        \
  }()
