#pragma once

// Helpers for the host code on top of the Torch stable ABI.

#include <cuda_runtime.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/shim_utils.h>

// The shim's stream accessor is guarded by USE_CUDA, so declare it here.
extern "C" AOTITorchError
aoti_torch_get_current_cuda_stream(int32_t device_index, void **ret_stream);

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// Dispatches on the logits dtype, with the type available as `scalar_t`.
#define TDT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                           \
  THO_DISPATCH_SWITCH(                                                         \
      TYPE, NAME,                                                              \
      THO_PRIVATE_CASE_TYPE_USING_HINT(ScalarType::Float, scalar_t,            \
                                       __VA_ARGS__)                            \
          THO_PRIVATE_CASE_TYPE_USING_HINT(ScalarType::Half, scalar_t,         \
                                           __VA_ARGS__)                        \
              THO_PRIVATE_CASE_TYPE_USING_HINT(ScalarType::BFloat16, scalar_t, \
                                               __VA_ARGS__))

// Like STD_CUDA_KERNEL_LAUNCH_CHECK, which needs shim functions that are only
// declared with USE_CUDA.
#define TDT_CUDA_KERNEL_LAUNCH_CHECK()                                         \
  do {                                                                         \
    const cudaError_t err = cudaGetLastError();                                \
    STD_TORCH_CHECK(err == cudaSuccess,                                        \
                    "CUDA error: ", cudaGetErrorString(err));                  \
  } while (0)

namespace tdt_loss {

template <typename T> T *ptr(Tensor const &x) {
  return static_cast<T *>(x.data_ptr());
}

inline bool same_device(Tensor const &a, Tensor const &b) {
  return a.is_cuda() == b.is_cuda() &&
         a.get_device_index() == b.get_device_index();
}

inline bool same_sizes(Tensor const &a, Tensor const &b) {
  if (a.dim() != b.dim())
    return false;
  for (int64_t i = 0; i < a.dim(); ++i) {
    if (a.size(i) != b.size(i))
      return false;
  }
  return true;
}

inline cudaStream_t current_stream(Tensor const &x) {
  void *stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(x.get_device_index(), &stream));
  return static_cast<cudaStream_t>(stream);
}

} // namespace tdt_loss
