#pragma once

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

// The shim's stream accessor is guarded by USE_CUDA, so declare it here.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream);

// Returns the current CUDA/HIP stream of the tensor's device as an opaque
// pointer. Callers cast it to cudaStream_t in their .cu file, so that this
// header does not contain CUDA runtime names that hipify would have to rewrite.
inline void* current_stream_ptr(int32_t device_index) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return stream_ptr;
}

inline void* current_stream_ptr(const torch::stable::Tensor& t) {
  return current_stream_ptr(t.get_device_index());
}
