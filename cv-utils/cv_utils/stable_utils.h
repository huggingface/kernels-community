#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

// The shim's stream accessor is guarded by USE_CUDA, so declare it here.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream);

#include <cuda_runtime.h>

inline cudaStream_t get_current_cuda_stream(const torch::stable::Tensor& t) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}
