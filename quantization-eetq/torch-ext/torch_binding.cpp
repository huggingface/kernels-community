#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("w8_a16_gemm(Tensor input, Tensor weight, Tensor scale) -> Tensor");
  ops.impl("w8_a16_gemm", torch::kCUDA, &w8_a16_gemm_forward_cuda);
  ops.def("w8_a16_gemm_(Tensor input, Tensor weight, Tensor scale, Tensor! output,"
                      "int m, int n, int k) -> Tensor");
  ops.impl("w8_a16_gemm_", torch::kCUDA, &w8_a16_gemm_forward_cuda_);
  ops.def("preprocess_weights(Tensor origin_weight, bool is_int4) -> Tensor");
  ops.impl("preprocess_weights", torch::kCUDA, &preprocess_weights_cuda);
  ops.def("quant_weights(Tensor origin_weight, ScalarType quant_type,"
                      "bool return_unprocessed_quantized_tensor) -> Tensor[]");
  ops.impl("quant_weights", torch::kCPU, &symmetric_quantize_last_axis_of_tensor);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
