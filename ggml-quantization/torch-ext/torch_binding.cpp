#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "get_rows(Tensor blocks, Tensor indices, int ggml_type, int cols, ScalarType dtype) -> Tensor");
  ops.def("dequantize(Tensor blocks, int ggml_type, int rows, int cols, ScalarType dtype) -> Tensor");
  ops.def("mul_mat_vec(Tensor blocks, Tensor x, int ggml_type, int out_features) -> Tensor");
  ops.def(
      "mul_mat_id(Tensor blocks, Tensor x, Tensor ids, int ggml_type, int out_features) -> Tensor");
  // Takes no tensor, so it has no device to dispatch on and is registered as a catch-all. Each
  // backend's shared object is its own library namespace, so there is one implementation per build.
  ops.def("gemv_types() -> int[]");
  ops.impl("gemv_types", &gemv_types);

  // The schema is the same for every backend; only the implementation differs.
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("get_rows", torch::kCUDA, &get_rows);
  ops.impl("dequantize", torch::kCUDA, &dequantize);
  ops.impl("mul_mat_vec", torch::kCUDA, &mul_mat_vec);
#elif defined(METAL_KERNEL)
  ops.impl("get_rows", torch::kMPS, &get_rows);
  ops.impl("dequantize", torch::kMPS, &dequantize);
  ops.impl("mul_mat_vec", torch::kMPS, &mul_mat_vec);
  ops.impl("mul_mat_id", torch::kMPS, &mul_mat_id);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
