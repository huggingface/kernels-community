#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("affine_qmm_t(Tensor x, Tensor w, Tensor scales, Tensor biases, int group_size, int bits) -> Tensor");
  ops.impl("affine_qmm_t", torch::kMPS, &affine_qmm_t);
  // Not part of the Python API: the tests use it (through `_ops`) to check which kernel the
  // dispatch picks. Takes no tensor, so it is registered as a catch-all.
  ops.def("kernel_for(int M, int N, int K, int group_size, int bits, ScalarType dtype) -> str");
  ops.impl("kernel_for", &kernel_for);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
