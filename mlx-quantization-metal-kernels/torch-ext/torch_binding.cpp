#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // FP-quantized (MXFP4) operations
  ops.def("mxfp4_qmm_n(Tensor x, Tensor w, Tensor scales, int output_features) -> Tensor");
  ops.def("mxfp4_qmv(Tensor x, Tensor w, Tensor scales, int output_features) -> Tensor");

  // Affine quantized operations
  ops.def("affine_qmv(Tensor x, Tensor w, Tensor scales, Tensor biases, int group_size, int bits, int output_features) -> Tensor");
  ops.def("affine_qmm_t(Tensor x, Tensor w, Tensor scales, Tensor biases, int group_size, int bits) -> Tensor");
  ops.def("affine_qmm_n(Tensor x, Tensor w, Tensor scales, Tensor biases, int group_size, int bits, int output_features) -> Tensor");

}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, MPS, ops) {
  // FP-quantized (MXFP4)
  ops.impl("mxfp4_qmm_n", mxfp4_qmm_n);
  ops.impl("mxfp4_qmv", mxfp4_qmv);

  // Affine quantized
  ops.impl("affine_qmv", affine_qmv);
  ops.impl("affine_qmm_t", affine_qmm_t);
  ops.impl("affine_qmm_n", affine_qmm_n);

}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
