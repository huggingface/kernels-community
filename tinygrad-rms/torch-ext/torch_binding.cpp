#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("tinygrad_rms_norm(Tensor! out, Tensor! rms_inv, Tensor input, float epsilon) -> ()");
  ops.def("tinygrad_rms_norm_inplace(Tensor! out, Tensor input, float epsilon) -> ()");

  ops.impl("tinygrad_rms_norm", torch::kCUDA, &tinygrad_rms_norm);
  ops.impl("tinygrad_rms_norm_inplace", torch::kCUDA, &tinygrad_rms_norm_inplace);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
