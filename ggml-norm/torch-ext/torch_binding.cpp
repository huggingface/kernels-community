#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");

  // The schema is the same for every backend; only the implementation differs.
#if defined(METAL_KERNEL)
  ops.impl("rms_norm", torch::kMPS, &rms_norm);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
