#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("top_k(Tensor logits, int k, bool softmax) -> Tensor[]");

#if defined(METAL_KERNEL)
  ops.impl("top_k", torch::kMPS, &top_k);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
