#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("flash_attn(Tensor q, Tensor k, Tensor v, Tensor? mask, float scale) -> Tensor");
  // Takes no tensor, so it has no device to dispatch on and is registered as a catch-all. Each
  // backend's shared object is its own library namespace, so there is one implementation per build.
  ops.def("supports_flash_attn(int n_q, int head_dim_k, int head_dim_v) -> bool");
  ops.impl("supports_flash_attn", &supports_flash_attn);

  // The schema is the same for every backend; only the implementation differs.
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("flash_attn", torch::kCUDA, &flash_attn);
#elif defined(METAL_KERNEL)
  ops.impl("flash_attn", torch::kMPS, &flash_attn);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
