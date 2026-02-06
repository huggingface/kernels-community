#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
#elif defined(METAL_KERNEL)
  ops.impl("silu_and_mul", torch::kMPS, &silu_and_mul);
#endif

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("mul_and_silu", torch::kCUDA, &mul_and_silu);
#elif defined(METAL_KERNEL)
  ops.impl("mul_and_silu", torch::kMPS, &mul_and_silu);
#endif

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);
#elif defined(METAL_KERNEL)
  ops.impl("gelu_and_mul", torch::kMPS, &gelu_and_mul);
#endif

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);
#elif defined(METAL_KERNEL)
  ops.impl("gelu_tanh_and_mul", torch::kMPS, &gelu_tanh_and_mul);
#endif

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("fatrelu_and_mul", torch::kCUDA, &fatrelu_and_mul);
#elif defined(METAL_KERNEL)
  ops.impl("fatrelu_and_mul", torch::kMPS, &fatrelu_and_mul);
#endif

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);
#elif defined(METAL_KERNEL)
  ops.impl("gelu_new", torch::kMPS, &gelu_new);
#endif

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);
#elif defined(METAL_KERNEL)
  ops.impl("gelu_fast", torch::kMPS, &gelu_fast);
#endif

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gelu_quick", torch::kCUDA, &gelu_quick);
#elif defined(METAL_KERNEL)
  ops.impl("gelu_quick", torch::kMPS, &gelu_quick);
#endif

  // GELU with `tanh` approximation.
  ops.def("gelu_tanh(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gelu_tanh", torch::kCUDA, &gelu_tanh);
#elif defined(METAL_KERNEL)
  ops.impl("gelu_tanh", torch::kMPS, &gelu_tanh);
#endif

  // SiLU implementation.
  ops.def("silu(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("silu", torch::kCUDA, &silu);
#elif defined(METAL_KERNEL)
  ops.impl("silu", torch::kMPS, &silu);
#endif

  // GELU with none approximation.
  ops.def("gelu(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("gelu", torch::kCUDA, &gelu);
#elif defined(METAL_KERNEL)
  ops.impl("gelu", torch::kMPS, &gelu);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
