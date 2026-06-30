#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
#include <torch/csrc/stable/library.h>
#else
#include <torch/library.h>
#endif

#include "registration.h"
#include "torch_binding.h"

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)

// Stable-ABI registration
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");

  // GELU with `tanh` approximation.
  ops.def("gelu_tanh(Tensor! out, Tensor input) -> ()");

  // SiLU implementation.
  ops.def("silu(Tensor! out, Tensor input) -> ()");

  // GELU with none approximation.
  ops.def("gelu(Tensor! out, Tensor input) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  ops.impl("mul_and_silu", TORCH_BOX(&mul_and_silu));
  ops.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  ops.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  ops.impl("fatrelu_and_mul", TORCH_BOX(&fatrelu_and_mul));
  ops.impl("gelu_new", TORCH_BOX(&gelu_new));
  ops.impl("gelu_fast", TORCH_BOX(&gelu_fast));
  ops.impl("gelu_quick", TORCH_BOX(&gelu_quick));
  ops.impl("gelu_tanh", TORCH_BOX(&gelu_tanh));
  ops.impl("silu", TORCH_BOX(&silu));
  ops.impl("gelu", TORCH_BOX(&gelu));
}

#else

// Non-stable registration - Metal
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kMPS, &silu_and_mul);

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.impl("mul_and_silu", torch::kMPS, &mul_and_silu);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kMPS, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kMPS, &gelu_tanh_and_mul);

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  ops.impl("fatrelu_and_mul", torch::kMPS, &fatrelu_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kMPS, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kMPS, &gelu_fast);

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kMPS, &gelu_quick);

  // GELU with `tanh` approximation.
  ops.def("gelu_tanh(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh", torch::kMPS, &gelu_tanh);

  // SiLU implementation.
  ops.def("silu(Tensor! out, Tensor input) -> ()");
  ops.impl("silu", torch::kMPS, &silu);

  // GELU with none approximation.
  ops.def("gelu(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu", torch::kMPS, &gelu);
}

#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
