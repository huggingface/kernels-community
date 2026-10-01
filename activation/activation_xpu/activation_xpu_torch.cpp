#include <torch/all.h>

#include "activation_xpu.hpp"

// Global entry points referenced by torch-ext/torch_binding.cpp for the XPU
// backend. They forward to the SYCL implementations, which perform the shape
// and device validation.

void silu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::silu_and_mul(out, input);
}

void mul_and_silu(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::mul_and_silu(out, input);
}

void gelu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::gelu_and_mul(out, input);
}

void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::gelu_tanh_and_mul(out, input);
}

void fatrelu_and_mul(torch::Tensor &out, torch::Tensor &input,
                     double threshold) {
  activation_xpu::fatrelu_and_mul(out, input, threshold);
}

void gelu_new(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::gelu_new(out, input);
}

void gelu_fast(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::gelu_fast(out, input);
}

void gelu_quick(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::gelu_quick(out, input);
}

void gelu_tanh(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::gelu_tanh(out, input);
}

void silu(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::silu(out, input);
}

void gelu(torch::Tensor &out, torch::Tensor &input) {
  activation_xpu::gelu(out, input);
}
