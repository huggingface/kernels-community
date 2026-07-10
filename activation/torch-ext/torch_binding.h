#pragma once

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
// Stable-ABI signatures
#include <torch/csrc/stable/tensor.h>

using activation_tensor = torch::stable::Tensor;

void silu_and_mul(activation_tensor& out, activation_tensor const& input);

void mul_and_silu(activation_tensor& out, activation_tensor const& input);

void gelu_and_mul(activation_tensor& out, activation_tensor const& input);

void gelu_tanh_and_mul(activation_tensor& out, activation_tensor const& input);

void fatrelu_and_mul(activation_tensor& out, activation_tensor const& input,
                     double threshold);

void gelu_new(activation_tensor& out, activation_tensor const& input);

void gelu_fast(activation_tensor& out, activation_tensor const& input);

void gelu_quick(activation_tensor& out, activation_tensor const& input);

void gelu_tanh(activation_tensor& out, activation_tensor const& input);

void silu(activation_tensor& out, activation_tensor const& input);

void gelu(activation_tensor& out, activation_tensor const& input);

#else
// Non-stable signatures (used by the Metal backend).
#include <torch/torch.h>

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor &out, torch::Tensor &input);

void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input);

void fatrelu_and_mul(torch::Tensor &out, torch::Tensor &input,
                     double threshold);

void gelu_new(torch::Tensor &out, torch::Tensor &input);

void gelu_fast(torch::Tensor &out, torch::Tensor &input);

void gelu_quick(torch::Tensor &out, torch::Tensor &input);

void gelu_tanh(torch::Tensor &out, torch::Tensor &input);

void silu(torch::Tensor &out, torch::Tensor &input);

void gelu(torch::Tensor &out, torch::Tensor &input);

#endif
