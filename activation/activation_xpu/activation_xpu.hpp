#pragma once

#include <torch/torch.h>

namespace activation_xpu {

void silu_and_mul(torch::Tensor& out, const torch::Tensor& input);

void mul_and_silu(torch::Tensor& out, const torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, const torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, const torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, const torch::Tensor& input,
                     double threshold);

void gelu_new(torch::Tensor& out, const torch::Tensor& input);

void gelu_fast(torch::Tensor& out, const torch::Tensor& input);

void gelu_quick(torch::Tensor& out, const torch::Tensor& input);

void gelu(torch::Tensor& out, const torch::Tensor& input);

void gelu_tanh(torch::Tensor& out, const torch::Tensor& input);

void silu(torch::Tensor& out, const torch::Tensor& input);

}  // namespace activation_xpu
