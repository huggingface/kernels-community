#pragma once

#include <torch/all.h>

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
void mul_and_silu(torch::Tensor& out, torch::Tensor& input);
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_new(torch::Tensor& out, torch::Tensor& input);
void gelu_fast(torch::Tensor& out, torch::Tensor& input);
void gelu_quick(torch::Tensor& out, torch::Tensor& input);
void swigluoai_and_mul(torch::Tensor& out, torch::Tensor& input, double alpha, double limit);
