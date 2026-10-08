#pragma once

#include "utils.h"

void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void mul_and_silu(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_tanh_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_new(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_fast(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_quick(torch::stable::Tensor& out, torch::stable::Tensor& input);
void swigluoai_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input, double alpha, double limit);
