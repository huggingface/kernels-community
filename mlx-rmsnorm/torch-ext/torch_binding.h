#pragma once

#include <ATen/ATen.h>

void launch_forward_kernel(const at::Tensor& input,
                           const at::Tensor& weight,
                           at::Tensor& output,
                           float epsilon);

void launch_backward_kernel(const at::Tensor& input,
                           const at::Tensor& weight,
                           const at::Tensor& grad_output,
                           at::Tensor& grad_input,
                           at::Tensor* grad_weight,
                           float epsilon);