#pragma once
#include <torch/torch.h>

torch::Tensor rms_norm_impl(
    const torch::Tensor& input,
    at::IntArrayRef normalized_shape,
    const torch::Tensor& weight,
    double epsilon);