#pragma once

#include <torch/torch.h>

void tinygrad_rms_norm(
    torch::Tensor& output,
    torch::Tensor& rms_inv,
    const torch::Tensor& input,
    double epsilon);

void tinygrad_rms_norm_inplace(
    torch::Tensor& output,
    const torch::Tensor& input,
    double epsilon);
