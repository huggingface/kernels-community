#pragma once

#include <vector>

#include <torch/csrc/stable/tensor.h>

torch::stable::Tensor ms_deform_attn_cuda_forward(
    const torch::stable::Tensor &value,
    const torch::stable::Tensor &spatial_shapes,
    const torch::stable::Tensor &level_start_index,
    const torch::stable::Tensor &sampling_loc,
    const torch::stable::Tensor &attn_weight, const int64_t im2col_step);

std::vector<torch::stable::Tensor> ms_deform_attn_cuda_backward(
    const torch::stable::Tensor &value,
    const torch::stable::Tensor &spatial_shapes,
    const torch::stable::Tensor &level_start_index,
    const torch::stable::Tensor &sampling_loc,
    const torch::stable::Tensor &attn_weight,
    const torch::stable::Tensor &grad_output, const int64_t im2col_step);
