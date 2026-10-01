/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>
#include "deformable_detr/ms_deform_im2col_cuda.cuh"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

// The shim's stream accessor is guarded by USE_CUDA, so declare it here.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream);

#include <cuda.h>
#include <cuda_runtime.h>

using torch::stable::Tensor;

static cudaStream_t get_current_cuda_stream(const Tensor &t)
{
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr));
    return static_cast<cudaStream_t>(stream_ptr);
}

#define MS_DEFORM_ATTN_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                \
  THO_DISPATCH_SWITCH(                                                       \
      TYPE, NAME,                                                            \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__)   \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)    \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)     \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__))


Tensor ms_deform_attn_cuda_forward(
    const Tensor &value,
    const Tensor &spatial_shapes,
    const Tensor &level_start_index,
    const Tensor &sampling_loc,
    const Tensor &attn_weight,
    const int64_t im2col_step)
{
    const torch::stable::accelerator::DeviceGuard guard(value.get_device_index());

    STD_TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    STD_TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    STD_TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    STD_TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    STD_TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    STD_TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    STD_TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    STD_TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    STD_TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    STD_TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, static_cast<int>(im2col_step));

    STD_TORCH_CHECK(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto output = torch::stable::new_zeros(value, {batch, num_query, num_heads, channels});

    const int batch_n = im2col_step_;
    auto output_n = torch::stable::view(output, {batch/im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = torch::stable::select(output_n, 0, n);
        MS_DEFORM_ATTN_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
            ms_deformable_im2col_cuda(get_current_cuda_stream(value),
                value.const_data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.const_data_ptr<int64_t>(),
                level_start_index.const_data_ptr<int64_t>(),
                sampling_loc.const_data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.const_data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.mutable_data_ptr<scalar_t>());

        }));
    }

    output = torch::stable::view(output, {batch, num_query, num_heads*channels});

    return output;
}


std::vector<Tensor> ms_deform_attn_cuda_backward(
    const Tensor &value,
    const Tensor &spatial_shapes,
    const Tensor &level_start_index,
    const Tensor &sampling_loc,
    const Tensor &attn_weight,
    const Tensor &grad_output,
    const int64_t im2col_step)
{
    const torch::stable::accelerator::DeviceGuard guard(value.get_device_index());

    STD_TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    STD_TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    STD_TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    STD_TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    STD_TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    STD_TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    STD_TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    STD_TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    STD_TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    STD_TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    STD_TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    STD_TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, static_cast<int>(im2col_step));

    STD_TORCH_CHECK(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = torch::stable::new_zeros(value, value.sizes());
    auto grad_sampling_loc = torch::stable::new_zeros(sampling_loc, sampling_loc.sizes());
    auto grad_attn_weight = torch::stable::new_zeros(attn_weight, attn_weight.sizes());

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = torch::stable::view(grad_output, {batch/im2col_step_, batch_n, num_query, num_heads, channels});
    
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto grad_output_g = torch::stable::select(grad_output_n, 0, n);
        MS_DEFORM_ATTN_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(get_current_cuda_stream(value),
                                    grad_output_g.const_data_ptr<scalar_t>(),
                                    value.const_data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                                    spatial_shapes.const_data_ptr<int64_t>(),
                                    level_start_index.const_data_ptr<int64_t>(),
                                    sampling_loc.const_data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_weight.const_data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.mutable_data_ptr<scalar_t>() +  n * im2col_step_ * per_value_size,
                                    grad_sampling_loc.mutable_data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    grad_attn_weight.mutable_data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}
