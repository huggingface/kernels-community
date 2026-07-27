#pragma once

#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> scaled_fp4_quant(
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    bool is_sf_swizzled_layout);

torch::Tensor cutlass_scaled_fp4_mm(torch::Tensor const& A,
                                    torch::Tensor const& B,
                                    torch::Tensor const& A_sf,
                                    torch::Tensor const& B_sf,
                                    torch::Tensor const& alpha);

torch::Tensor nvfp4_gemv(torch::Tensor const& A, torch::Tensor const& B,
                         torch::Tensor const& B_sf, torch::Tensor const& alpha);
