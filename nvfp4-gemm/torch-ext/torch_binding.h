#pragma once

#include <tuple>

#include <torch/csrc/stable/tensor.h>

using nvfp4_tensor = torch::stable::Tensor;

std::tuple<nvfp4_tensor, nvfp4_tensor> scaled_fp4_quant(
    nvfp4_tensor const& input, nvfp4_tensor const& input_global_scale,
    bool is_sf_swizzled_layout);

nvfp4_tensor cutlass_scaled_fp4_mm(nvfp4_tensor const& A,
                                   nvfp4_tensor const& B,
                                   nvfp4_tensor const& A_sf,
                                   nvfp4_tensor const& B_sf,
                                   nvfp4_tensor const& alpha);

nvfp4_tensor nvfp4_gemv(nvfp4_tensor const& A, nvfp4_tensor const& B,
                        nvfp4_tensor const& B_sf, nvfp4_tensor const& alpha);

nvfp4_tensor nvfp4_gemv_swiglu(nvfp4_tensor const& A, nvfp4_tensor const& B,
                               nvfp4_tensor const& B_sf,
                               nvfp4_tensor const& alpha);

nvfp4_tensor nvfp4_gemv_gated(nvfp4_tensor const& A, nvfp4_tensor const& G,
                              nvfp4_tensor const& B, nvfp4_tensor const& B_sf,
                              nvfp4_tensor const& alpha);
