#pragma once


#include <iostream> 
#include <stdexcept> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/reference/fna_reference_forward.hpp> 
#include <natten/cuda/reference/fna_reference_backward.hpp> 
#include <natten_autogen/cuda/reference/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace reference { 
#define DISPATCH_REFERENCE_FNA_FORWARD_1D(dtype, is_causal, ...) \
  [&] { \
    if (dtype == at::kFloat) { \
      DISPATCH_REFERENCE_FNA_FORWARD_1D_float32(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kHalf) { \
      DISPATCH_REFERENCE_FNA_FORWARD_1D_float16(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_REFERENCE_FNA_FORWARD_1D_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Reference FNA kernel dispatch failed! Reference FNA-1D does not support this data type."); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_1D(dtype, is_causal, ...) \
  [&] { \
    if (dtype == at::kFloat) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_1D_float32(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kHalf) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_1D_float16(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_1D_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Reference FNA kernel dispatch failed! Reference FNA-1D does not support this data type."); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_2D(dtype, is_causal, ...) \
  [&] { \
    if (dtype == at::kFloat) { \
      DISPATCH_REFERENCE_FNA_FORWARD_2D_float32(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kHalf) { \
      DISPATCH_REFERENCE_FNA_FORWARD_2D_float16(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_REFERENCE_FNA_FORWARD_2D_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Reference FNA kernel dispatch failed! Reference FNA-2D does not support this data type."); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_2D(dtype, is_causal, ...) \
  [&] { \
    if (dtype == at::kFloat) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_2D_float32(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kHalf) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_2D_float16(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_2D_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Reference FNA kernel dispatch failed! Reference FNA-2D does not support this data type."); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_3D(dtype, is_causal, ...) \
  [&] { \
    if (dtype == at::kFloat) { \
      DISPATCH_REFERENCE_FNA_FORWARD_3D_float32(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kHalf) { \
      DISPATCH_REFERENCE_FNA_FORWARD_3D_float16(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_REFERENCE_FNA_FORWARD_3D_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Reference FNA kernel dispatch failed! Reference FNA-3D does not support this data type."); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_3D(dtype, is_causal, ...) \
  [&] { \
    if (dtype == at::kFloat) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_3D_float32(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kHalf) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_3D_float16(is_causal, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_3D_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Reference FNA kernel dispatch failed! Reference FNA-3D does not support this data type."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace reference 
#endif 

