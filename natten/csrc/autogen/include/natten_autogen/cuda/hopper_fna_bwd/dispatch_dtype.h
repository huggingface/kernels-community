#pragma once


#include <iostream> 
#include <stdexcept> 
#include <string> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_HOPPER_FNA
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/fna_hopper/fna_backward.cuh> 
#include <natten_autogen/cuda/hopper_fna_bwd/dispatch_head_dim.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_BACKWARD_1D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_hopper 
#endif 
#endif 

