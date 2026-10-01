#pragma once


#include <iostream> 
#include <stdexcept> 
#include <string> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/fna_blackwell/fna_forward.cuh> 
#include <natten_autogen/cuda/blackwell_fna/dispatch_head_dim.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_FORWARD_1D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e4m3fn) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e5m2) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e4m3fn) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e5m2) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e4m3fn) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e5m2) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

