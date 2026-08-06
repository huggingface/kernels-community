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
#include <natten/cuda/fmha_blackwell/fmha_forward.cuh> 
#include <natten_autogen/cuda/blackwell_fmha/dispatch_head_dim.h> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 
#define DISPATCH_BLACKWELL_FMHA_FORWARD(dtype, dim, q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_float16(dim, q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16(dim, q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e4m3fn) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3(dim, q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dtype == c10::ScalarType::Float8_e5m2) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2(dim, q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

