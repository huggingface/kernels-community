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
#include <natten/cuda/hopper_fmha_fna.h> 
#include <natten/cuda/fmha_hopper/fmha_forward.cuh> 
#include <natten_autogen/cuda/hopper_fmha/dispatch_head_dim.h> 
namespace natten { 
namespace cuda { 
namespace fmha_hopper { 
#define DISPATCH_HOPPER_FMHA_FORWARD(dtype, dim, q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (dtype == at::kHalf) { \
      DISPATCH_HOPPER_FMHA_FORWARD_float16(dim, q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else if (dtype == at::kBFloat16) { \
      DISPATCH_HOPPER_FMHA_FORWARD_bfloat16(dim, q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It does not support dtype " + std::string(c10::toString(dtype)) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_hopper 
#endif 
#endif 

