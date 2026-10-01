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
#include <natten_autogen/cuda/blackwell_fmha/dispatch_tile_size.h> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 
#define DISPATCH_BLACKWELL_FMHA_FORWARD_float16(dim, q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim32(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim64(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim128(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16(dim, q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3(dim, q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3_headdim32(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3_headdim64(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3_headdim128(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e4m3."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2(dim, q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2_headdim32(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2_headdim64(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2_headdim128(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e5m2."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

