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
#include <natten/cuda/fna_blackwell/fna_backward.cuh> 
#include <natten_autogen/cuda/blackwell_fna_bwd/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA backward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA backward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA backward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA backward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA backward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA backward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

