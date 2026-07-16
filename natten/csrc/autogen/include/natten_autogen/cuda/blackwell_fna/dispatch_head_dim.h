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
#include <natten_autogen/cuda/blackwell_fna/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e4m3."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e5m2."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e4m3."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e5m2."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e4m3."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (dim <= 32) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 64) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (dim <= 128) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for e5m2."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

