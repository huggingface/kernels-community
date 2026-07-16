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
#include <natten/cuda/fna_hopper/fna_forward.cuh> 
#include <natten_autogen/cuda/hopper_fna/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim256(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim256(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim256(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim256(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for float16."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256(is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA forward kernel dispatch failed! It does not support head dim " + std::to_string(dim) + " for bfloat16."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_hopper 
#endif 
#endif 

