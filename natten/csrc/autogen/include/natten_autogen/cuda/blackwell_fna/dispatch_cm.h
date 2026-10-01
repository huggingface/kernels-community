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
#include <natten_autogen/cuda/blackwell_fna/dispatch_tile_shape.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim32_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim32_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim64_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim64_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim128_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_float16_headdim128_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim32_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim32_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim64_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim64_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim128_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_bfloat16_headdim128_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim32_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim32_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim64_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim64_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim128_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e4m3_headdim128_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim32_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim32_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim64_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim64_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim128_causal0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D_e5m2_headdim128_causal1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-1D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_float16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_bfloat16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim32_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim32_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim32_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim32_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim64_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim64_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim64_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim64_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim128_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim128_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim128_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e4m3_headdim128_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim32_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim32_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim32_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim32_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim64_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim64_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim64_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim64_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim128_causal0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim128_causal0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim128_causal1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D_e5m2_headdim128_causal1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-2D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_float16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (float16, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_bfloat16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (bfloat16, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e4m3_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (e4m3, head_dim 128) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 32) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 64) got invalid causal mask!"); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D_e5m2_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
          throw std::runtime_error("Blackwell FNA-3D forward kernel dispatch failed! Causal mask dispatcher (e5m2, head_dim 128) got invalid causal mask!"); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

