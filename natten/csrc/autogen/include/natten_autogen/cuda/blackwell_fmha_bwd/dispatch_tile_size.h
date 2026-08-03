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
#include <natten/cuda/fmha_blackwell/fmha_backward.cuh> 
#include <natten_autogen/cuda/blackwell_fmha_bwd/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 
#define DISPATCH_BLACKWELL_FMHA_BACKWARD_float16_headdim32(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_float16_128x128x32(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_float16_headdim64(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_float16_128x128x64(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_float16_headdim128(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_float16_128x128x128(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_bfloat16_128x128x32(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_bfloat16_128x128x64(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_bfloat16_128x128x128(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

