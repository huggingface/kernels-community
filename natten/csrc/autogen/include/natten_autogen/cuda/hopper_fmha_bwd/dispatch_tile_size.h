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
#include <natten/cuda/fmha_hopper/fmha_backward.cuh> 
#include <natten_autogen/cuda/hopper_fmha_bwd/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha_hopper { 
#define DISPATCH_HOPPER_FMHA_BACKWARD_float16_headdim32(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_float16_64x128x32(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_float16_128x128x32(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_BACKWARD_float16_headdim64(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_float16_64x128x64(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_float16_128x128x64(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_BACKWARD_float16_headdim128(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_float16_64x128x128(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_BACKWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_bfloat16_64x128x32(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_bfloat16_128x128x32(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_BACKWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_bfloat16_64x128x64(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_bfloat16_128x128x64(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_BACKWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_hopper::hopper_fmha_backward_bfloat16_64x128x128(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_hopper 
#endif 
#endif 

