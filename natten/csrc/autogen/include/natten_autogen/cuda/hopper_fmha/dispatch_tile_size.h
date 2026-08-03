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
#include <natten_autogen/cuda/hopper_fmha/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha_hopper { 
#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim32(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_64x128x32(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (float16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim64(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_64x128x64(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (float16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim128(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_128x128x128_coop(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_128x128x128_pp(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (float16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim256(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_128x64x256_coop(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (float16, head_dim 256): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_64x128x32(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (bfloat16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_64x128x64(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (bfloat16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_128x128x128_coop(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_128x128x128_pp(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (bfloat16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim256(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_128x64x256_coop(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FMHA forward kernel dispatch failed! It got invalid Q tile, KV tile, and schedule combination (bfloat16, head_dim 256): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + ", schedule=" + natten::cuda::hopper::to_string(kernel_type) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_hopper 
#endif 
#endif 

