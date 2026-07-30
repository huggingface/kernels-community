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
#include <natten/cuda/fna_hopper/fna_backward.cuh> 
#include <natten_autogen/cuda/hopper_fna_bwd/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim32_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_64x128x32_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_128x128x32_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim32_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_64x128x32_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_128x128x32_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim64_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_64x128x64_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_128x128x64_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim64_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_64x128x64_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_128x128x64_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim128_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_64x128x128_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim128_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_float16_64x128x128_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim32_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_64x128x32_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_128x128x32_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim32_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_64x128x32_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_128x128x32_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim64_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_64x128x64_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_128x128x64_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim64_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_64x128x64_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_128x128x64_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim128_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_64x128x128_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim128_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_hopper::hopper_fna1d_backward_bfloat16_64x128x128_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x32_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x64_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_float16_64x128x128_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x32_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x64_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_hopper::hopper_fna2d_backward_bfloat16_64x128x128_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q2x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_float16_64x128x128_Q1x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q2x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_hopper::hopper_fna3d_backward_bfloat16_64x128x128_Q1x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Hopper FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_hopper 
#endif 
#endif 

