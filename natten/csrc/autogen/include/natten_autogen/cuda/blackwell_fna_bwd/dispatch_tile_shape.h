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
#include <natten_autogen/cuda/blackwell_fna_bwd/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim32_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_float16_128x128x32_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim32_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_float16_128x128x32_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim64_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_float16_128x128x64_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim64_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_float16_128x128x64_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim128_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_float16_128x128x128_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim128_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_float16_128x128x128_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim32_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_bfloat16_128x128x32_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim32_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_bfloat16_128x128x32_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim64_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_bfloat16_128x128x64_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim64_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_bfloat16_128x128x64_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim128_causal0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_bfloat16_128x128x128_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim128_causal1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128) { \
  natten::cuda::fna_blackwell::blackwell_fna1d_backward_bfloat16_128x128x128_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-1D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x32_Q8x16_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x64_Q8x16_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_float16_128x128x128_Q8x16_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x32_Q8x16_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x64_Q8x16_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q16x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna2d_backward_bfloat16_128x128x128_Q8x16_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-2D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q1x8x16_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x4x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x2x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q4x4x8_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x32_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q1x8x16_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x4x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x2x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q4x4x8_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x64_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q1x8x16_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x4x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x2x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q4x4x8_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_float16_128x128x128_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q1x8x16_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x4x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x2x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q4x4x8_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x32_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q1x8x16_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x4x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x2x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q4x4x8_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x64_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 1 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q1x8x16_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x4x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 2 && cute::get<2>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x2x16_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 16) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q4x4x8_KV2x4x16_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8) { \
  natten::cuda::fna_blackwell::blackwell_fna3d_backward_bfloat16_128x128x128_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          throw std::runtime_error("Blackwell FNA-3D backward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=(" + std::to_string(cute::get<0>(q_tile_shape)) + "," + std::to_string(cute::get<1>(q_tile_shape)) + "," + std::to_string(cute::get<2>(q_tile_shape)) + "), kv_tile=(" + std::to_string(cute::get<0>(kv_tile_shape)) + "," + std::to_string(cute::get<1>(kv_tile_shape)) + "," + std::to_string(cute::get<2>(kv_tile_shape)) + ")."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

