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
#include <natten_autogen/cuda/blackwell_fmha/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 
#define DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim32(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x32_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x32(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim64(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x64_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x64(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim128(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x128_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x128(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (float16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x32_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x32(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x64_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x64(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x128_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x128(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (bfloat16, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3_headdim32(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e4m3_256x128x32_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e4m3_256x128x32(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (e4m3, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3_headdim64(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e4m3_256x128x64_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e4m3_256x128x64(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (e4m3, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e4m3_headdim128(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e4m3_256x128x128_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e4m3_256x128x128(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (e4m3, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2_headdim32(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e5m2_256x128x32_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e5m2_256x128x32(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (e5m2, head_dim 32): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2_headdim64(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e5m2_256x128x64_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e5m2_256x128x64(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (e5m2, head_dim 64): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_e5m2_headdim128(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e5m2_256x128x128_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_e5m2_256x128x128(__VA_ARGS__); \
  } \
} \
    else { \
          throw std::runtime_error("Blackwell FMHA forward kernel dispatch failed! It got invalid Q tile and KV tile combination (e5m2, head_dim 128): q_tile=" + std::to_string(q_tile_size) + ", kv_tile=" + std::to_string(kv_tile_size) + "."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

