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
#include <natten_autogen/cuda/blackwell_fna_bwd/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_BACKWARD(rank, dtype, dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if constexpr (rank == 1) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if constexpr (rank == 2) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if constexpr (rank == 3) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Blackwell FNA backward kernel dispatch failed! It only supports NA1D, 2D, and 3D!"); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

