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
#include <natten_autogen/cuda/hopper_fna/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_FORWARD(rank, dtype, dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if constexpr (rank == 1) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if constexpr (rank == 2) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if constexpr (rank == 3) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      throw std::runtime_error("Hopper FNA forward kernel dispatch failed! It only supports NA1D, 2D, and 3D!"); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_hopper 
#endif 
#endif 

