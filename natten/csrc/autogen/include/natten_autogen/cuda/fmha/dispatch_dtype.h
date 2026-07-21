#pragma once


#include <iostream> 
#include <stdexcept> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/cuda/fmha/kernel_forward.h> 
#include <natten/cuda/fmha/kernel_backward.h> 
#include <natten_autogen/cuda/fmha/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha { 
#define DISPATCH_FMHA_FORWARD_SM50(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_sm50_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_sm50_float16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FMHA_FORWARD_SM70(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_sm70_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_sm70_float16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FMHA_FORWARD_SM75(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_sm75_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_sm75_float16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FMHA_FORWARD_SM80(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_sm80_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_sm80_float16(cb); \
    } \
    else if (dtype == at::kBFloat16) { \
      fmha_sm80_bfloat16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM80."); \
    } \
}();

#define DISPATCH_FMHA_BACKWARD_SM50(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_backward_sm50_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_backward_sm50_float16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FMHA_BACKWARD_SM70(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_backward_sm70_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_backward_sm70_float16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FMHA_BACKWARD_SM75(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_backward_sm75_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_backward_sm75_float16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FMHA_BACKWARD_SM80(dtype, cb) \
  [&] { \
    if (dtype == at::kFloat) { \
      fmha_backward_sm80_float32(cb); \
    } \
    else if (dtype == at::kHalf) { \
      fmha_backward_sm80_float16(cb); \
    } \
    else if (dtype == at::kBFloat16) { \
      fmha_backward_sm80_bfloat16(cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FMHA does not support this data type on SM80."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha 

