#pragma once


#include <iostream> 
#include <stdexcept> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/cuda/fna/na_utils.cuh> 
#include <natten/cuda/fna/kernel_forward.h> 
#include <natten/cuda/fna/kernel_backward.h> 
#include <natten_autogen/cuda/fna/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace fna { 
#define DISPATCH_FNA_FORWARD_1D_SM50(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_1D_SM50_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_1D_SM50_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM70(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_1D_SM70_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_1D_SM70_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM75(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_1D_SM75_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_1D_SM75_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM80(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_1D_SM80_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_1D_SM80_float16(is_causal, cb); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_FNA_FORWARD_1D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM80."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM50(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_2D_SM50_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_2D_SM50_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM70(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_2D_SM70_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_2D_SM70_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM75(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_2D_SM75_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_2D_SM75_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM80(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_2D_SM80_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_2D_SM80_float16(is_causal, cb); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_FNA_FORWARD_2D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM80."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM50(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_3D_SM50_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_3D_SM50_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM70(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_3D_SM70_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_3D_SM70_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM75(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_3D_SM75_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_3D_SM75_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM80(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_FORWARD_3D_SM80_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_FORWARD_3D_SM80_float16(is_causal, cb); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_FNA_FORWARD_3D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM80."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM50(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_1D_SM50_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_1D_SM50_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM70(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_1D_SM70_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_1D_SM70_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM75(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_1D_SM75_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_1D_SM75_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM80(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_1D_SM80_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_1D_SM80_float16(is_causal, cb); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_FNA_BACKWARD_1D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-1D does not support this data type on SM80."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM50(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_2D_SM50_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_2D_SM50_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM70(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_2D_SM70_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_2D_SM70_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM75(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_2D_SM75_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_2D_SM75_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM80(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_2D_SM80_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_2D_SM80_float16(is_causal, cb); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_FNA_BACKWARD_2D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-2D does not support this data type on SM80."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM50(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_3D_SM50_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_3D_SM50_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM50."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM70(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_3D_SM70_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_3D_SM70_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM70."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM75(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_3D_SM75_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_3D_SM75_float16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM75."); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM80(dtype, is_causal, cb) \
  [&] { \
    if (dtype == torch::kFloat32) { \
      DISPATCH_FNA_BACKWARD_3D_SM80_float32(is_causal, cb); \
    } \
    else if (dtype == torch::kFloat16) { \
      DISPATCH_FNA_BACKWARD_3D_SM80_float16(is_causal, cb); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_FNA_BACKWARD_3D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      throw std::runtime_error("NATTEN kernel dispatch failed! FNA-3D does not support this data type on SM80."); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna 

