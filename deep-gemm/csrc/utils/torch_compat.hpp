#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

// DeepGEMM upstream uses the torch:: C++ API from torch/python.h. Kernel Hub
// builds use Python's limited ABI, so CUDA TUs must avoid Python/pybind headers.
namespace torch {
namespace indexing = at::indexing;

using at::Tensor;
using at::TensorOptions;
using c10::ScalarType;

using at::arange;
using at::empty;
using at::empty_like;
using at::empty_strided;
using at::from_blob;
using at::tensor;
using at::zeros;

inline constexpr auto kBFloat16 = at::kBFloat16;
inline constexpr auto kByte = at::kByte;
inline constexpr auto kFloat = at::kFloat;
inline constexpr auto kFloat8_e4m3fn = at::kFloat8_e4m3fn;
inline constexpr auto kFloat32 = at::kFloat;
inline constexpr auto kInt = at::kInt;
inline constexpr auto kInt8 = at::kChar;
inline constexpr auto kInt32 = at::kInt;
inline constexpr auto kInt64 = at::kLong;
inline constexpr auto kUInt8 = at::kByte;
inline constexpr auto kCUDA = c10::kCUDA;
} // namespace torch
