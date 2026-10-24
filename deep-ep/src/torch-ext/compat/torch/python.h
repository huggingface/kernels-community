#pragma once

// kernels-community shim for DeepEP's host sources.
//
// Upstream includes <torch/python.h> for the libtorch C++ API. Hub kernels
// cannot use Torch's Python bindings (they are built against Python's limited
// ABI), so this header provides the same C++ API without any Python headers.
// See pybind11/pybind11.h in this directory for the pybind11 value types.

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "../pybind11/pybind11.h"

namespace torch {

using c10::Event;
using c10::Stream;

namespace cuda {
using c10::cuda::CUDAStream;
}  // namespace cuda

}  // namespace torch
