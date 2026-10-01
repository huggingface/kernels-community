// Self-contained CUDA error-check macros.
//
// Replaces c10's <c10/cuda/CUDAException.h> (C10_CUDA_CHECK /
// C10_CUDA_KERNEL_LAUNCH_CHECK), which hard-errors in stable-ABI builds under
// TORCH_STABLE_ONLY / TORCH_TARGET_VERSION. These are used from .cu translation
// units compiled by nvcc, where the CUDA runtime is available.

#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define C10_CUDA_CHECK(EXPR)                                                    \
  do {                                                                          \
    const cudaError_t __err = (EXPR);                                           \
    if (__err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(__err));                                       \
      abort();                                                                  \
    }                                                                           \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
