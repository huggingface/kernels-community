#pragma once

// atomicAdd overloads for Half and BFloat16, which the col2im kernels need.
// They used to come from <THC/THCAtomics.cuh> (ATen/cuda/Atomic.cuh), which is
// not part of the stable ABI. Torch 2.14 moved them to
// <torch/headeronly/cuda/Atomic.h>, but older Torch versions do not ship that
// header, so this mirrors its Half/BFloat16 implementation.

#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
#include <cuda_bf16.h>
#endif

namespace deformable_detr_atomics {

// Compare-and-swap on the 32-bit word that contains the 16-bit value at
// `address`.
template <typename T>
inline __device__ T atomic_add_16bit_cas(T* address, T val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  T sum;
  do {
    assumed = old;
    sum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    sum = sum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (sum.x << 16)
                              : (old & 0xffff0000) | sum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  sum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return sum;
}

} // namespace deformable_detr_atomics

inline __device__ torch::headeronly::Half atomicAdd(
    torch::headeronly::Half* address,
    torch::headeronly::Half val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  return deformable_detr_atomics::atomic_add_16bit_cas(address, val);
#else
  return atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
}

inline __device__ torch::headeronly::BFloat16 atomicAdd(
    torch::headeronly::BFloat16* address,
    torch::headeronly::BFloat16 val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  return deformable_detr_atomics::atomic_add_16bit_cas(address, val);
#else
  __nv_bfloat16 r = atomicAdd(
      reinterpret_cast<__nv_bfloat16*>(address),
      *reinterpret_cast<__nv_bfloat16*>(&val));
  return *reinterpret_cast<torch::headeronly::BFloat16*>(&r);
#endif
}
