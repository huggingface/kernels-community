/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************************/


// Define CPU_CAPABILITY_AVX512 before including ATen headers
// This enables AVX512-specific code paths in PyTorch's Vectorized classes
#if defined(__AVX512F__) && defined(__AVX512BF16__) && defined(__AMX_BF16__)
#define CPU_CAPABILITY_AVX512
#endif

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>

#include "fmha_fwd.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

// [NOTE]: Flash Attention Varlen CPU Implementation
//
// This is a CPU implementation of flash attention using AMX/BRGEMM.
// Supports both prefill (seqlen_q == seqlen_k) and decode (seqlen_q < seqlen_k) scenarios.
//

namespace {

//==============================================================================
// Common utilities (from common.h)
//==============================================================================

#define CHECK_CPU(x) TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CPU(x);                            \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

inline int get_thread_num() {
#if defined(_OPENMP)
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// balance payload across each thread
template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
  T n_my = div_up(n, nth);
  n_start = ith * n_my;
  n_end = std::min(n_start + n_my, n);
}

template <typename func_t>
inline void parallel_for(int n, const func_t& f) {
#if defined(_OPENMP)
#pragma omp parallel
  {
    int nth = omp_get_num_threads();
    int ith = omp_get_thread_num();
    int tbegin, tend;
    balance211(n, nth, ith, tbegin, tend);
    f(tbegin, tend);
  }
#else
  f(0, n);
#endif
}

// data indexing for dimension collapse
template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

// forced unroll for perf critical path
#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

template <int n>
struct Unroll {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    Unroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

template <>
struct Unroll<1> {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};

//==============================================================================
// GEMM utilities (from gemm.h)
//==============================================================================

// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

//==============================================================================
// Vector utilities (from vec.h)
//==============================================================================

using namespace at::vec;

template <typename scalar_t, typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline Vectorized<scalar_t> convert_from_float_ext(const Vectorized<float>& a, const Vectorized<float>& b) {
  return at::vec::convert_from_float<scalar_t>(a, b);
}

#if defined(CPU_CAPABILITY_AVX512)

// `at::vec::convert_from_float<>` from PyTorch doesn't have avx512-bf16 intrinsics
// use native instruction for float32->bfloat16 conversion
template <>
inline Vectorized<at::BFloat16>
convert_from_float_ext<at::BFloat16>(const Vectorized<float>& a, const Vectorized<float>& b) {
  return (__m512i)(_mm512_cvtne2ps_pbh(__m512(b), __m512(a)));
}

//==============================================================================
// Transpose utilities (from vec.h)
//==============================================================================

inline void transpose_16x16_32bit(__m512i* v) {
  __m512i v1[16];
  v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
  v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
  v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
  v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
  v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
  v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
  v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
  v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
  v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
  v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
  v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
  v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
  v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
  v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
  v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
  v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

  v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
  v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
  v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
  v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
  v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
  v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
  v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
  v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
  v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
  v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
  v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
  v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
  v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
  v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
  v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
  v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

  v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
  v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
  v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
  v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
  v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
  v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
  v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
  v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
  v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
  v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
  v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
  v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
  v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
  v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
  v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
  v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

  v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
  v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
  v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
  v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
  v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
  v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
  v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
  v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
  v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
  v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
  v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
  v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
  v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
  v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
  v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
  v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

// remove warning : ignoring attributes on template argument '__m512i' [-Wignored-attributes]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

// transpose from [2, 32] to [32, 2]
inline std::tuple<__m512i, __m512i> transpose_2x32_16bit(__m512i r0, __m512i r1) {
  __m512i d0 = _mm512_unpacklo_epi16(r0, r1);
  __m512i d1 = _mm512_unpackhi_epi16(r0, r1);
  r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);
  r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);
  d0 = _mm512_shuffle_i32x4(r0, r1, 0x88);
  d1 = _mm512_shuffle_i32x4(r0, r1, 0xdd);
  return std::make_tuple(d0, d1);
}
#pragma GCC diagnostic pop

#endif  // CPU_CAPABILITY_AVX512

//==============================================================================
// VNNI packing utilities (from vec_pack.h)
//==============================================================================

template <typename index_t>
inline index_t get_index(index_t* ind, int i) {
  return (ind == nullptr) ? (index_t)i : ind[i];
}

#if defined(CPU_CAPABILITY_AVX512)

// key: from [N, 32] to [32/2, N, 2]
template <typename scalar_t, typename index_t>
inline void pack_vnni_Nx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[16];

  int n = 0;
  for (; n < N; ++n) {
    index_t index = get_index(ind, n);
    vinputs[n] = _mm512_loadu_si512(src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask, vinputs[k]);
  }
}

template <typename scalar_t, typename index_t>
inline void pack_vnni_N_remainder(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[16];

  int K2 = K >> 1;
  const __mmask16 vmask = (1 << K2) - 1;

  int n = 0;
  for (; n < N; ++n) {
    index_t index = get_index(ind, n);
    vinputs[n] = _mm512_maskz_loadu_epi32(vmask, src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask2 = (1 << N) - 1;
  for (int k = 0; k < K2; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask2, vinputs[k]);
  }
}

// value: from [K, 32] to [K/2, 32, 2]
template <typename scalar_t, typename index_t>
inline void pack_vnni_Kx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[2];

  int k = 0;
  for (; k < K; ++k) {
    index_t index = get_index(ind, k);
    vinputs[k] = _mm512_loadu_si512(src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  // pack value
  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2, d0);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2 + 32, d1);
}

template <typename scalar_t, typename index_t>
inline void pack_vnni_K_remainder(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[2];

  const __mmask32 vmask = (1 << N) - 1;

  int k = 0;
  for (; k < K; ++k) {
    index_t index = get_index(ind, k);
    vinputs[k] = _mm512_maskz_loadu_epi16(vmask, src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  // pack value
  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);

  if (N <= 16) {
    // 2N * 16bits: N * 32bits
    const __mmask16 vmask2 = (1 << N) - 1;
    _mm512_mask_storeu_epi32(dst + 0 * ld_dst * 2, vmask2, d0);
  } else {
    // 2(N-16) * 16bits: (N-16) * 32bits
    const __mmask16 vmask2 = (1 << (N - 16)) - 1;
    _mm512_storeu_epi32(dst + 0 * ld_dst * 2, d0);
    _mm512_mask_storeu_epi32(dst + 0 * ld_dst * 2 + 32, vmask2, d1);
  }
}
#endif  // CPU_CAPABILITY_AVX512

// convert to vnni format
// from [N, K/2, 2] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t, bool is_indexed>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;
  const int K_remainder = K - KB * 32;

  for (int nb = 0; nb < NB; ++nb) {
    int nb_size = std::min(N - nb * 16, 16);
    for (int kb = 0; kb < KB; ++kb) {
      // handle 16x512bits each block
      pack_vnni_Nx32<scalar_t, index_t>(
          /*    dst */ dst + ((kb * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          /*    src */ src + kb * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          /*    ind */ is_indexed ? ind + nb * 16 : nullptr,
          /*      N */ nb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
    if (K_remainder > 0) {
      pack_vnni_N_remainder<scalar_t, index_t>(
          /*    dst */ dst + ((KB * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          /*    src */ src + KB * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          /*    ind */ is_indexed ? ind + nb * 16 : nullptr,
          /*      N */ nb_size,
          /*      K */ K_remainder,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  for (int n = 0; n < N; ++n) {
    index_t index = get_index(ind, n);
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst[k * ld_dst * 2 + n * 2 + d] = src[index * ld_src + k * 2 + d];
      }
    }
  }
#endif
}

template <typename scalar_t>
void pack_vnni(scalar_t* __restrict__ dst, const scalar_t* __restrict__ src, int N, int K, int ld_src, int ld_dst) {
  pack_vnni<scalar_t, int32_t, false>(dst, src, nullptr, N, K, ld_src, ld_dst);
}

template <typename scalar_t, typename index_t>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  assert(ind != nullptr);
  pack_vnni<scalar_t, index_t, true>(dst, src, ind, N, K, ld_src, ld_dst);
}

// convert to vnni format
// from [K/2, 2, N] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t, bool is_indexed>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int KB = div_up(K, 2);
  const int NB = N / 32;
  const int N_remainder = N - NB * 32;

  for (int kb = 0; kb < KB; ++kb) {
    int kb_size = std::min(K - kb * 2, 2);
    for (int nb = 0; nb < NB; ++nb) {
      // handle 2x512bits each block
      pack_vnni_Kx32<scalar_t, index_t>(
          /*    dst */ dst + ((kb * 2) >> 1) * ld_dst * 2 + nb * 32 * 2,
          /*    src */ src + (is_indexed ? 0 : kb * 2 * ld_src) + nb * 32,
          /*    ind */ is_indexed ? ind + kb * 2 : nullptr,
          /*      K */ kb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
    if (N_remainder > 0) {
      pack_vnni_K_remainder(
          /*    dst */ dst + ((kb * 2) >> 1) * ld_dst * 2 + NB * 32 * 2,
          /*    src */ src + (is_indexed ? 0 : kb * 2 * ld_src) + NB * 32,
          /*    ind */ is_indexed ? ind + kb * 2 : nullptr,
          /*      K */ kb_size,
          /*      N */ N_remainder,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    index_t index0 = get_index(ind, k + 0);
    index_t index1 = get_index(ind, k + 1);
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[index0 * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[index1 * ld_src + n];
    }
  }
  if (K % 2 != 0) {
    index_t index = get_index(ind, K - 1);
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[index * ld_src + n];
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
    k += 2;
  }
#endif
}

template <typename scalar_t>
void pack_vnni2(scalar_t* __restrict__ dst, const scalar_t* __restrict__ src, int K, int N, int ld_src, int ld_dst) {
  pack_vnni2<scalar_t, int32_t, false>(dst, src, nullptr, K, N, ld_src, ld_dst);
}

template <typename scalar_t, typename index_t>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  assert(ind != nullptr);
  pack_vnni2<scalar_t, index_t, true>(dst, src, ind, K, N, ld_src, ld_dst);
}

//==============================================================================
// Flash Attention Kernel Implementation
//==============================================================================

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t, int BLOCK_N>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    // for COLS = 2, 4 use 512bit store
    if constexpr (col % 2 == 0) {
      fVec a_fvec0 = fVec::loadu(input + col * 16);
      fVec a_fvec1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(s);
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a_fvec0 = fVec::loadu(acc + d) * s_fvec;
    fVec a_fvec1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * s);
  }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
void flash_attn_varlen_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k,
    void* __restrict__ buffer,
    int32_t* __restrict__ indices,
    int max_seqlen_q,
    int max_seqlen_k,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float sm_scale,
    int buffer_size_per_thread,
    bool causal) {
  using Vec = at::vec::Vectorized<float>;

  // Ensure BLOCK_M <= BLOCK_N to prevent potential buffer overflows during causal masking
  static_assert(BLOCK_M <= BLOCK_N);

  // strides
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // compute index (bs, mb_offset) for Query blocks
  // do this sequentially as usually problem size won't be big
  int idx = 0;
  for (int32_t bs = 0; bs < batches; ++bs) {
    int32_t seqlen_q = cu_seqlens_q[bs + 1] - cu_seqlens_q[bs];
    int32_t seqlen_k = cu_seqlens_k[bs + 1] - cu_seqlens_k[bs];
    TORCH_CHECK(seqlen_q <= max_seqlen_q && seqlen_k <= max_seqlen_k);

    int32_t blocks = div_up(seqlen_q, BLOCK_M);
    for (int32_t offset = 0; offset < blocks; ++offset) {
      indices[idx * 2 + 0] = bs;
      indices[idx * 2 + 1] = offset;
      idx++;
    }
  }
  // number of query blocks
  int MB = idx;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // parallel on [MB, num_heads]
  parallel_for(num_heads * MB, [&](int begin, int end) {
    int head_id{0}, mb{0};
    data_index_init(begin, head_id, num_heads, mb, MB);

    int tid = get_thread_num();
    // s_i and s_delta: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    float* __restrict__ s_delta = s_i;

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // s_delta2: [BLOCK_M, BLOCK_N]; copy of s_delta in scalar_t
    scalar_t* __restrict__ s_delta2 = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = s_delta2 + BLOCK_M * BLOCK_N;

    // init Btmp just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      int32_t bs = indices[mb * 2 + 0];
      int32_t seq_q_start_loc = cu_seqlens_q[bs];
      int32_t seq_k_start_loc = cu_seqlens_k[bs];
      int32_t seqlen_q = cu_seqlens_q[bs + 1] - cu_seqlens_q[bs];

      // offset and size in MB
      int m = indices[mb * 2 + 1] * BLOCK_M;
      int m_size = std::min(BLOCK_M, seqlen_q - m);

      assert(m_size > 0);

      int head_kv_id = head_id / num_groups;

      // get query
      const scalar_t* __restrict__ q_ptr = q + (seq_q_start_loc + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

      int seqlen_k = cu_seqlens_k[bs + 1] - cu_seqlens_k[bs];
      // For causal attention, each query at position q_pos can only attend to keys at positions <= q_pos
      // The query positions in this block are [m, m + m_size), so the last query is at position (m + m_size - 1)
      // Therefore, we need all keys from [0, m + m_size) to attend properly
      // But we also need to consider when seqlen_q != seqlen_k (e.g., decoding with KV cache)
      // In decoding: seqlen_q = 1, and the query should attend to all seqlen_k keys
      // The actual query position in the full sequence is (seqlen_k - seqlen_q + m + row)
      int num_keys = causal ? std::min(seqlen_k - seqlen_q + m + m_size, seqlen_k) : seqlen_k;
      for (int n = 0; n < num_keys; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t>(
            /*    dst */ Btmp,
            /*    src */ k + (seq_k_start_loc + n) * k_strideN + head_kv_id * k_strideH,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ k_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        // apply causal mask
        // For causal attention with different seqlen_q and seqlen_k (e.g., decoding with KV cache):
        // Query at local position 'row' (within this block) corresponds to global position (m + row) in query sequence
        // Its actual position in the key sequence is: seqlen_k - seqlen_q + m + row
        // It can only attend to keys at positions <= its position
        if (causal && num_keys - n <= BLOCK_N) {
          for (int row = 0; row < m_size; ++row) {
            // Global query position in key sequence
            int q_pos_in_k = seqlen_k - seqlen_q + m + row;
            int last_col = q_pos_in_k - n;
            // fill [last_col + 1, n_size) to -inf
            if (last_col < n_size - 1) {
              float* row_ptr = s_i + row * BLOCK_N;
              int start_col = std::max(0, last_col + 1);
              fill_stub(row_ptr + start_col, -std::numeric_limits<float>::infinity(), n_size - start_col);
            }
          }
        }

        const Vec scale_vec = Vec(sm_scale);
        for (int row = 0; row < m_size; ++row) {
          // s_i <- s_i * scale
          at::vec::map<float>(
              [scale_vec](Vec x) { return x * scale_vec; }, s_i + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + row * BLOCK_N, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          float m_delta = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); }, s_delta + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta;
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          at::vec::map<float>(
              [m_delta](Vec x) { return x * Vec(m_delta); },
              v_prime + row * head_size_v,
              v_prime + row * head_size_v,
              head_size_v);

          // pad s_delta with 0 first and then convert to scalar_t
          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        // get value and pack
        pack_vnni2<scalar_t>(
            /*    dst */ Btmp,
            /*    src */ v + (seq_k_start_loc + n) * v_strideN + head_kv_id * v_strideH,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ v_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta2,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seqlen_k

      scalar_t* __restrict__ out_ptr = out + (seq_q_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

      // move to the next index
      data_index_step(head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

template <int BLOCK_M, int BLOCK_N>
inline int resize_buffer(at::Tensor& buffer, int num_threads, int head_size, int head_size_v) {
  const int size_per_thread =
      /* s_i     */ BLOCK_M * BLOCK_N * sizeof(float) +
      /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
      /* s_delta */ BLOCK_M * BLOCK_N * sizeof(uint16_t) +
      /* Btmp    */ BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  buffer.resize_({num_threads, size_per_thread});
  return size_per_thread;
}

template <int BLOCK_M>
inline void resize_indices(at::Tensor& indices, int num_seqs, int max_seqlen_q) {
  // we allocate memory based on max seqlen
  indices.resize_({num_seqs, div_up(max_seqlen_q, BLOCK_M), 2});
}

// [NOTE]: `fmha_fwd_varlen_impl` - Flash Attention Varlen Implementation using AMX
//
// Inputs:
//   q: [num_tokens, num_heads, head_size]
//   k: [num_tokens, num_heads_kv, head_size]
//   v: [num_tokens, num_heads_kv, head_size_v]
//   out: [num_tokens, num_heads, head_size_v] - output tensor
//   cu_seqlens_q: [num_seqs + 1]
//   cu_seqlens_k: [num_seqs + 1]
//   max_seqlen_q: maximum query sequence length
//   max_seqlen_k: maximum key sequence length
//   softmax_scale: scaling factor (use 0 for default 1/sqrt(head_size))
//   is_causal: whether to apply causal masking
//
void fmha_fwd_varlen_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool is_causal) {

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);
  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_INPUT(cu_seqlens_q);
  CHECK_INPUT(cu_seqlens_k);
  CHECK_EQ(cu_seqlens_q.scalar_type(), at::kInt);
  CHECK_EQ(cu_seqlens_k.scalar_type(), at::kInt);

  int num_seqs = cu_seqlens_q.size(0) - 1;
  int num_tokens = q.size(0);
  int num_heads = q.size(1);
  int num_heads_kv = k.size(1);
  int head_size = q.size(2);
  int head_size_v = v.size(2);

  // strides for q, k and v
  int q_strideM = q.stride(0);
  int q_strideH = q.stride(1);
  int k_strideN = k.stride(0);
  int k_strideH = k.stride(1);
  int v_strideN = v.stride(0);
  int v_strideH = v.stride(1);

  // check sizes
  CHECK_EQ(k.size(2), head_size);
  CHECK_EQ(v.size(1), num_heads_kv);
  CHECK_EQ(cu_seqlens_k.size(0), num_seqs + 1);

  // D and DV need to be even as we transpose by 512-bit
  TORCH_CHECK(head_size % 2 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 2 == 0, "invalid head_size_v ", head_size_v);

  // Use default softmax_scale if not provided (0 or NaN)
  float sm_scale = softmax_scale;
  if (sm_scale == 0.0f || std::isnan(sm_scale)) {
    sm_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  }

  int num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({}, q.options().dtype(at::kChar));
  at::Tensor indices = at::empty({}, q.options().dtype(at::kInt));

  // Resize output if needed
  if (out.numel() == 0) {
    out.resize_({num_tokens, num_heads, head_size_v});
  }

  constexpr int BLOCK_M = 256;
  constexpr int BLOCK_N = 768;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "fmha_fwd_varlen_impl", [&] {
    int sz = resize_buffer<BLOCK_M, BLOCK_N>(buffer, num_threads, head_size, head_size_v);
    resize_indices<BLOCK_M>(indices, num_seqs, max_seqlen_q);

    flash_attn_varlen_kernel_impl<scalar_t, BLOCK_M, BLOCK_N>(
        out.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        cu_seqlens_q.data_ptr<int32_t>(),
        cu_seqlens_k.data_ptr<int32_t>(),
        buffer.data_ptr(),
        indices.data_ptr<int32_t>(),
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        num_heads,
        num_heads_kv,
        head_size,
        head_size_v,
        q_strideM,
        q_strideH,
        k_strideN,
        k_strideH,
        v_strideN,
        v_strideH,
        sm_scale,
        sz,
        is_causal);
  });
}
