/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU Common Utilities
//
// Shared utility functions for all CPU MoE implementations.

#define CPU_CAPABILITY_AVX512

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace megablocks {
namespace cpu {

// ============================================================================
// Macros
// ============================================================================

#define UNUSED(x) (void)(x)

#define CHECK_CPU(x) TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

// ============================================================================
// Constants
// ============================================================================

constexpr int GRAIN_SIZE = 1024;
constexpr int MAX_CACHE_BLOCK_SIZE = 4;

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

constexpr int block_size_m() {
  return 2 * TILE_M;
}

constexpr int block_size_n() {
  return 2 * TILE_N;
}

// ============================================================================
// Utility Functions
// ============================================================================

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

template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
  T n_my = div_up(n, nth);
  n_start = ith * n_my;
  n_end = std::min(n_start + n_my, n);
}

inline int adjust_num_threads(int m) {
  int actual_nth = at::get_num_threads();
  if (m == 1) {
    return actual_nth;
  }
  return std::max(1, (actual_nth >> 1) * 2);
}

// ============================================================================
// Parallel Utilities
// ============================================================================

template <typename func_t>
inline void parallel_2d(int m, int n, const func_t& f) {
  int nth = adjust_num_threads(m);
  float r = float(m) / n;
  int nth_m = std::ceil(std::sqrt(r * nth));
  int nth_n = 1;
  for (; nth_m > 0; --nth_m) {
    nth_n = nth / nth_m;
    if (nth_m * nth_n == nth) {
      break;
    }
  }

#if defined(_OPENMP)
#pragma omp parallel num_threads(nth)
  {
    int ith = omp_get_thread_num();
    int ith_m = ith / nth_n;
    int ith_n = ith % nth_n;

    int thread_block_m = div_up(m, nth_m);
    int thread_block_n = div_up(n, nth_n);

    int begin_m = ith_m * thread_block_m;
    int end_m = std::min(m, begin_m + thread_block_m);
    int begin_n = ith_n * thread_block_n;
    int end_n = std::min(n, begin_n + thread_block_n);

    f(begin_m, end_m, begin_n, end_n);
  }
#else
  f(0, m, 0, n);
#endif
}

// ============================================================================
// Cache Blocking
// ============================================================================

template <typename T>
inline int get_cache_blocks(int chunk_size) {
  const int L2_size = 2048 * 1024 >> 1;
  return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
}

template <>
inline int get_cache_blocks<at::Float8_e4m3fn>(int chunk_size) {
  int cache_block_size = get_cache_blocks<at::BFloat16>(chunk_size);
  return std::min(MAX_CACHE_BLOCK_SIZE, cache_block_size);
}

template <typename T, typename func_t>
inline void loop_2d(int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1, int64_t chunk_size, const func_t& f) {
  int64_t cache_blocks_nb = get_cache_blocks<T>(chunk_size);
  for (int64_t nbb = nb0; nbb < nb1; nbb += cache_blocks_nb) {
    for (int64_t mb = mb0; mb < mb1; ++mb) {
      for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, nb1); ++nb) {
        f(mb, nb, nb - nbb);
      }
    }
  }
}

// ============================================================================
// Data Index Helpers
// ============================================================================

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

// ============================================================================
// Unroll Helper
// ============================================================================

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

// ============================================================================
// Vectorization Helpers
// ============================================================================

using namespace at::vec;

template <typename scalar_t, typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline Vectorized<scalar_t> convert_from_float_ext(const Vectorized<float>& a, const Vectorized<float>& b) {
  return at::vec::convert_from_float<scalar_t>(a, b);
}

template <typename scalar_t>
inline void convert_from_float_and_store(scalar_t* out, const Vectorized<float>& a) {
  float out_buffer[at::vec::Vectorized<float>::size()];
  a.store(out_buffer);
  for (int i = 0; i < 16; i++) {
    out[i] = (scalar_t)out_buffer[i];
  }
}

#if defined(CPU_CAPABILITY_AVX512)

template <>
inline Vectorized<at::BFloat16>
convert_from_float_ext<at::BFloat16>(const Vectorized<float>& a, const Vectorized<float>& b) {
  return (__m512i)(_mm512_cvtne2ps_pbh(__m512(b), __m512(a)));
}

template <>
inline void convert_from_float_and_store<at::BFloat16>(at::BFloat16* out, const Vectorized<float>& a) {
  _mm256_storeu_si256((__m256i*)out, (__m256i)(_mm512_cvtneps_pbh(__m512(a))));
}

#define CVT_BF16_TO_FP32(a) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16))

#endif // CPU_CAPABILITY_AVX512

// ============================================================================
// BRGEMM Helpers
// ============================================================================

template <typename T>
inline bool can_use_brgemm(int M);

template <>
inline bool can_use_brgemm<at::BFloat16>(int M) {
  return M > 4;
}

template <>
inline bool can_use_brgemm<at::Half>(int M) {
  return true;
}

template <>
inline bool can_use_brgemm<int8_t>(int M) {
  return M > 4;
}

template <>
inline bool can_use_brgemm<uint8_t>(int M) {
  return M > 4;
}

template <>
inline bool can_use_brgemm<at::Float8_e4m3fn>(int M) {
  return M > 4;
}

// ============================================================================
// Row Size Helpers
// ============================================================================

template <typename T>
inline int64_t get_row_size(int64_t K) {
  return K;
}

template <>
inline int64_t get_row_size<int8_t>(int64_t K) {
  return K + sizeof(int32_t);
}

template <>
inline int64_t get_row_size<uint8_t>(int64_t K) {
  return K >> 1;
}

inline int64_t get_row_size(int64_t K, bool use_int8_w8a8) {
  return use_int8_w8a8 ? K + sizeof(int32_t) : K;
}

// ============================================================================
// Copy and Arithmetic Stubs
// ============================================================================

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  int64_t i = 0;
  for (; i <= size - Vec::size(); i += Vec::size()) {
    Vec::loadu(input + i).store(out + i);
  }
  for (; i < size; ++i) {
    out[i] = input[i];
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  int64_t i = 0;
  for (; i <= size - bVec::size(); i += bVec::size()) {
    fVec v0 = fVec::loadu(input + i);
    fVec v1 = fVec::loadu(input + i + fVec::size());
    convert_from_float_ext<scalar_t>(v0, v1).store(out + i);
  }
  for (; i < size; ++i) {
    out[i] = (scalar_t)input[i];
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, float weight, int64_t size) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  fVec vweight = fVec(weight);
  int64_t i = 0;
  for (; i <= size - bVec::size(); i += bVec::size()) {
    fVec v0 = fVec::loadu(input + i) * vweight;
    fVec v1 = fVec::loadu(input + i + fVec::size()) * vweight;
    convert_from_float_ext<scalar_t>(v0, v1).store(out + i);
  }
  for (; i < size; ++i) {
    out[i] = (scalar_t)(input[i] * weight);
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, float weight, int64_t size) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  fVec vweight = fVec(weight);
  int64_t i = 0;
  for (; i <= size - bVec::size(); i += bVec::size()) {
    auto [v0, v1] = convert_to_float<scalar_t>(bVec::loadu(input + i));
    v0 = v0 * vweight;
    v1 = v1 * vweight;
    convert_from_float_ext<scalar_t>(v0, v1).store(out + i);
  }
  for (; i < size; ++i) {
    out[i] = (scalar_t)((float)input[i] * weight);
  }
}

template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;

  // load first row
  int64_t k = 0;
  for (; k <= K - bVec::size(); k += bVec::size()) {
    auto [v0, v1] = convert_to_float<scalar_t>(bVec::loadu(input + k));
    for (int64_t j = 1; j < topk; ++j) {
      auto [u0, u1] = convert_to_float<scalar_t>(bVec::loadu(input + j * K + k));
      v0 = v0 + u0;
      v1 = v1 + u1;
    }
    convert_from_float_ext<scalar_t>(v0, v1).store(out + k);
  }
  for (; k < K; ++k) {
    float sum = 0.0f;
    for (int64_t j = 0; j < topk; ++j) {
      sum += (float)input[j * K + k];
    }
    out[k] = (scalar_t)sum;
  }
}

template <typename scalar_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float weight,
    int64_t size) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  fVec vweight = fVec(weight);
  int64_t i = 0;
  for (; i <= size - bVec::size(); i += bVec::size()) {
    auto [v0, v1] = convert_to_float<scalar_t>(bVec::loadu(input + i));
    auto [u0, u1] = convert_to_float<scalar_t>(bVec::loadu(input2 + i));
    v0 = v0 + u0 * vweight;
    v1 = v1 + u1 * vweight;
    convert_from_float_ext<scalar_t>(v0, v1).store(out + i);
  }
  for (; i < size; ++i) {
    out[i] = (scalar_t)((float)input[i] + (float)input2[i] * weight);
  }
}

template <typename scalar_t>
inline void silu_and_mul_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input0,
    const scalar_t* __restrict__ input1,
    int64_t size) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  fVec one = fVec(1.0f);
  int64_t i = 0;
  for (; i <= size - bVec::size(); i += bVec::size()) {
    auto [v0, v1] = convert_to_float<scalar_t>(bVec::loadu(input0 + i));
    auto [u0, u1] = convert_to_float<scalar_t>(bVec::loadu(input1 + i));
    v0 = v0 / (one + v0.neg().exp()) * u0;
    v1 = v1 / (one + v1.neg().exp()) * u1;
    convert_from_float_ext<scalar_t>(v0, v1).store(out + i);
  }
  for (; i < size; ++i) {
    float x = (float)input0[i];
    float y = (float)input1[i];
    out[i] = (scalar_t)(x / (1.0f + std::exp(-x)) * y);
  }
}

template <typename scalar_t>
inline void add_bias_stub(float* __restrict__ input, const scalar_t* __restrict__ input2, int64_t size) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  int64_t i = 0;
  for (; i <= size - bVec::size(); i += bVec::size()) {
    fVec v0 = fVec::loadu(input + i);
    fVec v1 = fVec::loadu(input + i + fVec::size());
    auto [u0, u1] = convert_to_float<scalar_t>(bVec::loadu(input2 + i));
    v0 = v0 + u0;
    v1 = v1 + u1;
    v0.store(input + i);
    v1.store(input + i + fVec::size());
  }
  for (; i < size; ++i) {
    input[i] += (float)input2[i];
  }
}

}  // namespace cpu
}  // namespace megablocks
