/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU Fused MoE Implementation for FP8/MXFP4
//
// Based on sglang implementation.

#define CPU_CAPABILITY_AVX512
#include "moe_ops.h"
#include "common.h"
#include <ATen/record_function.h>
#include <ATen/native/CPUBlas.h>

namespace megablocks {
namespace cpu {

using namespace at::vec;

// BLOCK_K constant for FP8/MXFP4
constexpr int64_t kBlockK = 128;

namespace {

// ============================================================================
// MXFP4 and FP8 conversion macros and functions
// ============================================================================

#if defined(CPU_CAPABILITY_AVX512)

#define MXFP4_VALUES \
  -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.0f, 6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.0f

// FP8 bias for conversion to bf16: 1/256 in float32
#define kFP8_BIAS 0x3b800000

// convert 64 mxfp4 to 2x bf16 vectors, expect input 32-way packing
inline std::tuple<__m512bh, __m512bh> cvt_mxfp4_e2m1_bf16_intrinsic_lut(__m256i a, __m512i s0, __m512i s1) {
  // LUT
  const __m512 values = _mm512_set_ps(MXFP4_VALUES);
  const __m512i lut = (__m512i)(_mm512_cvtne2ps_pbh(values, values));

  const __m512i abs_mask = _mm512_set1_epi16(0x7FFF);
  const __m512i zero = _mm512_setzero_si512();

  // expand values to 16-bit integers
  __m512i x0 = _mm512_cvtepu8_epi16(a);
  __m512i x1 = _mm512_srli_epi32(x0, 4);

  // LUT to convert mxfp4 values to bf16
  x0 = _mm512_permutexvar_epi16(x0, lut);
  x1 = _mm512_permutexvar_epi16(x1, lut);

  // check for zeros
  __mmask32 mask0 = _mm512_cmp_epi16_mask(_mm512_and_si512(x0, abs_mask), zero, _MM_CMPINT_EQ);
  __mmask32 mask1 = _mm512_cmp_epi16_mask(_mm512_and_si512(x1, abs_mask), zero, _MM_CMPINT_EQ);

  // emulate bf16 mul with scale factor
  x0 = _mm512_add_epi16(x0, s0);
  x1 = _mm512_add_epi16(x1, s1);

  // blend with zero
  x0 = _mm512_mask_blend_epi16(mask0, x0, zero);
  x1 = _mm512_mask_blend_epi16(mask1, x1, zero);

  return std::make_tuple(__m512bh(x0), __m512bh(x1));
}

#define CVT_MXFP4_TO_BF16(a, s0, s1) cvt_mxfp4_e2m1_bf16_intrinsic_lut(a, s0, s1)

// FP8 to BF16 conversion - fast version
inline __attribute__((always_inline)) __m512bh CVT_FP8_TO_BF16_EXT(__m256i a) {
  const __m512i mask0 = _mm512_set1_epi16(0x80);  // sign bit
  const __m512i mask1 = _mm512_set1_epi16(0x7F);  // exponent and mantissa
  const __m512i mask2 = _mm512_set1_epi16(0x4000);

  __m512i x = _mm512_cvtepu8_epi16(a);
  __m512i vsign = _mm512_and_si512(x, mask0);
  vsign = _mm512_slli_epi16(vsign, 8);

  __m512i vexp_and_mant = _mm512_and_si512(x, mask1);
  vexp_and_mant = _mm512_slli_epi16(vexp_and_mant, 4);

  // _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C: 0b11111110
  return (__m512bh)(_mm512_ternarylogic_epi32(vsign, mask2, vexp_and_mant, 0b11111110));
}

// transpose from [2, 32] to [32, 2]
inline std::tuple<__m512i, __m512i> transpose_2x32_16bit(__m512i r0, __m512i r1) {
  // r0: {a0, a1, ..., a31}
  // r1: {b0, b1, ..., b31}
  //
  // d0: {a0,   b0, ..., a15, b15}
  // d1: {a16, b16, ..., a31, b31}
  //
  __m512i d0 = _mm512_unpacklo_epi16(r0, r1);
  __m512i d1 = _mm512_unpackhi_epi16(r0, r1);
  r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);
  r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);
  d0 = _mm512_shuffle_i32x4(r0, r1, 0x88);
  d1 = _mm512_shuffle_i32x4(r0, r1, 0xdd);
  return std::make_tuple(d0, d1);
}

#endif // CPU_CAPABILITY_AVX512

// ============================================================================
// Unpacking functions for FP8 and MXFP4
// ============================================================================

// FP8 unpacking
inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int64_t N,
    int64_t K,
    int64_t ldb,
    int64_t ldb_tmp,
    float scale) {
#if defined(CPU_CAPABILITY_AVX512)
  // [K/2, N, 2]
  const int64_t K2 = K >> 1;
  const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);
  const __m512 vexp = _mm512_castsi512_ps(_mm512_set1_epi32(kFP8_BIAS));
  const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(scale), vexp);

  constexpr int BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  // prefetch distance
  constexpr int PREFETCH_SIZE_K = 64;

#pragma GCC unroll 4
  for (int64_t k = 0; k < K2; ++k) {
    __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2);
    if constexpr (PREFETCH_SIZE_K > 0) {
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2, _MM_HINT_T0);
    }

    __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
    __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);

    __m512bh bf16_0 = CVT_FP8_TO_BF16_EXT(b8_0);
    __m512bh bf16_1 = CVT_FP8_TO_BF16_EXT(b8_1);

    // Apply scale
    __m512 f0_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 0));
    __m512 f0_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 1));
    __m512 f1_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 0));
    __m512 f1_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 1));

    f0_lo = _mm512_mul_ps(f0_lo, vd);
    f0_hi = _mm512_mul_ps(f0_hi, vd);
    f1_lo = _mm512_mul_ps(f1_lo, vd);
    f1_hi = _mm512_mul_ps(f1_hi, vd);

    bf16_0 = _mm512_cvtne2ps_pbh(f0_hi, f0_lo);
    bf16_1 = _mm512_cvtne2ps_pbh(f1_hi, f1_lo);

    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 0, (__m512i)bf16_0);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 32, (__m512i)bf16_1);
  }
#else
  TORCH_CHECK(false, "unpack_B: FP8 scalar path not implemented!");
#endif
}

// MXFP4 unpacking
inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const uint8_t* __restrict__ packed_B,
    int64_t N,
    int64_t K,
    int64_t ldb,
    int64_t ldb_tmp,
    const uint8_t* __restrict__ scale) {
#if defined(CPU_CAPABILITY_AVX512)
  // [K/2, N, 2]
  const int64_t K2 = K >> 1;
  const int64_t ldb2 = ldb;                                           // ldb * 2 >> 1;
  const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(packed_B);  // 2 * 4 bit = 8 bit

  constexpr int BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  // prefetch distance
  constexpr int PREFETCH_SIZE_K = 64;

  // exponent bias 127
  const __m512i off = _mm512_set1_epi16(0x7F);

  // load 32 bytes only once for each block
  __m256i s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scale));
  __m512i s16 = _mm512_slli_epi16(_mm512_sub_epi16(_mm512_cvtepu8_epi16(s8), off), 0x7);

  // holds Nx2(64) scales, interleaved as 2 belongs to K dimension
  // e.g. vs0: { s0,  s0,  s1,  s1, ..., s15, s15}
  //      vs1: {s16, s16, s17, s17, ..., s31, s31}
  auto [vscale0, vscale1] = transpose_2x32_16bit(s16, s16);

#pragma GCC unroll 4
  for (int64_t k = 0; k < K2; ++k) {
    __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2));
    if constexpr (PREFETCH_SIZE_K > 0) {
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2, _MM_HINT_T0);
    }
    auto [vb0, vb1] = CVT_MXFP4_TO_BF16(b4, vscale0, vscale1);

    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 0, (__m512i)vb0);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 32, (__m512i)vb1);
  }
#else
  TORCH_CHECK(false, "unpack_B: MXFP4 scalar path not implemented!");
#endif
}

// ============================================================================
// Specialized helper functions for FP8/MXFP4 kernel
// ============================================================================

template <typename scalar_t, int BLOCK_N>
inline void clamp_sigmoid_and_mul(
    scalar_t* __restrict__ output,
    const float* __restrict__ input0,
    int64_t m_size,
    int64_t N,
    const float alpha,
    const float limit,
    int64_t offset) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  const fVec one = fVec(1.f);
  const fVec zero = fVec(0.f);
  const fVec limit_v = fVec(limit);
  const fVec nlimit_v = fVec(-limit);
  const fVec alpha_v = fVec(alpha);

  // no remainder
  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out = output + m * N;
    const float* __restrict__ cur_ptr = input0 + m * BLOCK_N;
    for (int64_t d = 0; d < BLOCK_N; d += bVec::size()) {
      float tmp_glu0[fVec::size()];     // 16
      float tmp_linear0[fVec::size()];  // 16

      // interleaved: x[2i] = glu, x[2i+1] = linear
      for (int j = 0; j < fVec::size(); ++j) {
        // x0 [0,2,..30]
        tmp_glu0[j] = cur_ptr[d + j * 2];
        // y0 [1,3,...31]
        tmp_linear0[j] = cur_ptr[d + j * 2 + 1];
      }
      fVec x0 = fVec::loadu(tmp_glu0);
      fVec y0 = fVec::loadu(tmp_linear0);

      // clamp
      x0 = at::vec::minimum(x0, limit_v);
      y0 = at::vec::minimum(limit_v, at::vec::maximum(nlimit_v, y0));
      // x * sigmoid(x * alpha)
      x0 = x0 / (one + (x0 * alpha_v).neg().exp_u20());
      // (y + 1) * x
      y0 = y0 + one;
      x0 = x0 * y0;
      // // convert
      convert_from_float_and_store<scalar_t>(out + d / 2 + offset, x0);
    }
  }
}

// ============================================================================
// tinygemm_kernel implementation for FP8 and MXFP4
// ============================================================================

// Kernel template structure for output to scalar_t (bf16/fp16)
template <typename scalar_t, typename packed_t, typename param_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const param_t* __restrict__ scale,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K,
      bool do_unpack) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: primary template not implemented!");
  }
};

// Kernel template structure for output to float
template <typename scalar_t, typename packed_t, typename param_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2 {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      float* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      const param_t* __restrict__ scale,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K,
      bool do_unpack) {
    TORCH_CHECK(false, "tinygemm_kernel_nn2: primary template not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)

// ===== FP8 specialization (output to bf16) =====
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::Float8_e4m3fn, float, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ scale,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K,
      bool do_unpack) {
    
    constexpr int64_t BLOCK_K_LOCAL = 128;
    constexpr int64_t BLOCK_K2 = BLOCK_K_LOCAL >> 1;
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    constexpr int ldb_tmp = BLOCK_N;
    
    const int64_t KB = div_up(K, BLOCK_K_LOCAL);

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vsum[ROWS * COLS];
    __m512 vscale;

    const __m512 vexp = _mm512_castsi512_ps(_mm512_set1_epi32(kFP8_BIAS));

    auto loadc = [&](auto i) {
      vc[i] = _mm512_setzero_ps();
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(B);

    // Unpack B if needed
    if (do_unpack) {
      for (int64_t k = 0; k < K; k += BLOCK_K_LOCAL) {
        int64_t kb_size = std::min(BLOCK_K_LOCAL, K - k);
        int64_t idx = k >> 7;
        unpack_B(Btmp + k * ldb_tmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scale[idx]);
      }
    }

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(Btmp + k * ldb_tmp * 2 + col * 16));
          vb[col + 0] = (__m512bh)(_mm512_castsi256_si512(_mm512_extracti32x8_epi32(b_vec, 0)));
          vb[col + 1] = (__m512bh)(_mm512_castsi256_si512(_mm512_extracti32x8_epi32(b_vec, 1)));
        }
      }
      vsum[i] = _mm512_dpbf16_ps(vsum[i], va, vb[col]);
    };

    for (int64_t kb = 0; kb < KB; ++kb) {
      int64_t kb_start = kb * BLOCK_K2;
      int64_t kb_end = std::min(K >> 1, kb_start + BLOCK_K2);
      
      // Block computation
      Unroll<ROWS * COLS>{}([&](auto i) { vsum[i] = _mm512_setzero_ps(); });
      for (int k = kb_start; k < kb_end; ++k) {
        Unroll<ROWS * COLS>{}(compute, k);
      }
      Unroll<ROWS * COLS>{}([&](auto i) { vc[i] = _mm512_add_ps(vc[i], vsum[i]); });
    }

    // Store results: convert float to bf16
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; n += 32) {
        __m512 f0 = vc[m * COLS + n / 16];
        __m512 f1 = vc[m * COLS + n / 16 + 1];
        __m512bh bf = _mm512_cvtne2ps_pbh(f1, f0);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(C + m * ldc + n), (__m512i)bf);
      }
    }
  }
};

// ===== MXFP4 specialization (output to bf16) =====
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, uint8_t, uint8_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const uint8_t* __restrict__ scale,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K,
      bool do_unpack) {
    // mxfp4 supports only group size of 32
    // expect weight packed in 32-way, vnni2 format Nx2(64)
    (void)Btmp;  // not used in inline conversion
    (void)Ctmp;
    (void)do_unpack;  // always do inline conversion
    
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    // holds Nx2(64) scales, interleaved as 2 belongs to K dimension
    __m512i vscale[COLS];

    // exponent bias 127
    const __m512i off = _mm512_set1_epi16(0x7F);

    auto loadc = [&](auto i) {
      vc[i] = _mm512_setzero_ps();
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        // load 32 * 2 (64) int4 at a time
        if constexpr (col % 2 == 0) {
          __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2 + col * 16));
          std::tie(vb[col + 0], vb[col + 1]) = CVT_MXFP4_TO_BF16(b4, vscale[col + 0], vscale[col + 1]);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };

    for (int64_t k = 0; k < K2; ++k) {
      // update scales every 16x2 K
      if ((k & 15) == 0) {
        __m256i s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scale + (k >> 4) * 32));
        __m512i s16 = _mm512_slli_epi16(_mm512_sub_epi16(_mm512_cvtepu8_epi16(s8), off), 0x7);
        std::tie(vscale[0], vscale[1]) = transpose_2x32_16bit(s16, s16);
      }
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2,4 use 512bit store
      if constexpr (col % 2 == 0) {
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};

// ===== FP8 specialization (output to float) =====
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2<at::BFloat16, at::Float8_e4m3fn, float, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      float* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      const float* __restrict__ scale,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K,
      bool do_unpack) {
    
    constexpr int64_t BLOCK_K = 128;
    constexpr int64_t BLOCK_K2 = BLOCK_K >> 1;
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    constexpr int ldb_tmp = BLOCK_N;
    
    const int64_t KB = div_up(K, (int64_t)BLOCK_K);

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vsum[ROWS * COLS];

    auto loadc = [&](auto i) {
      vc[i] = _mm512_setzero_ps();
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t lda2 = lda >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);

    // Unpack B if needed
    if (do_unpack) {
      for (int k = 0; k < K; k += BLOCK_K) {
        int kb_size = std::min(BLOCK_K, K - k);
        int idx = k >> 7;
        unpack_B(Btmp + k * ldb_tmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scale[idx]);
      }
    }

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(Btmp + k * ldb_tmp * 2 + col * 16));
          vb[col + 0] = (__m512bh)(_mm512_castsi256_si512(_mm512_extracti32x8_epi32(b_vec, 0)));
          vb[col + 1] = (__m512bh)(_mm512_castsi256_si512(_mm512_extracti32x8_epi32(b_vec, 1)));
        }
      }
      vsum[i] = _mm512_dpbf16_ps(vsum[i], va, vb[col]);
    };

    for (int64_t kb = 0; kb < KB; ++kb) {
      int64_t kb_start = kb * BLOCK_K2;
      int64_t kb_end = std::min(K >> 1, kb_start + BLOCK_K2);
      
      Unroll<ROWS * COLS>{}([&](auto i) { vsum[i] = _mm512_setzero_ps(); });
      for (int k = kb_start; k < kb_end; ++k) {
        Unroll<ROWS * COLS>{}(compute, k);
      }
      Unroll<ROWS * COLS>{}([&](auto i) { vc[i] = _mm512_add_ps(vc[i], vsum[i]); });
    }

    // Store float results
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; n += 16) {
        _mm512_storeu_ps(C + m * ldc + n, vc[m * COLS + n / 16]);
      }
    }
  }
};

// ===== MXFP4 specialization (output to float) =====
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2<at::BFloat16, uint8_t, uint8_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      float* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      const uint8_t* __restrict__ scale,
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K,
      bool do_unpack) {
    // mxfp4 supports only group size of 32
    // expect weight packed in 32-way, vnni2 format Nx2(64)
    (void)Btmp;  // not used in inline conversion
    (void)do_unpack;  // always do inline conversion
    
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    // holds Nx2(64) scales, interleaved as 2 belongs to K dimension
    __m512i vscale[COLS];

    // exponent bias 127
    const __m512i off = _mm512_set1_epi16(0x7F);

    auto loadc = [&](auto i) {
      vc[i] = _mm512_setzero_ps();
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        // load 32 * 2 (64) int4 at a time
        if constexpr (col % 2 == 0) {
          __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2 + col * 16));
          std::tie(vb[col + 0], vb[col + 1]) = CVT_MXFP4_TO_BF16(b4, vscale[col + 0], vscale[col + 1]);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };

    for (int64_t k = 0; k < K2; ++k) {
      // update scales every 16x2 K
      if ((k & 15) == 0) {
        __m256i s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scale + (k >> 4) * 32));
        __m512i s16 = _mm512_slli_epi16(_mm512_sub_epi16(_mm512_cvtepu8_epi16(s8), off), 0x7);
        std::tie(vscale[0], vscale[1]) = transpose_2x32_16bit(s16, s16);
      }
      Unroll<ROWS * COLS>{}(compute, k);
    }

    // Store float results
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; n += 16) {
        _mm512_storeu_ps(C + m * ldc + n, vc[m * COLS + n / 16]);
      }
    }
  }
};

// ============================================================================
// brgemm structures for MXFP4 and FP8
// ============================================================================

// Helper: unpack_B dispatcher based on packed_t type
template <typename packed_t, typename param_t>
struct unpack_helper;

// FP8 unpack helper
template <>
struct unpack_helper<at::Float8_e4m3fn, float> {
  static inline void unpack(
      at::BFloat16* Btmp, const at::Float8_e4m3fn* B,
      int N, int K, int ldb, int ldb_tmp, const float* scale) {
    constexpr int64_t BLOCK_K_LOCAL = 128;
    for (int k = 0; k < K; k += BLOCK_K_LOCAL) {
      int64_t kb_size = std::min((int64_t)BLOCK_K_LOCAL, (int64_t)(K - k));
      int idx = k >> 7;
      unpack_B(Btmp + k * ldb_tmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scale[idx]);
    }
  }
};

// MXFP4 unpack helper
template <>
struct unpack_helper<uint8_t, uint8_t> {
  static inline void unpack(
      at::BFloat16* Btmp, const uint8_t* B,
      int N, int K, int ldb, int ldb_tmp, const uint8_t* scale) {
    constexpr int BLOCK_N = block_size_n();
    for (int k = 0; k < K; k += 32) {
      unpack_B(Btmp + k * ldb_tmp, B + k * (ldb >> 1), N, 32, ldb, ldb_tmp, scale + (k >> 5) * BLOCK_N);
    }
  }
};

// Unified brgemm template (output to scalar_t) - primary template throws error
template <typename scalar_t, typename packed_t, typename param_t>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const param_t* __restrict__ scale,
      int M, int N, int K,
      int lda, int ldb, int ldc,
      bool do_unpack = true) {
    TORCH_CHECK(false, "brgemm: primary template not implemented, only BFloat16 is supported!");
  }
};

// FP8 specialization for BFloat16
template <>
struct brgemm<at::BFloat16, at::Float8_e4m3fn, float> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ scale,
      int M, int N, int K,
      int lda, int ldb, int ldc,
      bool do_unpack = true) {
    constexpr int BLOCK_N = block_size_n();
    const int ldb_tmp = BLOCK_N;

    if (do_unpack) {
      unpack_helper<at::Float8_e4m3fn, float>::unpack(Btmp, B, N, K, ldb, ldb_tmp, scale);
    }

    at::native::cpublas::brgemm(M, N, K, lda, ldb_tmp, BLOCK_N, false, A, Btmp, Ctmp);

    for (int m = 0; m < M; ++m) {
      copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
    }
  }
};

// MXFP4 specialization for BFloat16
template <>
struct brgemm<at::BFloat16, uint8_t, uint8_t> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const uint8_t* __restrict__ scale,
      int M, int N, int K,
      int lda, int ldb, int ldc,
      bool do_unpack = true) {
    constexpr int BLOCK_N = block_size_n();
    const int ldb_tmp = BLOCK_N;

    if (do_unpack) {
      unpack_helper<uint8_t, uint8_t>::unpack(Btmp, B, N, K, ldb, ldb_tmp, scale);
    }

    at::native::cpublas::brgemm(M, N, K, lda, ldb_tmp, BLOCK_N, false, A, Btmp, Ctmp);

    for (int m = 0; m < M; ++m) {
      copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
    }
  }
};

// Unified brgemm2 template (output to float) - primary template throws error
template <typename scalar_t, typename packed_t, typename param_t>
struct brgemm2 {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      float* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      const param_t* __restrict__ scale,
      int M, int N, int K,
      int lda, int ldb, int ldc,
      bool do_unpack = true) {
    TORCH_CHECK(false, "brgemm2: primary template not implemented, only BFloat16 is supported!");
  }
};

// FP8 specialization for BFloat16
template <>
struct brgemm2<at::BFloat16, at::Float8_e4m3fn, float> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      float* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      const float* __restrict__ scale,
      int M, int N, int K,
      int lda, int ldb, int ldc,
      bool do_unpack = true) {
    constexpr int BLOCK_N = block_size_n();
    const int ldb_tmp = BLOCK_N;

    if (do_unpack) {
      unpack_helper<at::Float8_e4m3fn, float>::unpack(Btmp, B, N, K, ldb, ldb_tmp, scale);
    }

    at::native::cpublas::brgemm(M, N, K, lda, ldb_tmp, BLOCK_N, false, A, Btmp, C);
  }
};

// MXFP4 specialization for BFloat16
template <>
struct brgemm2<at::BFloat16, uint8_t, uint8_t> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      float* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      const uint8_t* __restrict__ scale,
      int M, int N, int K,
      int lda, int ldb, int ldc,
      bool do_unpack = true) {
    constexpr int BLOCK_N = block_size_n();
    const int ldb_tmp = BLOCK_N;

    if (do_unpack) {
      unpack_helper<uint8_t, uint8_t>::unpack(Btmp, B, N, K, ldb, ldb_tmp, scale);
    }

    at::native::cpublas::brgemm(M, N, K, lda, ldb_tmp, BLOCK_N, false, A, Btmp, C);
  }
};

#endif // CPU_CAPABILITY_AVX512

// ===== tinygemm_kernel interface functions =====

// Macro to dispatch tinygemm_kernel_nn based on block size
#define DISPATCH_TINYGEMM_NN(SCALAR_T, PACKED_T, PARAM_T, MB_SIZE)                    \
  tinygemm_kernel_nn<SCALAR_T, PACKED_T, PARAM_T, MB_SIZE, 32>::apply(                \
      A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start,            \
      Btmp, Ctmp, scale, M, N, K, lda, ldb, ldc, block_size_K, do_unpack_block)

// Macro to dispatch tinygemm_kernel_nn2 based on block size
#define DISPATCH_TINYGEMM_NN2(SCALAR_T, PACKED_T, PARAM_T, MB_SIZE)                   \
  tinygemm_kernel_nn2<SCALAR_T, PACKED_T, PARAM_T, MB_SIZE, 32>::apply(               \
      A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start,            \
      Btmp, scale, M, N, K, lda, ldb, ldc, block_size_K, do_unpack_block)

// Unified tinygemm_kernel interface (output to scalar_t)
template <typename scalar_t, typename packed_t, typename param_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const param_t* __restrict__ scale,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    bool brg, int64_t block_size_K, bool do_unpack) {
  
  if (brg) {
    brgemm<scalar_t, packed_t, param_t>::apply(
        A, B, C, Btmp, Ctmp, scale, M, N, K, lda, ldb, ldc, do_unpack);
    return;
  }

  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);
      bool do_unpack_block = (mb == 0) && do_unpack;

      switch (mb_size << 4 | nb_size >> 4) {
        case 0x12: DISPATCH_TINYGEMM_NN(scalar_t, packed_t, param_t, 1); break;
        case 0x22: DISPATCH_TINYGEMM_NN(scalar_t, packed_t, param_t, 2); break;
        case 0x32: DISPATCH_TINYGEMM_NN(scalar_t, packed_t, param_t, 3); break;
        case 0x42: DISPATCH_TINYGEMM_NN(scalar_t, packed_t, param_t, 4); break;
        default: TORCH_CHECK(false, "Unexpected block size");
      }
    }
  }
}

// Unified tinygemm_kernel interface (output to float)
template <typename scalar_t, typename packed_t, typename param_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    float* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    const param_t* __restrict__ scale,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    bool brg, int64_t block_size_K, bool do_unpack) {
  
  if (brg) {
    brgemm2<scalar_t, packed_t, param_t>::apply(
        A, B, C, Btmp, scale, M, N, K, lda, ldb, ldc, do_unpack);
    return;
  }

  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);
      bool do_unpack_block = (mb == 0) && do_unpack;

      switch (mb_size << 4 | nb_size >> 4) {
        case 0x12: DISPATCH_TINYGEMM_NN2(scalar_t, packed_t, param_t, 1); break;
        case 0x22: DISPATCH_TINYGEMM_NN2(scalar_t, packed_t, param_t, 2); break;
        case 0x32: DISPATCH_TINYGEMM_NN2(scalar_t, packed_t, param_t, 3); break;
        case 0x42: DISPATCH_TINYGEMM_NN2(scalar_t, packed_t, param_t, 4); break;
        default: TORCH_CHECK(false, "Unexpected block size");
      }
    }
  }
}

#undef DISPATCH_TINYGEMM_NN
#undef DISPATCH_TINYGEMM_NN2

}  // anonymous namespace

template <typename scalar_t, typename packed_t, typename param_t, bool is_mxfp4>
void fused_experts_fp_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const packed_t* __restrict__ packed_w1,
    const packed_t* __restrict__ packed_w2,
    const scalar_t* __restrict__ w1_bias,
    const scalar_t* __restrict__ w2_bias,
    const param_t* __restrict__ w1s,
    const param_t* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad,
    float alpha,
    float limit,
    CPUAcTMethod act_func,
    bool with_bias) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 1: intermediate_cache0 = hidden_states @ w1
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(2 * N, BLOCK_N);
  int64_t scale_size_N = div_up(2 * N, block_size_N);
  int64_t scale_size_K = div_up(K, block_size_K);
  int64_t blocks_n_per_group = block_size_N / BLOCK_N;
  std::function<int64_t(int64_t)> scale_offset_per_block;
  if constexpr (is_mxfp4) {
    scale_offset_per_block = [&](int64_t a) { return a * BLOCK_N; };
  } else {
    scale_offset_per_block = [&](int64_t a) { return a / blocks_n_per_group; };
  }

  const int64_t packed_K = get_row_size<packed_t>(K);

  const int64_t stride_e = 2 * N * packed_K;
  const int64_t stride_n = packed_K;

  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  // const bool use_brgemm = can_use_brgemm<packed_t>(avg_M);
  const bool use_brgemm = true;

  int64_t B_tmp_offset_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * 2 * N * K;
  int64_t B_tmp_size_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * 3 * N * K;

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    

    loop_2d<packed_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t n_size = std::min(2 * N - nb * BLOCK_N, BLOCK_N);

      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const packed_t* __restrict__ B = packed_w1 + expert_id * stride_e + nb * BLOCK_N * stride_n;
      const param_t* __restrict__ Bs =
          w1s + expert_id * scale_size_N * scale_size_K + scale_offset_per_block(nb) * scale_size_K;

      // do unpacking for the first row or a new expert
      int32_t pre_expert_id = mb == 0 ? -1 : expert_ids[mb - 1];
      bool do_unpack = (mb == mb0) || (expert_id != pre_expert_id);

      // 1.a load A
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, input + index * K, K);
      }

      const int64_t offset = offsets[mb];
      tinygemm_kernel<scalar_t>(
          /*   A            */ A,
          /*   B            */ B,
          /*   C            */ C0,
          /*   Btmp         */ B_tmp + tid * B_tmp_size_per_thread + nb_offset * BLOCK_N * K,
          /*   scale        */ Bs,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ K,
          /*   lda          */ K,
          /*   ldb          */ n_size,
          /*   ldc          */ BLOCK_N,
          /*   brg          */ use_brgemm,
          /*   block_size_K */ block_size_K,
          /*   do_unpack    */ do_unpack);
      if (with_bias) {
        const scalar_t* __restrict__ B_bias = w1_bias + expert_id * 2 * N + nb * BLOCK_N;
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C0 + m * BLOCK_N, B_bias, n_size);
        }
      }
      if (act_func == CPUAcTMethod::swiglu) {
        clamp_sigmoid_and_mul<scalar_t, BLOCK_N>(
            ic1 + offset * N,
            C0,
            m_size,
            N,
            alpha,
            limit,
            nb * (BLOCK_N / 2));
      } else if (act_func == CPUAcTMethod::silu_and_mul) {
        // copy C0 to ic0
        for (int64_t m = 0; m < m_size; ++m) {
          copy_stub(ic0 + (offset + m) * N * 2 + nb * BLOCK_N, C0 + m * BLOCK_N, n_size);
        }
      }

    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 1.5: intermediate_cache1 = silu(intermediate_cache0)
  if (act_func == CPUAcTMethod::silu_and_mul) {
    at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
      }
    });
  }
  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  scale_size_N = div_up(K, block_size_N);
  scale_size_K = div_up(N, block_size_K);
  const int64_t packed_IC = get_row_size<packed_t>(IC);
  const int64_t stride_e2 = OC * packed_IC;
  const int64_t stride_oc = packed_IC;

  // parallel on [MB2, NB2]
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<packed_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const packed_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const param_t* __restrict__ Bs =
          w2s + expert_id * scale_size_N * scale_size_K + scale_offset_per_block(nb) * scale_size_K;

      // do unpacking for the first row or a new expert
      int32_t pre_expert_id = mb == 0 ? -1 : expert_ids[mb - 1];
      bool do_unpack = (mb == mb0) || (expert_id != pre_expert_id);

      tinygemm_kernel<scalar_t>(
          /*   A            */ A,
          /*   B            */ B,
          /*   C            */ C0,
          /*   Btmp         */ B_tmp + tid * B_tmp_size_per_thread + B_tmp_offset_per_thread + nb_offset * BLOCK_N * IC,
          /*   scale        */ Bs,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ IC,
          /*   lda          */ IC,
          /*   ldb          */ n_size,
          /*   ldc          */ BLOCK_N,
          /*   brg          */ use_brgemm,
          /*   block_size_K */ block_size_K,
          /*   do_unpack    */ do_unpack);
     
      if (with_bias) {
        const scalar_t* __restrict__ B_bias = w2_bias + expert_id * OC + nb * BLOCK_N;
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C0 + m * BLOCK_N, B_bias, n_size);
        }
      }
      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C0 + m * BLOCK_N, weight, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
  // stage 3: out = intermediate_cache2.sum(dim=1)
  //   from [M, topk, K] to [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#define INSTANTIATE_MOE_FP_TEMPLATE(TYPE1, TYPE2, TYPE3, IS_MXFP4)           \
  template void fused_experts_fp_kernel_impl<TYPE1, TYPE2, TYPE3, IS_MXFP4>( \
      TYPE1* __restrict__ output,                                            \
      TYPE1* __restrict__ ic0,                                               \
      TYPE1* __restrict__ ic1,                                               \
      TYPE1* __restrict__ ic2,                                               \
      TYPE1* __restrict__ A_tmp,                                             \
      TYPE1* __restrict__ B_tmp,                                             \
      float* __restrict__ C_tmp,                                             \
      const TYPE1* __restrict__ input,                                       \
      const TYPE2* __restrict__ packed_w1,                                   \
      const TYPE2* __restrict__ packed_w2,                                   \
      const TYPE1* __restrict__ w1_bias,                                     \
      const TYPE1* __restrict__ w2_bias,                                     \
      const TYPE3* __restrict__ w1s,                                         \
      const TYPE3* __restrict__ w2s,                                         \
      int64_t block_size_N,                                                  \
      int64_t block_size_K,                                                  \
      const float* __restrict__ topk_weights,                                \
      const int32_t* __restrict__ sorted_ids,                                \
      const int32_t* __restrict__ expert_ids,                                \
      const int32_t* __restrict__ offsets,                                   \
      int64_t M,                                                             \
      int64_t N,                                                             \
      int64_t K,                                                             \
      int64_t E,                                                             \
      int64_t topk,                                                          \
      int64_t num_tokens_post_pad,                                           \
      float alpha,                                                           \
      float limit,                                                           \
      CPUAcTMethod act_func,                                                 \
      bool with_bias)

INSTANTIATE_MOE_FP_TEMPLATE(at::BFloat16, at::Float8_e4m3fn, float, false);
INSTANTIATE_MOE_FP_TEMPLATE(at::Half, at::Float8_e4m3fn, float, false);
INSTANTIATE_MOE_FP_TEMPLATE(at::BFloat16, uint8_t, uint8_t, true);
INSTANTIATE_MOE_FP_TEMPLATE(at::Half, uint8_t, uint8_t, true);

template <typename scalar_t>
void shared_expert_fp8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 1: intermediate_cache0 = hidden_states @ w1
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(2 * N, BLOCK_N);
  int64_t scale_size_K = div_up(K, block_size_K);
  int64_t blocks_n_per_group = block_size_N / BLOCK_N;

  const bool use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(M);

  int64_t B_tmp_size_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * 3 * N * K;

  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();

    loop_2d<at::Float8_e4m3fn>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);
      int64_t n_size = std::min(2 * N - nb * BLOCK_N, BLOCK_N);

      // do unpacking for the first row
      bool do_unpack = (mb == mb0);

      tinygemm_kernel<scalar_t>(
          /*   A            */ input + mb * BLOCK_M * K,
          /*   B            */ packed_w1 + nb * BLOCK_N * K,
          /*   C            */ ic0 + mb * BLOCK_M * 2 * N + nb * BLOCK_N,
          /*   Btmp         */ B_tmp + tid * B_tmp_size_per_thread + nb_offset * BLOCK_N * K,
          /*   Ctmp         */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N,
          /*   scale        */ w1s + (nb / blocks_n_per_group) * scale_size_K,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ K,
          /*   lda          */ K,
          /*   ldb          */ n_size,
          /*   ldc          */ 2 * N,
          /*   brg          */ use_brgemm,
          /*   block_size_K */ block_size_K,
          /*   do_unpack    */ do_unpack);
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 1.5: intermediate_cache1 = silu(intermediate_cache0)
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
    }
  });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [K, N] as [OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(K, BLOCK_N);
  scale_size_K = div_up(N, block_size_K);
  int64_t B_tmp_offset_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * 2 * N * K;

  // parallel on [MB2, NB2]
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    alignas(64) scalar_t C[BLOCK_M * BLOCK_N];

    loop_2d<at::Float8_e4m3fn>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // do unpacking for the first row
      bool do_unpack = (mb == mb0);

      // 2.a gemm: C = A @ B
      tinygemm_kernel<scalar_t>(
          /*   A            */ ic1 + mb * BLOCK_M * N,
          /*   B            */ packed_w2 + nb * BLOCK_N * N,
          /*   C            */ C,
          /*   Btmp         */ B_tmp + tid * B_tmp_size_per_thread + B_tmp_offset_per_thread + nb_offset * BLOCK_N * IC,
          /*   Ctmp         */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N,
          /*   scale        */ w2s + (nb / blocks_n_per_group) * scale_size_K,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ IC,
          /*   lda          */ IC,
          /*   ldb          */ n_size,
          /*   ldc          */ BLOCK_N,
          /*   brg          */ use_brgemm,
          /*   block_size_K */ block_size_K,
          /*   do_unpack    */ do_unpack);

      // 2.b copy from C to output and add fused_experts_out
      scalar_t* __restrict__ out = output + mb * BLOCK_M * K + nb * BLOCK_N;
      const scalar_t* __restrict__ fused_out = fused_experts_out + mb * BLOCK_M * K + nb * BLOCK_N;
      for (int64_t m = 0; m < m_size; ++m) {
        add_mul_stub(out + m * K, C + m * BLOCK_N, fused_out + m * K, routed_scaling_factor, n_size);
      }
    });
  });

  if (use_brgemm) {
    at::native::cpublas::brgemm_release();
  }
}

#define INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(TYPE)   \
  template void shared_expert_fp8_kernel_impl<TYPE>(   \
      TYPE* __restrict__ output,                       \
      TYPE* __restrict__ ic0,                          \
      TYPE* __restrict__ ic1,                          \
      TYPE* __restrict__ B_tmp,                        \
      float* __restrict__ C_tmp,                       \
      const TYPE* __restrict__ input,                  \
      const at::Float8_e4m3fn* __restrict__ packed_w1, \
      const at::Float8_e4m3fn* __restrict__ packed_w2, \
      const float* __restrict__ w1s,                   \
      const float* __restrict__ w2s,                   \
      int64_t block_size_N,                            \
      int64_t block_size_K,                            \
      const TYPE* __restrict__ fused_experts_out,      \
      float routed_scaling_factor,                     \
      int64_t M,                                       \
      int64_t N,                                       \
      int64_t K)

INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(at::BFloat16);
INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(at::Half);

}  // namespace cpu
}  // namespace megablocks
