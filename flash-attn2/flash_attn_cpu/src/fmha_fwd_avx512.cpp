/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************************/

// AVX512 implementation - compile with -mavx512f -mavx512bf16 -mamx-bf16 -mamx-tile

#define CPU_CAPABILITY_AVX512

#include <immintrin.h>
#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>

#include "fmha_fwd_common.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_avx512.hpp"

namespace flash_attn_cpu {
namespace avx512 {

using namespace at::vec;

//==============================================================================
// AVX512 specific: float32->bfloat16/float16 conversion
//==============================================================================

template <typename scalar_t, typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline Vectorized<scalar_t> convert_from_float_avx512(const Vectorized<float>& a, const Vectorized<float>& b) {
  return at::vec::convert_from_float<scalar_t>(a, b);
}

// BFloat16 specialization using native AVX512-BF16 instruction
template <>
inline Vectorized<at::BFloat16>
convert_from_float_avx512<at::BFloat16>(const Vectorized<float>& a, const Vectorized<float>& b) {
  return (__m512i)(_mm512_cvtne2ps_pbh(__m512(b), __m512(a)));
}

// Float16 specialization using AVX512-FP16 or fallback
template <>
inline Vectorized<at::Half>
convert_from_float_avx512<at::Half>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // Use PyTorch's vectorized conversion which handles fp16 correctly
  return at::vec::convert_from_float<at::Half>(a, b);
}

//==============================================================================
// AVX512 Transpose utilities
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

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

//==============================================================================
// AVX512 VNNI packing utilities
//==============================================================================

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
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

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
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask2 = (1 << N) - 1;
  for (int k = 0; k < K2; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask2, vinputs[k]);
  }
}

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
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

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
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);

  if (N <= 16) {
    const __mmask16 vmask2 = (1 << N) - 1;
    _mm512_mask_storeu_epi32(dst + 0 * ld_dst * 2, vmask2, d0);
  } else {
    const __mmask16 vmask2 = (1 << (N - 16)) - 1;
    _mm512_storeu_epi32(dst + 0 * ld_dst * 2, d0);
    _mm512_mask_storeu_epi32(dst + 0 * ld_dst * 2 + 32, vmask2, d1);
  }
}

template <typename scalar_t, typename index_t, bool is_indexed>
void pack_vnni_impl(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  const int NB = div_up(N, 16);
  const int KB = K / 32;
  const int K_remainder = K - KB * 32;

  for (int nb = 0; nb < NB; ++nb) {
    int nb_size = std::min(N - nb * 16, 16);
    for (int kb = 0; kb < KB; ++kb) {
      pack_vnni_Nx32<scalar_t, index_t>(
          dst + ((kb * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          src + kb * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          is_indexed ? ind + nb * 16 : nullptr,
          nb_size,
          ld_src,
          ld_dst);
    }
    if (K_remainder > 0) {
      pack_vnni_N_remainder<scalar_t, index_t>(
          dst + ((KB * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          src + KB * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          is_indexed ? ind + nb * 16 : nullptr,
          nb_size,
          K_remainder,
          ld_src,
          ld_dst);
    }
  }
}

template <typename scalar_t, typename index_t, bool is_indexed>
void pack_vnni2_impl(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  const int KB = div_up(K, 2);
  const int NB = N / 32;
  const int N_remainder = N - NB * 32;

  for (int kb = 0; kb < KB; ++kb) {
    int kb_size = std::min(K - kb * 2, 2);
    for (int nb = 0; nb < NB; ++nb) {
      pack_vnni_Kx32<scalar_t, index_t>(
          dst + ((kb * 2) >> 1) * ld_dst * 2 + nb * 32 * 2,
          src + (is_indexed ? 0 : kb * 2 * ld_src) + nb * 32,
          is_indexed ? ind + kb * 2 : nullptr,
          kb_size,
          ld_src,
          ld_dst);
    }
    if (N_remainder > 0) {
      pack_vnni_K_remainder(
          dst + ((kb * 2) >> 1) * ld_dst * 2 + NB * 32 * 2,
          src + (is_indexed ? 0 : kb * 2 * ld_src) + NB * 32,
          is_indexed ? ind + kb * 2 : nullptr,
          kb_size,
          N_remainder,
          ld_src,
          ld_dst);
    }
  }
}

}  // namespace avx512

//==============================================================================
// Pack functions implementation (called from fmha_fwd_kernel.hpp)
// These are in flash_attn_cpu namespace to match declarations in fmha_fwd_kernel.hpp
//==============================================================================

template <typename scalar_t>
void pack_vnni(scalar_t* dst, const scalar_t* src, int N, int K, int ld_src, int ld_dst) {
  avx512::pack_vnni_impl<scalar_t, int32_t, false>(dst, src, nullptr, N, K, ld_src, ld_dst);
}

template <typename scalar_t>
void pack_vnni2(scalar_t* dst, const scalar_t* src, int K, int N, int ld_src, int ld_dst) {
  avx512::pack_vnni2_impl<scalar_t, int32_t, false>(dst, src, nullptr, K, N, ld_src, ld_dst);
}

template <typename scalar_t, int BLOCK_N>
void copy_stub_block(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    if constexpr (col % 2 == 0) {
      fVec a_fvec0 = fVec::loadu(input + col * 16);
      fVec a_fvec1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = avx512::convert_from_float_avx512<scalar_t>(a_fvec0, a_fvec1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

// Explicit template instantiations
template void pack_vnni<at::BFloat16>(at::BFloat16*, const at::BFloat16*, int, int, int, int);
template void pack_vnni<at::Half>(at::Half*, const at::Half*, int, int, int, int);
template void pack_vnni2<at::BFloat16>(at::BFloat16*, const at::BFloat16*, int, int, int, int);
template void pack_vnni2<at::Half>(at::Half*, const at::Half*, int, int, int, int);
template void copy_stub_block<at::BFloat16, 768>(at::BFloat16*, const float*);
template void copy_stub_block<at::Half, 768>(at::Half*, const float*);

//==============================================================================
// Public API wrapper in avx512 namespace
//==============================================================================

namespace avx512 {

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
  flash_attn_cpu::fmha_fwd_varlen_kernel(
      q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal);
}

}  // namespace avx512
}  // namespace flash_attn_cpu
