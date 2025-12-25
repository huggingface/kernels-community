/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************************/

// Fallback implementation - no special CPU features required
// Uses PyTorch's brgemm which has multiple backend implementations

#include <ATen/ATen.h>

#include "fmha_fwd_common.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_fallback.hpp"

namespace flash_attn_cpu {
namespace fallback {

//==============================================================================
// Scalar VNNI packing utilities (no AVX512 intrinsics)
//==============================================================================

template <typename scalar_t>
void pack_vnni_scalar(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst[k * ld_dst * 2 + n * 2 + d] = src[n * ld_src + k * 2 + d];
      }
    }
  }
}

template <typename scalar_t>
void pack_vnni2_scalar(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[(k + 0) * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[(k + 1) * ld_src + n];
    }
  }
  if (K % 2 != 0) {
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[(K - 1) * ld_src + n];
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
  }
}

//==============================================================================
// Fallback PackPolicy
//==============================================================================

struct FallbackPackPolicy {
  template <typename scalar_t>
  static void pack_vnni(scalar_t* dst, const scalar_t* src, int N, int K, int ld_src, int ld_dst) {
    pack_vnni_scalar<scalar_t>(dst, src, N, K, ld_src, ld_dst);
  }

  template <typename scalar_t>
  static void pack_vnni2(scalar_t* dst, const scalar_t* src, int K, int N, int ld_src, int ld_dst) {
    pack_vnni2_scalar<scalar_t>(dst, src, K, N, ld_src, ld_dst);
  }

  template <typename scalar_t, int BLOCK_N>
  static void copy_stub_block(scalar_t* __restrict__ out, const float* __restrict__ input) {
    for (int i = 0; i < BLOCK_N; ++i) {
      out[i] = static_cast<scalar_t>(input[i]);
    }
  }
};

//==============================================================================
// Public API
//==============================================================================

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
  fmha_fwd_varlen_impl_template<FallbackPackPolicy>(
      q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal);
}

}  // namespace fallback
}  // namespace flash_attn_cpu
