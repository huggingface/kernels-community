/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************************/

// Shared flash attention kernel implementation
// This template is parameterized by PackPolicy which provides pack_vnni, pack_vnni2, copy_stub_block

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>

#include "fmha_fwd_common.hpp"

#include <cmath>
#include <limits>

namespace flash_attn_cpu {

//==============================================================================
// Flash Attention Kernel Implementation (shared)
// PackPolicy must provide:
//   - template<scalar_t> pack_vnni(dst, src, N, K, ld_src, ld_dst)
//   - template<scalar_t> pack_vnni2(dst, src, K, N, ld_src, ld_dst)
//   - template<scalar_t, BLOCK_N> copy_stub_block(out, input)
//==============================================================================

template <typename scalar_t, int BLOCK_M, int BLOCK_N, typename PackPolicy>
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

  static_assert(BLOCK_M <= BLOCK_N);

  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // compute index (bs, mb_offset) for Query blocks
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
  int MB = idx;

  const int ldb_tmp = std::max(head_size, head_size_v);
  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  parallel_for(num_heads * MB, [&](int begin, int end) {
    int head_id{0}, mb{0};
    data_index_init(begin, head_id, num_heads, mb, MB);

    int tid = get_thread_num();
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    float* __restrict__ s_delta = s_i;
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;
    scalar_t* __restrict__ s_delta2 = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);
    scalar_t* __restrict__ Btmp = s_delta2 + BLOCK_M * BLOCK_N;

    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      int32_t bs = indices[mb * 2 + 0];
      int32_t seq_q_start_loc = cu_seqlens_q[bs];
      int32_t seq_k_start_loc = cu_seqlens_k[bs];
      int32_t seqlen_q = cu_seqlens_q[bs + 1] - cu_seqlens_q[bs];

      int m = indices[mb * 2 + 1] * BLOCK_M;
      int m_size = std::min(BLOCK_M, seqlen_q - m);
      assert(m_size > 0);

      int head_kv_id = head_id / num_groups;
      const scalar_t* __restrict__ q_ptr = q + (seq_q_start_loc + m) * q_strideM + head_id * q_strideH;

      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<float>::infinity(), m_size);

      int seqlen_k = cu_seqlens_k[bs + 1] - cu_seqlens_k[bs];
      int num_keys = causal ? std::min(seqlen_k - seqlen_q + m + m_size, seqlen_k) : seqlen_k;

      for (int n = 0; n < num_keys; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n);
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // pack key
        PackPolicy::template pack_vnni<scalar_t>(
            Btmp,
            k + (seq_k_start_loc + n) * k_strideN + head_kv_id * k_strideH,
            n_size,
            head_size,
            k_strideN,
            BLOCK_N);

        // Q @ K
        at::native::cpublas::brgemm(
            m_size, n_size, head_size,
            q_strideM, BLOCK_N, BLOCK_N,
            false,
            q_ptr, Btmp, s_i);

        // causal mask - apply to every block where masking is needed
        if (causal) {
          for (int row = 0; row < m_size; ++row) {
            // q_pos_in_k is the position in K that this query can attend up to (inclusive)
            int q_pos_in_k = seqlen_k - seqlen_q + m + row;
            // last_col is the last valid column in the current K block
            int last_col = q_pos_in_k - n;
            if (last_col < n_size - 1) {
              float* row_ptr = s_i + row * BLOCK_N;
              int start_col = std::max(0, last_col + 1);
              fill_stub(row_ptr + start_col, -std::numeric_limits<float>::infinity(), n_size - start_col);
            }
          }
        }

        const Vec scale_vec = Vec(sm_scale);
        for (int row = 0; row < m_size; ++row) {
          at::vec::map<float>(
              [scale_vec](Vec x) { return x * scale_vec; }, s_i + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + row * BLOCK_N, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // Handle case where entire row is masked (m_i == -inf)
          // In this case, skip the update to avoid NaN from exp(-inf - (-inf))
          if (std::isinf(m_i) && m_i < 0) {
            // Entire row is masked, fill s_delta with zeros
            fill_stub(s_delta + row * BLOCK_N, 0.f, padded_n_size);
            PackPolicy::template copy_stub_block<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
            continue;
          }

          float m_delta = std::exp(m_prime[row] - m_i);

          at::vec::map<float>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); }, s_delta + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          s_prime[row] *= m_delta;
          s_prime[row] += at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          at::vec::map<float>(
              [m_delta](Vec x) { return x * Vec(m_delta); },
              v_prime + row * head_size_v, v_prime + row * head_size_v, head_size_v);

          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          PackPolicy::template copy_stub_block<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        // pack value
        PackPolicy::template pack_vnni2<scalar_t>(
            Btmp,
            v + (seq_k_start_loc + n) * v_strideN + head_kv_id * v_strideH,
            n_size,
            head_size_v,
            v_strideN,
            head_size_v);

        // s_delta @ V + V'
        at::native::cpublas::brgemm(
            m_size, head_size_v, padded_n_size,
            BLOCK_N, head_size_v, head_size_v,
            true,
            s_delta2, Btmp, v_prime);
      }

      scalar_t* __restrict__ out_ptr = out + (seq_q_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        // When s_prime[row] == 0 (entire row masked by causal), output zeros instead of NaN
        if (s_prime[row] == 0.f) {
          fill_stub(out_ptr + row * o_strideM, static_cast<scalar_t>(0), head_size_v);
        } else {
          float s = 1 / s_prime[row];
          copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
        }
      }

      data_index_step(head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

//==============================================================================
// Buffer management (shared)
//==============================================================================

template <int BLOCK_M, int BLOCK_N>
inline int resize_buffer(at::Tensor& buffer, int num_threads, int head_size, int head_size_v) {
  const int size_per_thread =
      BLOCK_M * BLOCK_N * sizeof(float) +
      BLOCK_M * head_size_v * sizeof(float) +
      BLOCK_M * BLOCK_N * sizeof(uint16_t) +
      BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  buffer.resize_({num_threads, size_per_thread});
  return size_per_thread;
}

template <int BLOCK_M>
inline void resize_indices(at::Tensor& indices, int num_seqs, int max_seqlen_q) {
  indices.resize_({num_seqs, div_up(max_seqlen_q, BLOCK_M), 2});
}

//==============================================================================
// Shared fmha_fwd_varlen_impl template
//==============================================================================

template <typename PackPolicy>
void fmha_fwd_varlen_impl_template(
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

  int q_strideM = q.stride(0);
  int q_strideH = q.stride(1);
  int k_strideN = k.stride(0);
  int k_strideH = k.stride(1);
  int v_strideN = v.stride(0);
  int v_strideH = v.stride(1);

  CHECK_EQ(k.size(2), head_size);
  CHECK_EQ(v.size(1), num_heads_kv);
  CHECK_EQ(cu_seqlens_k.size(0), num_seqs + 1);

  TORCH_CHECK(head_size % 2 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 2 == 0, "invalid head_size_v ", head_size_v);

  float sm_scale = softmax_scale;
  if (sm_scale == 0.0f || std::isnan(sm_scale)) {
    sm_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  }

  int num_threads = at::get_num_threads();
  at::Tensor buffer = at::empty({}, q.options().dtype(at::kChar));
  at::Tensor indices = at::empty({}, q.options().dtype(at::kInt));

  if (out.numel() == 0) {
    out.resize_({num_tokens, num_heads, head_size_v});
  }

  constexpr int BLOCK_M = 256;
  constexpr int BLOCK_N = 768;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "fmha_fwd_varlen_impl", [&] {
    int sz = resize_buffer<BLOCK_M, BLOCK_N>(buffer, num_threads, head_size, head_size_v);
    resize_indices<BLOCK_M>(indices, num_seqs, max_seqlen_q);

    flash_attn_varlen_kernel_impl<scalar_t, BLOCK_M, BLOCK_N, PackPolicy>(
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

}  // namespace flash_attn_cpu
