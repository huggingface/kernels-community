// Copyright © 2024 Apple Inc.
//
// Variable-length (cu_seqlens) decode attention. These are MLX's vector
// kernels (sdpa_vector.h) adapted to packed [total_tokens, heads, head_dim]
// inputs, used when every sequence has only a few query tokens. Differences
// from upstream:
//
// - Sequence bounds come from cu_seqlens_q/cu_seqlens_k, and inputs are
//   addressed by (token, head) strides with 64-bit offsets.
// - Optional tanh softcapping.
// - Attention sinks are float32 and also supported by the GQA read-once
//   variant.
// - Causal masking is aligned to the bottom-right, like the steel kernel.
//   Query rows that see no keys produce zeros.
// - No array masks.

#include <metal_simdgroup>

// clang-format off
#include "utils.h"
#include "varlen_params.h"

using namespace metal;

constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
constant bool has_softcap [[function_constant(303)]];
constant int blocks [[function_constant(304)]];

template <typename U>
METAL_FUNC U apply_softcap(U score, const constant VarlenAttnParams* params) {
  if (has_softcap) {
    score = params->softcap * metal::precise::tanh(score / params->softcap);
  }
  return score;
}

template <typename T, int D>
[[kernel]] void sdpa_vector_varlen(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant VarlenAttnParams* params [[buffer(4)]],
    const device int* cu_seqlens_q [[buffer(5)]],
    const device int* cu_seqlens_k [[buffer(6)]],
    const device float* sinks [[buffer(7), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) { // clang-format on
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int seq_idx = tid.z;

  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int qL = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  // The grid is sized for the longest sequence. The whole threadgroup exits
  // here, before any barrier.
  if (q_seq_idx >= qL) {
    return;
  }
  const int k_seq_start = cu_seqlens_k[seq_idx];
  const int N = cu_seqlens_k[seq_idx + 1] - k_seq_start;

  const int kv_head_idx = head_idx / (params->H / params->H_kv);
  const int64_t q_row = int64_t(q_seq_start) + q_seq_idx;
  const int k_seq_stride = int(params->k_strides[0]);
  const int v_seq_stride = int(params->v_strides[0]);
  const int inner_k_stride = BN * k_seq_stride;
  const int inner_v_stride = BN * v_seq_stride;

  queries += q_row * params->q_strides[0] + head_idx * params->q_strides[1] +
      simd_lid * qk_per_thread;
  keys += (int64_t(k_seq_start) + simd_gid) * params->k_strides[0] +
      kv_head_idx * params->k_strides[1] + simd_lid * qk_per_thread;
  values += (int64_t(k_seq_start) + simd_gid) * params->v_strides[0] +
      kv_head_idx * params->v_strides[1] + simd_lid * v_per_thread;
  out += q_row * params->o_strides[0] + head_idx * params->o_strides[1] +
      simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(params->scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - qL + q_seq_idx);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = apply_softcap(simd_sum(score), params);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// First pass of the 2-pass kernel. Each threadgroup handles one key/value
// head and one of `blocks` strided slices of the key sequence, for every
// query head of the group and every query token of the sequence. The partial
// outputs are laid out as [total_q_tokens, H, blocks, D].
// clang-format off
template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_1_varlen(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant VarlenAttnParams* params [[buffer(4)]],
    const device int* cu_seqlens_q [[buffer(5)]],
    const device int* cu_seqlens_k [[buffer(6)]],
    const device float* sinks [[buffer(7), function_constant(has_sinks)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) { // clang-format on
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int seq_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;

  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int qL = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  // There is no barrier in this kernel, so simdgroups without a query token
  // can exit on their own.
  if (q_seq_idx >= qL) {
    return;
  }
  const int k_seq_start = cu_seqlens_k[seq_idx];
  const int N = cu_seqlens_k[seq_idx + 1] - k_seq_start;

  const int64_t q_row = int64_t(q_seq_start) + q_seq_idx;
  const int64_t o_offset = q_row * params->H + q_head_idx;
  const int k_seq_stride = int(params->k_strides[0]);
  const int v_seq_stride = int(params->v_strides[0]);

  queries += q_row * params->q_strides[0] + q_head_idx * params->q_strides[1] +
      simd_lid * qk_per_thread;
  keys += (int64_t(k_seq_start) + block_idx) * params->k_strides[0] +
      kv_head_idx * params->k_strides[1] + simd_lid * qk_per_thread;
  values += (int64_t(k_seq_start) + block_idx) * params->v_strides[0] +
      kv_head_idx * params->v_strides[1] + simd_lid * v_per_thread;
  out += o_offset * blocks * D + block_idx * D + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(params->scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - qL + q_seq_idx);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * keys[j];
      }
      score = apply_softcap(simd_sum(score), params);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * k_seq_stride;
    values += blocks * v_seq_stride;
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Duplication-free variant for high gqa_factor decode: each simdgroup owns a
// contiguous token sub-chunk and computes HPT of its group's query heads, so
// each K/V byte is read G / HPT times instead of G times. Single-token
// queries only; the partials layout matches sdpa_vector_2pass_2_varlen.
// clang-format off
template <typename T, int D, int G, int HPT>
[[kernel]] void sdpa_vector_2pass_1_gqa_varlen(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant VarlenAttnParams* params [[buffer(4)]],
    const device int* cu_seqlens_q [[buffer(5)]],
    const device int* cu_seqlens_k [[buffer(6)]],
    const device float* sinks [[buffer(7), function_constant(has_sinks)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) { // clang-format on
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int NT = G / HPT;

  typedef float U;

  const int kv_head_idx = tid.x;
  const int seq_idx = tid.y;
  const int block_idx = tid.z;
  const int num_blocks = tpg.z;
  const int g = tidtg.y;
  const int cchunk = g / NT;
  const int h0 = (g % NT) * HPT;

  const int q_seq_start = cu_seqlens_q[seq_idx];
  // Sequences without a query token have nothing to do. The whole
  // threadgroup exits here, before the barrier.
  if (cu_seqlens_q[seq_idx + 1] == q_seq_start) {
    return;
  }
  const int k_seq_start = cu_seqlens_k[seq_idx];
  const int N = cu_seqlens_k[seq_idx + 1] - k_seq_start;

  const int64_t q_row = q_seq_start;
  const int base_head = kv_head_idx * G;
  const int k_seq_stride = int(params->k_strides[0]);
  const int v_seq_stride = int(params->v_strides[0]);

  const int chunk = (N + num_blocks - 1) / num_blocks;
  const int kstart = block_idx * chunk;
  const int kend = min(N, kstart + chunk);
  const int sub = (chunk + HPT - 1) / HPT;
  const int s0 = kstart + cchunk * sub;
  const int s1 = min(kend, s0 + sub);

  const device T* kp = keys +
      (int64_t(k_seq_start) + s0) * params->k_strides[0] +
      kv_head_idx * params->k_strides[1] + simd_lid * qk_per_thread;
  const device T* vp = values +
      (int64_t(k_seq_start) + s0) * params->v_strides[0] +
      kv_head_idx * params->v_strides[1] + simd_lid * v_per_thread;

  U q[HPT][qk_per_thread];
  for (int j = 0; j < HPT; j++) {
    const device T* qp = queries + q_row * params->q_strides[0] +
        (base_head + h0 + j) * params->q_strides[1] + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[j][i] = static_cast<U>(params->scale) * qp[i];
    }
  }

  U max_score[HPT];
  U sum_exp_score[HPT];
  U o[HPT][v_per_thread];
  for (int j = 0; j < HPT; j++) {
    max_score[j] = Limits<U>::finite_min;
    sum_exp_score[j] = 0;
    for (int i = 0; i < v_per_thread; i++) {
      o[j][i] = 0;
    }
  }

  for (int t = s0; t < s1; t++) {
    U kr[qk_per_thread];
    U vr[v_per_thread];
    for (int i = 0; i < qk_per_thread; i++) {
      kr[i] = kp[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      vr[i] = vp[i];
    }
    kp += k_seq_stride;
    vp += v_seq_stride;
    for (int j = 0; j < HPT; j++) {
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[j][i] * kr[i];
      }
      score = apply_softcap(simd_sum(score), params);
      U new_max = max(max_score[j], score);
      U factor = fast::exp(max_score[j] - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score[j] = new_max;
      sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;
      for (int i = 0; i < v_per_thread; i++) {
        o[j][i] = o[j][i] * factor + exp_score * vr[i];
      }
    }
  }

  threadgroup U o_sh[G * HPT * D];
  threadgroup U se_sh[G * HPT];
  threadgroup U mx_sh[G * HPT];
  for (int j = 0; j < HPT; j++) {
    int slot = (h0 + j) * HPT + cchunk;
    U inv = sum_exp_score[j] > 0 ? 1 / sum_exp_score[j] : 0;
    for (int i = 0; i < v_per_thread; i++) {
      o_sh[slot * D + simd_lid * v_per_thread + i] = o[j][i] * inv;
    }
    if (simd_lid == 0) {
      se_sh[slot] = sum_exp_score[j];
      mx_sh[slot] = max_score[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The sink of each query head is counted once, in the first block, as a
  // score without a value.
  const bool add_sink = has_sinks && block_idx == 0;
  const U sink = add_sink ? static_cast<U>(sinks[base_head + g]) : 0;

  U gmax = add_sink ? sink : Limits<U>::finite_min;
  for (int s = 0; s < HPT; s++) {
    gmax = max(gmax, mx_sh[g * HPT + s]);
  }
  U denom = add_sink ? fast::exp(sink - gmax) : 0;
  U acc[v_per_thread] = {0};
  for (int s = 0; s < HPT; s++) {
    U w = se_sh[g * HPT + s] * fast::exp(mx_sh[g * HPT + s] - gmax);
    denom += w;
    for (int i = 0; i < v_per_thread; i++) {
      acc[i] += w * o_sh[(g * HPT + s) * D + simd_lid * v_per_thread + i];
    }
  }

  const int64_t o_offset = q_row * params->H + base_head + g;
  device T* op = out + o_offset * num_blocks * D + block_idx * D +
      simd_lid * v_per_thread;
  for (int i = 0; i < v_per_thread; i++) {
    op[i] = static_cast<T>(acc[i]);
  }
  if (simd_lid == 0) {
    sums[o_offset * num_blocks + block_idx] = denom;
    maxs[o_offset * num_blocks + block_idx] = gmax;
  }
}

// Second pass: combines the `blocks` partial outputs of each query row.
// clang-format off
template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2_varlen(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant VarlenAttnParams* params [[buffer(4)]],
    const constant int& num_blocks [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) { // clang-format on
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int64_t q_row = tid.y;
  const int64_t q_offset = q_row * params->H + head_idx;
  partials +=
      q_offset * num_blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * num_blocks;
  maxs += q_offset * num_blocks;
  out += q_row * params->o_strides[0] + head_idx * params->o_strides[1] +
      simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < num_blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < num_blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < num_blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// clang-format off
#define instantiate_sdpa_vector(tname, type, head_dim)            \
  instantiate_kernel(                                             \
      "sdpa_vector_varlen_" #tname "_" #head_dim,                 \
      sdpa_vector_varlen, type, head_dim)                         \
  instantiate_kernel(                                             \
      "sdpa_vector_2pass_1_varlen_" #tname "_" #head_dim,         \
      sdpa_vector_2pass_1_varlen, type, head_dim)                 \
  instantiate_kernel(                                             \
      "sdpa_vector_2pass_2_varlen_" #tname "_" #head_dim,         \
      sdpa_vector_2pass_2_varlen, type, head_dim)

#define instantiate_sdpa_vector_gqa(tname, type, head_dim, g, hpt) \
  instantiate_kernel(                                              \
      "sdpa_vector_2pass_1_gqa_" #g "_varlen_" #tname "_" #head_dim, \
      sdpa_vector_2pass_1_gqa_varlen, type, head_dim, g, hpt)

// Same head dims and GQA variants as upstream MLX, without the asymmetric
// (192, 128) and 512 variants; head_dim 32 is a local addition.
#define instantiate_sdpa_vector_heads(tname, type)      \
  instantiate_sdpa_vector(tname, type, 32)              \
  instantiate_sdpa_vector(tname, type, 64)              \
  instantiate_sdpa_vector(tname, type, 96)              \
  instantiate_sdpa_vector(tname, type, 128)             \
  instantiate_sdpa_vector(tname, type, 192)             \
  instantiate_sdpa_vector(tname, type, 256)             \
  instantiate_sdpa_vector_gqa(tname, type, 64, 8, 8)    \
  instantiate_sdpa_vector_gqa(tname, type, 128, 8, 4)   \
  instantiate_sdpa_vector_gqa(tname, type, 128, 12, 4)  \
  instantiate_sdpa_vector_gqa(tname, type, 128, 16, 2)

instantiate_sdpa_vector_heads(float32, float)
instantiate_sdpa_vector_heads(bfloat16, bfloat16_t)
instantiate_sdpa_vector_heads(float16, half)
// clang-format on
