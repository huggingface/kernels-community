#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace tdt_loss {

constexpr int kWarpSize = 32;
// Upper bound on the number of durations, only used to size shared memory.
constexpr int kMaxDurations = 64;

__device__ __forceinline__ float neg_inf() { return -INFINITY; }

// log(exp(a) + exp(b)), safe when either argument is -inf.
__device__ __forceinline__ double log_add(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  return fmax(a, b) + log1p(exp(-fabs(a - b)));
}

// Merge two partial (max, sum(exp(x - max))) pairs of an online softmax.
__device__ __forceinline__ void merge_max_sum(float &m, float &s, float m_other,
                                              float s_other) {
  const float m_new = fmaxf(m, m_other);
  if (m_new == -INFINITY) {
    m = m_new;
    s = 0.f;
    return;
  }
  s = s * expf(m - m_new) + s_other * expf(m_other - m_new);
  m = m_new;
}

// Block-wide reduction of (max, sum) pairs. The result is valid in all threads.
// `shm_m` and `shm_s` must hold at least blockDim.x / kWarpSize floats.
__device__ __forceinline__ void block_reduce_max_sum(float &m, float &s,
                                                     float *shm_m,
                                                     float *shm_s) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const float m_other = __shfl_xor_sync(0xffffffff, m, offset);
    const float s_other = __shfl_xor_sync(0xffffffff, s, offset);
    merge_max_sum(m, s, m_other, s_other);
  }
  if (lane == 0) {
    shm_m[warp] = m;
    shm_s[warp] = s;
  }
  __syncthreads();
  if (warp == 0) {
    m = lane < num_warps ? shm_m[lane] : -INFINITY;
    s = lane < num_warps ? shm_s[lane] : 0.f;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      const float m_other = __shfl_xor_sync(0xffffffff, m, offset);
      const float s_other = __shfl_xor_sync(0xffffffff, s, offset);
      merge_max_sum(m, s, m_other, s_other);
    }
    if (lane == 0) {
      shm_m[0] = m;
      shm_s[0] = s;
    }
  }
  __syncthreads();
  m = shm_m[0];
  s = shm_s[0];
}

// Clamp the per-sample lengths to the padded lattice so that malformed inputs
// cannot cause out-of-bounds accesses. Returns T (frames) and U (labels); the
// lattice for the sample spans t in [0, T) and u in [0, U].
__device__ __forceinline__ void sample_lengths(const int *source_lengths,
                                               const int *target_lengths,
                                               int b, int max_T, int max_U,
                                               int &T, int &U) {
  T = min(max(source_lengths[b], 0), max_T);
  U = min(max(target_lengths[b], 0), max_U - 1);
}

}  // namespace tdt_loss
