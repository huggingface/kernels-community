// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU Fused MoE Implementation
//
// Optimized with brgemm (batch-reduced GEMM) for Intel AMX
// Key optimizations from sglang:
//   1. VNNI weight packing for brgemm
//   2. BLOCK_M=32 matching AMX tile size
//   3. sorted token order for contiguous GEMM2 input
//   4. parallel_2d on (MB, NB) for better parallelism
//   5. fused silu_and_mul with GEMM1 output

#include "moe_ops.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace megablocks {
namespace cpu {

namespace {

// AMX tile sizes
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 32;

// Block sizes for tiling - 2 tiles per block for AMX efficiency
constexpr int64_t BLOCK_M = 2 * TILE_M;  // 32
constexpr int64_t BLOCK_N = 2 * TILE_N;  // 32

// Grain size for parallel_for
constexpr int64_t GRAIN_SIZE = 512;

inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// Determine if brgemm should be used based on M size
template <typename scalar_t>
inline bool can_use_brgemm(int64_t M) {
  return false;
}

template <>
inline bool can_use_brgemm<at::BFloat16>(int64_t M) {
  return M > 4;
}

template <>
inline bool can_use_brgemm<at::Half>(int64_t M) {
  return true;
}

template <>
inline bool can_use_brgemm<float>(int64_t M) {
  return M > 4;
}

// ======================== VNNI Weight Packing ========================
// Convert weight from [N, K] to VNNI format [K/2, N, 2] for bf16/fp16
// VNNI format is required by brgemm for efficient AMX computation
template <typename scalar_t>
void pack_vnni(scalar_t* __restrict__ packed, const scalar_t* __restrict__ weight, int64_t N, int64_t K) {
  constexpr int VNNI_BLK = 2;  // bf16/fp16 uses 2
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}

// Float doesn't need VNNI packing, just transpose
template <>
void pack_vnni<float>(float* __restrict__ packed, const float* __restrict__ weight, int64_t N, int64_t K) {
  // For float, brgemm expects column-major B, so we transpose
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      packed[k * N + n] = weight[n * K + k];
    }
  }
}

// Convert weight tensor to packed format
// weight: [E, OC, IC] -> packed: [E, OC, IC] in VNNI format
template <typename scalar_t>
void convert_weight_packed_impl(
    scalar_t* __restrict__ packed,
    const scalar_t* __restrict__ weight,
    int64_t E, int64_t OC, int64_t IC) {
  
  const int64_t NB = div_up(OC, BLOCK_N);
  const int64_t stride = OC * IC;
  
  // Parallel on {E, NB}
  at::parallel_for(0, E * NB, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t e = i / NB;
      int64_t nb = i % NB;
      
      int64_t n = nb * BLOCK_N;
      int64_t n_size = std::min(BLOCK_N, OC - n);
      
      pack_vnni<scalar_t>(
          packed + e * stride + n * IC,
          weight + e * stride + n * IC,
          n_size, IC);
    }
  });
}

// ======================== Helper Stubs ========================

// Vectorized copy
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d <= size - Vec::size(); d += Vec::size()) {
    Vec::loadu(input + d).store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = input[d];
  }
}

// Copy with weight multiplication (float -> scalar_t)
template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, float weight, int64_t size) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  const fVec weight_vec(weight);
  
  if constexpr (std::is_same_v<scalar_t, float>) {
    int64_t d = 0;
    for (; d <= size - fVec::size(); d += fVec::size()) {
      fVec data = fVec::loadu(input + d) * weight_vec;
      data.store(out + d);
    }
    for (; d < size; ++d) {
      out[d] = input[d] * weight;
    }
  } else {
    constexpr int kVecSize = bVec::size();
    int64_t d = 0;
    for (; d <= size - kVecSize; d += kVecSize) {
      fVec data0 = fVec::loadu(input + d) * weight_vec;
      fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
      bVec out_vec = at::vec::convert_from_float<scalar_t>(data0, data1);
      out_vec.store(out + d);
    }
    for (; d < size; ++d) {
      out[d] = static_cast<scalar_t>(input[d] * weight);
    }
  }
}

// Sum from [topk, K] to [K]
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  
  if (topk == 1) {
    copy_stub(out, input, K);
    return;
  }
  
  if constexpr (std::is_same_v<scalar_t, float>) {
    int64_t d = 0;
    for (; d <= K - fVec::size(); d += fVec::size()) {
      fVec sum_vec = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        sum_vec = sum_vec + fVec::loadu(input + t * K + d);
      }
      sum_vec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += input[t * K + d];
      }
      out[d] = sum_val;
    }
  } else {
    constexpr int kVecSize = bVec::size();
    int64_t d = 0;
    for (; d <= K - kVecSize; d += kVecSize) {
      fVec sum0 = fVec(0.f);
      fVec sum1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x = bVec::loadu(input + t * K + d);
        fVec x0, x1;
        std::tie(x0, x1) = at::vec::convert_to_float<scalar_t>(x);
        sum0 = sum0 + x0;
        sum1 = sum1 + x1;
      }
      bVec out_vec = at::vec::convert_from_float<scalar_t>(sum0, sum1);
      out_vec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

// Add bias (in float)
template <typename scalar_t>
inline void add_bias_stub(float* __restrict__ data, const scalar_t* __restrict__ bias, int64_t size) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  
  if constexpr (std::is_same_v<scalar_t, float>) {
    int64_t d = 0;
    for (; d <= size - fVec::size(); d += fVec::size()) {
      fVec x = fVec::loadu(data + d);
      fVec b = fVec::loadu(bias + d);
      (x + b).store(data + d);
    }
    for (; d < size; ++d) {
      data[d] += bias[d];
    }
  } else {
    constexpr int kVecSize = bVec::size();
    int64_t d = 0;
    for (; d <= size - kVecSize; d += kVecSize) {
      fVec x0 = fVec::loadu(data + d);
      fVec x1 = fVec::loadu(data + d + fVec::size());
      bVec b_bvec = bVec::loadu(bias + d);
      fVec b0, b1;
      std::tie(b0, b1) = at::vec::convert_to_float<scalar_t>(b_bvec);
      (x0 + b0).store(data + d);
      (x1 + b1).store(data + d + fVec::size());
    }
    for (; d < size; ++d) {
      data[d] += static_cast<float>(bias[d]);
    }
  }
}

// ======================== Activation Functions ========================

// SiLU and mul: out = silu(gate) * up
// Input C0, C1 are in float with stride BLOCK_N
// Output is in scalar_t with stride N
template <typename scalar_t>
inline void silu_and_mul(
    scalar_t* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int64_t m_size,
    int64_t n_size,
    int64_t ldc) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  const fVec one(1.0f);
  
  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* out = output + m * ldc;
    const float* g = gate + m * BLOCK_N;
    const float* u = up + m * BLOCK_N;
    
    if constexpr (std::is_same_v<scalar_t, float>) {
      int64_t d = 0;
      for (; d <= n_size - fVec::size(); d += fVec::size()) {
        fVec gv = fVec::loadu(g + d);
        fVec uv = fVec::loadu(u + d);
        fVec silu = gv / (one + gv.neg().exp());
        (silu * uv).store(out + d);
      }
      for (; d < n_size; ++d) {
        float gval = g[d];
        float silu = gval / (1.0f + std::exp(-gval));
        out[d] = silu * u[d];
      }
    } else {
      constexpr int kVecSize = bVec::size();
      int64_t d = 0;
      for (; d <= n_size - kVecSize; d += kVecSize) {
        fVec g0 = fVec::loadu(g + d);
        fVec g1 = fVec::loadu(g + d + fVec::size());
        fVec u0 = fVec::loadu(u + d);
        fVec u1 = fVec::loadu(u + d + fVec::size());
        
        fVec silu0 = g0 / (one + g0.neg().exp());
        fVec silu1 = g1 / (one + g1.neg().exp());
        
        bVec out_vec = at::vec::convert_from_float<scalar_t>(silu0 * u0, silu1 * u1);
        out_vec.store(out + d);
      }
      for (; d < n_size; ++d) {
        float gval = g[d];
        float silu = gval / (1.0f + std::exp(-gval));
        out[d] = static_cast<scalar_t>(silu * u[d]);
      }
    }
  }
}

// SwigluOAI activation for GptOss models
template <typename scalar_t>
inline void swigluoai_and_mul(
    scalar_t* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int64_t m_size,
    int64_t n_size,
    int64_t ldc,
    float alpha,
    float limit) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  const fVec one(1.0f);
  const fVec limit_v(limit);
  const fVec nlimit_v(-limit);
  const fVec alpha_v(alpha);
  
  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* out = output + m * ldc;
    const float* g = gate + m * BLOCK_N;
    const float* u = up + m * BLOCK_N;
    
    if constexpr (std::is_same_v<scalar_t, float>) {
      int64_t d = 0;
      for (; d <= n_size - fVec::size(); d += fVec::size()) {
        fVec gv = fVec::loadu(g + d);
        fVec uv = fVec::loadu(u + d);
        gv = at::vec::minimum(gv, limit_v);
        uv = at::vec::minimum(at::vec::maximum(uv, nlimit_v), limit_v);
        fVec glu = gv / (one + (gv * alpha_v).neg().exp());
        ((uv + one) * glu).store(out + d);
      }
      for (; d < n_size; ++d) {
        float gval = std::min(g[d], limit);
        float uval = std::clamp(u[d], -limit, limit);
        float glu = gval / (1.0f + std::exp(-gval * alpha));
        out[d] = (uval + 1.0f) * glu;
      }
    } else {
      constexpr int kVecSize = bVec::size();
      int64_t d = 0;
      for (; d <= n_size - kVecSize; d += kVecSize) {
        fVec g0 = fVec::loadu(g + d);
        fVec g1 = fVec::loadu(g + d + fVec::size());
        fVec u0 = fVec::loadu(u + d);
        fVec u1 = fVec::loadu(u + d + fVec::size());
        
        g0 = at::vec::minimum(g0, limit_v);
        g1 = at::vec::minimum(g1, limit_v);
        u0 = at::vec::minimum(at::vec::maximum(u0, nlimit_v), limit_v);
        u1 = at::vec::minimum(at::vec::maximum(u1, nlimit_v), limit_v);
        
        fVec glu0 = g0 / (one + (g0 * alpha_v).neg().exp());
        fVec glu1 = g1 / (one + (g1 * alpha_v).neg().exp());
        
        bVec out_vec = at::vec::convert_from_float<scalar_t>((u0 + one) * glu0, (u1 + one) * glu1);
        out_vec.store(out + d);
      }
      for (; d < n_size; ++d) {
        float gval = std::min(g[d], limit);
        float uval = std::clamp(u[d], -limit, limit);
        float glu = gval / (1.0f + std::exp(-gval * alpha));
        out[d] = static_cast<scalar_t>((uval + 1.0f) * glu);
      }
    }
  }
}

// ======================== Token Sorting ========================

// Align block size: sort tokens by expert, compute offsets
// Returns num_tokens_post_pad (padded to BLOCK_M)
int moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int64_t num_experts,
    int64_t numel) {
  
  // Count tokens per expert
  std::vector<int32_t> expert_counts(num_experts, 0);
  for (int64_t i = 0; i < numel; ++i) {
    expert_counts[topk_ids[i]]++;
  }
  
  // Compute cumulative sums (padded to BLOCK_M)
  cumsums[0] = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    cumsums[e + 1] = cumsums[e] + div_up(expert_counts[e], BLOCK_M) * BLOCK_M;
    // Fill expert_ids for each block
    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
    }
  }
  int num_tokens_post_pad = cumsums[num_experts];
  
  // Sort tokens by expert
  std::vector<int32_t> write_offsets(num_experts, 0);
  for (int64_t i = 0; i < numel; ++i) {
    int32_t expert_id = topk_ids[i];
    int32_t pos = cumsums[expert_id] + write_offsets[expert_id];
    sorted_ids[pos] = i;
    write_offsets[expert_id]++;
  }
  
  // Fill padding with numel (invalid token)
  for (int64_t e = 0; e < num_experts; ++e) {
    int32_t start = cumsums[e] + expert_counts[e];
    int32_t end = cumsums[e + 1];
    for (int32_t i = start; i < end; ++i) {
      sorted_ids[i] = numel;
    }
  }
  
  // Compute offsets: cumulative count of valid tokens
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  offsets[0] = 0;
  for (int mb = 0; mb < num_token_blocks; ++mb) {
    int count = 0;
    for (int d = 0; d < BLOCK_M; ++d) {
      if (sorted_ids[mb * BLOCK_M + d] != numel) {
        count++;
      }
    }
    offsets[mb + 1] = offsets[mb] + count;
  }
  
  return num_tokens_post_pad;
}

} // namespace (anonymous)

// ======================== External API: Weight Packing ========================

torch::Tensor convert_weight_packed(torch::Tensor weight) {
  TORCH_CHECK(weight.is_cpu(), "weight must be CPU tensor");
  TORCH_CHECK(weight.dim() == 3, "weight must be 3D tensor [E, OC, IC]");
  
  auto weight_c = weight.contiguous();
  auto output = torch::empty_like(weight_c);
  
  const int64_t E = weight_c.size(0);
  const int64_t OC = weight_c.size(1);
  const int64_t IC = weight_c.size(2);
  
  AT_DISPATCH_SWITCH(weight_c.scalar_type(), "convert_weight_packed",
    AT_DISPATCH_CASE(at::kFloat, [&] {
      convert_weight_packed_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          weight_c.data_ptr<scalar_t>(),
          E, OC, IC);
    })
    AT_DISPATCH_CASE(at::kBFloat16, [&] {
      convert_weight_packed_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          weight_c.data_ptr<scalar_t>(),
          E, OC, IC);
    })
    AT_DISPATCH_CASE(at::kHalf, [&] {
      convert_weight_packed_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          weight_c.data_ptr<scalar_t>(),
          E, OC, IC);
    })
  );
  
  return output;
}

namespace {

// ======================== Main Computation ========================

template <typename scalar_t>
void fused_moe_impl(
    scalar_t* __restrict__ out_ptr,
    const scalar_t* __restrict__ h_ptr,
    const scalar_t* __restrict__ w1_ptr,
    const scalar_t* __restrict__ w2_ptr,
    const float* __restrict__ wt_ptr,
    const int32_t* __restrict__ id_ptr,
    const scalar_t* __restrict__ w1_bias_ptr,
    const scalar_t* __restrict__ w2_bias_ptr,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t inter_size,
    int64_t topk,
    bool is_vnni,
    bool use_swiglu,
    float alpha,
    float limit) {
  
  const int64_t M = num_tokens;
  const int64_t K = hidden_size;
  const int64_t N = inter_size;
  const int64_t E = num_experts;
  const int64_t numel = M * topk;
  
  // ======================== Step 1: Weight preparation ========================
  // If is_vnni=true, weights are already packed; otherwise pack them now
  std::vector<scalar_t> packed_w1_buf;
  std::vector<scalar_t> packed_w2_buf;
  const scalar_t* packed_w1;
  const scalar_t* packed_w2;
  
  if (is_vnni) {
    // Weights already in VNNI format
    packed_w1 = w1_ptr;
    packed_w2 = w2_ptr;
  } else {
    // Pack weights on-the-fly (slower, but works for non-packed weights)
    packed_w1_buf.resize(E * 2 * N * K);
    packed_w2_buf.resize(E * K * N);
    convert_weight_packed_impl<scalar_t>(packed_w1_buf.data(), w1_ptr, E, 2 * N, K);
    convert_weight_packed_impl<scalar_t>(packed_w2_buf.data(), w2_ptr, E, K, N);
    packed_w1 = packed_w1_buf.data();
    packed_w2 = packed_w2_buf.data();
  }
  
  // ======================== Step 2: Token sorting ========================
  std::vector<int32_t> sorted_ids(numel + E * BLOCK_M);
  std::vector<int32_t> expert_ids(div_up(numel, BLOCK_M) + E);
  std::vector<int32_t> cumsums(E + 1);
  std::vector<int32_t> offsets(div_up(numel, BLOCK_M) + E + 1);
  
  int num_tokens_post_pad = moe_align_block_size(
      sorted_ids.data(),
      expert_ids.data(),
      id_ptr,
      cumsums.data(),
      offsets.data(),
      E, numel);
  
  const int64_t MB = num_tokens_post_pad / BLOCK_M;
  const int64_t NB = div_up(N, BLOCK_N);
  
  // ======================== Step 3: Allocate buffers ========================
  // ic1: [numel, N] - GEMM1 + activation output (sorted order)
  // ic2: [numel, K] - GEMM2 output (original order, weighted)
  std::vector<scalar_t> ic1(numel * N);
  std::vector<scalar_t> ic2(numel * K);
  
  int num_threads = at::get_num_threads();
  std::vector<scalar_t> A_tmp(num_threads * BLOCK_M * K);
  std::vector<float> C_tmp(num_threads * 2 * BLOCK_M * BLOCK_N);
  
  int64_t avg_M = std::max(int64_t(1), numel / E);
  const bool use_brgemm = can_use_brgemm<scalar_t>(avg_M);
  const bool with_bias = (w1_bias_ptr != nullptr);
  
  // Strides for packed w1: [E, 2N, K] in VNNI format
  const int64_t stride_e1 = 2 * N * K;
  
  // ======================== Stage 1: GEMM1 + Activation ========================
  at::parallel_for(0, MB * NB, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    scalar_t* A = A_tmp.data() + tid * BLOCK_M * K;
    float* C0 = C_tmp.data() + tid * 2 * BLOCK_M * BLOCK_N;
    float* C1 = C0 + BLOCK_M * BLOCK_N;
    
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t mb = idx / NB;
      int64_t nb = idx % NB;
      
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      if (m_size == 0) continue;
      
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);
      int32_t expert_id = expert_ids[mb];
      const int32_t* A_ids = sorted_ids.data() + mb * BLOCK_M;
      
      // Load A: gather tokens (only for first NB block to avoid redundant copies)
      if (nb == 0) {
        for (int64_t m = 0; m < m_size; ++m) {
          int32_t index = A_ids[m] / topk;
          copy_stub(A + m * K, h_ptr + index * K, K);
        }
      }
      
      // B pointers: gate from [0, N), up from [N, 2N)
      const scalar_t* B0 = packed_w1 + expert_id * stride_e1 + nb * BLOCK_N * K;
      const scalar_t* B1 = packed_w1 + expert_id * stride_e1 + (NB + nb) * BLOCK_N * K;
      
      if (use_brgemm) {
        at::native::cpublas::brgemm(
            m_size, n_size, K,
            K, n_size, BLOCK_N,
            false, A, B0, C0);
        
        at::native::cpublas::brgemm(
            m_size, n_size, K,
            K, n_size, BLOCK_N,
            false, A, B1, C1);
      } else {
        // Fallback: naive GEMM
        std::fill(C0, C0 + m_size * BLOCK_N, 0.0f);
        std::fill(C1, C1 + m_size * BLOCK_N, 0.0f);
        
        for (int64_t m = 0; m < m_size; ++m) {
          for (int64_t k = 0; k < K; ++k) {
            float a_val = static_cast<float>(A[m * K + k]);
            for (int64_t n = 0; n < n_size; ++n) {
              // Note: B is in VNNI/transposed format
              C0[m * BLOCK_N + n] += a_val * static_cast<float>(B0[k * n_size + n]);
              C1[m * BLOCK_N + n] += a_val * static_cast<float>(B1[k * n_size + n]);
            }
          }
        }
      }
      
      // Add bias
      if (with_bias) {
        const scalar_t* bias0 = w1_bias_ptr + expert_id * 2 * N + nb * BLOCK_N;
        const scalar_t* bias1 = w1_bias_ptr + expert_id * 2 * N + N + nb * BLOCK_N;
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C0 + m * BLOCK_N, bias0, n_size);
          add_bias_stub(C1 + m * BLOCK_N, bias1, n_size);
        }
      }
      
      // Activation: silu(C0) * C1 -> ic1
      int64_t offset = offsets[mb];
      if (use_swiglu) {
        swigluoai_and_mul<scalar_t>(
            ic1.data() + offset * N + nb * BLOCK_N,
            C0, C1, m_size, n_size, N, alpha, limit);
      } else {
        silu_and_mul<scalar_t>(
            ic1.data() + offset * N + nb * BLOCK_N,
            C0, C1, m_size, n_size, N);
      }
    }
    
    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
  
  // ======================== Stage 2: GEMM2 ========================
  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t stride_e2 = OC * IC;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const bool with_bias2 = (w2_bias_ptr != nullptr);
  
  at::parallel_for(0, MB * NB2, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    float* C = C_tmp.data() + tid * 2 * BLOCK_M * BLOCK_N;
    
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t mb = idx / NB2;
      int64_t nb = idx % NB2;
      
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      if (m_size == 0) continue;
      
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);
      int32_t expert_id = expert_ids[mb];
      const int32_t* A_ids = sorted_ids.data() + mb * BLOCK_M;
      
      // A is in ic1, already sorted
      const scalar_t* A = ic1.data() + offsets[mb] * N;
      const scalar_t* B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * IC;
      
      if (use_brgemm) {
        at::native::cpublas::brgemm(
            m_size, n_size, IC,
            IC, n_size, BLOCK_N,
            false, A, B, C);
      } else {
        std::fill(C, C + m_size * BLOCK_N, 0.0f);
        for (int64_t m = 0; m < m_size; ++m) {
          for (int64_t k = 0; k < IC; ++k) {
            float a_val = static_cast<float>(A[m * IC + k]);
            for (int64_t n = 0; n < n_size; ++n) {
              C[m * BLOCK_N + n] += a_val * static_cast<float>(B[k * n_size + n]);
            }
          }
        }
      }
      
      // Add bias
      if (with_bias2) {
        const scalar_t* B_bias = w2_bias_ptr + expert_id * OC + nb * BLOCK_N;
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C + m * BLOCK_N, B_bias, n_size);
        }
      }
      
      // Scatter with weight to ic2
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = wt_ptr[index];
        copy_mul_stub(ic2.data() + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    }
    
    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
  
  // ======================== Stage 3: Sum reduction ========================
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(out_ptr + m * K, ic2.data() + m * topk * K, topk, K);
    }
  });
}

} // namespace

// ======================== Entry Point ========================

torch::Tensor fused_moe_cpu(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    const c10::optional<torch::Tensor>& w1_bias,
    const c10::optional<torch::Tensor>& w2_bias,
    bool is_vnni,
    const std::string& activation,
    float alpha,
    float limit) {
  
  TORCH_CHECK(hidden_states.is_cpu(), "hidden_states must be CPU");
  TORCH_CHECK(w1.is_cpu() && w2.is_cpu(), "weights must be CPU");
  
  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t num_experts = w1.size(0);
  const int64_t inter_size = w2.size(1);
  const int64_t topk = topk_weights.size(1);
  
  const bool use_swiglu = (activation == "swigluoai");
  
  TORCH_CHECK(hidden_states.scalar_type() == at::kFloat ||
              hidden_states.scalar_type() == at::kBFloat16 ||
              hidden_states.scalar_type() == at::kHalf,
              "fused_moe_cpu only supports float32, bfloat16, and float16");
  
  // Ensure contiguous
  auto h = hidden_states.contiguous();
  auto w1_c = w1.contiguous();
  auto w2_c = w2.contiguous();
  auto tw = topk_weights.contiguous().to(at::kFloat);
  auto ti = topk_ids.contiguous().to(at::kInt);
  
  auto output = torch::zeros_like(h);
  
  AT_DISPATCH_SWITCH(h.scalar_type(), "fused_moe_cpu",
    AT_DISPATCH_CASE(at::kFloat, [&] {
      fused_moe_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          h.data_ptr<scalar_t>(),
          w1_c.data_ptr<scalar_t>(),
          w2_c.data_ptr<scalar_t>(),
          tw.data_ptr<float>(),
          ti.data_ptr<int32_t>(),
          w1_bias.has_value() ? w1_bias.value().data_ptr<scalar_t>() : nullptr,
          w2_bias.has_value() ? w2_bias.value().data_ptr<scalar_t>() : nullptr,
          num_tokens, hidden_size, num_experts, inter_size, topk,
          is_vnni, use_swiglu, alpha, limit);
    })
    AT_DISPATCH_CASE(at::kBFloat16, [&] {
      fused_moe_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          h.data_ptr<scalar_t>(),
          w1_c.data_ptr<scalar_t>(),
          w2_c.data_ptr<scalar_t>(),
          tw.data_ptr<float>(),
          ti.data_ptr<int32_t>(),
          w1_bias.has_value() ? w1_bias.value().data_ptr<scalar_t>() : nullptr,
          w2_bias.has_value() ? w2_bias.value().data_ptr<scalar_t>() : nullptr,
          num_tokens, hidden_size, num_experts, inter_size, topk,
          is_vnni, use_swiglu, alpha, limit);
    })
    AT_DISPATCH_CASE(at::kHalf, [&] {
      fused_moe_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          h.data_ptr<scalar_t>(),
          w1_c.data_ptr<scalar_t>(),
          w2_c.data_ptr<scalar_t>(),
          tw.data_ptr<float>(),
          ti.data_ptr<int32_t>(),
          w1_bias.has_value() ? w1_bias.value().data_ptr<scalar_t>() : nullptr,
          w2_bias.has_value() ? w2_bias.value().data_ptr<scalar_t>() : nullptr,
          num_tokens, hidden_size, num_experts, inter_size, topk,
          is_vnni, use_swiglu, alpha, limit);
    })
  );
  
  return output;
}

} // namespace cpu
} // namespace megablocks
