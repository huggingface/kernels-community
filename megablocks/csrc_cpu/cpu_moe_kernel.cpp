// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU Fused MoE Implementation
//
// Strictly follows sglang implementation:
//   1. parallel_2d + loop_2d for cache-friendly 2D parallelism
//   2. Unroll template for compile-time unrolling
//   3. exp_u20() for fast exp
//   4. convert_from_float_ext with AVX512 intrinsics
//   5. tinygemm_kernel_nn2 with _mm512_dpbf16_ps
//   6. moe_align_block_size with thread-local counting

#define CPU_CAPABILITY_AVX512
#include "moe_ops.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX512F__) && defined(__AVX512BF16__) && defined(__AMX_BF16__)
#define CPU_CAPABILITY_AVX512
#endif

namespace megablocks {
namespace cpu {

namespace {

// ======================== Constants ========================
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 32;

template <typename T>
constexpr int64_t block_size_m() { return 2 * TILE_M; }  // 32
template <typename T>
constexpr int64_t block_size_n() { return 2 * TILE_N; }  // 32

constexpr int64_t GRAIN_SIZE = 1024;

// ======================== Utility Functions ========================
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

// Determine if brgemm should be used
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

// ======================== Unroll Template ========================
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

// ======================== Convert Functions ========================
template <typename scalar_t, typename std::enable_if_t<at::vec::is_reduced_floating_point_v<scalar_t>, int> = 0>
inline at::vec::Vectorized<scalar_t> convert_from_float_ext(
    const at::vec::Vectorized<float>& a, 
    const at::vec::Vectorized<float>& b) {
  return at::vec::convert_from_float<scalar_t>(a, b);
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline at::vec::Vectorized<at::BFloat16> convert_from_float_ext<at::BFloat16>(
    const at::vec::Vectorized<float>& a, 
    const at::vec::Vectorized<float>& b) {
  return (__m512i)(_mm512_cvtne2ps_pbh(__m512(b), __m512(a)));
}
#endif

// ======================== Parallel Utilities ========================
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

template <typename T>
inline int get_cache_blocks(int chunk_size) {
  const int L2_size = 2048 * 1024 >> 1;
  return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
}

template <typename T, typename func_t>
inline void loop_2d(int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1, 
                    int64_t chunk_size, const func_t& f) {
  int64_t cache_blocks_nb = get_cache_blocks<T>(chunk_size);
  for (int64_t nbb = nb0; nbb < nb1; nbb += cache_blocks_nb) {
    for (int64_t mb = mb0; mb < mb1; ++mb) {
      for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, nb1); ++nb) {
        f(mb, nb, nb - nbb);
      }
    }
  }
}

// ======================== Vectorized Stubs ========================
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, 
                          float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * weight_vec;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, 
                     int64_t topk, int64_t K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  
  if (topk == 1) {
    copy_stub(out, input, K);
    return;
  }
  
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= K - kVecSize; d += kVecSize) {
    fVec sum_fvec0 = fVec(0.f);
    fVec sum_fvec1 = fVec(0.f);
    for (int t = 0; t < topk; ++t) {
      bVec x_bvec = bVec::loadu(input + t * K + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);
      sum_fvec0 = sum_fvec0 + x_fvec0;
      sum_fvec1 = sum_fvec1 + x_fvec1;
    }
    bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < K; ++d) {
    float sum_val = 0.f;
    for (int t = 0; t < topk; ++t) {
      sum_val += static_cast<float>(input[t * K + d]);
    }
    out[d] = static_cast<scalar_t>(sum_val);
  }
}

template <typename scalar_t>
inline void add_bias_stub(float* __restrict__ input, const scalar_t* __restrict__ input2, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec x0 = fVec::loadu(input + d);
    fVec x1 = fVec::loadu(input + d + fVec::size());
    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);
    x0 = x0 + y0;
    x1 = x1 + y1;
    x0.store(input + d);
    x1.store(input + d + fVec::size());
  }
  for (; d < size; ++d) {
    input[d] = input[d] + float(input2[d]);
  }
}

// ======================== Activation Functions ========================
// silu_and_mul using exp_u20() for speed
template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul(
    scalar_t* __restrict__ output,
    const float* __restrict__ input0,
    const float* __restrict__ input1,
    int64_t m_size,
    int64_t N) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);

  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out = output + m * N;
    const float* __restrict__ x = input0 + m * BLOCK_N;
    const float* __restrict__ y = input1 + m * BLOCK_N;

    for (int64_t d = 0; d < BLOCK_N; d += bVec::size()) {
      fVec x0 = fVec::loadu(x + d);
      fVec x1 = fVec::loadu(x + d + fVec::size());
      fVec y0 = fVec::loadu(y + d);
      fVec y1 = fVec::loadu(y + d + fVec::size());
      // silu with exp_u20()
      x0 = x0 / (one + x0.neg().exp_u20());
      x1 = x1 / (one + x1.neg().exp_u20());
      // mul
      x0 = x0 * y0;
      x1 = x1 * y1;
      // convert
      bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
      out_vec.store(out + d);
    }
  }
}

// clamp_sigmoid_and_mul for swiglu (gpt-oss style)
template <typename scalar_t, int BLOCK_N>
inline void clamp_sigmoid_and_mul(
    scalar_t* __restrict__ output,
    const float* __restrict__ input0,
    int64_t m_size,
    int64_t N,
    float alpha,
    float limit,
    int64_t offset) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  const fVec limit_v = fVec(limit);
  const fVec nlimit_v = fVec(-limit);
  const fVec alpha_v = fVec(alpha);

  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out = output + m * N;
    const float* __restrict__ cur_ptr = input0 + m * BLOCK_N;
    for (int64_t d = 0; d < BLOCK_N; d += bVec::size()) {
      float tmp_glu0[fVec::size()];
      float tmp_linear0[fVec::size()];
      for (int j = 0; j < fVec::size(); ++j) {
        tmp_glu0[j] = cur_ptr[d + j * 2];
        tmp_linear0[j] = cur_ptr[d + j * 2 + 1];
      }
      fVec x0 = fVec::loadu(tmp_glu0);
      fVec y0 = fVec::loadu(tmp_linear0);
      x0 = at::vec::minimum(x0, limit_v);
      y0 = at::vec::minimum(limit_v, at::vec::maximum(nlimit_v, y0));
      x0 = x0 / (one + (x0 * alpha_v).neg().exp_u20());
      y0 = y0 + one;
      x0 = x0 * y0;
      bVec out_vec = convert_from_float_ext<scalar_t>(x0, fVec(0.f));
      // Store only first half since we deinterleaved
      at::vec::Vectorized<scalar_t>::loadu(out + d / 2 + offset); // placeholder
      // Actually store half
      for (int j = 0; j < fVec::size(); ++j) {
        out[d / 2 + offset + j] = static_cast<scalar_t>(x0[j]);
      }
    }
  }
}

// ======================== VNNI Weight Packing ========================
template <typename scalar_t>
void pack_vnni(scalar_t* __restrict__ packed, const scalar_t* __restrict__ weight, 
               int64_t N, int64_t K) {
  constexpr int VNNI_BLK = 2;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}

template <>
void pack_vnni<float>(float* __restrict__ packed, const float* __restrict__ weight, 
                      int64_t N, int64_t K) {
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      packed[k * N + n] = weight[n * K + k];
    }
  }
}

template <typename scalar_t>
void convert_weight_packed_impl(
    scalar_t* __restrict__ packed,
    const scalar_t* __restrict__ weight,
    int64_t E, int64_t OC, int64_t IC) {
  
  constexpr int64_t BLOCK_N = block_size_n<scalar_t>();
  const int64_t NB = div_up(OC, BLOCK_N);
  const int64_t stride = OC * IC;
  
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

// ======================== TinyGEMM Kernels ========================
#if defined(CPU_CAPABILITY_AVX512)

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2 {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B0,
      const scalar_t* __restrict__ B1,
      scalar_t* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn2: scalar path not implemented!");
  }
};

template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B0,
      const at::BFloat16* __restrict__ B1,
      at::BFloat16* __restrict__ C,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    __m512bh va;
    __m512bh vb0[COLS];
    __m512bh vb1[COLS];
    __m512 vc0[ROWS * COLS];
    __m512 vc1[ROWS * COLS];

    auto loadc = [&](auto i) {
      vc0[i] = _mm512_set1_ps(0.f);
      vc1[i] = _mm512_set1_ps(0.f);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b0_ptr = reinterpret_cast<const float*>(B0);
    const float* b1_ptr = reinterpret_cast<const float*>(B1);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb0[col] = (__m512bh)(_mm512_loadu_si512(b0_ptr + k * ldb2 + col * 16));
        vb1[col] = (__m512bh)(_mm512_loadu_si512(b1_ptr + k * ldb2 + col * 16));
      }
      vc0[i] = _mm512_dpbf16_ps(vc0[i], va, vb0[col]);
      vc1[i] = _mm512_dpbf16_ps(vc1[i], va, vb1[col]);
    };
    for (int64_t k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    using Vec = at::vec::Vectorized<float>;
    const Vec one = Vec(1.f);
    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      if constexpr (col % 2 == 0) {
        Vec x0 = vc0[row * COLS + col + 0];
        Vec x1 = vc0[row * COLS + col + 1];
        Vec y0 = vc1[row * COLS + col + 0];
        Vec y1 = vc1[row * COLS + col + 1];
        // silu with exp_u20
        x0 = x0 / (one + x0.neg().exp_u20());
        x1 = x1 / (one + x1.neg().exp_u20());
        // mul
        x0 = x0 * y0;
        x1 = x1 * y1;

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(__m512(x1), __m512(x0))));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};

#endif // CPU_CAPABILITY_AVX512

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE) \
  tinygemm_kernel_nn2<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B0 + nb_start * 2, B1 + nb_start * 2, \
      C + mb_start * ldc + nb_start, K, lda, ldb, ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B0,
    const scalar_t* __restrict__ B1,
    scalar_t* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  constexpr int64_t BLOCK_M_TINY = 4;
  constexpr int64_t BLOCK_N_TINY = 32;
  const int64_t MB = div_up(M, BLOCK_M_TINY);
  const int64_t NB = div_up(N, BLOCK_N_TINY);
  
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M_TINY;
    int64_t mb_size = std::min(BLOCK_M_TINY, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N_TINY;
      int64_t nb_size = std::min(BLOCK_N_TINY, N - nb_start);

#if defined(CPU_CAPABILITY_AVX512)
      switch (mb_size << 4 | nb_size >> 4) {
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        default:
          TORCH_CHECK(false, "Unexpected block size");
      }
#else
      TORCH_CHECK(false, "tinygemm requires AVX512");
#endif
    }
  }
}

// ======================== Token Sorting ========================
#define T_INDEX(tt) total_cnts + (tt) * num_experts

template <int BLOCK_M>
int moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
  
  // Accumulate count of expert ids locally (thread-parallel)
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = T_INDEX(tid + 1);
    for (int i = begin; i < end; ++i) {
      local_cnts[topk_ids[i]]++;
    }
  });

  // Prefix sum across threads
  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; }, 
        T_INDEX(t + 1), T_INDEX(t + 1), T_INDEX(t), num_experts);
  }

  int32_t* total_cnts_t_1 = T_INDEX(num_threads);

  cumsums[0] = 0;
  for (int e = 0; e < num_experts; ++e) {
    cumsums[e + 1] = cumsums[e] + div_up(total_cnts_t_1[e], BLOCK_M) * BLOCK_M;
    for (int k = cumsums[e]; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
    }
  }
  int num_tokens_post_pad = cumsums[num_experts];

  // Sort tokens by expert (thread-parallel)
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_offsets = T_INDEX(tid);
    for (int i = begin; i < end; ++i) {
      int32_t expert_id = topk_ids[i];
      int32_t b_offset = cumsums[expert_id];
      int32_t t_offset = local_offsets[expert_id];
      sorted_ids[b_offset + t_offset] = i;
      local_offsets[expert_id]++;
    }
  });

  // Padding value for sorted_ids: numel
  auto sorted_id_size = [=](const int32_t* sorted_ids_ptr) {
    for (int d = 0; d < BLOCK_M; ++d) {
      if (sorted_ids_ptr[d] == numel) {
        return d;
      }
    }
    return BLOCK_M;
  };

  // Fill padding
  for (int e = 0; e < num_experts; ++e) {
    int start = cumsums[e] + total_cnts_t_1[e];
    int end = cumsums[e + 1];
    for (int i = start; i < end; ++i) {
      sorted_ids[i] = numel;
    }
  }

  // Compute offsets
  offsets[0] = 0;
  const int num_token_blocks = num_tokens_post_pad / BLOCK_M;
  at::parallel_for(0, num_token_blocks, GRAIN_SIZE / BLOCK_M, [&](int begin, int end) {
    for (int mb = begin; mb < end; ++mb) {
      offsets[mb + 1] = sorted_id_size(sorted_ids + mb * BLOCK_M);
    }
  });
  
  // Prefix sum for offsets
  for (int mb = 0; mb < num_token_blocks; ++mb) {
    offsets[mb + 1] += offsets[mb];
  }

  return num_tokens_post_pad;
}

#undef T_INDEX

// ======================== Fused Experts Kernel ========================
template <typename scalar_t>
void fused_experts_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ packed_w1,
    const scalar_t* __restrict__ packed_w2,
    const scalar_t* __restrict__ w1_bias,
    const scalar_t* __restrict__ w2_bias,
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
    bool use_swiglu,
    bool with_bias) {
  
  constexpr int64_t BLOCK_M = block_size_m<scalar_t>();
  constexpr int64_t BLOCK_N = block_size_n<scalar_t>();
  
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  
  const int64_t stride_e = 2 * N * K;
  const int64_t stride_n = K;
  
  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  const bool use_brgemm = can_use_brgemm<scalar_t>(avg_M);

  // Stage 1: GEMM1 + Activation
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    scalar_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t nb_upper = nb, nb_lower = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);

      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb_upper * BLOCK_N * stride_n;
      const scalar_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + nb_lower * BLOCK_N * stride_n;
      const scalar_t* __restrict__ B0_bias = with_bias ? w1_bias + expert_id * 2 * N + nb_upper * BLOCK_N : nullptr;
      const scalar_t* __restrict__ B1_bias = with_bias ? w1_bias + expert_id * 2 * N + nb_lower * BLOCK_N : nullptr;

      // Load A
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, input + index * K, K);
      }

      if (use_brgemm) {
        at::native::cpublas::brgemm(m_size, n_size, K, K, n_size, BLOCK_N, false, A, B0, C0);
        at::native::cpublas::brgemm(m_size, n_size, K, K, n_size, BLOCK_N, false, A, B1, C1);
      } else {
        if (!use_swiglu) {
          tinygemm_kernel<scalar_t>(
              A, B0, B1, ic1 + offsets[mb] * N + nb * BLOCK_N,
              m_size, n_size, K, K, n_size, N);
          // Continue to next block since tinygemm fuses silu_and_mul
          if (with_bias) {
            // TODO: add bias before activation in tinygemm
          }
          return;
        }
        // For swiglu, need separate GEMMs
        tinygemm_kernel<scalar_t>(A, B0, B1, ic1 + offsets[mb] * N + nb * BLOCK_N,
                                   m_size, n_size, K, K, n_size, N);
      }

      if (with_bias && use_brgemm) {
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C0 + m * BLOCK_N, B0_bias, n_size);
          add_bias_stub(C1 + m * BLOCK_N, B1_bias, n_size);
        }
      }

      const int64_t offset = offsets[mb];
      if (!use_swiglu && use_brgemm) {
        silu_and_mul<scalar_t, BLOCK_N>(ic1 + offset * N + nb * BLOCK_N, C0, C1, m_size, N);
      } else if (use_swiglu) {
        clamp_sigmoid_and_mul<scalar_t, BLOCK_N>(ic1 + offset * N, C0, m_size, N, alpha, limit, nb * BLOCK_N / 2);
        clamp_sigmoid_and_mul<scalar_t, BLOCK_N>(ic1 + offset * N, C1, m_size, N, alpha, limit, N / 2 + nb * BLOCK_N / 2);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // Stage 2: GEMM2
  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * IC;
  const int64_t stride_oc = IC;

  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      int32_t expert_id = expert_ids[mb];
      const scalar_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const scalar_t* __restrict__ B_bias = with_bias ? w2_bias + expert_id * OC + nb * BLOCK_N : nullptr;

      if (use_brgemm) {
        at::native::cpublas::brgemm(m_size, n_size, IC, IC, n_size, BLOCK_N, false, A, B, C);
      } else {
        // Fallback for small M
        for (int64_t m = 0; m < m_size; ++m) {
          for (int64_t n = 0; n < n_size; ++n) {
            float acc = 0.f;
            for (int64_t k = 0; k < IC; ++k) {
              acc += float(A[m * IC + k]) * float(B[k * n_size + n]);
            }
            C[m * BLOCK_N + n] = acc;
          }
        }
      }

      if (with_bias) {
        for (int64_t m = 0; m < m_size; ++m) {
          add_bias_stub(C + m * BLOCK_N, B_bias, n_size);
        }
      }

      // Scatter with weight
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // Stage 3: Sum reduction
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

} // namespace

// ======================== External APIs ========================

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
  
  const int64_t M = hidden_states.size(0);
  const int64_t K = hidden_states.size(1);
  const int64_t E = w1.size(0);
  const int64_t N = w2.size(1);
  const int64_t topk = topk_weights.size(1);
  const int64_t numel = M * topk;
  
  const bool use_swiglu = (activation == "swigluoai");
  const bool with_bias = w1_bias.has_value();
  
  auto h = hidden_states.contiguous();
  auto w1_c = w1.contiguous();
  auto w2_c = w2.contiguous();
  auto tw = topk_weights.contiguous().to(at::kFloat);
  auto ti = topk_ids.contiguous().to(at::kInt);
  
  auto output = torch::zeros_like(h);
  
  AT_DISPATCH_SWITCH(h.scalar_type(), "fused_moe_cpu",
    AT_DISPATCH_CASE(at::kBFloat16, [&] {
      constexpr int64_t BLOCK_M = block_size_m<scalar_t>();
      constexpr int64_t BLOCK_N = block_size_n<scalar_t>();
      
      const int num_threads = at::get_num_threads();
      const int64_t max_blocks = div_up(numel, BLOCK_M) + E;
      
      // Allocate buffers
      std::vector<scalar_t> ic1(numel * N);
      std::vector<scalar_t> ic2(numel * K);
      std::vector<scalar_t> A_tmp(num_threads * BLOCK_M * K);
      std::vector<float> C_tmp(num_threads * 2 * BLOCK_M * BLOCK_N);
      
      // Allocate sorting buffers
      std::vector<int32_t> sorted_ids(numel + E * BLOCK_M, numel);
      std::vector<int32_t> expert_ids(max_blocks);
      std::vector<int32_t> total_cnts((num_threads + 1) * E, 0);
      std::vector<int32_t> cumsums(E + 1);
      std::vector<int32_t> offsets(max_blocks + 1);
      
      // Pack weights if needed
      std::vector<scalar_t> packed_w1_buf;
      std::vector<scalar_t> packed_w2_buf;
      const scalar_t* packed_w1;
      const scalar_t* packed_w2;
      
      if (is_vnni) {
        packed_w1 = w1_c.data_ptr<scalar_t>();
        packed_w2 = w2_c.data_ptr<scalar_t>();
      } else {
        packed_w1_buf.resize(E * 2 * N * K);
        packed_w2_buf.resize(E * K * N);
        convert_weight_packed_impl<scalar_t>(packed_w1_buf.data(), w1_c.data_ptr<scalar_t>(), E, 2 * N, K);
        convert_weight_packed_impl<scalar_t>(packed_w2_buf.data(), w2_c.data_ptr<scalar_t>(), E, K, N);
        packed_w1 = packed_w1_buf.data();
        packed_w2 = packed_w2_buf.data();
      }
      
      // Sort tokens
      int num_tokens_post_pad = moe_align_block_size<BLOCK_M>(
          sorted_ids.data(),
          expert_ids.data(),
          ti.data_ptr<int32_t>(),
          total_cnts.data(),
          cumsums.data(),
          offsets.data(),
          E, numel, num_threads);
      
      // Run kernel
      fused_experts_kernel_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          ic1.data(),
          ic2.data(),
          A_tmp.data(),
          C_tmp.data(),
          h.data_ptr<scalar_t>(),
          packed_w1,
          packed_w2,
          with_bias ? w1_bias.value().data_ptr<scalar_t>() : nullptr,
          with_bias ? w2_bias.value().data_ptr<scalar_t>() : nullptr,
          tw.data_ptr<float>(),
          sorted_ids.data(),
          expert_ids.data(),
          offsets.data(),
          M, N, K, E, topk, num_tokens_post_pad,
          alpha, limit, use_swiglu, with_bias);
    })
    AT_DISPATCH_CASE(at::kFloat, [&] {
      constexpr int64_t BLOCK_M = block_size_m<scalar_t>();
      constexpr int64_t BLOCK_N = block_size_n<scalar_t>();
      
      const int num_threads = at::get_num_threads();
      const int64_t max_blocks = div_up(numel, BLOCK_M) + E;
      
      std::vector<scalar_t> ic1(numel * N);
      std::vector<scalar_t> ic2(numel * K);
      std::vector<scalar_t> A_tmp(num_threads * BLOCK_M * K);
      std::vector<float> C_tmp(num_threads * 2 * BLOCK_M * BLOCK_N);
      
      std::vector<int32_t> sorted_ids(numel + E * BLOCK_M, numel);
      std::vector<int32_t> expert_ids(max_blocks);
      std::vector<int32_t> total_cnts((num_threads + 1) * E, 0);
      std::vector<int32_t> cumsums(E + 1);
      std::vector<int32_t> offsets(max_blocks + 1);
      
      std::vector<scalar_t> packed_w1_buf;
      std::vector<scalar_t> packed_w2_buf;
      const scalar_t* packed_w1;
      const scalar_t* packed_w2;
      
      if (is_vnni) {
        packed_w1 = w1_c.data_ptr<scalar_t>();
        packed_w2 = w2_c.data_ptr<scalar_t>();
      } else {
        packed_w1_buf.resize(E * 2 * N * K);
        packed_w2_buf.resize(E * K * N);
        convert_weight_packed_impl<scalar_t>(packed_w1_buf.data(), w1_c.data_ptr<scalar_t>(), E, 2 * N, K);
        convert_weight_packed_impl<scalar_t>(packed_w2_buf.data(), w2_c.data_ptr<scalar_t>(), E, K, N);
        packed_w1 = packed_w1_buf.data();
        packed_w2 = packed_w2_buf.data();
      }
      
      int num_tokens_post_pad = moe_align_block_size<BLOCK_M>(
          sorted_ids.data(),
          expert_ids.data(),
          ti.data_ptr<int32_t>(),
          total_cnts.data(),
          cumsums.data(),
          offsets.data(),
          E, numel, num_threads);
      
      fused_experts_kernel_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          ic1.data(),
          ic2.data(),
          A_tmp.data(),
          C_tmp.data(),
          h.data_ptr<scalar_t>(),
          packed_w1,
          packed_w2,
          with_bias ? w1_bias.value().data_ptr<scalar_t>() : nullptr,
          with_bias ? w2_bias.value().data_ptr<scalar_t>() : nullptr,
          tw.data_ptr<float>(),
          sorted_ids.data(),
          expert_ids.data(),
          offsets.data(),
          M, N, K, E, topk, num_tokens_post_pad,
          alpha, limit, use_swiglu, with_bias);
    })
    AT_DISPATCH_CASE(at::kHalf, [&] {
      constexpr int64_t BLOCK_M = block_size_m<scalar_t>();
      constexpr int64_t BLOCK_N = block_size_n<scalar_t>();
      
      const int num_threads = at::get_num_threads();
      const int64_t max_blocks = div_up(numel, BLOCK_M) + E;
      
      std::vector<scalar_t> ic1(numel * N);
      std::vector<scalar_t> ic2(numel * K);
      std::vector<scalar_t> A_tmp(num_threads * BLOCK_M * K);
      std::vector<float> C_tmp(num_threads * 2 * BLOCK_M * BLOCK_N);
      
      std::vector<int32_t> sorted_ids(numel + E * BLOCK_M, numel);
      std::vector<int32_t> expert_ids(max_blocks);
      std::vector<int32_t> total_cnts((num_threads + 1) * E, 0);
      std::vector<int32_t> cumsums(E + 1);
      std::vector<int32_t> offsets(max_blocks + 1);
      
      std::vector<scalar_t> packed_w1_buf;
      std::vector<scalar_t> packed_w2_buf;
      const scalar_t* packed_w1;
      const scalar_t* packed_w2;
      
      if (is_vnni) {
        packed_w1 = w1_c.data_ptr<scalar_t>();
        packed_w2 = w2_c.data_ptr<scalar_t>();
      } else {
        packed_w1_buf.resize(E * 2 * N * K);
        packed_w2_buf.resize(E * K * N);
        convert_weight_packed_impl<scalar_t>(packed_w1_buf.data(), w1_c.data_ptr<scalar_t>(), E, 2 * N, K);
        convert_weight_packed_impl<scalar_t>(packed_w2_buf.data(), w2_c.data_ptr<scalar_t>(), E, K, N);
        packed_w1 = packed_w1_buf.data();
        packed_w2 = packed_w2_buf.data();
      }
      
      int num_tokens_post_pad = moe_align_block_size<BLOCK_M>(
          sorted_ids.data(),
          expert_ids.data(),
          ti.data_ptr<int32_t>(),
          total_cnts.data(),
          cumsums.data(),
          offsets.data(),
          E, numel, num_threads);
      
      fused_experts_kernel_impl<scalar_t>(
          output.data_ptr<scalar_t>(),
          ic1.data(),
          ic2.data(),
          A_tmp.data(),
          C_tmp.data(),
          h.data_ptr<scalar_t>(),
          packed_w1,
          packed_w2,
          with_bias ? w1_bias.value().data_ptr<scalar_t>() : nullptr,
          with_bias ? w2_bias.value().data_ptr<scalar_t>() : nullptr,
          tw.data_ptr<float>(),
          sorted_ids.data(),
          expert_ids.data(),
          offsets.data(),
          M, N, K, E, topk, num_tokens_post_pad,
          alpha, limit, use_swiglu, with_bias);
    })
  );
  
  return output;
}

} // namespace cpu
} // namespace megablocks
