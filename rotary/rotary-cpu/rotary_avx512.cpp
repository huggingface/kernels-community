// AVX512 implementation of the rotary embedding op for BF16.
// Compile with -mavx512f -mavx512vl -mavx512dq -mavx512bw -mf16c
// -DCPU_CAPABILITY_AVX512 so that at::vec::Vectorized actually lowers to AVX512
// (otherwise it falls back to the scalar vec_base.h implementation).
#include "rotary_avx512.hpp"

#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <tuple>

using namespace at::vec;

namespace rotary_cpu {
namespace avx512 {

// Per-row strides (in elements) for a 4-D tensor laid out as
// (batch, head, seq, rot_dim). The last dim is required to be contiguous so it
// can be loaded as a vector; the other three dims may have arbitrary strides
// (e.g. a transposed-then-cloned q/k tensor), so row pointers are computed from
// (b, h, s) explicitly rather than assuming uniform spacing.
struct Strides4 {
  int64_t b, h, s;
};

// Process rows [r0, r1). Each "row" is one (batch, head, seq) position holding
// `rot` contiguous BF16 elements. cos/sin are shared across heads, so their
// pointer is derived from (batch, seq) only.
//
// Parallelization is over (batch, seq) positions. cos/sin are shared across the
// H heads of a position, so they are converted to fp32 once per position and
// reused for all heads (cutting cos/sin loads + bf16->fp32 conversions by H).
template <bool Conj>
static inline void apply_rotary_core(
    const at::BFloat16 *x1b, const at::BFloat16 *x2b, const at::BFloat16 *cosb,
    const at::BFloat16 *sinb, at::BFloat16 *o1b, at::BFloat16 *o2b, int64_t p0,
    int64_t p1, int64_t rot, int64_t H, int64_t S, Strides4 sx1, Strides4 sx2,
    Strides4 so1, Strides4 so2, int64_t cb, int64_t cs, int64_t sb, int64_t ss) {
  using Vb = Vectorized<at::BFloat16>;
  using Vf = Vectorized<float>;
  constexpr int kVb = Vb::size(); // 32 bf16 lanes = 2 x 16 fp32
  constexpr int kMaxChunks = 8; // cache cos/sin for rot up to 256
  const int64_t nfull = rot / kVb;
  const bool cache = nfull <= kMaxChunks;

  Vf cos0[kMaxChunks], cos1[kMaxChunks], sin0[kMaxChunks], sin1[kMaxChunks];

  for (int64_t p = p0; p < p1; ++p) {
    const int64_t b = p / S;
    const int64_t s = p - b * S;

    const at::BFloat16 *cosp = cosb + b * cb + s * cs;
    const at::BFloat16 *sinp = sinb + b * sb + s * ss;

    // Convert cos/sin once for this (batch, seq) position; reuse over heads.
    if (cache) {
      for (int64_t ci = 0; ci < nfull; ++ci) {
        const int64_t jj = ci * kVb;
        std::tie(cos0[ci], cos1[ci]) =
            convert_bfloat16_float(Vb::loadu(cosp + jj));
        std::tie(sin0[ci], sin1[ci]) =
            convert_bfloat16_float(Vb::loadu(sinp + jj));
      }
    }

    const at::BFloat16 *x1h = x1b + b * sx1.b + s * sx1.s;
    const at::BFloat16 *x2h = x2b + b * sx2.b + s * sx2.s;
    at::BFloat16 *o1h = o1b + b * so1.b + s * so1.s;
    at::BFloat16 *o2h = o2b + b * so2.b + s * so2.s;

    for (int64_t h = 0; h < H; ++h) {
      const at::BFloat16 *x1p = x1h + h * sx1.h;
      const at::BFloat16 *x2p = x2h + h * sx2.h;
      at::BFloat16 *o1p = o1h + h * so1.h;
      at::BFloat16 *o2p = o2h + h * so2.h;

      int64_t j = 0;
      for (int64_t ci = 0; ci < nfull; ++ci, j += kVb) {
        Vb av = Vb::loadu(x1p + j);
        Vb cv = Vb::loadu(x2p + j);

        Vf a0, a1, c0, c1, co0, co1, si0, si1;
        std::tie(a0, a1) = convert_bfloat16_float(av);
        std::tie(c0, c1) = convert_bfloat16_float(cv);
        if (cache) {
          co0 = cos0[ci]; co1 = cos1[ci];
          si0 = sin0[ci]; si1 = sin1[ci];
        } else {
          std::tie(co0, co1) = convert_bfloat16_float(Vb::loadu(cosp + j));
          std::tie(si0, si1) = convert_bfloat16_float(Vb::loadu(sinp + j));
        }

        Vf o1_0, o1_1, o2_0, o2_1;
        if constexpr (Conj) {
          // o1 = a*cos + c*sin ; o2 = c*cos - a*sin
          o1_0 = fmadd(a0, co0, c0 * si0);
          o1_1 = fmadd(a1, co1, c1 * si1);
          o2_0 = fmsub(c0, co0, a0 * si0);
          o2_1 = fmsub(c1, co1, a1 * si1);
        } else {
          // o1 = a*cos - c*sin ; o2 = a*sin + c*cos
          o1_0 = fmsub(a0, co0, c0 * si0);
          o1_1 = fmsub(a1, co1, c1 * si1);
          o2_0 = fmadd(a0, si0, c0 * co0);
          o2_1 = fmadd(a1, si1, c1 * co1);
        }

        convert_float_bfloat16(o1_0, o1_1).store(o1p + j);
        convert_float_bfloat16(o2_0, o2_1).store(o2p + j);
      }

      for (; j < rot; ++j) {
        float a = x1p[j];
        float c = x2p[j];
        float co = cosp[j];
        float si = sinp[j];
        if constexpr (Conj) {
          o1p[j] = at::BFloat16(a * co + c * si);
          o2p[j] = at::BFloat16(c * co - a * si);
        } else {
          o1p[j] = at::BFloat16(a * co - c * si);
          o2p[j] = at::BFloat16(a * si + c * co);
        }
      }
    }
  }
}

void apply_rotary(const torch::Tensor &x1, const torch::Tensor &x2,
                  const torch::Tensor &cos, const torch::Tensor &sin,
                  torch::Tensor &out1, torch::Tensor &out2, bool conj) {
  const int64_t rot = x1.size(-1);
  const int64_t S = x1.size(-2);
  const int64_t H = x1.size(-3);
  const int64_t B = x1.size(0);
  const int64_t P = B * S; // number of (batch, seq) positions

  const at::BFloat16 *x1b = x1.data_ptr<at::BFloat16>();
  const at::BFloat16 *x2b = x2.data_ptr<at::BFloat16>();
  const at::BFloat16 *cosb = cos.data_ptr<at::BFloat16>();
  const at::BFloat16 *sinb = sin.data_ptr<at::BFloat16>();
  at::BFloat16 *o1b = out1.data_ptr<at::BFloat16>();
  at::BFloat16 *o2b = out2.data_ptr<at::BFloat16>();

  // Full (batch, head, seq) strides so any layout works (e.g. transposed q/k).
  const Strides4 sx1{x1.stride(0), x1.stride(1), x1.stride(2)};
  const Strides4 sx2{x2.stride(0), x2.stride(1), x2.stride(2)};
  const Strides4 so1{out1.stride(0), out1.stride(1), out1.stride(2)};
  const Strides4 so2{out2.stride(0), out2.stride(1), out2.stride(2)};
  // cos/sin are (batch, 1, seq, rot) broadcast over heads.
  const int64_t cb = cos.stride(0), cs = cos.stride(2);
  const int64_t sb = sin.stride(0), ss = sin.stride(2);

  const int64_t num_threads = at::get_num_threads();
  // Each position does H heads * rot elements of work.
  const int64_t pos_per_grain =
      std::max<int64_t>(1, 8192 / std::max<int64_t>(H * rot, 1));
  const int64_t grain_size =
      std::max<int64_t>(pos_per_grain, P / std::max<int64_t>(num_threads, 1));

  at::parallel_for(0, P, grain_size, [&](int64_t begin, int64_t end) {
    if (conj) {
      apply_rotary_core<true>(x1b, x2b, cosb, sinb, o1b, o2b, begin, end, rot, H,
                              S, sx1, sx2, so1, so2, cb, cs, sb, ss);
    } else {
      apply_rotary_core<false>(x1b, x2b, cosb, sinb, o1b, o2b, begin, end, rot,
                               H, S, sx1, sx2, so1, so2, cb, cs, sb, ss);
    }
  });
}

} // namespace avx512
} // namespace rotary_cpu
