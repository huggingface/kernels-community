// AVX512 + AVX512-BF16 implementation of SiLU-and-Mul for BF16.
// Compile with -mavx512f -mavx512bf16 -mavx512vl -DCPU_CAPABILITY_AVX512 so that
// at::vec::Vectorized actually lowers to AVX512 (otherwise it falls back to the
// scalar vec_base.h implementation).
#include "silu_and_mul_avx512.hpp"

#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <tuple>

using namespace at::vec;

namespace activation_cpu {
namespace avx512 {

// Core computation for a flat element range [begin, end) of the OUTPUT.
// Math is done entirely in fp32: bf16 is converted to float ONCE on load and
// back ONCE on store, instead of round-tripping through bf16 on every op.
static inline void silu_and_mul_core(const at::BFloat16 *in_data,
                                     at::BFloat16 *out_data, int64_t begin,
                                     int64_t end, int64_t d) {
  using Vb = Vectorized<at::BFloat16>;
  using Vf = Vectorized<float>;
  constexpr int kVb = Vb::size(); // 32 bf16 lanes = 2 x 16 fp32
  const Vf one(1.0f);

  int64_t token_idx = begin / d;
  int64_t offset = begin % d;
  int64_t i = begin;

  while (i < end) {
    int64_t elements_to_process = std::min(d - offset, end - i);

    const at::BFloat16 *a_ptr = in_data + token_idx * (2 * d) + offset;
    const at::BFloat16 *b_ptr = a_ptr + d;
    at::BFloat16 *out_ptr_local = out_data + i;

    int64_t j = 0;
    for (; j <= elements_to_process - kVb; j += kVb) {
      Vb a_bf = Vb::loadu(a_ptr + j);
      Vb b_bf = Vb::loadu(b_ptr + j);

      Vf a0, a1, b0, b1;
      std::tie(a0, a1) = convert_bfloat16_float(a_bf);
      std::tie(b0, b1) = convert_bfloat16_float(b_bf);

      // silu(a) = a / (1 + exp(-a)), in fp32.
      Vf s0 = a0 / (one + a0.neg().exp());
      Vf s1 = a1 / (one + a1.neg().exp());

      // Match the reference F.silu(x1) * x2 exactly: it rounds silu to bf16
      // BEFORE multiplying. Round-trip silu through bf16, then multiply.
      std::tie(s0, s1) =
          convert_bfloat16_float(convert_float_bfloat16(s0, s1));

      convert_float_bfloat16(s0 * b0, s1 * b1).store(out_ptr_local + j);
    }

    for (; j < elements_to_process; ++j) {
      float a = a_ptr[j];
      float b = b_ptr[j];
      float s = static_cast<float>(at::BFloat16(a / (1.0f + std::exp(-a))));
      out_ptr_local[j] = at::BFloat16(s * b);
    }

    i += elements_to_process;
    token_idx++;
    offset = 0;
  }
}

void silu_and_mul(torch::Tensor &out, const torch::Tensor &input) {
  int64_t d = out.size(-1);
  int64_t total_elements = out.numel();

  const at::BFloat16 *in_data = input.data_ptr<at::BFloat16>();
  at::BFloat16 *out_data = out.data_ptr<at::BFloat16>();

  // at::parallel_for runs serially when total_elements <= grain_size, which
  // covers the decode/tiny case without any fork/join overhead.
  int64_t num_threads = at::get_num_threads();
  int64_t grain_size = std::max((int64_t)8192,
                                total_elements /
                                    std::max<int64_t>(num_threads, 1));

  at::parallel_for(0, total_elements, grain_size,
                   [&](int64_t begin, int64_t end) {
                     silu_and_mul_core(in_data, out_data, begin, end, d);
                   });
}

} // namespace avx512
} // namespace activation_cpu
