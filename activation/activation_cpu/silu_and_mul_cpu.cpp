#include "silu_and_mul_cpu.hpp"
#include "cpu_features.hpp"
#include "silu_and_mul_avx512.hpp"

#include <ATen/ATen.h>

namespace activation_cpu {

void silu_and_mul(torch::Tensor &out, const torch::Tensor &input) {
  const int64_t d = out.size(-1);

  const bool can_use_avx512 =
      input.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 &&
      input.is_contiguous() && out.is_contiguous() &&
      CPUFeatures::hasAVX512BF16();

  if (can_use_avx512) {
    avx512::silu_and_mul(out, input);
    return;
  }

  // Generic ATen fallback: works for any dtype on any CPU and matches the
  // reference F.silu(x[..., :d]) * x[..., d:] (silu is rounded to the input
  // dtype before the multiply, just like PyTorch eager).
  auto input_c = input.is_contiguous() ? input : input.contiguous();
  auto x1 = input_c.narrow(-1, 0, d);
  auto x2 = input_c.narrow(-1, d, d);
  out.copy_(at::silu(x1) * x2);
}

} // namespace activation_cpu
