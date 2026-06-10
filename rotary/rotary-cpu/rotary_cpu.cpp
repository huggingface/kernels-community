#include "rotary_cpu.hpp"
#include "cpu_features.hpp"
#include "rotary_avx512.hpp"

#include <ATen/ATen.h>

namespace rotary_cpu {

void apply_rotary(const torch::Tensor &x1, const torch::Tensor &x2,
                  const torch::Tensor &cos, const torch::Tensor &sin,
                  torch::Tensor &out1, torch::Tensor &out2, bool conj) {
  const int64_t rot = x1.size(-1);

  // Fast path: contiguous-last-dim BF16, 4-D (batch, heads, seq, rot) tensors
  // with cos/sin broadcast over the head dimension, on an AVX512 CPU.
  const bool last_dim_contig =
      x1.stride(-1) == 1 && x2.stride(-1) == 1 && cos.stride(-1) == 1 &&
      sin.stride(-1) == 1 && out1.stride(-1) == 1 && out2.stride(-1) == 1;

  const bool all_bf16 =
      x1.scalar_type() == at::kBFloat16 && x2.scalar_type() == at::kBFloat16 &&
      cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16 &&
      out1.scalar_type() == at::kBFloat16 && out2.scalar_type() == at::kBFloat16;

  const bool shape_ok =
      x1.dim() == 4 && cos.dim() == 4 && cos.size(-3) == 1 && sin.size(-3) == 1;

  if (all_bf16 && last_dim_contig && shape_ok && CPUFeatures::hasAVX512()) {
    avx512::apply_rotary(x1, x2, cos, sin, out1, out2, conj);
    return;
  }

  // Generic ATen fallback: any dtype/layout on any CPU. Arithmetic in fp32,
  // results computed into fresh tensors so in-place aliasing (out1==x1) is safe.
  auto x1f = x1.to(at::kFloat);
  auto x2f = x2.to(at::kFloat);
  auto cosf = cos.to(at::kFloat);
  auto sinf = sin.to(at::kFloat);

  torch::Tensor o1, o2;
  if (conj) {
    o1 = x1f * cosf + x2f * sinf;
    o2 = x2f * cosf - x1f * sinf;
  } else {
    o1 = x1f * cosf - x2f * sinf;
    o2 = x1f * sinf + x2f * cosf;
  }
  out1.copy_(o1);
  out2.copy_(o2);
}

} // namespace rotary_cpu
