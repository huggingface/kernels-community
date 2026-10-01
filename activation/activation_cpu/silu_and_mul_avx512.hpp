#pragma once

#include <torch/all.h>

namespace activation_cpu {
namespace avx512 {

// Optimized SiLU-and-Mul for BF16 on AVX512 (+AVX512-BF16) CPUs.
// Requires `out` and `input` to be contiguous BF16 tensors with
// input.size(-1) == 2 * out.size(-1).
void silu_and_mul(torch::Tensor &out, const torch::Tensor &input);

} // namespace avx512
} // namespace activation_cpu
