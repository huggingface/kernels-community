#pragma once

#include <torch/all.h>

namespace rotary_cpu {
namespace avx512 {

// Optimized rotary embedding for BF16 on AVX512 CPUs.
//
// Requires:
//   * x1, x2, cos, sin, out1, out2 are BF16
//   * all tensors are 4-D (batch, heads, seq, rot_dim) with a contiguous last
//     dim and uniform row spacing (slices of a contiguous parent are fine)
//   * cos/sin broadcast over the head dimension (size(1) == 1)
//
// out1/out2 may alias x1/x2.
void apply_rotary(const torch::Tensor &x1, const torch::Tensor &x2,
                  const torch::Tensor &cos, const torch::Tensor &sin,
                  torch::Tensor &out1, torch::Tensor &out2, bool conj);

} // namespace avx512
} // namespace rotary_cpu
