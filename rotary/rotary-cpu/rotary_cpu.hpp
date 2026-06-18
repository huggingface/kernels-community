#pragma once

#include <torch/all.h>

namespace rotary_cpu {

// Dispatcher for the rotary embedding op on CPU.
//
// Computes, for non-conjugated (conj == false):
//     out1 = x1 * cos - x2 * sin
//     out2 = x1 * sin + x2 * cos
// and for conj == true:
//     out1 =  x1 * cos + x2 * sin
//     out2 = -x1 * sin + x2 * cos
// All arithmetic is done in fp32 and rounded back to the tensor dtype, matching
// the CUDA/XPU kernels. `cos`/`sin` broadcast over the head dimension. `out1`
// and `out2` may alias `x1`/`x2` (in-place rotation).
//
// Selects an AVX512 BF16 path when inputs are contiguous BF16 with cos/sin
// broadcast over heads on an AVX512 CPU; otherwise uses a generic ATen fallback.
void apply_rotary(const torch::Tensor &x1, const torch::Tensor &x2,
                  const torch::Tensor &cos, const torch::Tensor &sin,
                  torch::Tensor &out1, torch::Tensor &out2, bool conj);

} // namespace rotary_cpu
