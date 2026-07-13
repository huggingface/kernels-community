#pragma once

#include <torch/all.h>

namespace activation_cpu {

// Dispatcher: selects the AVX512-BF16 kernel when the input is contiguous BF16
// and the CPU supports AVX512-BF16; otherwise uses a generic ATen fallback that
// works for any dtype on any CPU.
//
// Semantics: out = silu(input[..., :d]) * input[..., d:], where
// d = out.size(-1) and input.size(-1) == 2 * d.
void silu_and_mul(torch::Tensor &out, const torch::Tensor &input);

} // namespace activation_cpu
