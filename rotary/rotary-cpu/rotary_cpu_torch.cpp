#include <torch/all.h>

#include "rotary_cpu.hpp"

// Global entry point referenced by torch-ext/torch_binding.cpp for the CPU
// backend. The `apply_rotary` wrapper there validates devices/dtypes/shapes and
// forwards to this function.
void _apply_rotary(torch::Tensor const &x1, torch::Tensor const &x2,
                   torch::Tensor const &cos, torch::Tensor const &sin,
                   torch::Tensor &out1, torch::Tensor &out2, bool const conj) {
  rotary_cpu::apply_rotary(x1, x2, cos, sin, out1, out2, conj);
}
