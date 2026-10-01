#include <torch/all.h>

#include "silu_and_mul_cpu.hpp"

// Global entry point referenced by torch-ext/torch_binding.cpp for the CPU
// backend. Validates shapes and forwards to the CPU dispatcher.
void silu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(out.device().is_cpu(), "out must be a CPU tensor");
  TORCH_CHECK(input.size(-1) == 2 * out.size(-1),
              "input last dim must be twice the output last dim");

  activation_cpu::silu_and_mul(out, input);
}
