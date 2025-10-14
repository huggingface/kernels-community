#pragma once

#include <torch/all.h>

namespace megablocks {

// Public interface function for radix sorting with indices
void sort(torch::Tensor x,
          int end_bit,
          torch::Tensor x_out,
          torch::Tensor iota_out);

} // namespace megablocks