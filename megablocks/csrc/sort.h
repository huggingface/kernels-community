#pragma once

#include "stable_utils.h"

namespace megablocks {

// Public interface function for radix sorting with indices
void sort(torch::stable::Tensor x,
          int end_bit,
          torch::stable::Tensor x_out,
          torch::stable::Tensor iota_out);

} // namespace megablocks