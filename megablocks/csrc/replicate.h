#pragma once

#include "stable_utils.h"

namespace megablocks {

// Forward pass: replicate values from x according to bin sizes
void replicate_forward(torch::stable::Tensor x,
                       torch::stable::Tensor bins,
                       torch::stable::Tensor out);

// Backward pass: reduce gradients back to bins using segmented reduction
void replicate_backward(torch::stable::Tensor grad,
                        torch::stable::Tensor bins,
                        torch::stable::Tensor out);

} // namespace megablocks