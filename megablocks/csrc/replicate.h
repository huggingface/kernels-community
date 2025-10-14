#pragma once

#include <torch/all.h>

namespace megablocks {

// Forward pass: replicate values from x according to bin sizes
void replicate_forward(torch::Tensor x,
                       torch::Tensor bins,
                       torch::Tensor out);

// Backward pass: reduce gradients back to bins using segmented reduction
void replicate_backward(torch::Tensor grad,
                        torch::Tensor bins,
                        torch::Tensor out);

} // namespace megablocks