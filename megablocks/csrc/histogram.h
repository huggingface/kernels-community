#pragma once

#include <torch/all.h>

namespace megablocks {

// Public interface function for computing histograms
torch::Tensor histogram(torch::Tensor x, int num_bins);

} // namespace megablocks