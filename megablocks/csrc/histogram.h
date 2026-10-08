#pragma once

#include "stable_utils.h"

namespace megablocks {

// Public interface function for computing histograms
torch::stable::Tensor histogram(torch::stable::Tensor x, int num_bins);

} // namespace megablocks