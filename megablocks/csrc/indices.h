#pragma once

#include <torch/all.h>

namespace megablocks {

// Public interface function for constructing indices from padded bins
void indices(torch::Tensor padded_bins,
             int block_size,
             int output_block_rows,
             int output_block_columns,
             torch::Tensor out);

} // namespace megablocks