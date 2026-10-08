#pragma once

#include "stable_utils.h"

namespace megablocks {

// Public interface function for constructing indices from padded bins
void indices(torch::stable::Tensor padded_bins,
             int block_size,
             int output_block_rows,
             int output_block_columns,
             torch::stable::Tensor out);

} // namespace megablocks