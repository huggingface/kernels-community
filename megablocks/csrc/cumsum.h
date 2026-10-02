#pragma once

#include "stable_utils.h"

namespace megablocks {

// Forward declarations for the public interface functions
void exclusive_cumsum(torch::stable::Tensor x, int dim, torch::stable::Tensor out);
void inclusive_cumsum(torch::stable::Tensor x, int dim, torch::stable::Tensor out);

} // namespace megablocks