#pragma once

#include <torch/all.h>

namespace megablocks {

// Forward declarations for the public interface functions
void exclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out);
void inclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out);

} // namespace megablocks