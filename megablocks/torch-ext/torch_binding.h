#pragma once

#include <torch/torch.h>

torch::Tensor exclusive_cumsum_wrapper(torch::Tensor x, int64_t dim, torch::Tensor out);
// torch::Tensor inclusive_cumsum_wrapper(torch::Tensor x, int64_t dim, torch::Tensor out);