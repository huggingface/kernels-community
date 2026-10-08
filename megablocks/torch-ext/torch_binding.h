#pragma once

#include <torch/csrc/stable/tensor.h>

torch::stable::Tensor exclusive_cumsum_wrapper(torch::stable::Tensor x, int64_t dim, torch::stable::Tensor out);
// torch::stable::Tensor inclusive_cumsum_wrapper(torch::stable::Tensor x, int64_t dim, torch::stable::Tensor out);
