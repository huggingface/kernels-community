#pragma once

#include <torch/torch.h>

void count_cumsum_cuda(const torch::Tensor &x, torch::Tensor &count_output,
                       torch::Tensor &cumsum_output, bool do_cumsum);
