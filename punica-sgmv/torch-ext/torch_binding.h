#pragma once

#include <torch/torch.h>

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                   torch::Tensor indicies, int64_t layer_idx, double scale);

void dispatch_sgmv_cutlass(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                           torch::Tensor s_start, torch::Tensor s_end,
                           torch::Tensor tmp, int64_t layer_idx);

void dispatch_sgmv_shrink(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                          torch::Tensor s_start, torch::Tensor s_end, torch::Tensor tmp, int64_t layer_idx);


int64_t sgmv_tmp_size(int64_t num_problems);
