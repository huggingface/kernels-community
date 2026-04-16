#pragma once

#include <torch/torch.h>

torch::Tensor sdpa_int4(
    torch::Tensor queries,
    torch::Tensor k_quant,
    torch::Tensor k_scales,
    torch::Tensor k_biases,
    torch::Tensor v_quant,
    torch::Tensor v_scales,
    torch::Tensor v_biases,
    int64_t gqa_factor,
    int64_t N,
    double scale,
    int64_t sliding_window);
