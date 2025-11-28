#pragma once

#include "cpu_features.hpp"
#include "bitsandbytes_avx512.hpp"
#include <torch/all.h>
#include <stdexcept>
#include <ATen/ATen.h>

namespace bitsandbytes_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void gemm_4bit(const torch::Tensor &input, const torch::Tensor &weight, 
        const torch::Tensor &absmax, torch::Tensor &out, int64_t blocksize, int64_t quant_type);
} // namespace bitsandbytes_cpu