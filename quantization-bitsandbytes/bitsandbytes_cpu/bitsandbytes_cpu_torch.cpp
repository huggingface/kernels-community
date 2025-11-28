#include <torch/all.h>
#include "bitsandbytes_cpu.hpp"

// Forward implementation for CPU
torch::Tensor gemm_4bit_cpu_forward(
    const torch::Tensor &input, const torch::Tensor &weight, 
    const torch::Tensor &absmax, int64_t blocksize, int64_t quant_type)
{
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(absmax.is_contiguous(), "absmax must be contiguous");

    auto output = at::empty({input.size(0), weight.size(0)}, input.options());

    bitsandbytes_cpu::gemm_4bit(
        input,
        weight,
        absmax,
        output,
        blocksize,
        quant_type
    );

    return output;
}
