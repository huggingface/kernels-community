#include <torch/all.h>
#include "gptq_cpu.hpp"

// Forward implementation for CPU
torch::Tensor gemm_int4_cpu_forward(
    const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &zeros,
    const torch::Tensor &absmax, int64_t blocksize)
{
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(zeros.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(absmax.is_contiguous(), "absmax must be contiguous");

    auto output = at::empty({input.size(0), weight.size(0)}, input.options());

    gptq_cpu::gemm_int4(
        input,
        weight,
        zeros,
        absmax,
        output,
        blocksize
    );

    return output;
}
