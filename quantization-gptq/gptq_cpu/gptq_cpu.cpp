#include "gptq_cpu.hpp"

using gptq_cpu::avx512::DataType_t;

namespace gptq_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void gemm_int4(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &zeros,
        const torch::Tensor &absmax, torch::Tensor &out, int64_t blocksize)
    {
        int64_t M = input.size(0);
        int64_t N = weight.size(0);
        int64_t K = input.size(1);
        // strides
        int64_t x_strideM = input.stride(0);
        int64_t out_strideM = out.stride(0);
        // Runtime CPU feature detection and dispatch
        if (CPUFeatures::hasAVX512BF16())
        {
            // Use AVX512 optimized implementation
            gptq_cpu::avx512::gemm_int4_inference<at::BFloat16>(
                M, N, K,
                input.data_ptr<at::BFloat16>(),
                weight.data_ptr<unsigned char>(),
                zeros.data_ptr<uint8_t>(),
                absmax.data_ptr<at::BFloat16>(),
                out.data_ptr<at::BFloat16>(),
                blocksize, x_strideM, out_strideM);
        }
        else
        {
            // raise error for unsupported CPU
            throw std::runtime_error("[gptq] gemm_int4: CPU does not support AVX512BF16 instruction set required for 4-bit quantization operations.");
        }
    }
} // namespace gptq_cpu
