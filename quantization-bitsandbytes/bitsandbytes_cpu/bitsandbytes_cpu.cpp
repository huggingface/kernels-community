#include "bitsandbytes_cpu.hpp"

using bitsandbytes_cpu::avx512::DataType_t;

namespace bitsandbytes_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void gemm_4bit(const torch::Tensor &input, const torch::Tensor &weight, 
        const torch::Tensor &absmax, torch::Tensor &out, int64_t blocksize, int64_t quant_type)
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
            if (quant_type == 1) {
                bitsandbytes_cpu::avx512::gemm_4bit_inference<at::BFloat16, DataType_t::FP4>(
                    M, N, K,
                    input.data_ptr<at::BFloat16>(),
                    weight.data_ptr<unsigned char>(),
                    absmax.data_ptr<at::BFloat16>(),
                    out.data_ptr<at::BFloat16>(),
                    blocksize, x_strideM, out_strideM);
            }
            else {
                bitsandbytes_cpu::avx512::gemm_4bit_inference<at::BFloat16, DataType_t::NF4>(
                    M, N, K,
                    input.data_ptr<at::BFloat16>(),
                    weight.data_ptr<unsigned char>(),
                    absmax.data_ptr<at::BFloat16>(),
                    out.data_ptr<at::BFloat16>(),
                    blocksize, x_strideM, out_strideM);
            }
        }
        else
        {
            // raise error for unsupported CPU
            throw std::runtime_error("[bitsandbytes] gemm_4bit: CPU does not support AVX512BF16 instruction set required for 4-bit quantization operations.");
        }
    }
} // namespace bitsandbytes_cpu
