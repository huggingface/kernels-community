#include "bitsandbytes_cpu.hpp"

using bitsandbytes_cpu::avx512::DataType_t;

namespace bitsandbytes_cpu
{

    static const std::vector<float> NF4_DATA = {
        -1.00000000f, -0.69619280f, -0.52507305f, -0.39491749f, 
        -0.28444138f, -0.18477343f, -0.09105004f,  0.00000000f,
        0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f, 
        0.44070983f,  0.56261700f,  0.72295684f,  1.00000000f
    };

    static const std::vector<float> FP4_DATA = {
        0.0000f,  0.0052f,  0.6667f,  1.0000f, 
        0.3333f,  0.5000f,  0.1667f,  0.2500f,
        0.0000f, -0.0052f, -0.6667f, -1.0000f, 
        -0.3333f, -0.5000f, -0.1667f, -0.2500f
    };

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
            const int64_t packing_block_n = 32;
            int64_t N = weight.size(0);
            int64_t K_half = weight.size(1);
            auto device = weight.device();
            auto qw = weight.reshape({-1, packing_block_n});
            auto low = torch::bitwise_and(qw, 0x0F);
            auto high = torch::bitwise_and(torch::bitwise_right_shift(qw, 4), 0x0F);
            auto restored = torch::cat({low, high}, 1);
            restored = restored.reshape({N / packing_block_n, K_half, packing_block_n, 2});
            restored = restored.transpose(-3, -2);
            auto unpacked_weight = restored.reshape({N, K_half * 2});
            torch::Tensor table;
            if (quant_type == 1) {
                table = torch::tensor(FP4_DATA, torch::dtype(torch::kFloat32)).to(device); // FP4
            } else {
                table = torch::tensor(NF4_DATA, torch::dtype(torch::kFloat32)).to(device); // NF4
            }
            auto indices = unpacked_weight.to(torch::kLong); 
            auto dequantized_weight = table.index({indices});
            auto scales_expanded = absmax.t().repeat_interleave(blocksize, 1);
            auto original_weight = dequantized_weight * scales_expanded;
            auto weight_final = original_weight.t().to(input.dtype());
            torch::matmul_out(out, input, weight_final);
        }
    }
} // namespace bitsandbytes_cpu
