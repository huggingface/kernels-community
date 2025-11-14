#include "cpu_features.hpp"
#include "rmsnorm_avx2.hpp"
#include "rmsnorm_avx512.hpp"
#include <stdexcept>
#include <ATen/ATen.h>

namespace rmsnorm_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void rmsnorm(torch::Tensor &out, const torch::Tensor &input, const torch::Tensor &weight,
                 float epsilon)
    {
        // Runtime CPU feature detection and dispatch
        if (CPUFeatures::hasAVX512BF16())
        {
            // Use AVX512 optimized implementation
            rmsnorm_cpu::avx512::rms_norm(out, input, weight, epsilon);
        }
        else if (CPUFeatures::hasAVX2())
        {
            // Use AVX2 optimized implementation
            rmsnorm_cpu::avx2::rms_norm(out, input, weight, epsilon);
        }
        else
        {
            // Fallback to ATen implementation
            auto input1 = input.to(at::kFloat);
            auto variance = at::mean(at::pow(input1, 2), -1, true);
            auto hidden_states = at::rsqrt(at::add(variance, epsilon));
            out = at::mul(weight, at::mul(input1, hidden_states)).to(input.scalar_type());
        }
    }

    void rmsnorm_backward(
        torch::Tensor &grad_input,
        torch::Tensor &grad_weight,
        const torch::Tensor &grad_output,
        const torch::Tensor &hidden_states,
        const torch::Tensor &weight,
        float variance_epsilon)
    {
        // Runtime CPU feature detection and dispatch
        if (CPUFeatures::hasAVX512BF16())
        {
            // Use AVX512 optimized implementation
            rmsnorm_cpu::avx512::rms_norm_backward(grad_input, grad_weight, grad_output, hidden_states, weight, variance_epsilon);
        }
        else if (CPUFeatures::hasAVX2())
        {
            // Use AVX2 optimized implementation
            rmsnorm_cpu::avx2::rms_norm_backward(grad_input, grad_weight, grad_output, hidden_states, weight, variance_epsilon);
        }
        else
        {
            // Fallback to ATen implementation (compute gradients in FP32)
            auto g = grad_output.to(at::kFloat);   // (N, H)
            auto x = hidden_states.to(at::kFloat); // (N, H)
            auto w = weight.to(at::kFloat);        // (H,)

            const int64_t H = x.size(-1);

            // inv_rms per token: rsqrt(mean(x^2, -1) + eps) -> shape (N,1)
            auto variance = x.pow(2).mean(-1, /*keepdim=*/true);
            auto inv_rms = (variance + variance_epsilon).rsqrt();

            // S = sum_k g * w * x over last dim -> shape (N,1)
            auto S = (g * x * w).sum(-1, /*keepdim=*/true);

            // grad_input = g * w * inv_rms - (x * inv_rms^3 * S) / H
            auto inv_rms3 = inv_rms.pow(3);
            auto grad_input1 = g * w * inv_rms - x * inv_rms3 * S / static_cast<float>(H);

            // grad_weight = sum over tokens of g * x * inv_rms -> shape (H,)
            auto grad_weight1 = (g * x * inv_rms).sum(0);

            // copy back to requested dtypes / tensors
            grad_input.copy_(grad_input1.to(grad_input.scalar_type()));
            grad_weight.copy_(grad_weight1.to(grad_weight.scalar_type()));
        }
    }
} // namespace rmsnorm_cpu
