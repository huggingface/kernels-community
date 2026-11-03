/*
 * Fused RadialGELU Kernel - Compilation Unit
 */

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "fused_gelu_kernel.cuh"

extern "C" void fused_gelu_fwd_launcher(
    const float* x, float* out,
    int total_pairs, int half_D, float eps
) {
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    radial_gelu_fwd_kernel<<<blocks, threads>>>(x, out, total_pairs, half_D, eps);
}

extern "C" void fused_gelu_bwd_launcher(
    const float* grad_out, const float* x, float* grad_x,
    int total_pairs, int half_D, float eps
) {
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    radial_gelu_bwd_kernel<<<blocks, threads>>>(grad_out, x, grad_x, total_pairs, half_D, eps);
}
