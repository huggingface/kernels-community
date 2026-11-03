/*
 * Fused PearlNorm Kernel - Compilation Unit
 */

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "fused_norm_kernel.cuh"

extern "C" void fused_norm_fwd_launcher(
    const float* x, const float* scale, float* out, float* rms_inv_out,
    int N, int D, float eps
) {
    int half_D = D / 2;
    int threads = (half_D < 256) ? half_D : 256;
    if (threads < 32) threads = 32;
    int smem_bytes = (threads / NORM_WARP_SIZE + 1) * sizeof(float);
    pearl_norm_fwd_kernel<<<N, threads, smem_bytes>>>(
        x, scale, out, rms_inv_out, N, D, eps
    );
}

extern "C" void fused_norm_bwd_launcher(
    const float* grad_out, const float* x, const float* scale,
    const float* rms_inv, float* grad_x,
    int N, int D, float eps
) {
    int half_D = D / 2;
    int threads = (half_D < 256) ? half_D : 256;
    if (threads < 32) threads = 32;
    int smem_bytes = (threads / NORM_WARP_SIZE + 1) * sizeof(float);
    pearl_norm_bwd_kernel<<<N, threads, smem_bytes>>>(
        grad_out, x, scale, rms_inv, grad_x, N, D, eps
    );
}
