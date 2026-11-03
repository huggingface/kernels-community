/*
 * Fused z-loss Kernel - Compilation Unit
 */

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "fused_zloss_kernel.cuh"

extern "C" void fused_zloss_fwd_launcher(
    const float* logits, const int* mask, float* lse_out,
    int N, int V
) {
    constexpr int THREADS = 256;
    zloss_fwd_kernel<THREADS><<<N, THREADS>>>(logits, mask, lse_out, V);
}

extern "C" void fused_zloss_bwd_launcher(
    const float* logits, const float* lse, const int* mask,
    float grad_scale, float* grad_logits,
    int N, int V
) {
    constexpr int THREADS = 256;
    zloss_bwd_kernel<THREADS><<<N, THREADS>>>(logits, lse, mask, grad_scale, grad_logits, V);
}
