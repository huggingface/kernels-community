/*
 * Fused Embedding Kernel - Compilation Unit
 */

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "fused_embedding_kernel.cuh"

extern "C" void fused_embedding_fwd_launcher(
    const float* phase_angles,
    const float* log_magnitude,
    const int* token_ids,
    float* output,
    int N, int P
) {
    // One block per token, P threads per block
    // For d_model=256, P=128 — good occupancy
    fused_embedding_fwd_kernel<<<N, P>>>(
        phase_angles, log_magnitude, token_ids, output, N, P
    );
}

extern "C" void fused_embedding_bwd_launcher(
    const float* phase_angles,
    const float* log_magnitude,
    const int* token_ids,
    const float* grad_output,
    float* grad_phase_angles,
    float* grad_log_magnitude,
    int N, int P
) {
    fused_embedding_bwd_kernel<<<N, P>>>(
        phase_angles, log_magnitude, token_ids, grad_output,
        grad_phase_angles, grad_log_magnitude, N, P
    );
}
