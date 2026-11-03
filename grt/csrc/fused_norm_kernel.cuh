/*
 * Fused PearlNorm Kernel — pair-aware RMSNorm in one launch.
 *
 * Replaces 3+ PyTorch kernels (square+sum, mean+sqrt, div+mul) with 1.
 * Uses warp-level reduction (__shfl_down_sync) for fast RMS computation.
 */

#ifndef FUSED_NORM_KERNEL_H
#define FUSED_NORM_KERNEL_H

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NORM_WARP_SIZE 32

// ROCm HIP requires a 64-bit mask for __shfl_down_sync
#ifdef USE_ROCM
#define FULL_MASK 0xffffffffffffffffULL
#else
#define FULL_MASK 0xffffffff
#endif

// Forward: x[n, D] -> out[n, D] = x * rms_inv * scale
// One block per token position (n), cooperative reduction across pairs
__global__ void pearl_norm_fwd_kernel(
    const float* __restrict__ x,       // [N, D]
    const float* __restrict__ scale,   // [D]
    float* __restrict__ out,           // [N, D]
    float* __restrict__ rms_inv_out,   // [N] — saved for backward
    int N, int D, float eps
) {
    int n = blockIdx.x;
    if (n >= N) return;

    int half_D = D / 2;

    // Phase 1: compute sum of pair radii squared (cooperative reduction)
    float local_sum = 0.0f;
    for (int p = threadIdx.x; p < half_D; p += blockDim.x) {
        float x0 = x[n * D + 2 * p];
        float x1 = x[n * D + 2 * p + 1];
        local_sum += x0 * x0 + x1 * x1;
    }

    // Warp-level reduction
    for (int offset = NORM_WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);

    // Cross-warp reduction via shared memory
    extern __shared__ float smem[];
    int warp_id = threadIdx.x / NORM_WARP_SIZE;
    int lane_id = threadIdx.x % NORM_WARP_SIZE;
    if (lane_id == 0) smem[warp_id] = local_sum;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        int n_warps = (blockDim.x + NORM_WARP_SIZE - 1) / NORM_WARP_SIZE;
        float val = (lane_id < n_warps) ? smem[lane_id] : 0.0f;
        for (int offset = NORM_WARP_SIZE / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(FULL_MASK, val, offset);
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();
    total = smem[0];

    // rms_inv = 1 / sqrt(mean(r_sq) + eps)
    float rms_inv = rsqrtf(total / half_D + eps);

    // Save for backward
    if (threadIdx.x == 0) {
        rms_inv_out[n] = rms_inv;
    }

    // Phase 2: apply normalization and scale
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        out[n * D + i] = x[n * D + i] * rms_inv * scale[i];
    }
}

// Backward: compute grad_x only (grad_scale computed in Python to avoid atomicAdd contention)
__global__ void pearl_norm_bwd_kernel(
    const float* __restrict__ grad_out,  // [N, D]
    const float* __restrict__ x,         // [N, D]
    const float* __restrict__ scale,     // [D]
    const float* __restrict__ rms_inv,   // [N]
    float* __restrict__ grad_x,          // [N, D]
    int N, int D, float eps
) {
    int n = blockIdx.x;
    if (n >= N) return;

    int half_D = D / 2;
    float rms_inv_n = rms_inv[n];

    // Phase 1: compute dot(grad_out * scale, x) for the correction term
    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        local_dot += grad_out[n * D + i] * scale[i] * x[n * D + i];
    }

    // Warp-level reduction
    for (int offset = NORM_WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_dot += __shfl_down_sync(FULL_MASK, local_dot, offset);

    extern __shared__ float smem[];
    int warp_id = threadIdx.x / NORM_WARP_SIZE;
    int lane_id = threadIdx.x % NORM_WARP_SIZE;
    if (lane_id == 0) smem[warp_id] = local_dot;
    __syncthreads();

    float total_dot = 0.0f;
    if (warp_id == 0) {
        int n_warps = (blockDim.x + NORM_WARP_SIZE - 1) / NORM_WARP_SIZE;
        float val = (lane_id < n_warps) ? smem[lane_id] : 0.0f;
        for (int offset = NORM_WARP_SIZE / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(FULL_MASK, val, offset);
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();
    total_dot = smem[0];

    // Phase 2: compute grad_x only
    // grad_x[i] = (grad_out[i] * scale[i] - x[i] * total_dot * rms_inv^2 / half_D) * rms_inv
    float coeff = total_dot * rms_inv_n * rms_inv_n / half_D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float go = grad_out[n * D + i];
        float xi = x[n * D + i];
        float si = scale[i];
        grad_x[n * D + i] = (go * si - xi * coeff) * rms_inv_n;
    }
}

#endif // FUSED_NORM_KERNEL_H
