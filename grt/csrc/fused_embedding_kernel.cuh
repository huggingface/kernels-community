/*
 * Fused Embedding Kernel — sincosf + exp + scale in one launch.
 *
 * Replaces 5 PyTorch kernels (cos, sin, stack, exp, mul) with 1.
 * Uses sincosf() hardware instruction for simultaneous sin+cos.
 */

#ifndef FUSED_EMBEDDING_KERNEL_H
#define FUSED_EMBEDDING_KERNEL_H

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Forward: phase_angles[token_id, p] -> (cos, sin) * exp(clamp(log_mag))
__global__ void fused_embedding_fwd_kernel(
    const float* __restrict__ phase_angles,   // [V, P]
    const float* __restrict__ log_magnitude,  // [V]
    const int*   __restrict__ token_ids,      // [N]
    float* __restrict__ output,               // [N, D]  (D = 2*P)
    int N, int P
) {
    int n = blockIdx.x;
    int p = threadIdx.x;
    if (n >= N || p >= P) return;

    int token_id = token_ids[n];
    float angle = phase_angles[token_id * P + p];

    // sincosf: hardware instruction, ~1.5x faster than separate sin+cos
    float c, s;
    sincosf(angle, &s, &c);

    // Fused magnitude: exp(clamp(log_mag, -5, 5))
    float log_mag = log_magnitude[token_id];
    log_mag = fminf(fmaxf(log_mag, -5.0f), 5.0f);
    float mag = expf(log_mag);

    // Write interleaved [cos*mag, sin*mag, cos*mag, sin*mag, ...]
    output[n * 2 * P + 2 * p]     = c * mag;
    output[n * 2 * P + 2 * p + 1] = s * mag;
}

// Warp-level reduction for embedding kernel (self-contained, no external deps)
__device__ inline float emb_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#ifdef USE_ROCM
        val += __shfl_down(val, offset, 32);
#else
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, 32);
#endif
    }
    return val;
}

// Backward: compute gradients for phase_angles and log_magnitude
__global__ void fused_embedding_bwd_kernel(
    const float* __restrict__ phase_angles,   // [V, P]
    const float* __restrict__ log_magnitude,  // [V]
    const int*   __restrict__ token_ids,      // [N]
    const float* __restrict__ grad_output,    // [N, D]
    float* __restrict__ grad_phase_angles,    // [V, P]
    float* __restrict__ grad_log_magnitude,   // [V]
    int N, int P
) {
    int n = blockIdx.x;
    int p = threadIdx.x;
    if (n >= N || p >= P) return;

    int token_id = token_ids[n];
    float angle = phase_angles[token_id * P + p];

    float c, s;
    sincosf(angle, &s, &c);

    float log_mag = log_magnitude[token_id];
    log_mag = fminf(fmaxf(log_mag, -5.0f), 5.0f);
    float mag = expf(log_mag);

    float g0 = grad_output[n * 2 * P + 2 * p];      // grad for cos*mag
    float g1 = grad_output[n * 2 * P + 2 * p + 1];  // grad for sin*mag

    // d/d(angle) of [cos(a)*m, sin(a)*m] = [-sin(a)*m, cos(a)*m]
    float grad_angle = g0 * (-s * mag) + g1 * (c * mag);
    atomicAdd(&grad_phase_angles[token_id * P + p], grad_angle);

    // d/d(log_mag) of [cos(a)*exp(lm), sin(a)*exp(lm)] = [cos(a)*m, sin(a)*m]
    // (only if log_mag is within clamp range)
    float raw_log_mag = log_magnitude[token_id];
    float grad_lm = 0.0f;
    if (raw_log_mag >= -5.0f && raw_log_mag <= 5.0f) {
        grad_lm = g0 * (c * mag) + g1 * (s * mag);
    }

    // Block-level reduction: warp reduce first, then inter-warp via shared memory
    __shared__ float warp_sums[8];  // max 256 threads = 8 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    grad_lm = emb_warp_reduce_sum(grad_lm);
    if (lane_id == 0) {
        warp_sums[warp_id] = grad_lm;
    }
    __syncthreads();

    // Thread 0 sums across warps and does single atomicAdd
    if (threadIdx.x == 0) {
        int n_warps = (P + 31) / 32;
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) {
            total += warp_sums[w];
        }
        if (total != 0.0f) {
            atomicAdd(&grad_log_magnitude[token_id], total);
        }
    }
}

#endif // FUSED_EMBEDDING_KERNEL_H
