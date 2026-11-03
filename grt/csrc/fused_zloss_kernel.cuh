/*
 * Fused z-loss Kernel — logsumexp^2 mean in one launch.
 *
 * Replaces torch.logsumexp(logits, -1).pow(2).mean() with a single kernel.
 * One block per row, parallel reduction within block.
 *
 * Forward:
 *   For each valid row n: lse[n] = log(sum_v exp(logits[n,v] - max_n)) + max_n
 *   Returns: z_loss = mean_n(lse[n]^2)
 *   Saves: lse values for backward
 *
 * Backward:
 *   d/d(logits[n,v]) of (lse[n]^2 / N_valid) = (2 * lse[n] / N_valid) * softmax(logits)[n,v]
 *   = (2 * lse[n] / N_valid) * exp(logits[n,v] - lse[n])
 */

#ifndef FUSED_ZLOSS_KERNEL_H
#define FUSED_ZLOSS_KERNEL_H

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Block-wide max reduction via shared mem.
// THREADS must be a power of two <= 1024.
template <int THREADS>
__device__ inline float block_max(float v, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    return sdata[0];
}

template <int THREADS>
__device__ inline float block_sum(float v, float* sdata) {
    int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

// Forward: compute lse[n] per row, write out lse (shape [N]).
// mask[n] = 0 means skip (ignored target); lse[n] written as 0 in that case.
// Caller will sum lse^2 over valid rows and divide by n_valid.
template <int THREADS>
__global__ void zloss_fwd_kernel(
    const float* __restrict__ logits,  // [N, V]
    const int* __restrict__ mask,      // [N]: 1 if valid, 0 if ignored
    float* __restrict__ lse_out,       // [N]
    int V
) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    const float* row = logits + (long long)n * V;

    if (mask[n] == 0) {
        if (tid == 0) lse_out[n] = 0.0f;
        return;
    }

    __shared__ float sdata[THREADS];

    // pass 1: max
    float local_max = -INFINITY;
    for (int v = tid; v < V; v += THREADS) {
        local_max = fmaxf(local_max, row[v]);
    }
    float row_max = block_max<THREADS>(local_max, sdata);

    // pass 2: sum exp(x - max)
    float local_sum = 0.0f;
    for (int v = tid; v < V; v += THREADS) {
        local_sum += expf(row[v] - row_max);
    }
    float row_sum = block_sum<THREADS>(local_sum, sdata);

    if (tid == 0) {
        lse_out[n] = logf(row_sum) + row_max;
    }
}

// Backward: grad_logits[n,v] = grad_z_loss * (2 * lse[n] / n_valid) * exp(logits[n,v] - lse[n])
// grad_z_loss is a scalar (upstream grad of the scalar z_loss).
template <int THREADS>
__global__ void zloss_bwd_kernel(
    const float* __restrict__ logits,   // [N, V]
    const float* __restrict__ lse,      // [N]
    const int* __restrict__ mask,       // [N]
    float grad_scale,                   // grad_z_loss * (2 / n_valid)
    float* __restrict__ grad_logits,    // [N, V]
    int V
) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    const float* row = logits + (long long)n * V;
    float* grow = grad_logits + (long long)n * V;

    if (mask[n] == 0) {
        for (int v = tid; v < V; v += THREADS) grow[v] = 0.0f;
        return;
    }

    float lse_n = lse[n];
    float coef = grad_scale * lse_n;  // (2*grad_z_loss/n_valid) * lse[n]

    for (int v = tid; v < V; v += THREADS) {
        grow[v] = coef * expf(row[v] - lse_n);
    }
}

#endif  // FUSED_ZLOSS_KERNEL_H
