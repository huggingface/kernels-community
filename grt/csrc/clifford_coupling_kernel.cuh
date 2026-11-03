/*
 * Clifford Coupling Features - CUDA Kernels
 * Optimized: pairs cached in shared memory for fast neighbor access.
 */

#ifndef CLIFFORD_COUPLING_KERNEL_H
#define CLIFFORD_COUPLING_KERNEL_H

// PyTorch's hipify rewrites .cu but NOT .cuh includes,
// so we need an explicit guard for ROCm headers.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Clifford coupling features kernel - Shared memory optimized
// ---------------------------------------------------------------------------

__global__ void clifford_coupling_fwd_kernel(
    const float* __restrict__ x_pairs,  // [batch, P, 2]
    float* __restrict__ features,       // [batch, 4*P]
    int batch,
    int P
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y + blockIdx.z * gridDim.y;

    if (pair_idx >= P || batch_idx >= batch) return;

    // Shared memory: cache all pairs for this batch element
    extern __shared__ float smem[];  // [blockDim.x * 2]

    int base_in = (batch_idx * P + pair_idx) * 2;

    // Load own pair into shared memory
    float x0 = x_pairs[base_in];
    float x1 = x_pairs[base_in + 1];
    smem[2 * threadIdx.x]     = x0;
    smem[2 * threadIdx.x + 1] = x1;
    __syncthreads();

    // Bivector: read next pair from shared memory (fast, no bank conflicts)
    int next_local = (threadIdx.x + 1) % P;  // Use P for cyclic wrap (not blockDim.x)
    float x0_next = smem[2 * next_local];
    float x1_next = smem[2 * next_local + 1];
    float b12 = x0 * x1_next - x1 * x0_next;

    // Store features in planar layout
    int base_out = batch_idx * 4 * P;
    features[base_out + pair_idx] = x0 * x0 + x1 * x1;  // scalar
    features[base_out + P + pair_idx] = x0;               // v1
    features[base_out + 2 * P + pair_idx] = x1;           // v2
    features[base_out + 3 * P + pair_idx] = b12;          // bivector
}

// ---------------------------------------------------------------------------
// Clifford coupling backward kernel - Shared memory optimized
// ---------------------------------------------------------------------------

__global__ void clifford_coupling_bwd_kernel(
    const float* __restrict__ x_pairs,        // [batch, P, 2]
    const float* __restrict__ grad_features,  // [batch, 4*P]
    float* __restrict__ grad_x_pairs,         // [batch, P, 2]
    int batch,
    int P
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y + blockIdx.z * gridDim.y;

    if (pair_idx >= P || batch_idx >= batch) return;

    // Shared memory: cache pairs + bivector gradients for neighbor access
    extern __shared__ float smem[];  // [blockDim.x * 3]: x0, x1, gb12
    float* smem_x0   = smem;
    float* smem_x1   = smem + blockDim.x;
    float* smem_gb12 = smem + 2 * blockDim.x;

    int base_in = (batch_idx * P + pair_idx) * 2;
    int base_grad = batch_idx * 4 * P;

    float x0 = x_pairs[base_in];
    float x1 = x_pairs[base_in + 1];
    float gb12 = grad_features[base_grad + 3 * P + pair_idx];

    // Load into shared memory
    smem_x0[threadIdx.x]   = x0;
    smem_x1[threadIdx.x]   = x1;
    smem_gb12[threadIdx.x] = gb12;
    __syncthreads();

    // Read gradients
    float gs = grad_features[base_grad + pair_idx];
    float gv1 = grad_features[base_grad + P + pair_idx];
    float gv2 = grad_features[base_grad + 2 * P + pair_idx];

    // Gradient for s = x0² + x1²
    float gx0 = 2.0f * x0 * gs;
    float gx1 = 2.0f * x1 * gs;

    // Gradient for v1 = x0, v2 = x1
    gx0 += gv1;
    gx1 += gv2;

    // Gradient for bivector: read next pair from shared memory
    int next_local = (threadIdx.x + 1) % P;
    float x0_next = smem_x0[next_local];
    float x1_next = smem_x1[next_local];

    gx0 += gb12 * x1_next;
    gx1 += gb12 * (-x0_next);

    // Gradient from previous pair's bivector: read from shared memory
    int prev_local = (threadIdx.x - 1 + P) % P;
    float x0_prev = smem_x0[prev_local];
    float x1_prev = smem_x1[prev_local];
    float gb12_prev = smem_gb12[prev_local];

    gx0 += gb12_prev * (-x1_prev);
    gx1 += gb12_prev * x0_prev;

    // Store gradients
    grad_x_pairs[base_in] = gx0;
    grad_x_pairs[base_in + 1] = gx1;
}

#endif // CLIFFORD_COUPLING_KERNEL_H
