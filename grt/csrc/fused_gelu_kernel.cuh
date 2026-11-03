/*
 * Fused RadialGELU Kernel — radial gating in one launch.
 *
 * Replaces 4+ PyTorch kernels (norm, gelu, div, mul) with 1.
 * One thread per pair — embarrassingly parallel, no reductions needed.
 */

#ifndef FUSED_GELU_KERNEL_H
#define FUSED_GELU_KERNEL_H

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Forward: for each pair (x0, x1), compute r = sqrt(x0²+x1²),
// gate = gelu(r)/(r+eps), output = (x0*gate, x1*gate)
__global__ void radial_gelu_fwd_kernel(
    const float* __restrict__ x,     // [N, D]
    float* __restrict__ out,         // [N, D]
    int total_pairs, int half_D, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int n = idx / half_D;
    int p = idx % half_D;
    int offset = n * 2 * half_D + 2 * p;

    float x0 = x[offset];
    float x1 = x[offset + 1];

    float r = sqrtf(x0 * x0 + x1 * x1);

    // GELU using erf: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    float gelu_r = 0.5f * r * (1.0f + erff(r * 0.7071067811865476f));
    float gate = gelu_r / (r + eps);

    out[offset]     = x0 * gate;
    out[offset + 1] = x1 * gate;
}

// Backward: compute grad_x given grad_output
// d/dx0 of (x0 * gate(r)) = gate + x0 * dgate/dx0
//   where dgate/dr = (gelu'(r)*r - gelu(r)) / (r+eps)^2
//   and dr/dx0 = x0/r
__global__ void radial_gelu_bwd_kernel(
    const float* __restrict__ grad_out,  // [N, D]
    const float* __restrict__ x,         // [N, D]
    float* __restrict__ grad_x,          // [N, D]
    int total_pairs, int half_D, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int n = idx / half_D;
    int p = idx % half_D;
    int offset = n * 2 * half_D + 2 * p;

    float x0 = x[offset];
    float x1 = x[offset + 1];

    float r = sqrtf(x0 * x0 + x1 * x1);
    float r_eps = r + eps;

    // GELU and its derivative
    float erf_val = erff(r * 0.7071067811865476f);
    float gelu_r = 0.5f * r * (1.0f + erf_val);
    float gate = gelu_r / r_eps;

    // gelu'(r) = 0.5 * (1 + erf(r/sqrt2)) + r * exp(-r²/2) / sqrt(2*pi)
    float exp_val = expf(-0.5f * r * r);
    float gelu_prime = 0.5f * (1.0f + erf_val) + r * exp_val * 0.3989422804014327f;

    // dgate/dr = (gelu'(r) * r_eps - gelu_r) / r_eps^2
    //          = (gelu'(r) - gate) / r_eps
    float dgate_dr = (gelu_prime - gate) / r_eps;

    float g0 = grad_out[offset];
    float g1 = grad_out[offset + 1];

    if (r > 1e-12f) {
        float inv_r = 1.0f / r;
        // dr/dx0 = x0/r, dr/dx1 = x1/r
        // d/dx0 = gate * g0 + (x0 * dgate_dr * x0/r) * g0 + (x1 * dgate_dr * x0/r) * g1
        //       = gate * g0 + dgate_dr * x0/r * (x0*g0 + x1*g1)
        float common = dgate_dr * inv_r * (x0 * g0 + x1 * g1);
        grad_x[offset]     = gate * g0 + x0 * common;
        grad_x[offset + 1] = gate * g1 + x1 * common;
    } else {
        // Near zero: gate approaches gelu(0)/eps ≈ 0, grad ≈ 0
        grad_x[offset]     = gate * g0;
        grad_x[offset + 1] = gate * g1;
    }
}

#endif // FUSED_GELU_KERNEL_H
