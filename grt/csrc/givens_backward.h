/*
 * Givens Rotation Kernel - Backward Pass
 * Uses sincosf() for fused sin+cos and STAGE_BARRIER for warp-level sync.
 */

#ifndef GIVENS_BACKWARD_H
#define GIVENS_BACKWARD_H

#include "givens_common.h"

template<typename DataT>
LAUNCH_BOUNDS_BWD
__global__ void givens_bwd_kernel(
    const DataT* __restrict__ grad_out,
    const DataT* __restrict__ x_in,
    const float* __restrict__ angles,
    const float* __restrict__ alpha,
    const float* __restrict__ scales,
    DataT* __restrict__ grad_x,
    float* __restrict__ grad_angles,
    float* __restrict__ grad_alpha,
    float* __restrict__ grad_scales,
    const int* __restrict__ shifts,
    float* __restrict__ save_buf,
    int N, int H, int D, int K
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];
    float* work = smem;

    float* save = save_buf + (long long)block_id * K * D;   // 64-bit: N*K*D passes 2^31 at batch 2 x 4K x d_ff 8192 x 36 stages
    int x_offset = (n * H + h) * D;

    // Phase 1: Forward recomputation with saving
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        float v0, v1;
        load_pair(x_in, x_offset + 2 * pair_idx, v0, v1);
        work[SHMEM_IDX(2 * pair_idx)] = v0;
        work[SHMEM_IDX(2 * pair_idx + 1)] = v1;
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            save[k * D + 2 * pair_idx] = work[SHMEM_IDX(2 * pair_idx)];
            save[k * D + 2 * pair_idx + 1] = work[SHMEM_IDX(2 * pair_idx + 1)];
        }
        __syncthreads();

        int shift = shifts[k];
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            int p0, p1;
            compute_pair_indices(pair_idx, shift, D, p0, p1);

            float x0 = work[SHMEM_IDX(p0)];
            float x1 = work[SHMEM_IDX(p1)];

            int cs_offset = (k * H + h) * half_D + pair_idx;
            float phase = ATAN2F(x1, x0);
            float theta_eff = angles[cs_offset] + alpha[cs_offset] * phase;
            float c, s;
            sincosf(theta_eff, &s, &c);

            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            int sc_offset = (k * H + h) * D;
            r0 *= scales[sc_offset + 2 * pair_idx];
            r1 *= scales[sc_offset + 2 * pair_idx + 1];

            work[SHMEM_IDX(p0)] = r0;
            work[SHMEM_IDX(p1)] = r1;
        }
        STAGE_BARRIER(half_D);
    }

    // Phase 2: Backward pass — atomicAdd gradients directly to global memory
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        float v0, v1;
        load_pair(grad_out, x_offset + 2 * pair_idx, v0, v1);
        work[SHMEM_IDX(2 * pair_idx)] = v0;
        work[SHMEM_IDX(2 * pair_idx + 1)] = v1;
    }
    __syncthreads();

    for (int k = K - 1; k >= 0; k--) {
        int shift = shifts[k];
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            int p0, p1;
            compute_pair_indices(pair_idx, shift, D, p0, p1);

            float g_out0 = work[SHMEM_IDX(p0)];
            float g_out1 = work[SHMEM_IDX(p1)];

            float x0 = save[k * D + p0];
            float x1 = save[k * D + p1];

            int cs_offset = (k * H + h) * half_D + pair_idx;
            float phase = ATAN2F(x1, x0);
            float a_val = alpha[cs_offset];
            float theta_eff = angles[cs_offset] + a_val * phase;
            float c, s;
            sincosf(theta_eff, &s, &c);

            int sc_offset = (k * H + h) * D;
            float s0 = scales[sc_offset + 2 * pair_idx];
            float s1 = scales[sc_offset + 2 * pair_idx + 1];

            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            // Scale gradients — direct atomicAdd to global
            atomicAdd(&grad_scales[sc_offset + 2 * pair_idx], r0 * g_out0);
            atomicAdd(&grad_scales[sc_offset + 2 * pair_idx + 1], r1 * g_out1);

            float g_r0 = g_out0 * s0;
            float g_r1 = g_out1 * s1;
            float g_theta = g_r0 * (-r1) + g_r1 * r0;

            // Angle and alpha gradients — direct atomicAdd to global
            atomicAdd(&grad_angles[cs_offset], g_theta);
            atomicAdd(&grad_alpha[cs_offset], g_theta * phase);

            float g_x0 = g_r0 * c + g_r1 * s;
            float g_x1 = -g_r0 * s + g_r1 * c;

            float norm_sq = x0 * x0 + x1 * x1;
            if (norm_sq > 1e-12f) {
                float inv_norm_sq = a_val / norm_sq;
                g_x0 += g_theta * (-x1 * inv_norm_sq);
                g_x1 += g_theta * (x0 * inv_norm_sq);
            }

            work[SHMEM_IDX(p0)] = g_x0;
            work[SHMEM_IDX(p1)] = g_x1;
        }
        STAGE_BARRIER(half_D);
    }

    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        store_pair(grad_x, x_offset + 2 * pair_idx,
                   work[SHMEM_IDX(2 * pair_idx)],
                   work[SHMEM_IDX(2 * pair_idx + 1)]);
    }
}

// ---------------------------------------------------------------------------
// Precomputed trig backward kernel (alpha=0 fast path)
//
// Matches givens_fwd_precomp_kernel. Since alpha=0:
// - No atan2/sincosf in forward recomputation (use precomputed cos/sin)
// - No alpha gradient (always zero)
// - Angle gradient simplifies: g_theta flows directly to grad_angles
// ---------------------------------------------------------------------------
template<typename DataT>
LAUNCH_BOUNDS_BWD
__global__ void givens_bwd_precomp_kernel(
    const DataT* __restrict__ grad_out,
    const DataT* __restrict__ x_in,
    const float* __restrict__ cos_angles,   // precomputed cos(angles), [K, H, P]
    const float* __restrict__ sin_angles,   // precomputed sin(angles), [K, H, P]
    const float* __restrict__ scales,
    DataT* __restrict__ grad_x,
    float* __restrict__ grad_cos,           // gradient w.r.t. cos(angles)
    float* __restrict__ grad_sin,           // gradient w.r.t. sin(angles)
    float* __restrict__ grad_scales,
    const int* __restrict__ shifts,
    float* __restrict__ save_buf,
    int N, int H, int D, int K
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];
    float* work = smem;

    float* save = save_buf + (long long)block_id * K * D;   // 64-bit: N*K*D passes 2^31 at batch 2 x 4K x d_ff 8192 x 36 stages
    int x_offset = (n * H + h) * D;

    // Phase 1: Forward recomputation with saving (using precomputed trig)
    // Vectorized load from global memory
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        float v0, v1;
        load_pair(x_in, x_offset + 2 * pair_idx, v0, v1);
        work[SHMEM_IDX(2 * pair_idx)] = v0;
        work[SHMEM_IDX(2 * pair_idx + 1)] = v1;
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            save[k * D + 2 * pair_idx] = work[SHMEM_IDX(2 * pair_idx)];
            save[k * D + 2 * pair_idx + 1] = work[SHMEM_IDX(2 * pair_idx + 1)];
        }
        __syncthreads();

        int shift = shifts[k];
        int cs_base = (k * H + h) * half_D;
        int sc_base = (k * H + h) * D;
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            int p0, p1;
            compute_pair_indices(pair_idx, shift, D, p0, p1);

            float x0 = work[SHMEM_IDX(p0)];
            float x1 = work[SHMEM_IDX(p1)];

            float c = cos_angles[cs_base + pair_idx];
            float s = sin_angles[cs_base + pair_idx];

            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            r0 *= scales[sc_base + 2 * pair_idx];
            r1 *= scales[sc_base + 2 * pair_idx + 1];

            work[SHMEM_IDX(p0)] = r0;
            work[SHMEM_IDX(p1)] = r1;
        }
        STAGE_BARRIER(half_D);
    }

    // Phase 2: Backward pass
    // Vectorized load
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        float v0, v1;
        load_pair(grad_out, x_offset + 2 * pair_idx, v0, v1);
        work[SHMEM_IDX(2 * pair_idx)] = v0;
        work[SHMEM_IDX(2 * pair_idx + 1)] = v1;
    }
    __syncthreads();

    for (int k = K - 1; k >= 0; k--) {
        int shift = shifts[k];
        int cs_base = (k * H + h) * half_D;
        int sc_base = (k * H + h) * D;
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            int p0, p1;
            compute_pair_indices(pair_idx, shift, D, p0, p1);

            float g_out0 = work[SHMEM_IDX(p0)];
            float g_out1 = work[SHMEM_IDX(p1)];

            float x0 = save[k * D + p0];
            float x1 = save[k * D + p1];

            float c = cos_angles[cs_base + pair_idx];
            float s = sin_angles[cs_base + pair_idx];

            float s0 = scales[sc_base + 2 * pair_idx];
            float s1 = scales[sc_base + 2 * pair_idx + 1];

            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            // Scale gradients
            atomicAdd(&grad_scales[sc_base + 2 * pair_idx], r0 * g_out0);
            atomicAdd(&grad_scales[sc_base + 2 * pair_idx + 1], r1 * g_out1);

            float g_r0 = g_out0 * s0;
            float g_r1 = g_out1 * s1;

            // Gradient w.r.t. cos/sin (chain rule through rotation)
            // r0 = x0*c - x1*s → dr0/dc = x0, dr0/ds = -x1
            // r1 = x0*s + x1*c → dr1/dc = x1, dr1/ds = x0
            float g_c = g_r0 * x0 + g_r1 * x1;
            float g_s = g_r0 * (-x1) + g_r1 * x0;
            atomicAdd(&grad_cos[cs_base + pair_idx], g_c);
            atomicAdd(&grad_sin[cs_base + pair_idx], g_s);

            // Gradient w.r.t. input x
            float g_x0 = g_r0 * c + g_r1 * s;
            float g_x1 = -g_r0 * s + g_r1 * c;

            work[SHMEM_IDX(p0)] = g_x0;
            work[SHMEM_IDX(p1)] = g_x1;
        }
        STAGE_BARRIER(half_D);
    }

    // Vectorized store
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        store_pair(grad_x, x_offset + 2 * pair_idx,
                   work[SHMEM_IDX(2 * pair_idx)],
                   work[SHMEM_IDX(2 * pair_idx + 1)]);
    }
}

// ---------------------------------------------------------------------------
// Radix-4 Precomputed trig backward kernel
//
// Mirrors givens_fwd_radix4_precomp_kernel. Fuses pairs of stages (s, 2s)
// in both forward-recompute (Phase 1) and backward (Phase 2).
// Uses reverse_pair_idx() from givens_forward.h for pair_idx lookups.
// ---------------------------------------------------------------------------

template<typename DataT>
LAUNCH_BOUNDS_R4_BWD
__global__ void givens_bwd_radix4_precomp_kernel(
    const DataT* __restrict__ grad_out,
    const DataT* __restrict__ x_in,
    const float* __restrict__ cos_angles,
    const float* __restrict__ sin_angles,
    const float* __restrict__ scales,
    DataT* __restrict__ grad_x,
    float* __restrict__ grad_cos,
    float* __restrict__ grad_sin,
    float* __restrict__ grad_scales,
    const int* __restrict__ shifts,
    float* __restrict__ save_buf,
    int N, int H, int D, int K
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];
    float* work = smem;
    float* save = save_buf + (long long)block_id * K * D;   // 64-bit: N*K*D passes 2^31 at batch 2 x 4K x d_ff 8192 x 36 stages
    int x_offset = (n * H + h) * D;

    // ===== Phase 1: Forward recompute with saving (Radix-4 fused) =====
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        work[SHMEM_IDX(i)] = load_f(x_in, x_offset + i);
    }
    __syncthreads();

    int k = 0;
    while (k < K) {
        // Save pre-stage state
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            save[k * D + i] = work[SHMEM_IDX(i)];
        }
        if (k + 1 < K) {
            for (int i = threadIdx.x; i < D; i += blockDim.x) {
                save[(k+1) * D + i] = 0.0f;  // will be overwritten below
            }
        }
        __syncthreads();

        int s1 = -shifts[k];

        if (k + 1 < K && (-shifts[k+1]) == s1 * 2) {
            int s2 = s1 * 2;
            int group_size = s1 * 4;
            int n_groups = D / group_size;
            int n_work = n_groups * s1;
            int cs_k  = (k * H + h) * half_D;
            int sc_k  = (k * H + h) * D;
            int cs_k1 = ((k+1) * H + h) * half_D;
            int sc_k1 = ((k+1) * H + h) * D;

            for (int wi = threadIdx.x; wi < n_work; wi += blockDim.x) {
                int g = wi / s1;
                int j = wi - g * s1;
                int base = g * group_size;
                int e0 = base + j, e1 = base + j + s1, e2 = base + j + s2, e3 = base + j + s2 + s1;

                float r0 = work[SHMEM_IDX(e0)];
                float r1 = work[SHMEM_IDX(e1)];
                float r2 = work[SHMEM_IDX(e2)];
                float r3 = work[SHMEM_IDX(e3)];

                // Stage k (stride s1)
                {
                    int pi = reverse_pair_idx(e0, s1);
                    float c = cos_angles[cs_k + pi], s = sin_angles[cs_k + pi];
                    float t0 = r0*c - r1*s, t1 = r0*s + r1*c;
                    r0 = t0 * scales[sc_k + 2*pi]; r1 = t1 * scales[sc_k + 2*pi+1];
                }
                {
                    int pi = reverse_pair_idx(e2, s1);
                    float c = cos_angles[cs_k + pi], s = sin_angles[cs_k + pi];
                    float t0 = r2*c - r3*s, t1 = r2*s + r3*c;
                    r2 = t0 * scales[sc_k + 2*pi]; r3 = t1 * scales[sc_k + 2*pi+1];
                }

                // Save state between stage k and k+1
                save[(k+1) * D + e0] = r0;
                save[(k+1) * D + e1] = r1;
                save[(k+1) * D + e2] = r2;
                save[(k+1) * D + e3] = r3;

                // Stage k+1 (stride s2)
                {
                    int pi = reverse_pair_idx(e0, s2);
                    float c = cos_angles[cs_k1 + pi], s = sin_angles[cs_k1 + pi];
                    float t0 = r0*c - r2*s, t1 = r0*s + r2*c;
                    r0 = t0 * scales[sc_k1 + 2*pi]; r2 = t1 * scales[sc_k1 + 2*pi+1];
                }
                {
                    int pi = reverse_pair_idx(e1, s2);
                    float c = cos_angles[cs_k1 + pi], s = sin_angles[cs_k1 + pi];
                    float t0 = r1*c - r3*s, t1 = r1*s + r3*c;
                    r1 = t0 * scales[sc_k1 + 2*pi]; r3 = t1 * scales[sc_k1 + 2*pi+1];
                }

                work[SHMEM_IDX(e0)] = r0; work[SHMEM_IDX(e1)] = r1;
                work[SHMEM_IDX(e2)] = r2; work[SHMEM_IDX(e3)] = r3;
            }
            STAGE_BARRIER(half_D);
            k += 2;
        } else {
            // Radix-2 fallback
            int cs_base = (k * H + h) * half_D;
            int sc_base = (k * H + h) * D;
            int shift = shifts[k];
            for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
                int p0, p1;
                compute_pair_indices(pair_idx, shift, D, p0, p1);
                float x0 = work[SHMEM_IDX(p0)], x1 = work[SHMEM_IDX(p1)];
                float c = cos_angles[cs_base + pair_idx], s = sin_angles[cs_base + pair_idx];
                float r0 = x0*c - x1*s, r1 = x0*s + x1*c;
                r0 *= scales[sc_base + 2*pair_idx]; r1 *= scales[sc_base + 2*pair_idx+1];
                work[SHMEM_IDX(p0)] = r0; work[SHMEM_IDX(p1)] = r1;
            }
            STAGE_BARRIER(half_D);
            k += 1;
        }
    }

    // ===== Phase 2: Backward pass (reverse order, Radix-4 fused) =====
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        work[SHMEM_IDX(i)] = load_f(grad_out, x_offset + i);
    }
    __syncthreads();

    k = K - 1;
    while (k >= 0) {
        int s1 = -shifts[k];

        // Check if this stage was part of a Radix-4 fused pair (k and k-1)
        if (k >= 1 && (-shifts[k-1]) * 2 == s1) {
            // k-1 had stride s1/2, k has stride s1 — they were fused as (k-1, k)
            int s_small = s1 / 2;  // stride of stage k-1
            int s_big = s1;        // stride of stage k
            int group_size = s_small * 4;
            int n_groups = D / group_size;

            int cs_kbig  = (k * H + h) * half_D;
            int sc_kbig  = (k * H + h) * D;
            int cs_ksmall = ((k-1) * H + h) * half_D;
            int sc_ksmall = ((k-1) * H + h) * D;

            int n_work = n_groups * s_small;
            for (int wi = threadIdx.x; wi < n_work; wi += blockDim.x) {
                int g = wi / s_small;
                int j = wi - g * s_small;
                int base = g * group_size;
                int e0 = base + j, e1 = base + j + s_small, e2 = base + j + s_big, e3 = base + j + s_big + s_small;

                float g0 = work[SHMEM_IDX(e0)];
                float g1 = work[SHMEM_IDX(e1)];
                float g2 = work[SHMEM_IDX(e2)];
                float g3 = work[SHMEM_IDX(e3)];

                // --- Backward stage k (stride s_big): pairs (g0,g2) and (g1,g3) ---
                // Saved input to stage k is save[k * D + ...]
                {
                    int pi = reverse_pair_idx(e0, s_big);
                    float c = cos_angles[cs_kbig + pi], s = sin_angles[cs_kbig + pi];
                    float s0 = scales[sc_kbig + 2*pi], s1v = scales[sc_kbig + 2*pi+1];
                    float x0 = save[k * D + e0], x1 = save[k * D + e2];
                    float r0 = x0*c - x1*s, r1 = x0*s + x1*c;

                    atomicAdd(&grad_scales[sc_kbig + 2*pi], r0 * g0);
                    atomicAdd(&grad_scales[sc_kbig + 2*pi+1], r1 * g2);

                    float gr0 = g0 * s0, gr1 = g2 * s1v;
                    atomicAdd(&grad_cos[cs_kbig + pi], gr0*x0 + gr1*x1);
                    atomicAdd(&grad_sin[cs_kbig + pi], gr0*(-x1) + gr1*x0);

                    g0 = gr0*c + gr1*s;
                    g2 = -gr0*s + gr1*c;
                }
                {
                    int pi = reverse_pair_idx(e1, s_big);
                    float c = cos_angles[cs_kbig + pi], s = sin_angles[cs_kbig + pi];
                    float s0 = scales[sc_kbig + 2*pi], s1v = scales[sc_kbig + 2*pi+1];
                    float x0 = save[k * D + e1], x1 = save[k * D + e3];
                    float r0 = x0*c - x1*s, r1 = x0*s + x1*c;

                    atomicAdd(&grad_scales[sc_kbig + 2*pi], r0 * g1);
                    atomicAdd(&grad_scales[sc_kbig + 2*pi+1], r1 * g3);

                    float gr0 = g1 * s0, gr1 = g3 * s1v;
                    atomicAdd(&grad_cos[cs_kbig + pi], gr0*x0 + gr1*x1);
                    atomicAdd(&grad_sin[cs_kbig + pi], gr0*(-x1) + gr1*x0);

                    g1 = gr0*c + gr1*s;
                    g3 = -gr0*s + gr1*c;
                }

                // --- Backward stage k-1 (stride s_small): pairs (g0,g1) and (g2,g3) ---
                // NO BARRIER — all in registers
                {
                    int pi = reverse_pair_idx(e0, s_small);
                    float c = cos_angles[cs_ksmall + pi], s = sin_angles[cs_ksmall + pi];
                    float s0 = scales[sc_ksmall + 2*pi], s1v = scales[sc_ksmall + 2*pi+1];
                    float x0 = save[(k-1) * D + e0], x1 = save[(k-1) * D + e1];
                    float r0 = x0*c - x1*s, r1 = x0*s + x1*c;

                    atomicAdd(&grad_scales[sc_ksmall + 2*pi], r0 * g0);
                    atomicAdd(&grad_scales[sc_ksmall + 2*pi+1], r1 * g1);

                    float gr0 = g0 * s0, gr1 = g1 * s1v;
                    atomicAdd(&grad_cos[cs_ksmall + pi], gr0*x0 + gr1*x1);
                    atomicAdd(&grad_sin[cs_ksmall + pi], gr0*(-x1) + gr1*x0);

                    g0 = gr0*c + gr1*s;
                    g1 = -gr0*s + gr1*c;
                }
                {
                    int pi = reverse_pair_idx(e2, s_small);
                    float c = cos_angles[cs_ksmall + pi], s = sin_angles[cs_ksmall + pi];
                    float s0 = scales[sc_ksmall + 2*pi], s1v = scales[sc_ksmall + 2*pi+1];
                    float x0 = save[(k-1) * D + e2], x1 = save[(k-1) * D + e3];
                    float r0 = x0*c - x1*s, r1 = x0*s + x1*c;

                    atomicAdd(&grad_scales[sc_ksmall + 2*pi], r0 * g2);
                    atomicAdd(&grad_scales[sc_ksmall + 2*pi+1], r1 * g3);

                    float gr0 = g2 * s0, gr1 = g3 * s1v;
                    atomicAdd(&grad_cos[cs_ksmall + pi], gr0*x0 + gr1*x1);
                    atomicAdd(&grad_sin[cs_ksmall + pi], gr0*(-x1) + gr1*x0);

                    g2 = gr0*c + gr1*s;
                    g3 = -gr0*s + gr1*c;
                }

                work[SHMEM_IDX(e0)] = g0; work[SHMEM_IDX(e1)] = g1;
                work[SHMEM_IDX(e2)] = g2; work[SHMEM_IDX(e3)] = g3;
            }
            STAGE_BARRIER(half_D);
            k -= 2;
        } else {
            // Radix-2 fallback
            int cs_base = (k * H + h) * half_D;
            int sc_base = (k * H + h) * D;
            int shift = shifts[k];
            for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
                int p0, p1;
                compute_pair_indices(pair_idx, shift, D, p0, p1);

                float g_out0 = work[SHMEM_IDX(p0)];
                float g_out1 = work[SHMEM_IDX(p1)];
                float x0 = save[k * D + p0];
                float x1 = save[k * D + p1];
                float c = cos_angles[cs_base + pair_idx];
                float s = sin_angles[cs_base + pair_idx];
                float s0 = scales[sc_base + 2*pair_idx];
                float s1v = scales[sc_base + 2*pair_idx+1];

                float r0 = x0*c - x1*s, r1 = x0*s + x1*c;

                atomicAdd(&grad_scales[sc_base + 2*pair_idx], r0 * g_out0);
                atomicAdd(&grad_scales[sc_base + 2*pair_idx+1], r1 * g_out1);

                float gr0 = g_out0 * s0, gr1 = g_out1 * s1v;
                atomicAdd(&grad_cos[cs_base + pair_idx], gr0*x0 + gr1*x1);
                atomicAdd(&grad_sin[cs_base + pair_idx], gr0*(-x1) + gr1*x0);

                work[SHMEM_IDX(p0)] = gr0*c + gr1*s;
                work[SHMEM_IDX(p1)] = -gr0*s + gr1*c;
            }
            STAGE_BARRIER(half_D);
            k -= 1;
        }
    }

    // Store grad_x
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        store_f(grad_x, x_offset + i, work[SHMEM_IDX(i)]);
    }
}

#endif // GIVENS_BACKWARD_H
