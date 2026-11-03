/*
 * Givens Rotation Kernel - Mixture-of-Rotations (MoR)
 * Soft expert blending for FFN
 */

#ifndef GIVENS_MOR_H
#define GIVENS_MOR_H

#include "givens_common.h"

template<typename DataT>
LAUNCH_BOUNDS_FWD
__global__ void givens_mor_fwd_kernel(
    const DataT* __restrict__ x,
    const float* __restrict__ expert_angles,
    const float* __restrict__ expert_scales,
    const float* __restrict__ router_weights,
    DataT* __restrict__ out,
    const int* __restrict__ shifts,
    int N, int H, int D, int K, int E
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];
    float* work = smem;
    float* rw = smem + SHMEM_IDX(D);

    int x_offset = (n * H + h) * D;
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        float v0, v1;
        load_pair(x, x_offset + 2 * pair_idx, v0, v1);
        work[SHMEM_IDX(2 * pair_idx)] = v0;
        work[SHMEM_IDX(2 * pair_idx + 1)] = v1;
    }

    if (threadIdx.x < E) {
        rw[threadIdx.x] = router_weights[n * E + threadIdx.x];
    }
    __syncthreads();

    int kh_stride = K * H * half_D;
    int kh_stride_s = K * H * D;

    for (int k = 0; k < K; k++) {
        int shift = shifts[k];
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            int p0, p1;
            compute_pair_indices(pair_idx, shift, D, p0, p1);

            float x0 = work[SHMEM_IDX(p0)];
            float x1 = work[SHMEM_IDX(p1)];

            int ang_base = (k * H + h) * half_D + pair_idx;
            float theta = 0.0f;
            for (int e = 0; e < E; e++) {
                theta += rw[e] * expert_angles[e * kh_stride + ang_base];
            }

            float c, s;
            sincosf(theta, &s, &c);
            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            int sc_base = (k * H + h) * D;
            float s0 = 0.0f, s1 = 0.0f;
            for (int e = 0; e < E; e++) {
                int sc_off = e * kh_stride_s + sc_base;
                s0 += rw[e] * expert_scales[sc_off + 2 * pair_idx];
                s1 += rw[e] * expert_scales[sc_off + 2 * pair_idx + 1];
            }

            work[SHMEM_IDX(p0)] = r0 * s0;
            work[SHMEM_IDX(p1)] = r1 * s1;
        }
        STAGE_BARRIER(half_D);
    }

    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        store_pair(out, x_offset + 2 * pair_idx,
                   work[SHMEM_IDX(2 * pair_idx)],
                   work[SHMEM_IDX(2 * pair_idx + 1)]);
    }
}

template<typename DataT>
LAUNCH_BOUNDS_MOR_BWD
__global__ void givens_mor_bwd_kernel(
    const DataT* __restrict__ grad_out,
    const DataT* __restrict__ x_in,
    const float* __restrict__ expert_angles,
    const float* __restrict__ expert_scales,
    const float* __restrict__ router_weights,
    DataT* __restrict__ grad_x,
    float* __restrict__ grad_expert_angles,
    float* __restrict__ grad_expert_scales,
    float* __restrict__ grad_router,
    const int* __restrict__ shifts,
    float* __restrict__ save_buf,
    int N, int H, int D, int K, int E
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];
    float* work = smem;
    float* rw = smem + SHMEM_IDX(D);
    float* grw = rw + E;

    float* save = save_buf + (long long)block_id * K * D;   // 64-bit: N*K*D passes 2^31 at batch 2 x 4K x d_ff 8192 x 36 stages
    int x_offset = (n * H + h) * D;
    int kh_stride = K * H * half_D;
    int kh_stride_s = K * H * D;

    if (threadIdx.x < E) {
        rw[threadIdx.x] = router_weights[n * E + threadIdx.x];
        grw[threadIdx.x] = 0.0f;
    }

    // Phase 1: Forward recomputation
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

            int ang_base = (k * H + h) * half_D + pair_idx;
            float theta = 0.0f;
            for (int e = 0; e < E; e++) {
                theta += rw[e] * expert_angles[e * kh_stride + ang_base];
            }
            float c, s;
            sincosf(theta, &s, &c);
            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            int sc_base = (k * H + h) * D;
            float s0 = 0.0f, s1 = 0.0f;
            for (int e = 0; e < E; e++) {
                int sc_off = e * kh_stride_s + sc_base;
                s0 += rw[e] * expert_scales[sc_off + 2 * pair_idx];
                s1 += rw[e] * expert_scales[sc_off + 2 * pair_idx + 1];
            }

            work[SHMEM_IDX(p0)] = r0 * s0;
            work[SHMEM_IDX(p1)] = r1 * s1;
        }
        STAGE_BARRIER(half_D);
    }

    // Phase 2: Backward pass
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

            int ang_base = (k * H + h) * half_D + pair_idx;
            float theta = 0.0f;
            for (int e = 0; e < E; e++) {
                theta += rw[e] * expert_angles[e * kh_stride + ang_base];
            }
            float c, s;
            sincosf(theta, &s, &c);
            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            int sc_base = (k * H + h) * D;
            float s0_val = 0.0f, s1_val = 0.0f;
            for (int e = 0; e < E; e++) {
                int sc_off = e * kh_stride_s + sc_base;
                s0_val += rw[e] * expert_scales[sc_off + 2 * pair_idx];
                s1_val += rw[e] * expert_scales[sc_off + 2 * pair_idx + 1];
            }

            float g_r0 = g_out0 * s0_val;
            float g_r1 = g_out1 * s1_val;
            float g_s0_blend = g_out0 * r0;
            float g_s1_blend = g_out1 * r1;

            for (int e = 0; e < E; e++) {
                int sc_off = e * kh_stride_s + sc_base;
                atomicAdd(&grad_expert_scales[sc_off + 2 * pair_idx], rw[e] * g_s0_blend);
                atomicAdd(&grad_expert_scales[sc_off + 2 * pair_idx + 1], rw[e] * g_s1_blend);
            }

            float g_theta = g_r0 * (-r1) + g_r1 * r0;
            for (int e = 0; e < E; e++) {
                atomicAdd(&grad_expert_angles[e * kh_stride + ang_base], rw[e] * g_theta);
            }

            for (int e = 0; e < E; e++) {
                float gr_e = g_theta * expert_angles[e * kh_stride + ang_base];
                int sc_off = e * kh_stride_s + sc_base;
                gr_e += g_s0_blend * expert_scales[sc_off + 2 * pair_idx];
                gr_e += g_s1_blend * expert_scales[sc_off + 2 * pair_idx + 1];
                // Warp reduction: sum 32 threads before shared-mem atomic (32x fewer atomics)
                gr_e = warp_reduce_sum(gr_e);
                if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
                    atomicAdd(&grw[e], gr_e);
                }
            }

            float g_x0 = g_r0 * c + g_r1 * s;
            float g_x1 = -g_r0 * s + g_r1 * c;
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

    __syncthreads();
    if (threadIdx.x < E) {
        atomicAdd(&grad_router[n * E + threadIdx.x], grw[threadIdx.x]);
    }
}

#endif // GIVENS_MOR_H
