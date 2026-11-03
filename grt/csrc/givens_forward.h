/*
 * Givens Rotation Kernel - Forward Pass
 * Standard, tiled, and persistent thread block variants
 */

#ifndef GIVENS_FORWARD_H
#define GIVENS_FORWARD_H

#include "givens_common.h"

// ---------------------------------------------------------------------------
// Standard forward kernel (FFN)
// ---------------------------------------------------------------------------
template<typename DataT>
LAUNCH_BOUNDS_FWD
__global__ void givens_fwd_kernel(
    const DataT* __restrict__ x,
    const float* __restrict__ angles,
    const float* __restrict__ alpha,
    const float* __restrict__ scales,
    DataT* __restrict__ out,
    const int* __restrict__ shifts,
    int N, int H, int D, int K
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];

    int x_offset = (n * H + h) * D;
    // Vectorized load
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        float v0, v1;
        load_pair(x, x_offset + 2 * pair_idx, v0, v1);
        smem[SHMEM_IDX(2 * pair_idx)] = v0;
        smem[SHMEM_IDX(2 * pair_idx + 1)] = v1;
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        int shift = shifts[k];
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            int p0, p1;
            compute_pair_indices(pair_idx, shift, D, p0, p1);

            float x0 = smem[SHMEM_IDX(p0)];
            float x1 = smem[SHMEM_IDX(p1)];

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

            smem[SHMEM_IDX(p0)] = r0;
            smem[SHMEM_IDX(p1)] = r1;
        }
        STAGE_BARRIER(half_D);
    }

    // Vectorized store
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        store_pair(out, x_offset + 2 * pair_idx,
                   smem[SHMEM_IDX(2 * pair_idx)],
                   smem[SHMEM_IDX(2 * pair_idx + 1)]);
    }
}

// ---------------------------------------------------------------------------
// Multi-block tiled forward kernel for large D (Phase 2.1)
// ---------------------------------------------------------------------------
template<typename DataT>
LAUNCH_BOUNDS_FWD
__global__ void givens_fwd_tiled_kernel(
    const DataT* __restrict__ x,
    const float* __restrict__ angles,
    const float* __restrict__ alpha,
    const float* __restrict__ scales,
    DataT* __restrict__ out,
    const int* __restrict__ shifts,
    int N, int H, int D, int K, int tiles_per_th
) {
    int th_id = blockIdx.x / tiles_per_th;
    int tile_id = blockIdx.x % tiles_per_th;
    int n = th_id / H;
    int h = th_id % H;

    if (n >= N) return;

    int D_start = tile_id * D_TILE;
    int D_end = min(D_start + D_TILE, D);
    int local_D = D_end - D_start;
    int local_half_D = local_D / 2;

    if (local_D <= 0) return;

    extern __shared__ float smem[];
    int x_offset = (n * H + h) * D + D_start;

    for (int i = threadIdx.x; i < local_D; i += blockDim.x) {
        smem[SHMEM_IDX(i)] = load_f(x, x_offset + i);
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        int shift = shifts[k];
        for (int pair_idx = threadIdx.x; pair_idx < local_half_D; pair_idx += blockDim.x) {
            int global_pair = D_start / 2 + pair_idx;
            int p0_global, p1_global;
            compute_pair_indices(global_pair, shift, D, p0_global, p1_global);

            if (p0_global >= D_start && p0_global < D_end &&
                p1_global >= D_start && p1_global < D_end) {
                int p0 = p0_global - D_start;
                int p1 = p1_global - D_start;

                float x0 = smem[SHMEM_IDX(p0)];
                float x1 = smem[SHMEM_IDX(p1)];

                int cs_offset = (k * H + h) * (D / 2) + global_pair;
                float phase = ATAN2F(x1, x0);
                float theta_eff = angles[cs_offset] + alpha[cs_offset] * phase;
                float c, s;
                sincosf(theta_eff, &s, &c);

                float r0 = x0 * c - x1 * s;
                float r1 = x0 * s + x1 * c;

                int sc_offset = (k * H + h) * D;
                r0 *= scales[sc_offset + p0_global];
                r1 *= scales[sc_offset + p1_global];

                smem[SHMEM_IDX(p0)] = r0;
                smem[SHMEM_IDX(p1)] = r1;
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < local_D; i += blockDim.x) {
        store_f(out, x_offset + i, smem[SHMEM_IDX(i)]);
    }
}

// ---------------------------------------------------------------------------
// Persistent thread block forward kernel for small N (Phase 3.2)
// ---------------------------------------------------------------------------
template<typename DataT>
LAUNCH_BOUNDS_FWD
__global__ void givens_fwd_persistent_kernel(
    const DataT* __restrict__ x,
    const float* __restrict__ angles,
    const float* __restrict__ alpha,
    const float* __restrict__ scales,
    DataT* __restrict__ out,
    const int* __restrict__ shifts,
    int N, int H, int D, int K
) {
    int th_id = blockIdx.x;
    int total_blocks = gridDim.x;
    int half_D = D / 2;

    extern __shared__ float smem[];

    for (int start_th = th_id; start_th < N * H; start_th += total_blocks) {
        int n = start_th / H;
        int h = start_th % H;
        if (n >= N) continue;

        int x_offset = (n * H + h) * D;
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            float v0, v1;
            load_pair(x, x_offset + 2 * pair_idx, v0, v1);
            smem[SHMEM_IDX(2 * pair_idx)] = v0;
            smem[SHMEM_IDX(2 * pair_idx + 1)] = v1;
        }
        __syncthreads();

        for (int k = 0; k < K; k++) {
            int shift = shifts[k];
            for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
                int p0, p1;
                compute_pair_indices(pair_idx, shift, D, p0, p1);

                float x0 = smem[SHMEM_IDX(p0)];
                float x1 = smem[SHMEM_IDX(p1)];

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

                smem[SHMEM_IDX(p0)] = r0;
                smem[SHMEM_IDX(p1)] = r1;
            }
            STAGE_BARRIER(half_D);
        }

        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            store_pair(out, x_offset + 2 * pair_idx,
                       smem[SHMEM_IDX(2 * pair_idx)],
                       smem[SHMEM_IDX(2 * pair_idx + 1)]);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Precomputed trig forward kernel (alpha=0 fast path)
//
// When alpha=0 (FFN path), theta_eff = angles[idx] — no input dependence.
// Precompute cos/sin of all angles ONCE in shared memory, then each token
// just does 2 FMA per pair instead of atan2f + sincosf (~60→8 cycles).
// ---------------------------------------------------------------------------
template<typename DataT>
LAUNCH_BOUNDS_FWD
__global__ void givens_fwd_precomp_kernel(
    const DataT* __restrict__ x,
    const float* __restrict__ cos_angles,   // precomputed cos(angles), [K, H, P]
    const float* __restrict__ sin_angles,   // precomputed sin(angles), [K, H, P]
    const float* __restrict__ scales,
    DataT* __restrict__ out,
    const int* __restrict__ shifts,
    int N, int H, int D, int K
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];

    int x_offset = (n * H + h) * D;
    // Vectorized load: float2 for 2x bandwidth on global memory
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        float v0, v1;
        load_pair(x, x_offset + 2 * pair_idx, v0, v1);
        smem[SHMEM_IDX(2 * pair_idx)] = v0;
        smem[SHMEM_IDX(2 * pair_idx + 1)] = v1;
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        int shift = shifts[k];
        int cs_base = (k * H + h) * half_D;
        int sc_base = (k * H + h) * D;
        for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
            int p0, p1;
            compute_pair_indices(pair_idx, shift, D, p0, p1);

            float x0 = smem[SHMEM_IDX(p0)];
            float x1 = smem[SHMEM_IDX(p1)];

            // Precomputed trig — no atan2, no sincosf in hot loop
            float c = cos_angles[cs_base + pair_idx];
            float s = sin_angles[cs_base + pair_idx];

            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            r0 *= scales[sc_base + 2 * pair_idx];
            r1 *= scales[sc_base + 2 * pair_idx + 1];

            smem[SHMEM_IDX(p0)] = r0;
            smem[SHMEM_IDX(p1)] = r1;
        }
        STAGE_BARRIER(half_D);
    }

    // Vectorized store
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        store_pair(out, x_offset + 2 * pair_idx,
                   smem[SHMEM_IDX(2 * pair_idx)],
                   smem[SHMEM_IDX(2 * pair_idx + 1)]);
    }
}

// ---------------------------------------------------------------------------
// Radix-4 Precomputed trig forward kernel
//
// Fuses pairs of consecutive stages (stride s, 2s) by having each thread
// own a group of 4*s elements in registers. Both stages execute without
// shared memory round-trips or barriers between them.
//
// For K=21, D=256: reduces barriers from 21 to 12 (43% fewer).
// Stage pairs (1,2), (4,8), (16,32) fuse; stride 64 stays Radix-2.
//
// pair_idx reverse mapping (given element p0, stride s):
//   in_block = p0 % s
//   pair_idx = ((p0 - in_block) >> 1) + in_block
// scales indexed as: scales[sc_base + 2*pair_idx], scales[sc_base + 2*pair_idx + 1]
// ---------------------------------------------------------------------------

// Reverse compute_pair_indices: given element position p0 and stride s, find pair_idx
__device__ __forceinline__ int reverse_pair_idx(int p0, int s) {
    int in_block = p0 & (s - 1);  // p0 % s (s is power of 2)
    return ((p0 - in_block) >> 1) + in_block;
}

template<typename DataT>
LAUNCH_BOUNDS_R4_FWD
__global__ void givens_fwd_radix4_precomp_kernel(
    const DataT* __restrict__ x,
    const float* __restrict__ cos_angles,   // [K, H, P]
    const float* __restrict__ sin_angles,   // [K, H, P]
    const float* __restrict__ scales,       // [K, H, D]
    DataT* __restrict__ out,
    const int* __restrict__ shifts,
    int N, int H, int D, int K
) {
    int block_id = blockIdx.x;
    int n = block_id / H;
    int h = block_id % H;
    int half_D = D / 2;

    if (n >= N) return;

    extern __shared__ float smem[];
    int x_offset = (n * H + h) * D;

    // Load full vector into shared memory
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        smem[SHMEM_IDX(i)] = load_f(x, x_offset + i);
    }
    __syncthreads();

    int k = 0;
    while (k < K) {
        int s1 = -shifts[k];

        // Try Radix-4: fuse stages k and k+1 if strides are s and 2s
        if (k + 1 < K && (-shifts[k+1]) == s1 * 2) {
            int s2 = s1 * 2;
            int group_size = s1 * 4;
            int n_groups = D / group_size;
            int n_work = n_groups * s1;  // = D/4 always

            int cs_k  = (k * H + h) * half_D;
            int sc_k  = (k * H + h) * D;
            int cs_k1 = ((k+1) * H + h) * half_D;
            int sc_k1 = ((k+1) * H + h) * D;

            for (int wi = threadIdx.x; wi < n_work; wi += blockDim.x) {
                int g = wi / s1;           // which group
                int j = wi - g * s1;       // offset within group (wi % s1)
                int base = g * group_size;

                // 4 element positions: stride-s1 apart, offset by j
                int e0 = base + j;
                int e1 = base + j + s1;
                int e2 = base + j + s2;
                int e3 = base + j + s2 + s1;

                // Load 4 elements into registers
                float r0 = smem[SHMEM_IDX(e0)];
                float r1 = smem[SHMEM_IDX(e1)];
                float r2 = smem[SHMEM_IDX(e2)];
                float r3 = smem[SHMEM_IDX(e3)];

                // --- Stage k (stride s1): pairs (r0,r1) and (r2,r3) ---
                {
                    int pi = reverse_pair_idx(e0, s1);
                    float c = cos_angles[cs_k + pi];
                    float s = sin_angles[cs_k + pi];
                    float t0 = r0 * c - r1 * s;
                    float t1 = r0 * s + r1 * c;
                    r0 = t0 * scales[sc_k + 2 * pi];
                    r1 = t1 * scales[sc_k + 2 * pi + 1];
                }
                {
                    int pi = reverse_pair_idx(e2, s1);
                    float c = cos_angles[cs_k + pi];
                    float s = sin_angles[cs_k + pi];
                    float t0 = r2 * c - r3 * s;
                    float t1 = r2 * s + r3 * c;
                    r2 = t0 * scales[sc_k + 2 * pi];
                    r3 = t1 * scales[sc_k + 2 * pi + 1];
                }

                // --- Stage k+1 (stride s2): pairs (r0,r2) and (r1,r3) ---
                // NO BARRIER — all 4 elements in registers
                {
                    int pi = reverse_pair_idx(e0, s2);
                    float c = cos_angles[cs_k1 + pi];
                    float s = sin_angles[cs_k1 + pi];
                    float t0 = r0 * c - r2 * s;
                    float t1 = r0 * s + r2 * c;
                    r0 = t0 * scales[sc_k1 + 2 * pi];
                    r2 = t1 * scales[sc_k1 + 2 * pi + 1];
                }
                {
                    int pi = reverse_pair_idx(e1, s2);
                    float c = cos_angles[cs_k1 + pi];
                    float s = sin_angles[cs_k1 + pi];
                    float t0 = r1 * c - r3 * s;
                    float t1 = r1 * s + r3 * c;
                    r1 = t0 * scales[sc_k1 + 2 * pi];
                    r3 = t1 * scales[sc_k1 + 2 * pi + 1];
                }

                // Store back to shared memory
                smem[SHMEM_IDX(e0)] = r0;
                smem[SHMEM_IDX(e1)] = r1;
                smem[SHMEM_IDX(e2)] = r2;
                smem[SHMEM_IDX(e3)] = r3;
            }
            STAGE_BARRIER(half_D);
            k += 2;

        } else {
            // Fallback: standard Radix-2 precomp for this stage
            int cs_base = (k * H + h) * half_D;
            int sc_base = (k * H + h) * D;
            int shift = shifts[k];
            for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
                int p0, p1;
                compute_pair_indices(pair_idx, shift, D, p0, p1);

                float x0 = smem[SHMEM_IDX(p0)];
                float x1 = smem[SHMEM_IDX(p1)];

                float c = cos_angles[cs_base + pair_idx];
                float s = sin_angles[cs_base + pair_idx];
                float r0 = x0 * c - x1 * s;
                float r1 = x0 * s + x1 * c;
                r0 *= scales[sc_base + 2 * pair_idx];
                r1 *= scales[sc_base + 2 * pair_idx + 1];

                smem[SHMEM_IDX(p0)] = r0;
                smem[SHMEM_IDX(p1)] = r1;
            }
            STAGE_BARRIER(half_D);
            k += 1;
        }
    }

    // Vectorized store
    for (int pair_idx = threadIdx.x; pair_idx < half_D; pair_idx += blockDim.x) {
        store_pair(out, x_offset + 2 * pair_idx,
                   smem[SHMEM_IDX(2 * pair_idx)],
                   smem[SHMEM_IDX(2 * pair_idx + 1)]);
    }
}

#endif // GIVENS_FORWARD_H
