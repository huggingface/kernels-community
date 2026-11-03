/*
 * Tiled Coupling Forward Kernel for Attention (Clifford F_feat=4)
 *
 * Matches PyTorch reference _coupling_project when perms=None (identity permutation).
 *
 * Tiled design: each block processes TPB=16 tokens sharing one coupling tile
 * per stage. This eliminates ~94% of redundant global memory tile loads compared
 * to the 1-token-per-block design.
 *
 * blockDim = (half_D, TPB) = (32, 16) = 512 threads
 * gridDim  = B * ceil(N/TPB) * H
 *
 * Each token's 32 threads form one warp, so intra-token sync uses __syncwarp().
 * Full __syncthreads() only needed for coupling tile loads shared across tokens.
 *
 * Fuses K stages of:
 *   1. Direct pair extraction (identity permutation): (0,1), (2,3), ...
 *   2. Clifford feature compute [S0, V1, V2, B12]
 *   3. Coupling einsum: modulation[p] = sum_j features[j] * coupling[j, p]
 *   4. tanh + base angle -> effective rotation angle
 *   5. Givens rotation
 *   6. Scale
 *
 * Input x is [N, H, D] (shared across B projections)
 * Output is [B, N, H, D]
 * Params (angles, coupling, scales) are [B, K, H, ...]
 *
 * Shared memory layout (per block):
 *   coupling_tile[4*P*P]        -- shared across all TPB tokens (unpadded)
 *   work[TPB * padded_D]        -- D-vector per token with bank-conflict-free padding
 *   smem_x0[TPB * P]            -- extracted pair x0 per token
 *   smem_x1[TPB * P]            -- extracted pair x1 per token
 *   features[TPB * 4*P]         -- Clifford features per token
 */

#ifndef GIVENS_COUPLING_FORWARD_H
#define GIVENS_COUPLING_FORWARD_H

#include "givens_common.h"

LAUNCH_BOUNDS_COUPLING_FWD
__global__ void givens_coupling_fwd_tiled_kernel(
    const float* __restrict__ x,           // [N, H, D] -- shared input
    const float* __restrict__ angles,       // [B, K, H, P] base rotation angles
    const float* __restrict__ coupling,    // [B, K, H, 4*P, P]
    const float* __restrict__ scales,      // [B, K, H, D]
    const int* __restrict__ shifts,        // [K] butterfly stride/shift per stage
    float* __restrict__ out,               // [B, N, H, D]
    int N, int H, int D, int K, int B
) {
    int half_D = D / 2;
    int f_feat = 4;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int coupling_size = f_feat * half_D * half_D;

    // 2D thread block: threadIdx.x = pair index, threadIdx.y = token slot
    int p = threadIdx.x;            // [0, half_D-1]
    int token_slot = threadIdx.y;   // [0, TPB-1]
    int TPB = blockDim.y;

    // Block -> (b, n_tile, h)
    int n_tiles = (N + TPB - 1) / TPB;
    int block_id = blockIdx.x;
    int b = block_id / (n_tiles * H);
    int rest = block_id % (n_tiles * H);
    int n_tile = rest / H;
    int h = rest % H;

    int my_n = n_tile * TPB + token_slot;
    bool valid = (my_n < N);

    // Shared memory layout
    extern __shared__ float smem[];
    float* coupling_tile = smem;
    float* work_base     = smem + coupling_size;
    float* smem_x0_base  = work_base + TPB * padded_D;
    float* smem_x1_base  = smem_x0_base + TPB * half_D;
    float* feat_base     = smem_x1_base + TPB * half_D;

    // Per-token pointers
    float* my_work     = work_base + token_slot * padded_D;
    float* my_x0       = smem_x0_base + token_slot * half_D;
    float* my_x1       = smem_x1_base + token_slot * half_D;
    float* my_features = feat_base + token_slot * f_feat * half_D;

    // Load input x[my_n, h, :] into per-token work buffer
    if (valid) {
        int x_offset = (my_n * H + h) * D;
        my_work[SHMEM_IDX(2 * p)]     = x[x_offset + 2 * p];
        my_work[SHMEM_IDX(2 * p + 1)] = x[x_offset + 2 * p + 1];
    }
    __syncthreads();

    // Process K stages
    for (int k = 0; k < K; k++) {
        // --- Cooperative tile load: all 512 threads load coupling[b,k,h] ---
        // Forward einsum accesses coupling_tile[j * half_D + p] where p varies
        // across threads in a warp — each thread hits a different bank, no padding needed.
        int cp_base = ((b * K + k) * H + h) * coupling_size;
        int tid_linear = threadIdx.x + threadIdx.y * blockDim.x;
        int total_threads = blockDim.x * blockDim.y;
        for (int i = tid_linear; i < coupling_size; i += total_threads)
            coupling_tile[i] = coupling[cp_base + i];
        __syncthreads();

        // --- Extract pairs via butterfly permutation: (p0, p1) = perm(p, shifts[k]) ---
        int p0_perm, p1_perm;
        compute_pair_indices(p, shifts[k], D, p0_perm, p1_perm);
        if (valid) {
            my_x0[p] = my_work[SHMEM_IDX(p0_perm)];
            my_x1[p] = my_work[SHMEM_IDX(p1_perm)];
        }
        // Intra-token sync: all 32 threads of one token are in the same warp
        STAGE_BARRIER(half_D);

        // --- Compute Clifford features ---
        if (valid) {
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            float x0n = my_x0[next];
            float x1n = my_x1[next];

            my_features[p]                  = x0 * x0 + x1 * x1;   // S0
            my_features[half_D + p]         = x0;                    // V1
            my_features[2 * half_D + p]     = x1;                    // V2
            my_features[3 * half_D + p]     = x0 * x1n - x1 * x0n;  // B12
        }
        STAGE_BARRIER(half_D);

        // --- L2-normalize features (matches PyTorch: feat / (feat.norm + eps)) ---
        if (valid) {
            // Each thread accumulates partial sum-of-squares over 4 feature channels
            float sq = 0.0f;
            sq += my_features[p]              * my_features[p];
            sq += my_features[half_D + p]     * my_features[half_D + p];
            sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
            sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];

            // Per-segment butterfly reduction. Block dim is (half_D, TPB) so when
            // half_D < WARP_SIZE multiple token segments share a wavefront. SHFL_XOR
            // bounces values within each half_D-sized segment without leaking across
            // segments. After the loop, every lane within a segment holds that
            // segment's full sum. (SHFL_DOWN+SHFL_BROADCAST(0) was buggy: it broadcast
            // wavefront-lane-0 to all lanes, contaminating tokens 1..N in the warp.)
            for (int offset = half_D / 2; offset > 0; offset >>= 1)
                sq += SHFL_XOR(sq, offset);

            float inv_norm = rsqrtf(sq + 1e-6f);
            my_features[p]                  *= inv_norm;
            my_features[half_D + p]         *= inv_norm;
            my_features[2 * half_D + p]     *= inv_norm;
            my_features[3 * half_D + p]     *= inv_norm;
        }
        STAGE_BARRIER(half_D);

        // --- Coupling einsum + rotation + scale ---
        if (valid) {
            // Register-blocked 4-way accumulation
            int total = f_feat * half_D;
            float mod0 = 0.0f, mod1 = 0.0f, mod2 = 0.0f, mod3 = 0.0f;
            int j = 0;
            #pragma unroll 4
            for (; j + 3 < total; j += 4) {
                mod0 += my_features[j]     * coupling_tile[j       * half_D + p];
                mod1 += my_features[j + 1] * coupling_tile[(j + 1) * half_D + p];
                mod2 += my_features[j + 2] * coupling_tile[(j + 2) * half_D + p];
                mod3 += my_features[j + 3] * coupling_tile[(j + 3) * half_D + p];
            }
            float modulation = mod0 + mod1 + mod2 + mod3;
            for (; j < total; j++)
                modulation += my_features[j] * coupling_tile[j * half_D + p];

            // Effective angle: learned + content-dependent correction
            int a_off = ((b * K + k) * H + h) * half_D + p;
            float correction = FAST_TANHF(modulation);

            // cos/sin of learned base angle (same for all tokens)
            float c_base, s_base;
            sincosf(angles[a_off], &s_base, &c_base);
            // exact cos/sin of the tanh correction (was a Cayley approximation, T-109)
            float c_corr, s_corr;
            sincosf(correction, &s_corr, &c_corr);
            // Combine: cos(a+b) = ca*cb - sa*sb, sin(a+b) = sa*cb + ca*sb
            float c = c_base * c_corr - s_base * s_corr;
            float s = s_base * c_corr + c_base * s_corr;

            // Givens rotation
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            // Scale (eager applies scales[k][2p, 2p+1] in the permuted frame).
            int s_off = ((b * K + k) * H + h) * D;
            r0 *= scales[s_off + 2 * p];
            r1 *= scales[s_off + 2 * p + 1];

            // Write back through the inverse butterfly: store at the source positions
            // (p0_perm, p1_perm) so the output equals eager's perm -> rotate -> scale -> inv_perm.
            my_work[SHMEM_IDX(p0_perm)] = r0;
            my_work[SHMEM_IDX(p1_perm)] = r1;
        }
        __syncthreads();
    }

    // Write output: out[b, my_n, h, :]
    if (valid) {
        int out_offset = ((b * N + my_n) * H + h) * D;
        out[out_offset + 2 * p]     = my_work[SHMEM_IDX(2 * p)];
        out[out_offset + 2 * p + 1] = my_work[SHMEM_IDX(2 * p + 1)];
    }
}

#endif // GIVENS_COUPLING_FORWARD_H
