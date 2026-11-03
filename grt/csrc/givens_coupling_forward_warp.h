/*
 * Warp-Shuffle Coupling Forward Kernel for Attention  (HYBRID variant)
 *
 * Specialized for half_D <= WARP_SIZE (i.e., D <= 64).
 *
 * Hybrid data placement (informed by benchmarking the pure-shfl version,
 * which was 1.53x SLOWER than tiled because the einsum broadcast loop cost
 * 128 shfl/stage — far more than the barriers it saved):
 *
 *   IN REGISTERS (via __shfl, no barriers):
 *     - D-vector (v0, v1) — butterfly permutation via SHFL_BROADCAST
 *     - Extracted pair (x0, x1)
 *     - Clifford features (S0, V1, V2, B12) — neighbor via SHFL_BROADCAST
 *     - L2-norm reduction via SHFL_XOR
 *     - Rotation output (r0, r1) — inverse butterfly via SHFL_BROADCAST
 *
 *   IN SHARED MEMORY:
 *     - coupling_tile[4*P*P]   — shared across all TPB tokens, loaded once/stage
 *     - my_features[4*P]       — per-token feature spill, written by the 32
 *                                lanes of the token then read back by the
 *                                einsum.  Replaces 128 shfl broadcasts with
 *                                128 bank-conflict-free smem reads.
 *
 * This captures the barrier savings of the pure-warp design (3 of 5
 * barriers/stage eliminated: pair-extract, feature-compute, rotation-writeback)
 * without paying the shfl-broadcast penalty on the einsum that dominates
 * the stage cost.  Net per-stage: ~90 cycles saved vs tiled, ~500 cycles
 * saved vs pure-warp.
 *
 * blockDim = (half_D, TPB) = (32, TPB)
 * gridDim  = B * ceil(N/TPB) * H
 *
 * Constraint: half_D == WARP_SIZE == 32  (D == 64).
 */

#ifndef GIVENS_COUPLING_FORWARD_WARP_H
#define GIVENS_COUPLING_FORWARD_WARP_H

#include "givens_common.h"

// ---------------------------------------------------------------------------
// Warp-level butterfly shuffle helpers
//
// Thread p holds elements (2p, 2p+1).  A butterfly at stride s pairs
// element e with element e+s (or e-s).  We need to figure out:
//   1. Which two pairs this thread participates in (one for each element)
//   2. Which thread holds the partner element
//   3. Whether the partner is the other thread's v0 or v1
//
// For stride s and pair_idx p:
//   p0 = ((p & ~(s-1)) << 1) | (p & (s-1))
//   p1 = p0 + s
//
// Thread that owns element e: e >> 1  (integer divide by 2)
// Which register (v0 or v1): e & 1
// ---------------------------------------------------------------------------

// For a given pair_idx and stride, compute:
//   partner_lane: which thread holds the partner element
//   my_is_p0:     whether this thread's element is the "first" in the pair
//   partner_is_v1: whether the partner's contribution comes from v1 (else v0)
__device__ __forceinline__ void butterfly_shfl_info(
    int pair_idx, int stride, int D,
    int& p0_elem, int& p1_elem
) {
    int stride_mask = stride - 1;
    int block_start = (pair_idx & ~stride_mask) << 1;
    int in_block = pair_idx & stride_mask;
    p0_elem = block_start + in_block;
    p1_elem = p0_elem + stride;
}


LAUNCH_BOUNDS_COUPLING_WARP_FWD
__global__ void givens_coupling_fwd_warp_kernel(
    const float* __restrict__ x,           // [N, H, D] -- shared input
    const float* __restrict__ angles,       // [B, K, H, P] base rotation angles
    const float* __restrict__ coupling,    // [B, K, H, 4*P, P]
    const float* __restrict__ scales,      // [B, K, H, D]
    const int* __restrict__ shifts,        // [K] butterfly stride per stage
    float* __restrict__ out,               // [B, N, H, D]
    int N, int H, int D, int K, int B
) {
    int half_D = D / 2;     // == 32 == WARP_SIZE
    int f_feat = 4;
    int coupling_size = f_feat * half_D * half_D;  // 4*32*32 = 4096

    // 2D thread block: threadIdx.x = pair index (lane), threadIdx.y = token slot
    int p = threadIdx.x;            // [0, 31] — lane within warp
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

    // Shared memory: coupling tile (shared across tokens) + per-token
    // feature spill (written by each token's warp, read back by the einsum).
    extern __shared__ float smem[];
    float* coupling_tile = smem;
    float* feat_base     = smem + coupling_size;                 // [TPB, 4*P]
    float* my_features   = feat_base + token_slot * (f_feat * half_D);

    // ===== Load input into registers =====
    // Thread p holds elements (2p, 2p+1) of the D-vector
    float v0 = 0.0f, v1 = 0.0f;
    if (valid) {
        int x_offset = (my_n * H + h) * D;
        v0 = x[x_offset + 2 * p];
        v1 = x[x_offset + 2 * p + 1];
    }

    // ===== Process K rotation stages =====
    // NOTE: All warp shuffle calls must be executed by ALL lanes in the warp,
    // even for invalid tokens (my_n >= N).  Invalid threads participate with
    // v0=v1=0 and simply discard results.  Only global memory reads/writes
    // are guarded by `valid`.
    for (int k = 0; k < K; k++) {
        // --- Cooperative coupling tile load (all threads participate) ---
        int cp_base = ((b * K + k) * H + h) * coupling_size;
        int tid_linear = threadIdx.x + threadIdx.y * blockDim.x;
        int total_threads = blockDim.x * blockDim.y;
        for (int i = tid_linear; i < coupling_size; i += total_threads)
            coupling_tile[i] = coupling[cp_base + i];
        __syncthreads();

        int stride = -shifts[k];

        // --- Butterfly permutation via warp shuffle ---
        int p0_elem, p1_elem;
        butterfly_shfl_info(p, stride, D, p0_elem, p1_elem);

        int p0_thread = p0_elem >> 1;
        int p0_reg    = p0_elem & 1;
        int p1_thread = p1_elem >> 1;
        int p1_reg    = p1_elem & 1;

        // All lanes execute shuffles (required for warp convergence).
        float src0_v0 = SHFL_BROADCAST(v0, p0_thread);
        float src0_v1 = SHFL_BROADCAST(v1, p0_thread);
        float x0_val = p0_reg ? src0_v1 : src0_v0;

        float src1_v0 = SHFL_BROADCAST(v0, p1_thread);
        float src1_v1 = SHFL_BROADCAST(v1, p1_thread);
        float x1_val = p1_reg ? src1_v1 : src1_v0;

        // --- Clifford features (in registers) ---
        float my_S0  = x0_val * x0_val + x1_val * x1_val;
        float my_V1  = x0_val;
        float my_V2  = x1_val;

        int next = (p + 1) & (half_D - 1);
        float nb_x0  = SHFL_BROADCAST(x0_val, next);
        float nb_x1  = SHFL_BROADCAST(x1_val, next);
        float my_B12 = x0_val * nb_x1 - x1_val * nb_x0;

        // --- L2 normalize features across the warp ---
        float sq = my_S0 * my_S0 + my_V1 * my_V1 + my_V2 * my_V2 + my_B12 * my_B12;
        for (int offset = half_D / 2; offset > 0; offset >>= 1)
            sq += SHFL_XOR(sq, offset);
        float inv_norm = rsqrtf(sq + 1e-6f);
        my_S0  *= inv_norm;
        my_V1  *= inv_norm;
        my_V2  *= inv_norm;
        my_B12 *= inv_norm;

        // --- Spill normalized features to per-token smem ---
        // The pure-shfl version broadcast these 4 values across all 32 lanes
        // inside the einsum loop (128 shfl/stage).  Spilling to smem lets the
        // einsum read them with bank-conflict-free loads — same cost as the
        // tiled kernel — while the butterfly/norm stages above still run
        // register-only with no barriers.
        my_features[p]              = my_S0;
        my_features[half_D + p]     = my_V1;
        my_features[2 * half_D + p] = my_V2;
        my_features[3 * half_D + p] = my_B12;
        __syncwarp();   // intra-warp fence: all 32 lanes' writes visible to reads

        // --- Coupling einsum from smem (4-way accumulator, matches tiled kernel) ---
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

        // --- Angle computation (exact) ---
        float c_base = 1.0f, s_base = 0.0f;
        if (valid) {
            int a_off = ((b * K + k) * H + h) * half_D + p;
            sincosf(angles[a_off], &s_base, &c_base);
        }
        float correction = FAST_TANHF(modulation);
        // exact cos/sin of the tanh correction (was a Cayley approximation, T-109)
        float c_corr, s_corr;
        sincosf(correction, &s_corr, &c_corr);
        float c = c_base * c_corr - s_base * s_corr;
        float s = s_base * c_corr + c_base * s_corr;

        // --- Givens rotation ---
        float r0 = x0_val * c - x1_val * s;
        float r1 = x0_val * s + x1_val * c;

        // --- Scale ---
        if (valid) {
            int s_off = ((b * K + k) * H + h) * D;
            r0 *= scales[s_off + 2 * p];
            r1 *= scales[s_off + 2 * p + 1];
        }

        // --- Inverse butterfly: scatter r0/r1 back to owning threads ---
        int my_e0 = 2 * p;
        int my_e1 = 2 * p + 1;

        bool e0_is_p0 = (my_e0 & stride) == 0;
        int src_pair_e0 = e0_is_p0 ?
            reverse_pair_idx(my_e0, stride) :
            reverse_pair_idx(my_e0 - stride, stride);
        float e0_r0 = SHFL_BROADCAST(r0, src_pair_e0);
        float e0_r1 = SHFL_BROADCAST(r1, src_pair_e0);
        float new_v0 = e0_is_p0 ? e0_r0 : e0_r1;

        bool e1_is_p0 = (my_e1 & stride) == 0;
        int src_pair_e1 = e1_is_p0 ?
            reverse_pair_idx(my_e1, stride) :
            reverse_pair_idx(my_e1 - stride, stride);
        float e1_r0 = SHFL_BROADCAST(r0, src_pair_e1);
        float e1_r1 = SHFL_BROADCAST(r1, src_pair_e1);
        float new_v1 = e1_is_p0 ? e1_r0 : e1_r1;

        v0 = new_v0;
        v1 = new_v1;

        // Sync before next coupling tile load
        __syncthreads();
    }

    // ===== Write output =====
    if (valid) {
        int out_offset = ((b * N + my_n) * H + h) * D;
        out[out_offset + 2 * p]     = v0;
        out[out_offset + 2 * p + 1] = v1;
    }
}

#endif // GIVENS_COUPLING_FORWARD_WARP_H
