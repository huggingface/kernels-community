/*
 * Tiled Coupling Backward Kernel for Attention (Clifford F_feat=4)
 *
 * Matches PyTorch reference _coupling_project when perms=None (identity permutation).
 *
 * Tiled design: each block processes TPB=16 tokens sharing one coupling tile
 * per stage. Same block layout as forward: blockDim = (half_D, TPB) = (32, 16).
 *
 * Computes gradients for: x, angles, scales (in-kernel)
 * Saves features + grad_mod to buffers for grad_coupling via post-kernel batched matmul
 * (replaces O(N)-contention atomicAdd with efficient GEMM)
 *
 * Phase 1: Forward recomputation to populate save_buf with pre-stage states
 * Phase 2: Backward pass through K stages in reverse
 *
 * Key optimization: In-block gradient accumulation.
 * Instead of N per-token atomicAdd ops for grad_angles/grad_scales, each
 * block accumulates across TPB tokens in shared memory (using smem atomicAdd),
 * then emits a single global atomicAdd per element — 16x reduction in contention.
 *
 * Shared memory layout (per block):
 *   coupling_tile[4*P*(P+1)]        -- padded (backward has transpose access pattern)
 *   work[TPB * padded_D]            -- Phase 1: forward state; Phase 2: gradient buffer
 *   smem_x0[TPB * P]               -- extracted pairs
 *   smem_x1[TPB * P]               -- extracted pairs
 *   features[TPB * 4*P]            -- recomputed Clifford features
 *   grad_mod[TPB * P]              -- gradient of modulation
 *   gx0_rot[TPB * P]               -- rotation-only gradient for x0
 *   gx1_rot[TPB * P]               -- rotation-only gradient for x1
 *   grad_angles_accum[P]           -- in-block accumulation buffer
 *   grad_scales_accum[D]           -- in-block accumulation buffer
 */

#ifndef GIVENS_COUPLING_BACKWARD_H
#define GIVENS_COUPLING_BACKWARD_H

#include "givens_common.h"

LAUNCH_BOUNDS_COUPLING_BWD
__global__ void givens_coupling_bwd_tiled_kernel(
    const float* __restrict__ grad_out,      // [B, N, H, D]
    const float* __restrict__ x_in,          // [N, H, D]
    const float* __restrict__ angles,         // [B, K, H, P] base rotation angles
    const float* __restrict__ coupling,      // [B, K, H, 4*P, P]
    const float* __restrict__ scales,        // [B, K, H, D]
    const int* __restrict__ shifts,          // [K] butterfly stride/shift per stage
    float* __restrict__ grad_x,              // [N, H, D] -- atomicAdd across B projections
    float* __restrict__ grad_angles,         // [B, K, H, P]
    float* __restrict__ grad_scales,         // [B, K, H, D]
    float* __restrict__ feat_buf,            // [B*K*H, N, 4*P] -- for post-kernel matmul
    float* __restrict__ grad_mod_buf,        // [B*K*H, N, P]   -- for post-kernel matmul
    float* __restrict__ save_buf,            // [B*N*H, K, D]
    int N, int H, int D, int K, int B
) {
    int half_D = D / 2;
    int f_feat = 4;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int coupling_stride = half_D + 1;  // +1 padding to avoid 32-way bank conflicts on transpose
    int coupling_size = f_feat * half_D * half_D;  // global memory size (unpadded)
    int coupling_tile_size = f_feat * half_D * coupling_stride;  // smem size (padded)

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
    float* coupling_tile      = smem;
    float* work_base          = smem + coupling_tile_size;
    float* smem_x0_base       = work_base + TPB * padded_D;
    float* smem_x1_base       = smem_x0_base + TPB * half_D;
    float* feat_base          = smem_x1_base + TPB * half_D;
    float* grad_mod_base      = feat_base + TPB * f_feat * half_D;
    float* gx0_rot_base       = grad_mod_base + TPB * half_D;
    float* gx1_rot_base       = gx0_rot_base + TPB * half_D;
    float* grad_angles_accum  = gx1_rot_base + TPB * half_D;
    float* grad_scales_accum  = grad_angles_accum + half_D;

    // Per-token pointers
    float* my_work     = work_base + token_slot * padded_D;
    float* my_x0       = smem_x0_base + token_slot * half_D;
    float* my_x1       = smem_x1_base + token_slot * half_D;
    float* my_features = feat_base + token_slot * f_feat * half_D;
    float* my_grad_mod = grad_mod_base + token_slot * half_D;
    float* my_gx0_rot  = gx0_rot_base + token_slot * half_D;
    float* my_gx1_rot  = gx1_rot_base + token_slot * half_D;

    // Per-token save_buf pointer
    long long save_id = valid ? ((long long)(b * N + my_n) * H + h) : 0;
    float* my_save = save_buf + (long long)save_id * K * D;   // 64-bit: N*K*D passes 2^31 at batch 2 x 4K x d_ff 8192 x 36 stages
    int x_offset = valid ? ((my_n * H + h) * D) : 0;

    // Linearized thread ID for cooperative loads
    int tid_linear = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;

    // ================================================================
    // Phase 1: Forward recomputation -- populate save_buf
    // ================================================================

    if (valid) {
        my_work[SHMEM_IDX(2 * p)]     = x_in[x_offset + 2 * p];
        my_work[SHMEM_IDX(2 * p + 1)] = x_in[x_offset + 2 * p + 1];
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        // Save pre-stage state
        if (valid) {
            my_save[k * D + 2 * p]     = my_work[SHMEM_IDX(2 * p)];
            my_save[k * D + 2 * p + 1] = my_work[SHMEM_IDX(2 * p + 1)];
        }
        __syncthreads();

        // Load coupling tile with padding (all threads cooperate)
        // Global layout: [4*P rows, P cols], shared layout: [4*P rows, P+1 cols]
        {
            int total_rows = f_feat * half_D;
            int total_elems = total_rows * half_D;
            for (int i = tid_linear; i < total_elems; i += total_threads) {
                int row = i / half_D;
                int col = i & (half_D - 1);
                int cp_base = ((b * K + k) * H + h) * coupling_size;
                coupling_tile[row * coupling_stride + col] = coupling[cp_base + i];
            }
        }
        __syncthreads();

        // Extract pairs via butterfly permutation (matches eager perm -> read).
        int p0_perm_p1, p1_perm_p1;
        compute_pair_indices(p, shifts[k], D, p0_perm_p1, p1_perm_p1);
        if (valid) {
            my_x0[p] = my_work[SHMEM_IDX(p0_perm_p1)];
            my_x1[p] = my_work[SHMEM_IDX(p1_perm_p1)];
        }
        STAGE_BARRIER(half_D);

        // Compute features
        if (valid) {
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            my_features[p]                  = x0 * x0 + x1 * x1;
            my_features[half_D + p]         = x0;
            my_features[2 * half_D + p]     = x1;
            my_features[3 * half_D + p]     = x0 * my_x1[next] - x1 * my_x0[next];
        }
        STAGE_BARRIER(half_D);

        // L2-normalize features (match forward kernel)
        if (valid) {
            float sq = 0.0f;
            sq += my_features[p]              * my_features[p];
            sq += my_features[half_D + p]     * my_features[half_D + p];
            sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
            sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];
            // Per-segment butterfly reduction (see givens_coupling_forward.h
            // for the full bug explanation: SHFL_DOWN+SHFL_BROADCAST(0) leaks
            // wavefront-lane-0's value to all lanes when half_D < WARP_SIZE).
            for (int offset = half_D / 2; offset > 0; offset >>= 1)
                sq += SHFL_XOR(sq, offset);
            float inv_norm = rsqrtf(sq + 1e-6f);
            my_features[p]                  *= inv_norm;
            my_features[half_D + p]         *= inv_norm;
            my_features[2 * half_D + p]     *= inv_norm;
            my_features[3 * half_D + p]     *= inv_norm;
        }
        STAGE_BARRIER(half_D);

        // Coupling + rotation + scale (recompute forward)
        if (valid) {
            // Register blocking: 4 independent accumulators
            int total = f_feat * half_D;
            float mod0 = 0.0f, mod1 = 0.0f, mod2 = 0.0f, mod3 = 0.0f;
            int j = 0;
            #pragma unroll 4
            for (; j + 3 < total; j += 4) {
                mod0 += my_features[j]     * coupling_tile[j       * coupling_stride + p];
                mod1 += my_features[j + 1] * coupling_tile[(j + 1) * coupling_stride + p];
                mod2 += my_features[j + 2] * coupling_tile[(j + 2) * coupling_stride + p];
                mod3 += my_features[j + 3] * coupling_tile[(j + 3) * coupling_stride + p];
            }
            float mod = mod0 + mod1 + mod2 + mod3;
            for (; j < total; j++)
                mod += my_features[j] * coupling_tile[j * coupling_stride + p];

            int a_off = ((b * K + k) * H + h) * half_D + p;
            float correction = FAST_TANHF(mod);
            float c_base, s_base;
            sincosf(angles[a_off], &s_base, &c_base);
            // exact cos/sin of the tanh correction (was a Cayley approximation, T-109)
            float c_corr, s_corr;
            sincosf(correction, &s_corr, &c_corr);
            float c = c_base * c_corr - s_base * s_corr;
            float s = s_base * c_corr + c_base * s_corr;

            float x0 = my_x0[p];
            float x1 = my_x1[p];
            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            int s_off = ((b * K + k) * H + h) * D;
            r0 *= scales[s_off + 2 * p];
            r1 *= scales[s_off + 2 * p + 1];

            // Write back to source positions (rotation in permuted frame, then inv_perm).
            my_work[SHMEM_IDX(p0_perm_p1)] = r0;
            my_work[SHMEM_IDX(p1_perm_p1)] = r1;
        }
        __syncthreads();
    }

    // ================================================================
    // Phase 2: Backward pass
    // ================================================================

    // Load grad_out into work buffer
    if (valid) {
        int out_offset = ((b * N + my_n) * H + h) * D;
        my_work[SHMEM_IDX(2 * p)]     = grad_out[out_offset + 2 * p];
        my_work[SHMEM_IDX(2 * p + 1)] = grad_out[out_offset + 2 * p + 1];
    }
    __syncthreads();

    for (int k = K - 1; k >= 0; k--) {
        // --- Zero gradient accumulators (one per pair/element) ---
        // token_slot 0 threads zero the accumulators; all others wait.
        if (token_slot == 0) {
            grad_angles_accum[p] = 0.0f;
            grad_scales_accum[2 * p]     = 0.0f;
            grad_scales_accum[2 * p + 1] = 0.0f;
        }
        __syncthreads();

        // Load coupling tile with padding (all threads cooperate)
        {
            int total_rows = f_feat * half_D;
            int total_elems = total_rows * half_D;
            int cp_base = ((b * K + k) * H + h) * coupling_size;
            for (int i = tid_linear; i < total_elems; i += total_threads) {
                int row = i / half_D;
                int col = i & (half_D - 1);
                coupling_tile[row * coupling_stride + col] = coupling[cp_base + i];
            }
        }
        __syncthreads();

        // Load saved pre-stage state, extract pairs via butterfly permutation.
        int p0_perm, p1_perm;
        compute_pair_indices(p, shifts[k], D, p0_perm, p1_perm);
        if (valid) {
            my_x0[p] = my_save[k * D + p0_perm];
            my_x1[p] = my_save[k * D + p1_perm];
        }
        STAGE_BARRIER(half_D);

        // Recompute features + L2-normalize (must match forward)
        float phase2_inv_norm = 0.0f;  // saved for normalization gradient
        if (valid) {
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            my_features[p]                  = x0 * x0 + x1 * x1;
            my_features[half_D + p]         = x0;
            my_features[2 * half_D + p]     = x1;
            my_features[3 * half_D + p]     = x0 * my_x1[next] - x1 * my_x0[next];
        }
        STAGE_BARRIER(half_D);

        if (valid) {
            float sq = 0.0f;
            sq += my_features[p]              * my_features[p];
            sq += my_features[half_D + p]     * my_features[half_D + p];
            sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
            sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];
            // Per-segment butterfly reduction (same fix as phase-1 normalization).
            for (int offset = half_D / 2; offset > 0; offset >>= 1)
                sq += SHFL_XOR(sq, offset);
            phase2_inv_norm = rsqrtf(sq + 1e-6f);
            my_features[p]                  *= phase2_inv_norm;
            my_features[half_D + p]         *= phase2_inv_norm;
            my_features[2 * half_D + p]     *= phase2_inv_norm;
            my_features[3 * half_D + p]     *= phase2_inv_norm;
        }
        STAGE_BARRIER(half_D);

        // Recompute modulation/rotation, compute per-pair gradients.
        // Output grads live at the same permuted positions as the forward writes.
        if (valid) {
            float g_o0 = my_work[SHMEM_IDX(p0_perm)];
            float g_o1 = my_work[SHMEM_IDX(p1_perm)];

            float x0 = my_x0[p];
            float x1 = my_x1[p];

            // Recompute modulation with register blocking
            int total = f_feat * half_D;
            float mod0 = 0.0f, mod1 = 0.0f, mod2 = 0.0f, mod3 = 0.0f;
            int j = 0;
            #pragma unroll 4
            for (; j + 3 < total; j += 4) {
                mod0 += my_features[j]     * coupling_tile[j       * coupling_stride + p];
                mod1 += my_features[j + 1] * coupling_tile[(j + 1) * coupling_stride + p];
                mod2 += my_features[j + 2] * coupling_tile[(j + 2) * coupling_stride + p];
                mod3 += my_features[j + 3] * coupling_tile[(j + 3) * coupling_stride + p];
            }
            float mod = mod0 + mod1 + mod2 + mod3;
            for (; j < total; j++)
                mod += my_features[j] * coupling_tile[j * coupling_stride + p];

            int a_off = ((b * K + k) * H + h) * half_D + p;
            float tanh_mod = FAST_TANHF(mod);
            float c_base2, s_base2;
            sincosf(angles[a_off], &s_base2, &c_base2);
            // exact cos/sin of the tanh correction (was a Cayley approximation, T-109)
            float c_corr2, s_corr2;
            sincosf(tanh_mod, &s_corr2, &c_corr2);
            float c = c_base2 * c_corr2 - s_base2 * s_corr2;
            float s = s_base2 * c_corr2 + c_base2 * s_corr2;

            float r0 = x0 * c - x1 * s;
            float r1 = x0 * s + x1 * c;

            int s_off = ((b * K + k) * H + h) * D;
            float sc0 = scales[s_off + 2 * p];
            float sc1 = scales[s_off + 2 * p + 1];

            // Gradient through scale — accumulate in shared memory
            atomicAdd(&grad_scales_accum[2 * p],     g_o0 * r0);
            atomicAdd(&grad_scales_accum[2 * p + 1], g_o1 * r1);

            float g_r0 = g_o0 * sc0;
            float g_r1 = g_o1 * sc1;

            // Gradient through rotation
            float g_theta = g_r0 * (-r1) + g_r1 * r0;
            atomicAdd(&grad_angles_accum[p], g_theta);

            // Gradient through tanh
            float g_modulation = g_theta * (1.0f - tanh_mod * tanh_mod);
            my_grad_mod[p] = g_modulation;

            // Rotation-only input gradient
            my_gx0_rot[p] =  g_r0 * c + g_r1 * s;
            my_gx1_rot[p] = -g_r0 * s + g_r1 * c;
        }
        __syncthreads();

        // Save features + grad_mod for post-kernel batched matmul, then compute input gradient
        if (valid) {
            float g_mod_p = my_grad_mod[p];

            // Save features and grad_modulation to buffers
            // Layout: feat_buf[bkh, n, j], grad_mod_buf[bkh, n, p]
            {
                long long bkh = (long long)(b * K + k) * H + h;
                long long feat_off = (bkh * N + my_n) * (f_feat * half_D);
                for (int jj = 0; jj < f_feat; jj++)
                    feat_buf[feat_off + jj * half_D + p] = my_features[jj * half_D + p];
                long long gm_off = (bkh * N + my_n) * half_D;
                grad_mod_buf[gm_off + p] = g_mod_p;
            }

            // Feature gradients via coupling transpose (from padded shared mem)
            // These are gradients w.r.t. NORMALIZED features
            int prev = (p - 1 + half_D) & (half_D - 1);

            float gn_S0 = 0.0f, gn_V1 = 0.0f, gn_V2 = 0.0f;
            float gn_B12 = 0.0f, gn_B12_prev = 0.0f;

            for (int q = 0; q < half_D; q++) {
                float gm_q = my_grad_mod[q];
                gn_S0       += gm_q * coupling_tile[p * coupling_stride + q];
                gn_V1       += gm_q * coupling_tile[(half_D + p) * coupling_stride + q];
                gn_V2       += gm_q * coupling_tile[(2 * half_D + p) * coupling_stride + q];
                gn_B12      += gm_q * coupling_tile[(3 * half_D + p) * coupling_stride + q];
                gn_B12_prev += gm_q * coupling_tile[(3 * half_D + prev) * coupling_stride + q];
            }

            // Chain rule through L2 normalization: f_norm = f_raw / ||f_raw||
            // g_raw_i = (g_norm_i - f_norm_i * dot(g_norm, f_norm)) * inv_norm
            // my_features[] currently holds normalized features
            float fn_S0  = my_features[p];
            float fn_V1  = my_features[half_D + p];
            float fn_V2  = my_features[2 * half_D + p];
            float fn_B12 = my_features[3 * half_D + p];

            // Partial dot product: g_norm . f_norm (this thread's 4 elements)
            float dot_partial = gn_S0 * fn_S0 + gn_V1 * fn_V1
                              + gn_V2 * fn_V2 + gn_B12 * fn_B12;

            // Per-segment butterfly reduction (same fix as the L2-norm reductions
            // above; see givens_coupling_forward.h for full bug explanation).
            {
                for (int offset = half_D / 2; offset > 0; offset >>= 1)
                    dot_partial += SHFL_XOR(dot_partial, offset);
            }

            // Apply normalization gradient: g_raw = (g_norm - f_norm * dot) * inv_norm
            float g_S0  = (gn_S0  - fn_S0  * dot_partial) * phase2_inv_norm;
            float g_V1  = (gn_V1  - fn_V1  * dot_partial) * phase2_inv_norm;
            float g_V2  = (gn_V2  - fn_V2  * dot_partial) * phase2_inv_norm;
            float g_B12 = (gn_B12 - fn_B12 * dot_partial) * phase2_inv_norm;

            // B12_prev gradient also needs normalization chain rule
            // gn_B12_prev is the grad w.r.t. normalized B12 at position prev
            float fn_B12_prev = my_features[3 * half_D + prev];
            // The dot product contribution from prev's B12 is already in the full dot
            // But gn_B12_prev is a separate coupling transpose output for B12[prev],
            // it needs its own normalization adjustment using prev's normalized features.
            // Since the full norm is shared across all positions, the chain rule gives:
            float g_B12_prev = (gn_B12_prev - fn_B12_prev * dot_partial) * phase2_inv_norm;

            // Gradient through Clifford features -> input
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            float x0n = my_x0[next];
            float x1n = my_x1[next];
            float x0p = my_x0[prev];
            float x1p = my_x1[prev];

            // S0 = x0^2 + x1^2
            float g_x0 = g_S0 * 2.0f * x0;
            float g_x1 = g_S0 * 2.0f * x1;

            // V1 = x0, V2 = x1
            g_x0 += g_V1;
            g_x1 += g_V2;

            // B12[p] = x0[p]*x1[next] - x1[p]*x0[next]
            g_x0 += g_B12 * x1n;
            g_x1 += g_B12 * (-x0n);

            // B12[prev] = x0[prev]*x1[p] - x1[prev]*x0[p]
            g_x0 += g_B12_prev * (-x1p);
            g_x1 += g_B12_prev * x0p;

            // Add rotation-only gradient
            g_x0 += my_gx0_rot[p];
            g_x1 += my_gx1_rot[p];

            // Write total input gradient to work at the source positions
            // (so the previous stage k-1 reads it in the natural frame).
            my_work[SHMEM_IDX(p0_perm)] = g_x0;
            my_work[SHMEM_IDX(p1_perm)] = g_x1;
        }
        __syncthreads();

        // --- Emit accumulated gradients: one global atomicAdd per element ---
        // All P threads in token_slot=0 emit in parallel (not serialized)
        if (token_slot == 0) {
            int a_base = ((b * K + k) * H + h) * half_D;
            int s_base = ((b * K + k) * H + h) * D;
            atomicAdd(&grad_angles[a_base + p], grad_angles_accum[p]);
            atomicAdd(&grad_scales[s_base + 2 * p],     grad_scales_accum[2 * p]);
            atomicAdd(&grad_scales[s_base + 2 * p + 1], grad_scales_accum[2 * p + 1]);
        }
        __syncthreads();
    }

    // Write grad_x
    if (valid) {
        if (B == 1) {
            // Single projection — no contention, direct store is faster
            grad_x[x_offset + 2 * p]     = my_work[SHMEM_IDX(2 * p)];
            grad_x[x_offset + 2 * p + 1] = my_work[SHMEM_IDX(2 * p + 1)];
        } else {
            // Multiple projections share the same input — need atomic accumulation
            atomicAdd(&grad_x[x_offset + 2 * p],     my_work[SHMEM_IDX(2 * p)]);
            atomicAdd(&grad_x[x_offset + 2 * p + 1], my_work[SHMEM_IDX(2 * p + 1)]);
        }
    }
}

#endif // GIVENS_COUPLING_BACKWARD_H
