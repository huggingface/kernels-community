/*
 * Chunked Coupling Backward Kernel — handles large D where the coupling
 * matrix exceeds shared-memory capacity.
 *
 * Same chunking strategy as the forward (rows of the coupling matrix in
 * chunks of size CHUNK_ROWS). Backward has TWO einsum patterns over coupling:
 *
 *   (a) modulation recompute (same as forward):
 *       modulation[p] = sum_j features[j] * coupling[j, p]
 *
 *   (b) feature gradient:
 *       gn_S0[p]  = sum_q gm[q] * coupling[p,             q]   (p offset 0)
 *       gn_V1[p]  = sum_q gm[q] * coupling[half_D + p,    q]
 *       gn_V2[p]  = sum_q gm[q] * coupling[2*half_D + p,  q]
 *       gn_B12[p] = sum_q gm[q] * coupling[3*half_D + p,  q]
 *       gn_B12_prev[p] = sum_q gm[q] * coupling[3*half_D + prev, q]
 *
 * For (a) we accumulate the same way as forward.
 * For (b) we only need 4 specific rows per output dim p (S0/V1/V2/B12 at p,
 * and B12 at prev). We chunk the q-loop instead of the row-loop. Per chunk
 * we load grad_mod[col_chunk] into shared mem and 4 specific rows into
 * coupling_chunk; accumulate gn_X across chunks.
 *
 * NOTE: this requires loading the SAME coupling rows for each block (just
 * different column slices). The kernel still loads everything cooperatively,
 * just in chunks.
 *
 * Atomic accumulation, save_buf layout, and post-kernel grad_coupling matmul
 * are unchanged from the original kernel.
 *
 * blockDim: dim3(half_D, TPB), TPB chosen so half_D * TPB <= 1024.
 *
 * For D=256, half_D=128, TPB=8 -> 1024 threads.
 *   Coupling chunk: e.g. 32 rows * 128 cols = 16 KiB.
 *   Total smem ~ 80 KiB.
 */

#ifndef GIVENS_COUPLING_BACKWARD_CHUNKED_H
#define GIVENS_COUPLING_BACKWARD_CHUNKED_H

#include "givens_common.h"

LAUNCH_BOUNDS_BWD
__global__ void givens_coupling_bwd_chunked_kernel(
    const float* __restrict__ grad_out,      // [B, N, H, D]
    const float* __restrict__ x_in,          // [N, H, D]
    const float* __restrict__ angles,         // [B, K, H, P]
    const float* __restrict__ coupling,      // [B, K, H, 4*P, P]
    const float* __restrict__ scales,        // [B, K, H, D]
    const int* __restrict__ shifts,
    float* __restrict__ grad_x,              // [N, H, D]
    float* __restrict__ grad_angles,         // [B, K, H, P]
    float* __restrict__ grad_scales,         // [B, K, H, D]
    float* __restrict__ feat_buf,            // [B*K*H, N, 4*P]
    float* __restrict__ grad_mod_buf,        // [B*K*H, N, P]
    float* __restrict__ save_buf,            // [B*N*H, K, D]
    int N, int H, int D, int K, int B,
    int chunk_rows                           // CHUNK_ROWS for forward-style chunks
) {
    int half_D = D / 2;
    int f_feat = 4;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int total_rows = f_feat * half_D;
    int chunk_size = chunk_rows * half_D;
    int coupling_size_global = total_rows * half_D;

    int p = threadIdx.x;
    int token_slot = threadIdx.y;
    int TPB = blockDim.y;

    int n_tiles = (N + TPB - 1) / TPB;
    int block_id = blockIdx.x;
    int b = block_id / (n_tiles * H);
    int rest = block_id % (n_tiles * H);
    int n_tile = rest / H;
    int h = rest % H;

    int my_n = n_tile * TPB + token_slot;
    bool valid = (my_n < N);

    extern __shared__ float smem[];
    float* coupling_chunk     = smem;
    float* work_base          = smem + chunk_size;
    float* smem_x0_base       = work_base + TPB * padded_D;
    float* smem_x1_base       = smem_x0_base + TPB * half_D;
    float* feat_base          = smem_x1_base + TPB * half_D;
    float* grad_mod_base      = feat_base + TPB * f_feat * half_D;
    float* gx0_rot_base       = grad_mod_base + TPB * half_D;
    float* gx1_rot_base       = gx0_rot_base + TPB * half_D;
    float* grad_angles_accum  = gx1_rot_base + TPB * half_D;
    float* grad_scales_accum  = grad_angles_accum + half_D;

    float* my_work     = work_base + token_slot * padded_D;
    float* my_x0       = smem_x0_base + token_slot * half_D;
    float* my_x1       = smem_x1_base + token_slot * half_D;
    float* my_features = feat_base + token_slot * f_feat * half_D;
    float* my_grad_mod = grad_mod_base + token_slot * half_D;
    float* my_gx0_rot  = gx0_rot_base + token_slot * half_D;
    float* my_gx1_rot  = gx1_rot_base + token_slot * half_D;

    long long save_id = valid ? ((long long)(b * N + my_n) * H + h) : 0;
    float* my_save = save_buf + (long long)save_id * K * D;   // 64-bit: N*K*D passes 2^31 at batch 2 x 4K x d_ff 8192 x 36 stages
    int x_offset = valid ? ((my_n * H + h) * D) : 0;

    int tid_linear = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    int n_chunks = (total_rows + chunk_rows - 1) / chunk_rows;

    // ================================================================
    // Phase 1: forward recomputation
    // ================================================================
    if (valid) {
        my_work[SHMEM_IDX(2 * p)]     = x_in[x_offset + 2 * p];
        my_work[SHMEM_IDX(2 * p + 1)] = x_in[x_offset + 2 * p + 1];
    }
    __syncthreads();

    for (int k = 0; k < K; k++) {
        if (valid) {
            my_save[k * D + 2 * p]     = my_work[SHMEM_IDX(2 * p)];
            my_save[k * D + 2 * p + 1] = my_work[SHMEM_IDX(2 * p + 1)];
        }
        __syncthreads();

        int p0_perm, p1_perm;
        compute_pair_indices(p, shifts[k], D, p0_perm, p1_perm);
        if (valid) {
            my_x0[p] = my_work[SHMEM_IDX(p0_perm)];
            my_x1[p] = my_work[SHMEM_IDX(p1_perm)];
        }
        __syncthreads();

        if (valid) {
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            my_features[p]                  = x0 * x0 + x1 * x1;
            my_features[half_D + p]         = x0;
            my_features[2 * half_D + p]     = x1;
            my_features[3 * half_D + p]     = x0 * my_x1[next] - x1 * my_x0[next];
        }
        __syncthreads();

        // L2-normalize (cross-warp reduce via shmem)
        if (valid) {
            float sq = 0.0f;
            sq += my_features[p]              * my_features[p];
            sq += my_features[half_D + p]     * my_features[half_D + p];
            sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
            sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                sq += SHFL_XOR(sq, offset);
            if (half_D > WARP_SIZE) {
                __shared__ float warp_sums[16 * 16];
                int lane = p & (WARP_SIZE - 1);
                int warp = p / WARP_SIZE;
                int n_warps = half_D / WARP_SIZE;
                if (lane == 0) warp_sums[token_slot * n_warps + warp] = sq;
                __syncthreads();
                sq = 0.0f;
                for (int w = 0; w < n_warps; w++) sq += warp_sums[token_slot * n_warps + w];
            }
            float inv_norm = rsqrtf(sq + 1e-6f);
            my_features[p]                  *= inv_norm;
            my_features[half_D + p]         *= inv_norm;
            my_features[2 * half_D + p]     *= inv_norm;
            my_features[3 * half_D + p]     *= inv_norm;
        }
        __syncthreads();

        // Modulation via chunked coupling
        float modulation = 0.0f;
        int cp_base = ((b * K + k) * H + h) * coupling_size_global;
        for (int chunk = 0; chunk < n_chunks; chunk++) {
            int row_start = chunk * chunk_rows;
            int rows_this = (row_start + chunk_rows <= total_rows)
                            ? chunk_rows : (total_rows - row_start);
            int elems_this = rows_this * half_D;
            for (int i = tid_linear; i < elems_this; i += total_threads)
                coupling_chunk[i] = coupling[cp_base + row_start * half_D + i];
            __syncthreads();
            if (valid) {
                for (int j_local = 0; j_local < rows_this; j_local++) {
                    int j_global = row_start + j_local;
                    modulation += my_features[j_global] * coupling_chunk[j_local * half_D + p];
                }
            }
            __syncthreads();
        }

        if (valid) {
            int a_off = ((b * K + k) * H + h) * half_D + p;
            float correction = FAST_TANHF(modulation);
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

            my_work[SHMEM_IDX(p0_perm)] = r0;
            my_work[SHMEM_IDX(p1_perm)] = r1;
        }
        __syncthreads();
    }

    // ================================================================
    // Phase 2: backward
    // ================================================================
    if (valid) {
        int out_offset = ((b * N + my_n) * H + h) * D;
        my_work[SHMEM_IDX(2 * p)]     = grad_out[out_offset + 2 * p];
        my_work[SHMEM_IDX(2 * p + 1)] = grad_out[out_offset + 2 * p + 1];
    }
    __syncthreads();

    for (int k = K - 1; k >= 0; k--) {
        if (token_slot == 0) {
            grad_angles_accum[p] = 0.0f;
            grad_scales_accum[2 * p]     = 0.0f;
            grad_scales_accum[2 * p + 1] = 0.0f;
        }
        __syncthreads();

        // Load saved pre-stage state
        int p0_perm, p1_perm;
        compute_pair_indices(p, shifts[k], D, p0_perm, p1_perm);
        if (valid) {
            my_x0[p] = my_save[k * D + p0_perm];
            my_x1[p] = my_save[k * D + p1_perm];
        }
        __syncthreads();

        // Recompute features
        if (valid) {
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            my_features[p]                  = x0 * x0 + x1 * x1;
            my_features[half_D + p]         = x0;
            my_features[2 * half_D + p]     = x1;
            my_features[3 * half_D + p]     = x0 * my_x1[next] - x1 * my_x0[next];
        }
        __syncthreads();

        // L2-normalize features (same as phase 1)
        float phase2_inv_norm = 0.0f;
        if (valid) {
            float sq = 0.0f;
            sq += my_features[p]              * my_features[p];
            sq += my_features[half_D + p]     * my_features[half_D + p];
            sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
            sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                sq += SHFL_XOR(sq, offset);
            if (half_D > WARP_SIZE) {
                __shared__ float warp_sums[16 * 16];
                int lane = p & (WARP_SIZE - 1);
                int warp = p / WARP_SIZE;
                int n_warps = half_D / WARP_SIZE;
                if (lane == 0) warp_sums[token_slot * n_warps + warp] = sq;
                __syncthreads();
                sq = 0.0f;
                for (int w = 0; w < n_warps; w++) sq += warp_sums[token_slot * n_warps + w];
                phase2_inv_norm = rsqrtf(sq + 1e-6f);
            } else {
                phase2_inv_norm = rsqrtf(sq + 1e-6f);
            }
            my_features[p]                  *= phase2_inv_norm;
            my_features[half_D + p]         *= phase2_inv_norm;
            my_features[2 * half_D + p]     *= phase2_inv_norm;
            my_features[3 * half_D + p]     *= phase2_inv_norm;
        }
        __syncthreads();

        // Recompute modulation via chunked coupling (same as phase 1)
        float mod = 0.0f;
        int cp_base = ((b * K + k) * H + h) * coupling_size_global;
        for (int chunk = 0; chunk < n_chunks; chunk++) {
            int row_start = chunk * chunk_rows;
            int rows_this = (row_start + chunk_rows <= total_rows)
                            ? chunk_rows : (total_rows - row_start);
            int elems_this = rows_this * half_D;
            for (int i = tid_linear; i < elems_this; i += total_threads)
                coupling_chunk[i] = coupling[cp_base + row_start * half_D + i];
            __syncthreads();
            if (valid) {
                for (int j_local = 0; j_local < rows_this; j_local++) {
                    int j_global = row_start + j_local;
                    mod += my_features[j_global] * coupling_chunk[j_local * half_D + p];
                }
            }
            __syncthreads();
        }

        // Per-pair output gradient + per-pair scale/angle gradient accumulation
        if (valid) {
            float g_o0 = my_work[SHMEM_IDX(p0_perm)];
            float g_o1 = my_work[SHMEM_IDX(p1_perm)];

            float x0 = my_x0[p];
            float x1 = my_x1[p];

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

            atomicAdd(&grad_scales_accum[2 * p],     g_o0 * r0);
            atomicAdd(&grad_scales_accum[2 * p + 1], g_o1 * r1);

            float g_r0 = g_o0 * sc0;
            float g_r1 = g_o1 * sc1;

            float g_theta = g_r0 * (-r1) + g_r1 * r0;
            atomicAdd(&grad_angles_accum[p], g_theta);

            float g_modulation = g_theta * (1.0f - tanh_mod * tanh_mod);
            my_grad_mod[p] = g_modulation;

            my_gx0_rot[p] =  g_r0 * c + g_r1 * s;
            my_gx1_rot[p] = -g_r0 * s + g_r1 * c;
        }
        __syncthreads();

        // Save features + grad_mod for post-kernel matmul
        if (valid) {
            float g_mod_p = my_grad_mod[p];
            long long bkh = (long long)(b * K + k) * H + h;
            long long feat_off = (bkh * N + my_n) * (f_feat * half_D);
            for (int jj = 0; jj < f_feat; jj++)
                feat_buf[feat_off + jj * half_D + p] = my_features[jj * half_D + p];
            long long gm_off = (bkh * N + my_n) * half_D;
            grad_mod_buf[gm_off + p] = g_mod_p;
        }
        __syncthreads();

        // ----- Feature gradients via coupling transpose, chunked -----
        // gn_X[p] = sum_q gm[q] * coupling[X_row + p, q]
        // We chunk along the q axis (columns). Per chunk:
        //   - Load that q-slice of grad_mod into a small smem buffer (well actually
        //     grad_mod fits already in my_grad_mod[half_D] per token; we just iterate).
        //   - Load 4*half_D rows BUT only the q-slice of size chunk_cols per row.
        //
        // To keep the smem layout simple we instead chunk along ROWS (j-axis)
        // and iterate q across full rows. This requires us to load only 4 specific
        // rows per chunk: row p (S0), half_D+p (V1), 2*half_D+p (V2), 3*half_D+p (B12),
        // plus the prev-row 3*half_D+prev (B12_prev).
        //
        // However each thread needs DIFFERENT rows (its own p and its own prev),
        // so we can't share a single coupling_chunk across threads for this part.
        //
        // SIMPLEST CORRECT APPROACH: skip the smem chunk and load directly from
        // global memory for the gn_* computation. Each thread does 5 strided
        // loads of length half_D. This is O(half_D * 5) global loads per token
        // per stage, no smem dependency, no chunk-coordination bugs.
        //
        // For D=256, half_D=128: 5 * 128 = 640 global loads per (token, stage)
        // — bandwidth-bound but correct. Optimize later if needed.

        if (valid) {
            int prev = (p - 1 + half_D) & (half_D - 1);

            float gn_S0 = 0.0f, gn_V1 = 0.0f, gn_V2 = 0.0f;
            float gn_B12 = 0.0f, gn_B12_prev = 0.0f;

            int row_S0   = p;
            int row_V1   = half_D + p;
            int row_V2   = 2 * half_D + p;
            int row_B12  = 3 * half_D + p;
            int row_B12p = 3 * half_D + prev;

            // Direct global memory access — correct, simple, slower.
            // (cp_base is the start of coupling for this (b,k,h).)
            for (int q = 0; q < half_D; q++) {
                float gm_q = my_grad_mod[q];
                gn_S0       += gm_q * coupling[cp_base + row_S0   * half_D + q];
                gn_V1       += gm_q * coupling[cp_base + row_V1   * half_D + q];
                gn_V2       += gm_q * coupling[cp_base + row_V2   * half_D + q];
                gn_B12      += gm_q * coupling[cp_base + row_B12  * half_D + q];
                gn_B12_prev += gm_q * coupling[cp_base + row_B12p * half_D + q];
            }

            // Chain rule through L2 normalization
            float fn_S0  = my_features[p];
            float fn_V1  = my_features[half_D + p];
            float fn_V2  = my_features[2 * half_D + p];
            float fn_B12 = my_features[3 * half_D + p];

            float dot_partial = gn_S0 * fn_S0 + gn_V1 * fn_V1
                              + gn_V2 * fn_V2 + gn_B12 * fn_B12;
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot_partial += SHFL_XOR(dot_partial, offset);
            if (half_D > WARP_SIZE) {
                __shared__ float dot_warps[16 * 16];
                int lane = p & (WARP_SIZE - 1);
                int warp = p / WARP_SIZE;
                int n_warps = half_D / WARP_SIZE;
                if (lane == 0) dot_warps[token_slot * n_warps + warp] = dot_partial;
                __syncthreads();
                dot_partial = 0.0f;
                for (int w = 0; w < n_warps; w++) dot_partial += dot_warps[token_slot * n_warps + w];
            }

            float g_S0  = (gn_S0  - fn_S0  * dot_partial) * phase2_inv_norm;
            float g_V1  = (gn_V1  - fn_V1  * dot_partial) * phase2_inv_norm;
            float g_V2  = (gn_V2  - fn_V2  * dot_partial) * phase2_inv_norm;
            float g_B12 = (gn_B12 - fn_B12 * dot_partial) * phase2_inv_norm;

            float fn_B12_prev = my_features[3 * half_D + prev];
            float g_B12_prev = (gn_B12_prev - fn_B12_prev * dot_partial) * phase2_inv_norm;

            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            float x0n = my_x0[next];
            float x1n = my_x1[next];
            float x0p = my_x0[prev];
            float x1p = my_x1[prev];

            float g_x0 = g_S0 * 2.0f * x0;
            float g_x1 = g_S0 * 2.0f * x1;
            g_x0 += g_V1;
            g_x1 += g_V2;
            g_x0 += g_B12 * x1n;
            g_x1 += g_B12 * (-x0n);
            g_x0 += g_B12_prev * (-x1p);
            g_x1 += g_B12_prev * x0p;
            g_x0 += my_gx0_rot[p];
            g_x1 += my_gx1_rot[p];

            my_work[SHMEM_IDX(p0_perm)] = g_x0;
            my_work[SHMEM_IDX(p1_perm)] = g_x1;
        }
        __syncthreads();

        if (token_slot == 0) {
            int a_base = ((b * K + k) * H + h) * half_D;
            int s_base = ((b * K + k) * H + h) * D;
            atomicAdd(&grad_angles[a_base + p], grad_angles_accum[p]);
            atomicAdd(&grad_scales[s_base + 2 * p],     grad_scales_accum[2 * p]);
            atomicAdd(&grad_scales[s_base + 2 * p + 1], grad_scales_accum[2 * p + 1]);
        }
        __syncthreads();
    }

    if (valid) {
        if (B == 1) {
            grad_x[x_offset + 2 * p]     = my_work[SHMEM_IDX(2 * p)];
            grad_x[x_offset + 2 * p + 1] = my_work[SHMEM_IDX(2 * p + 1)];
        } else {
            atomicAdd(&grad_x[x_offset + 2 * p],     my_work[SHMEM_IDX(2 * p)]);
            atomicAdd(&grad_x[x_offset + 2 * p + 1], my_work[SHMEM_IDX(2 * p + 1)]);
        }
    }
}

#endif // GIVENS_COUPLING_BACKWARD_CHUNKED_H
