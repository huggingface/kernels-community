/*
 * Warp-Shuffle Coupling Backward Kernel for Attention (HYBRID variant)
 *
 * Specialized for half_D <= WARP_SIZE (D <= 64).
 * Same hybrid pattern as givens_coupling_forward_warp.h:
 *   - D-vector, butterfly permutation, features, norm: in registers via __shfl
 *   - Coupling tile, feature spill, grad_mod spill: in shared memory
 *
 * Phase 1: Forward recompute with save_buf writes
 * Phase 2: Backward pass in reverse stage order
 *
 * blockDim = (half_D, TPB) = (32, TPB)
 * gridDim  = B * ceil(N/TPB) * H
 *
 * Constraint: half_D == WARP_SIZE == 32  (D == 64).
 */

#ifndef GIVENS_COUPLING_BACKWARD_WARP_H
#define GIVENS_COUPLING_BACKWARD_WARP_H

#include "givens_common.h"

LAUNCH_BOUNDS_COUPLING_WARP_FWD  // same bounds as forward
__global__ void givens_coupling_bwd_warp_kernel(
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
    int N, int H, int D, int K, int B
) {
    int half_D = D / 2;
    int f_feat = 4;
    int coupling_size = f_feat * half_D * half_D;

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

    // Shared memory: coupling tile (padded for transpose access) +
    // per-token features + per-token grad_mod
    int coupling_stride = half_D + 1;  // +1 avoids 32-way bank conflicts on transpose
    int coupling_tile_size = f_feat * half_D * coupling_stride;
    extern __shared__ float smem[];
    float* coupling_tile = smem;
    float* feat_base     = smem + coupling_tile_size;
    float* gmod_base     = feat_base + TPB * f_feat * half_D;
    float* my_features   = feat_base + token_slot * (f_feat * half_D);
    float* my_grad_mod   = gmod_base + token_slot * half_D;

    int tid_linear = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;

    // Per-token save_buf pointer
    long long save_id = valid ? ((long long)(b * N + my_n) * H + h) : 0;
    float* my_save = save_buf + (long long)save_id * K * D;   // 64-bit: N*K*D passes 2^31 at batch 2 x 4K x d_ff 8192 x 36 stages

    // ================================================================
    // Phase 1: Forward recompute (warp-shuffle) with save_buf writes
    // ================================================================
    float v0 = 0.0f, v1 = 0.0f;
    if (valid) {
        int x_offset = (my_n * H + h) * D;
        v0 = x_in[x_offset + 2 * p];
        v1 = x_in[x_offset + 2 * p + 1];
    }

    for (int k = 0; k < K; k++) {
        // Save pre-stage state
        if (valid) {
            my_save[k * D + 2 * p]     = v0;
            my_save[k * D + 2 * p + 1] = v1;
        }

        // Load coupling tile with padding for transpose access
        int cp_base = ((b * K + k) * H + h) * coupling_size;
        {
            int total_rows = f_feat * half_D;
            int total_elems = total_rows * half_D;
            for (int i = tid_linear; i < total_elems; i += total_threads) {
                int row = i / half_D;
                int col = i & (half_D - 1);
                coupling_tile[row * coupling_stride + col] = coupling[cp_base + i];
            }
        }
        __syncthreads();

        int stride = -shifts[k];

        // Butterfly permutation via shfl
        int p0_elem, p1_elem;
        butterfly_shfl_info(p, stride, D, p0_elem, p1_elem);
        int p0_thread = p0_elem >> 1, p0_reg = p0_elem & 1;
        int p1_thread = p1_elem >> 1, p1_reg = p1_elem & 1;

        float src0_v0 = SHFL_BROADCAST(v0, p0_thread);
        float src0_v1 = SHFL_BROADCAST(v1, p0_thread);
        float x0_val = p0_reg ? src0_v1 : src0_v0;
        float src1_v0 = SHFL_BROADCAST(v0, p1_thread);
        float src1_v1 = SHFL_BROADCAST(v1, p1_thread);
        float x1_val = p1_reg ? src1_v1 : src1_v0;

        // Features + norm
        float my_S0  = x0_val * x0_val + x1_val * x1_val;
        float my_V1  = x0_val;
        float my_V2  = x1_val;
        int next = (p + 1) & (half_D - 1);
        float nb_x0 = SHFL_BROADCAST(x0_val, next);
        float nb_x1 = SHFL_BROADCAST(x1_val, next);
        float my_B12 = x0_val * nb_x1 - x1_val * nb_x0;

        float sq = my_S0*my_S0 + my_V1*my_V1 + my_V2*my_V2 + my_B12*my_B12;
        for (int offset = half_D/2; offset > 0; offset >>= 1)
            sq += SHFL_XOR(sq, offset);
        float inv_norm = rsqrtf(sq + 1e-6f);
        my_S0 *= inv_norm; my_V1 *= inv_norm; my_V2 *= inv_norm; my_B12 *= inv_norm;

        // Spill features to smem for einsum
        my_features[p]              = my_S0;
        my_features[half_D + p]     = my_V1;
        my_features[2*half_D + p]   = my_V2;
        my_features[3*half_D + p]   = my_B12;
        __syncwarp();

        // Einsum (uses padded stride)
        int total = f_feat * half_D;
        float mod0=0, mod1=0, mod2=0, mod3=0;
        int j = 0;
        #pragma unroll 4
        for (; j+3 < total; j+=4) {
            mod0 += my_features[j]   * coupling_tile[j     * coupling_stride + p];
            mod1 += my_features[j+1] * coupling_tile[(j+1) * coupling_stride + p];
            mod2 += my_features[j+2] * coupling_tile[(j+2) * coupling_stride + p];
            mod3 += my_features[j+3] * coupling_tile[(j+3) * coupling_stride + p];
        }
        float modulation = mod0+mod1+mod2+mod3;
        for (; j < total; j++)
            modulation += my_features[j] * coupling_tile[j * coupling_stride + p];

        // Angle + rotation + scale
        float c_base = 1.0f, s_base = 0.0f;
        if (valid) {
            int a_off = ((b*K+k)*H+h)*half_D + p;
            sincosf(angles[a_off], &s_base, &c_base);
        }
        float correction = FAST_TANHF(modulation);
        // exact cos/sin of the tanh correction (was a Cayley approximation, T-109)
        float c_corr, s_corr;
        sincosf(correction, &s_corr, &c_corr);
        float c = c_base * c_corr - s_base * s_corr;
        float s = s_base * c_corr + c_base * s_corr;

        float r0 = x0_val * c - x1_val * s;
        float r1 = x0_val * s + x1_val * c;

        if (valid) {
            int s_off = ((b*K+k)*H+h)*D;
            r0 *= scales[s_off + 2*p];
            r1 *= scales[s_off + 2*p + 1];
        }

        // Inverse butterfly
        int my_e0 = 2*p, my_e1 = 2*p+1;
        bool e0_is_p0 = (my_e0 & stride) == 0;
        int src_pair_e0 = e0_is_p0 ? reverse_pair_idx(my_e0, stride) : reverse_pair_idx(my_e0-stride, stride);
        float e0_r0 = SHFL_BROADCAST(r0, src_pair_e0);
        float e0_r1 = SHFL_BROADCAST(r1, src_pair_e0);
        float new_v0 = e0_is_p0 ? e0_r0 : e0_r1;

        bool e1_is_p0 = (my_e1 & stride) == 0;
        int src_pair_e1 = e1_is_p0 ? reverse_pair_idx(my_e1, stride) : reverse_pair_idx(my_e1-stride, stride);
        float e1_r0 = SHFL_BROADCAST(r0, src_pair_e1);
        float e1_r1 = SHFL_BROADCAST(r1, src_pair_e1);
        float new_v1 = e1_is_p0 ? e1_r0 : e1_r1;

        v0 = new_v0;
        v1 = new_v1;
        __syncthreads();
    }

    // ================================================================
    // Phase 2: Backward pass (reverse stage order, warp-shuffle)
    // ================================================================

    // Load grad_out into registers
    v0 = 0.0f; v1 = 0.0f;
    if (valid) {
        int out_offset = ((b * N + my_n) * H + h) * D;
        v0 = grad_out[out_offset + 2 * p];
        v1 = grad_out[out_offset + 2 * p + 1];
    }

    for (int k = K - 1; k >= 0; k--) {
        // Load coupling tile with padding
        int cp_base = ((b*K+k)*H+h) * coupling_size;
        {
            int total_rows = f_feat * half_D;
            int total_elems = total_rows * half_D;
            for (int i = tid_linear; i < total_elems; i += total_threads) {
                int row = i / half_D;
                int col = i & (half_D - 1);
                coupling_tile[row * coupling_stride + col] = coupling[cp_base + i];
            }
        }
        __syncthreads();

        int stride = -shifts[k];

        // --- Load saved state via shfl (from save_buf -> registers) ---
        // save_buf stores the pre-stage D-vector in natural order (2p, 2p+1)
        float saved_v0 = 0.0f, saved_v1 = 0.0f;
        if (valid) {
            saved_v0 = my_save[k * D + 2 * p];
            saved_v1 = my_save[k * D + 2 * p + 1];
        }

        // Butterfly permutation of saved state to get (x0, x1) pairs
        int p0_elem, p1_elem;
        butterfly_shfl_info(p, stride, D, p0_elem, p1_elem);
        int p0_thread = p0_elem >> 1, p0_reg = p0_elem & 1;
        int p1_thread = p1_elem >> 1, p1_reg = p1_elem & 1;

        float s0v0 = SHFL_BROADCAST(saved_v0, p0_thread);
        float s0v1 = SHFL_BROADCAST(saved_v1, p0_thread);
        float x0_val = p0_reg ? s0v1 : s0v0;
        float s1v0 = SHFL_BROADCAST(saved_v0, p1_thread);
        float s1v1 = SHFL_BROADCAST(saved_v1, p1_thread);
        float x1_val = p1_reg ? s1v1 : s1v0;

        // --- Recompute features + L2 norm (in registers) ---
        float my_S0  = x0_val * x0_val + x1_val * x1_val;
        float my_V1  = x0_val;
        float my_V2  = x1_val;
        int next = (p + 1) & (half_D - 1);
        int prev = (p - 1 + half_D) & (half_D - 1);
        float nb_x0 = SHFL_BROADCAST(x0_val, next);
        float nb_x1 = SHFL_BROADCAST(x1_val, next);
        float my_B12 = x0_val * nb_x1 - x1_val * nb_x0;

        float sq = my_S0*my_S0 + my_V1*my_V1 + my_V2*my_V2 + my_B12*my_B12;
        for (int offset = half_D/2; offset > 0; offset >>= 1)
            sq += SHFL_XOR(sq, offset);
        float phase2_inv_norm = rsqrtf(sq + 1e-6f);
        my_S0 *= phase2_inv_norm; my_V1 *= phase2_inv_norm;
        my_V2 *= phase2_inv_norm; my_B12 *= phase2_inv_norm;

        // Spill normalized features to smem
        my_features[p]              = my_S0;
        my_features[half_D + p]     = my_V1;
        my_features[2*half_D + p]   = my_V2;
        my_features[3*half_D + p]   = my_B12;
        __syncwarp();

        // --- Recompute modulation (einsum from smem, padded stride) ---
        int total = f_feat * half_D;
        float mod0=0, mod1=0, mod2=0, mod3=0;
        int j = 0;
        #pragma unroll 4
        for (; j+3 < total; j+=4) {
            mod0 += my_features[j]   * coupling_tile[j     * coupling_stride + p];
            mod1 += my_features[j+1] * coupling_tile[(j+1) * coupling_stride + p];
            mod2 += my_features[j+2] * coupling_tile[(j+2) * coupling_stride + p];
            mod3 += my_features[j+3] * coupling_tile[(j+3) * coupling_stride + p];
        }
        float mod = mod0+mod1+mod2+mod3;
        for (; j < total; j++)
            mod += my_features[j] * coupling_tile[j * coupling_stride + p];

        // --- Recompute rotation ---
        float c_base = 1.0f, s_base = 0.0f;
        int a_off = ((b*K+k)*H+h)*half_D + p;
        if (valid) {
            sincosf(angles[a_off], &s_base, &c_base);
        }
        float tanh_mod = FAST_TANHF(mod);
        // exact cos/sin of the tanh correction (was a Cayley approximation, T-109)
        float c_corr, s_corr;
        sincosf(tanh_mod, &s_corr, &c_corr);
        float c = c_base * c_corr - s_base * s_corr;
        float s = s_base * c_corr + c_base * s_corr;

        float r0 = x0_val * c - x1_val * s;
        float r1 = x0_val * s + x1_val * c;

        int s_off_base = ((b*K+k)*H+h)*D;
        float sc0 = valid ? scales[s_off_base + 2*p] : 1.0f;
        float sc1 = valid ? scales[s_off_base + 2*p+1] : 1.0f;

        // --- Get output gradients via butterfly shfl ---
        // grad is in (v0, v1) in natural order. Need it at permuted positions.
        float gv0_src = SHFL_BROADCAST(v0, p0_thread);
        float gv1_src = SHFL_BROADCAST(v1, p0_thread);
        float g_o0 = p0_reg ? gv1_src : gv0_src;
        float gv0_src1 = SHFL_BROADCAST(v0, p1_thread);
        float gv1_src1 = SHFL_BROADCAST(v1, p1_thread);
        float g_o1 = p1_reg ? gv1_src1 : gv0_src1;

        // --- Gradient through scale ---
        float g_sc0 = g_o0 * r0;
        float g_sc1 = g_o1 * r1;

        float g_r0 = g_o0 * sc0;
        float g_r1 = g_o1 * sc1;

        // --- Gradient through rotation ---
        float g_theta = g_r0 * (-r1) + g_r1 * r0;

        // --- Gradient through tanh ---
        float g_modulation = g_theta * (1.0f - tanh_mod * tanh_mod);

        // Accumulate grad_angles and grad_scales via atomicAdd
        if (valid) {
            atomicAdd(&grad_angles[a_off], g_theta);
            atomicAdd(&grad_scales[s_off_base + 2*p],   g_sc0);
            atomicAdd(&grad_scales[s_off_base + 2*p+1], g_sc1);
        }

        // --- Save features + grad_mod for post-kernel coupling gradient ---
        if (valid) {
            long long bkh = (long long)(b*K+k)*H + h;
            long long feat_off = (bkh * N + my_n) * (f_feat * half_D);
            for (int jj = 0; jj < f_feat; jj++)
                feat_buf[feat_off + jj*half_D + p] = my_features[jj*half_D + p];
            long long gm_off = (bkh * N + my_n) * half_D;
            grad_mod_buf[gm_off + p] = g_modulation;
        }

        // --- Spill grad_mod to smem for transpose einsum ---
        my_grad_mod[p] = g_modulation;
        __syncwarp();

        // --- Feature gradient via coupling transpose (einsum from smem, padded) ---
        float gn_S0 = 0.0f, gn_V1 = 0.0f, gn_V2 = 0.0f;
        float gn_B12 = 0.0f, gn_B12_prev = 0.0f;
        for (int q = 0; q < half_D; q++) {
            float gm_q = my_grad_mod[q];
            gn_S0       += gm_q * coupling_tile[p              * coupling_stride + q];
            gn_V1       += gm_q * coupling_tile[(half_D + p)   * coupling_stride + q];
            gn_V2       += gm_q * coupling_tile[(2*half_D + p) * coupling_stride + q];
            gn_B12      += gm_q * coupling_tile[(3*half_D + p) * coupling_stride + q];
            gn_B12_prev += gm_q * coupling_tile[(3*half_D + prev) * coupling_stride + q];
        }

        // --- Chain rule through L2 normalization ---
        float fn_S0  = my_S0,  fn_V1 = my_V1, fn_V2 = my_V2, fn_B12 = my_B12;
        float dot_partial = gn_S0*fn_S0 + gn_V1*fn_V1 + gn_V2*fn_V2 + gn_B12*fn_B12;
        for (int offset = half_D/2; offset > 0; offset >>= 1)
            dot_partial += SHFL_XOR(dot_partial, offset);

        float g_raw_S0  = (gn_S0  - fn_S0  * dot_partial) * phase2_inv_norm;
        float g_raw_V1  = (gn_V1  - fn_V1  * dot_partial) * phase2_inv_norm;
        float g_raw_V2  = (gn_V2  - fn_V2  * dot_partial) * phase2_inv_norm;
        float g_raw_B12 = (gn_B12 - fn_B12 * dot_partial) * phase2_inv_norm;

        float fn_B12_prev = SHFL_BROADCAST(fn_B12, prev);
        float g_raw_B12_prev = (gn_B12_prev - fn_B12_prev * dot_partial) * phase2_inv_norm;

        // --- Gradient through Clifford features -> input pair ---
        float nb_x0_prev = SHFL_BROADCAST(x0_val, prev);
        float nb_x1_prev = SHFL_BROADCAST(x1_val, prev);

        float g_x0 = g_raw_S0 * 2.0f * x0_val + g_raw_V1;
        float g_x1 = g_raw_S0 * 2.0f * x1_val + g_raw_V2;

        // B12[p] = x0*x1n - x1*x0n
        g_x0 += g_raw_B12 * nb_x1;   // d/dx0 of x0*x1n
        g_x1 += g_raw_B12 * (-nb_x0); // d/dx1 of -x1*x0n

        // B12[prev] = x0p*x1 - x1p*x0
        g_x0 += g_raw_B12_prev * (-nb_x1_prev); // d/dx0 of -x1p*x0
        g_x1 += g_raw_B12_prev * nb_x0_prev;     // d/dx1 of x0p*x1

        // Add rotation-only gradient
        float gx0_rot =  g_r0 * c + g_r1 * s;
        float gx1_rot = -g_r0 * s + g_r1 * c;
        g_x0 += gx0_rot;
        g_x1 += gx1_rot;

        // --- Inverse butterfly: scatter input grad back to owning threads ---
        // g_x0 goes to element p0_elem, g_x1 goes to element p1_elem
        int my_e0 = 2*p, my_e1 = 2*p+1;
        bool e0_is_p0 = (my_e0 & stride) == 0;
        int src_pair_e0 = e0_is_p0 ? reverse_pair_idx(my_e0, stride) : reverse_pair_idx(my_e0-stride, stride);
        float e0_gx0 = SHFL_BROADCAST(g_x0, src_pair_e0);
        float e0_gx1 = SHFL_BROADCAST(g_x1, src_pair_e0);
        float new_gv0 = e0_is_p0 ? e0_gx0 : e0_gx1;

        bool e1_is_p0 = (my_e1 & stride) == 0;
        int src_pair_e1 = e1_is_p0 ? reverse_pair_idx(my_e1, stride) : reverse_pair_idx(my_e1-stride, stride);
        float e1_gx0 = SHFL_BROADCAST(g_x0, src_pair_e1);
        float e1_gx1 = SHFL_BROADCAST(g_x1, src_pair_e1);
        float new_gv1 = e1_is_p0 ? e1_gx0 : e1_gx1;

        v0 = new_gv0;
        v1 = new_gv1;
        __syncthreads();
    }

    // Write grad_x
    if (valid) {
        int x_offset = (my_n * H + h) * D;
        if (B == 1) {
            grad_x[x_offset + 2*p]     = v0;
            grad_x[x_offset + 2*p + 1] = v1;
        } else {
            atomicAdd(&grad_x[x_offset + 2*p],     v0);
            atomicAdd(&grad_x[x_offset + 2*p + 1], v1);
        }
    }
}

#endif // GIVENS_COUPLING_BACKWARD_WARP_H
