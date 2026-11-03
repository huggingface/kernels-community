/*
 * Fused RotationalMixer Kernel — single stage, tiled TPB design.
 *
 * Does exactly one "k stage" of the PyTorch mixer forward:
 *   pairs  = x.view(..., n_groups, G, 2)
 *   feats  = clifford_features(pairs)                   # [..., n_groups, 4*G]
 *   feats /= (feats.norm(dim=-1, keepdim=True) + eps)   # per-group L2 norm
 *   mod    = einsum("...gj, gji -> ...gi", feats, coupling)  # [..., n_groups, G]
 *   theta  = angles + tanh(mod)                         # [n_groups, G]
 *   (r0,r1) = Givens(theta) @ (x0,x1)                   # elementwise per pair
 *   out    = r * scales                                 # per-dim scale
 *
 * Layout (inside kernel — already-permuted space):
 *   x_in, x_out:   [N, D]   where D = 2 * n_groups * G
 *   angles:        [n_groups, G]
 *   coupling:      [n_groups, 4*G, G]
 *   scales:        [n_groups, 2*G]   (already softplus'd by caller? no — raw, we apply softplus here)
 *
 * Butterfly permutation is applied by the Python caller around this kernel so
 * multi-stage loops stay clean.
 *
 * Block layout:
 *   threadIdx.x = pair index within group, p ∈ [0, G)
 *   threadIdx.y = token slot within tile,  t ∈ [0, TPB)
 *   gridDim.x   = ceil(N / TPB) * n_groups
 *
 * Each block processes TPB tokens for ONE group. The coupling tile for that
 * group (4*G*G floats, 1KB for G=16) fits easily in shared memory and is shared
 * across all TPB tokens — ~16x reduction in global reads of coupling.
 *
 * Writes output in-place at the same (token, group)-slice it read from.
 */

#ifndef FUSED_MIXER_KERNEL_H
#define FUSED_MIXER_KERNEL_H

#include "givens_common.h"

#ifndef MIXER_TPB
#define MIXER_TPB 16
#endif

// Softplus, matching torch.nn.functional.softplus (numerically stable).
__device__ inline float softplus_stable(float x) {
    // For large x, softplus(x) ~= x; for very negative x, ~= exp(x).
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}


// Forward: grid = (n_blocks_total,), block = (G, TPB).
// n_blocks_total = ceil(N/TPB) * n_groups
__launch_bounds__(256, 4)
__global__ void fused_mixer_fwd_kernel(
    const float* __restrict__ x_in,       // [N, D]
    const float* __restrict__ angles,     // [n_groups, G]
    const float* __restrict__ coupling,   // [n_groups, 4*G, G]
    const float* __restrict__ raw_scales, // [n_groups, 2*G]
    float*       __restrict__ x_out,      // [N, D]
    // save buffers (forward writes, backward reads) — may be nullptr if not saving
    float*       __restrict__ save_feat_norm,  // [N, n_groups, 4*G]  — normalized features
    float*       __restrict__ save_inv_norm,   // [N, n_groups]       — 1 / (L + eps) per (token, group)
    float*       __restrict__ save_theta,      // [N, n_groups, G]
    int N, int D, int n_groups, int G, float eps
) {
    const int TPB = MIXER_TPB;
    const int F_FEAT = 4;
    const int coup_tile_size = F_FEAT * G * G;  // floats per group

    int p = threadIdx.x;            // [0, G)
    int t = threadIdx.y;            // [0, TPB)

    // Decode block -> (token_tile, group_idx)
    int n_tiles = (N + TPB - 1) / TPB;
    int block_id = blockIdx.x;
    int token_tile = block_id / n_groups;
    int g = block_id % n_groups;

    int n = token_tile * TPB + t;
    bool valid = (n < N);

    // --- Shared memory ---
    extern __shared__ float smem[];
    float* s_coup  = smem;                              // [F_FEAT*G*G]     coupling tile (shared across tokens)
    float* s_x0    = s_coup + coup_tile_size;           // [TPB * G]        pair-0 per token
    float* s_x1    = s_x0   + TPB * G;                  // [TPB * G]        pair-1 per token
    float* s_feat  = s_x1   + TPB * G;                  // [TPB * F_FEAT*G] features per token
    float* s_scl   = s_feat + TPB * F_FEAT * G;         // [2*G]            group's per-pair scales (shared)

    // --- 1. Load coupling tile for this group (cooperative) ---
    // Coupling shape: [n_groups, F*G, G]. This group's slice: coupling[g, :, :].
    const float* coup_g = coupling + g * coup_tile_size;
    for (int i = p + t * G; i < coup_tile_size; i += G * TPB) {
        s_coup[i] = coup_g[i];
    }

    // --- 2. Load per-group scales (shared) and apply softplus ---
    if (t == 0) {
        s_scl[p]         = softplus_stable(raw_scales[g * 2 * G + 2 * p]);
        s_scl[p + G]     = softplus_stable(raw_scales[g * 2 * G + 2 * p + 1]);
    }

    // --- 3. Load per-token pairs for this group ---
    // Group g in token n has pairs (g*G + p). D index of pair p = 2 * (g*G + p).
    if (valid) {
        int pair_idx = g * G + p;           // pair index in [0, n_groups*G)
        int d_base = 2 * pair_idx;           // dim index in [0, D)
        s_x0[t * G + p] = x_in[n * D + d_base];
        s_x1[t * G + p] = x_in[n * D + d_base + 1];
    } else {
        s_x0[t * G + p] = 0.0f;
        s_x1[t * G + p] = 0.0f;
    }
    __syncthreads();

    // --- 4. Clifford features [S0, V1, V2, B12] for each token ---
    // S0 = x0^2 + x1^2     at index [t, 0*G + p]
    // V1 = x0              at index [t, 1*G + p]
    // V2 = x1              at index [t, 2*G + p]
    // B12 = x0 * x0_next - x1 * x1_next  (next = cyclic shift by 1 within group). This is the MIXER
    // bivector (ops.mixer_stage_ref, mixer_bivector=True); the attention kernels use x0*x1n - x1*x0n.
    // Until 2026-09-26 this kernel used the attention form: rel err 2.9e-2 vs reference (T-109).
    float x0 = s_x0[t * G + p];
    float x1 = s_x1[t * G + p];
    int p_next = (p + 1) % G;
    // We need x0_next and x1_next AFTER all threads have written — which they have via __syncthreads above.
    float x0_next = s_x0[t * G + p_next];
    float x1_next = s_x1[t * G + p_next];

    float s0_val  = x0 * x0 + x1 * x1;
    float v1_val  = x0;
    float v2_val  = x1;
    float b12_val = x0 * x0_next - x1 * x1_next;

    s_feat[t * F_FEAT * G + 0 * G + p] = s0_val;
    s_feat[t * F_FEAT * G + 1 * G + p] = v1_val;
    s_feat[t * F_FEAT * G + 2 * G + p] = v2_val;
    s_feat[t * F_FEAT * G + 3 * G + p] = b12_val;
    __syncthreads();

    // --- 5. L2-normalize features per (token, group) ---
    // One thread per token computes the norm; writes back into s_feat.
    // Thread p=0 computes, also saves inv_norm for backward.
    __shared__ float s_inv_norm[MIXER_TPB];
    if (p == 0) {
        float sq = 0.0f;
        for (int j = 0; j < F_FEAT * G; j++) {
            float v = s_feat[t * F_FEAT * G + j];
            sq += v * v;
        }
        float inv = 1.0f / (sqrtf(sq) + eps);
        s_inv_norm[t] = inv;
        for (int j = 0; j < F_FEAT * G; j++) {
            s_feat[t * F_FEAT * G + j] *= inv;
        }
    }
    __syncthreads();

    // --- 6. Modulation: mod[t, p] = sum_j feat[t, j] * coup[j, p] ---
    float mod = 0.0f;
    for (int j = 0; j < F_FEAT * G; j++) {
        mod += s_feat[t * F_FEAT * G + j] * s_coup[j * G + p];
    }

    // --- 7. Effective angle and rotation ---
    float theta = angles[g * G + p] + FAST_TANHF(mod);
    float c = cosf(theta);
    float s = sinf(theta);

    float r0 = x0 * c - x1 * s;
    float r1 = x0 * s + x1 * c;

    // --- 8. Scale and write output ---
    float scl0 = s_scl[p];
    float scl1 = s_scl[p + G];

    if (valid) {
        int pair_idx = g * G + p;
        int d_base = 2 * pair_idx;
        x_out[n * D + d_base]     = r0 * scl0;
        x_out[n * D + d_base + 1] = r1 * scl1;
    }

    // --- 9. Save features, inv_norm, theta for backward (if buffers provided) ---
    if (save_feat_norm != nullptr && valid) {
        // Save the normalized features (post-L2) — backward needs these.
        for (int j = p; j < F_FEAT * G; j += G) {
            save_feat_norm[n * n_groups * F_FEAT * G + g * F_FEAT * G + j] =
                s_feat[t * F_FEAT * G + j];
        }
    }
    if (save_inv_norm != nullptr && valid && p == 0) {
        save_inv_norm[n * n_groups + g] = s_inv_norm[t];
    }
    if (save_theta != nullptr && valid) {
        save_theta[n * n_groups * G + g * G + p] = theta;
    }
}



// ===========================================================================
// Backward kernel
// ===========================================================================
//
// Grid / block layout: same as forward — (n_groups * n_tiles, 1), (G, TPB).
// Inputs:
//   grad_out:      [N, D]
//   x_in:          [N, D]             (saved from forward — already permuted)
//   angles:        [n_groups, G]
//   coupling:      [n_groups, 4*G, G]
//   raw_scales:    [n_groups, 2*G]
//   save_feat_norm:[N, n_groups, 4*G]  — normalized features
//   save_inv_norm: [N, n_groups]       — 1/(L+eps) per token-group
//   save_theta:    [N, n_groups, G]    — angle+tanh(mod)
//
// Outputs (per-block, atomic-adds into the shared global accumulators):
//   grad_x:        [N, D]
//   grad_angles:   [n_groups, G]              (summed across tokens)
//   grad_coupling: [n_groups, 4*G, G]         (summed across tokens)
//   grad_raw_scales:[n_groups, 2*G]           (summed across tokens)
//
// In-block atomicAdd on shared-mem accumulators, then a single per-element
// global atomicAdd per block — TPB-x reduction in contention.
//
__launch_bounds__(256, 4)
__global__ void fused_mixer_bwd_kernel(
    const float* __restrict__ grad_out,       // [N, D]
    const float* __restrict__ x_in,           // [N, D]
    const float* __restrict__ angles,         // [n_groups, G]         — only for recomputing mod = theta - angle (for tanh' = 1 - tanh^2)
    const float* __restrict__ coupling,       // [n_groups, 4*G, G]
    const float* __restrict__ raw_scales,     // [n_groups, 2*G]
    const float* __restrict__ save_feat_norm, // [N, n_groups, 4*G]
    const float* __restrict__ save_inv_norm,  // [N, n_groups]
    const float* __restrict__ save_theta,     // [N, n_groups, G]
    float* __restrict__ grad_x,               // [N, D]
    float* __restrict__ grad_angles,          // [n_groups, G]
    float* __restrict__ grad_coupling,        // [n_groups, 4*G, G]
    float* __restrict__ grad_raw_scales,      // [n_groups, 2*G]
    int N, int D, int n_groups, int G, float eps
) {
    const int TPB = MIXER_TPB;
    const int F_FEAT = 4;
    const int coup_tile_size = F_FEAT * G * G;

    int p = threadIdx.x;
    int t = threadIdx.y;

    int n_tiles = (N + TPB - 1) / TPB;
    int block_id = blockIdx.x;
    int token_tile = block_id / n_groups;
    int g = block_id % n_groups;

    int n = token_tile * TPB + t;
    bool valid = (n < N);

    // --- Shared memory layout ---
    // s_coup            [F*G*G]              coupling tile for this group
    // s_featn           [TPB * F*G]          saved normalized features
    // s_x0, s_x1        [TPB * G]            input pair values
    // s_dmod            [TPB * G]            per-token dmod (along p)
    // s_dfeat           [TPB * F*G]          scratch for d_feat_norm -> d_feat_raw
    // s_gang_accum      [G]                  grad_angles accumulator (one per p)
    // s_gscl_accum      [2*G]                grad_raw_scales accumulator
    // s_gcoup_accum     [F*G*G]              grad_coupling accumulator
    extern __shared__ float smem[];
    float* s_coup     = smem;
    float* s_featn    = s_coup    + coup_tile_size;
    float* s_x0       = s_featn   + TPB * F_FEAT * G;
    float* s_x1       = s_x0      + TPB * G;
    float* s_dmod     = s_x1      + TPB * G;
    float* s_dfeat    = s_dmod    + TPB * G;
    float* s_gang     = s_dfeat   + TPB * F_FEAT * G;
    float* s_gscl     = s_gang    + G;
    float* s_gcoup    = s_gscl    + 2 * G;
    __shared__ float s_inv_norm[MIXER_TPB];

    // --- Load coupling tile ---
    const float* coup_g = coupling + g * coup_tile_size;
    for (int i = p + t * G; i < coup_tile_size; i += G * TPB) {
        s_coup[i] = coup_g[i];
    }

    // --- Zero the in-block accumulators (cooperative) ---
    for (int i = p + t * G; i < G; i += G * TPB) s_gang[i] = 0.0f;
    for (int i = p + t * G; i < 2 * G; i += G * TPB) s_gscl[i] = 0.0f;
    for (int i = p + t * G; i < coup_tile_size; i += G * TPB) s_gcoup[i] = 0.0f;

    // --- Load per-token x and saved tensors ---
    float x0 = 0.0f, x1 = 0.0f;
    float theta = 0.0f;
    if (valid) {
        int pair_idx = g * G + p;
        int d_base = 2 * pair_idx;
        x0 = x_in[n * D + d_base];
        x1 = x_in[n * D + d_base + 1];
        s_x0[t * G + p] = x0;
        s_x1[t * G + p] = x1;
        theta = save_theta[n * n_groups * G + g * G + p];
        // Load feat_norm for this thread's pair contributions.
        for (int j = p; j < F_FEAT * G; j += G) {
            s_featn[t * F_FEAT * G + j] =
                save_feat_norm[n * n_groups * F_FEAT * G + g * F_FEAT * G + j];
        }
    } else {
        s_x0[t * G + p] = 0.0f;
        s_x1[t * G + p] = 0.0f;
        for (int j = p; j < F_FEAT * G; j += G) {
            s_featn[t * F_FEAT * G + j] = 0.0f;
        }
    }
    if (p == 0 && valid) {
        s_inv_norm[t] = save_inv_norm[n * n_groups + g];
    } else if (p == 0) {
        s_inv_norm[t] = 0.0f;
    }

    // Softplus scales + their sigmoid for backward
    float raw_scl0 = raw_scales[g * 2 * G + 2 * p];
    float raw_scl1 = raw_scales[g * 2 * G + 2 * p + 1];
    float scl0 = softplus_stable(raw_scl0);
    float scl1 = softplus_stable(raw_scl1);
    // sigmoid(raw) — derivative of softplus
    float sig0 = 1.0f / (1.0f + expf(-raw_scl0));
    float sig1 = 1.0f / (1.0f + expf(-raw_scl1));

    __syncthreads();

    // --- Load grad_out for this token-pair ---
    float dout0 = 0.0f, dout1 = 0.0f;
    if (valid) {
        int pair_idx = g * G + p;
        int d_base = 2 * pair_idx;
        dout0 = grad_out[n * D + d_base];
        dout1 = grad_out[n * D + d_base + 1];
    }

    // --- 1. d_r from d_out (undo scale) ---
    float dr0 = dout0 * scl0;
    float dr1 = dout1 * scl1;

    // --- 2. grad_raw_scales (per-token contribution accumulates in s_gscl) ---
    // d(out0)/d(raw_scl0) = r0 * sigmoid(raw_scl0).  out0 = r0 * scl0.
    float c_angle = cosf(theta);
    float s_angle = sinf(theta);
    float r0 = x0 * c_angle - x1 * s_angle;
    float r1 = x0 * s_angle + x1 * c_angle;
    float grad_scl0_contrib = dout0 * r0 * sig0;
    float grad_scl1_contrib = dout1 * r1 * sig1;
    atomicAdd(&s_gscl[2 * p], grad_scl0_contrib);
    atomicAdd(&s_gscl[2 * p + 1], grad_scl1_contrib);

    // --- 3. d_theta from d_r ---
    // dtheta = -r1 * dr0 + r0 * dr1
    float dtheta = -r1 * dr0 + r0 * dr1;

    // --- 4. grad_angles (accumulate) ---
    atomicAdd(&s_gang[p], dtheta);

    // --- 5. d_r -> d_x (rotation-only path; feature-path comes later) ---
    float dx0_rot = dr0 * c_angle + dr1 * s_angle;
    float dx1_rot = -dr0 * s_angle + dr1 * c_angle;

    // --- 6. d_theta -> d_mod via tanh: theta = angle + tanh(mod), so theta - angle IS tanh(mod)
    // and tanh'(mod) = 1 - tanh(mod)^2. (Until 2026-09-26 this applied tanh a second time, which
    // silently corrupted grad_x and grad_coupling while grad_angles/grad_scales stayed right.) ---
    float tanh_mod = theta - angles[g * G + p];
    float dmod = dtheta * (1.0f - tanh_mod * tanh_mod);
    s_dmod[t * G + p] = dmod;

    __syncthreads();

    // --- 7. grad_coupling[g, j, p] = sum_t s_featn[t, j] * s_dmod[t, p] ---
    // Each (p, j) gets summed over the TPB tokens of this tile, then atomicAdd to global.
    // Here j loops; each (thread.x=p, thread.y=t) handles different (j) to spread work.
    // Simpler: thread (p, t) contributes to grad_coupling[g, *, p] via its own dmod[t, p]
    //          and its own feat_norm[t, *]. Accumulate into s_gcoup with atomics.
    {
        float dmod_p = s_dmod[t * G + p];
        for (int j = 0; j < F_FEAT * G; j++) {
            atomicAdd(&s_gcoup[j * G + p], s_featn[t * F_FEAT * G + j] * dmod_p);
        }
    }

    // --- 8. d_feat_norm[j] = sum_p s_dmod[t, p] * coup[j, p] ---
    // One thread per (t, j). Use linear index j = tid.x * (F_FEAT*G / G) + offset.
    // Simpler: each p loops over F_FEAT chunks. We want s_dfeat[t, j] for j in [0, F_FEAT*G).
    // Since we have G threads along x, each thread computes F_FEAT output j values.
    for (int jp = 0; jp < F_FEAT; jp++) {
        int j = jp * G + p;  // one per (jp, p)
        float df = 0.0f;
        for (int pp = 0; pp < G; pp++) {
            df += s_dmod[t * G + pp] * s_coup[j * G + pp];
        }
        s_dfeat[t * F_FEAT * G + j] = df;
    }
    __syncthreads();

    // --- 9. d_feat_norm -> d_feat_raw via L2-norm backward ---
    // feat_norm = feat_raw / (L + eps); let inv = 1/(L+eps), feat_raw = feat_norm / inv
    // d/d(feat_raw) [feat_norm] = (I - feat_norm * (feat_raw/L)^T) / (L + eps)
    //                           = inv * (I - feat_norm * feat_norm^T * (L+eps)/L)
    // Equivalently: d_feat_raw = inv * (d_feat_norm - feat_norm * (d_feat_norm · feat_raw) / L).
    // Using only quantities we have:
    //   L + eps = 1/inv_norm;  L = 1/inv_norm - eps
    //   feat_raw = feat_norm / inv
    //   dot = sum_j d_feat_norm[j] * feat_raw[j] = (sum_j d_feat_norm[j] * feat_norm[j]) / inv_norm
    //   d_feat_raw[j] = inv_norm * d_feat_norm[j] - feat_norm[j] * dot_norm_space
    //     where dot_norm_space = sum_j d_feat_norm[j] * feat_norm[j] * inv_norm^2 / L * L
    //   Simpler algebra: with L = 1/inv - eps,  (L+eps)/L = 1/(1 - eps*inv).
    //   Define a = sum_j d_feat_norm[j] * feat_norm[j]   (scalar per token-group).
    //   d_feat_raw[j] = inv_norm * (d_feat_norm[j] - feat_norm[j] * a / (1 - eps*inv_norm))
    //
    // For numerical safety when eps*inv_norm ~ 1 (L ~ 0), we clamp.
    //
    // Thread p=0 computes the scalar `a`; broadcast via shared mem.
    __shared__ float s_a[MIXER_TPB];
    if (p == 0) {
        float a_val = 0.0f;
        for (int j = 0; j < F_FEAT * G; j++) {
            a_val += s_dfeat[t * F_FEAT * G + j] * s_featn[t * F_FEAT * G + j];
        }
        s_a[t] = a_val;
    }
    __syncthreads();

    // Now compute d_feat_raw in place in s_dfeat.
    {
        float inv = s_inv_norm[t];
        float denom = 1.0f - eps * inv;
        // If L == 0 then denom == 0 and the correction blows up; in that case
        // features are zero anyway so d_feat_raw = inv * d_feat_norm is fine.
        float safe_inv_denom = (fabsf(denom) > 1e-8f) ? (1.0f / denom) : 0.0f;
        float a_val = s_a[t];
        for (int jp = 0; jp < F_FEAT; jp++) {
            int j = jp * G + p;
            float dfn = s_dfeat[t * F_FEAT * G + j];
            float fn = s_featn[t * F_FEAT * G + j];
            float dfr = inv * (dfn - fn * a_val * safe_inv_denom);
            s_dfeat[t * F_FEAT * G + j] = dfr;
        }
    }
    __syncthreads();

    // --- 10. d_feat_raw -> d_x0, d_x1 via Clifford features ---
    // feat layout: [S0(0..G), V1(G..2G), V2(2G..3G), B12(3G..4G)]
    // B12[p] = x0[p]*x0[p_next] - x1[p]*x1[p_next]  (mixer bivector), so
    // d_x0[p] = d_S0[p] * 2*x0[p] + d_V1[p] + d_B12[p] * x0[p_next] + d_B12[p_prev] * x0[p_prev]
    // d_x1[p] = d_S0[p] * 2*x1[p] + d_V2[p] - d_B12[p] * x1[p_next] - d_B12[p_prev] * x1[p_prev]
    int p_next = (p + 1) % G;
    int p_prev = (p - 1 + G) % G;

    float d_S0  = s_dfeat[t * F_FEAT * G + 0 * G + p];
    float d_V1  = s_dfeat[t * F_FEAT * G + 1 * G + p];
    float d_V2  = s_dfeat[t * F_FEAT * G + 2 * G + p];
    float d_B12_p      = s_dfeat[t * F_FEAT * G + 3 * G + p];
    float d_B12_pprev  = s_dfeat[t * F_FEAT * G + 3 * G + p_prev];

    float x0_next = s_x0[t * G + p_next];
    float x1_next = s_x1[t * G + p_next];
    float x0_prev = s_x0[t * G + p_prev];
    float x1_prev = s_x1[t * G + p_prev];

    float dx0_feat = d_S0 * 2.0f * x0 + d_V1 + d_B12_p * x0_next + d_B12_pprev * x0_prev;
    float dx1_feat = d_S0 * 2.0f * x1 + d_V2 - d_B12_p * x1_next - d_B12_pprev * x1_prev;

    // --- 11. Total d_x = rot + feat ---
    float dx0_total = dx0_rot + dx0_feat;
    float dx1_total = dx1_rot + dx1_feat;

    if (valid) {
        int pair_idx = g * G + p;
        int d_base = 2 * pair_idx;
        grad_x[n * D + d_base]     = dx0_total;
        grad_x[n * D + d_base + 1] = dx1_total;
    }

    __syncthreads();

    // --- 12. Flush in-block accumulators to global ---
    // One atomicAdd per element per block (TPB-x reduction).
    if (t == 0) {
        atomicAdd(&grad_angles[g * G + p], s_gang[p]);
        atomicAdd(&grad_raw_scales[g * 2 * G + 2 * p], s_gscl[2 * p]);
        atomicAdd(&grad_raw_scales[g * 2 * G + 2 * p + 1], s_gscl[2 * p + 1]);
    }
    // Grad coupling: F*G*G elements per group. Threads (p, t) cooperatively flush.
    for (int i = p + t * G; i < coup_tile_size; i += G * TPB) {
        atomicAdd(&grad_coupling[g * coup_tile_size + i], s_gcoup[i]);
    }
}


#endif  // FUSED_MIXER_KERNEL_H
