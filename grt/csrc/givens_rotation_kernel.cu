/*
 * Fused Givens Rotation Kernel - Main Entry Point
 * Includes forward, backward, MoR, and coupling kernels
 *
 * ROCm Compatible: Yes (AMD RDNA3 tested)
 * Hopper Features: Documented but disabled for ROCm compatibility
 */

// Include all kernel headers
#include "givens_common.h"
#include "givens_forward.h"
#include "givens_backward.h"
#include "givens_mor.h"
#include "givens_coupling_forward.h"
#include "givens_coupling_forward_warp.h"
#include "givens_coupling_backward.h"
#include "givens_coupling_backward_warp.h"
#include "givens_coupling_forward_chunked.h"
#include "givens_coupling_backward_chunked.h"

// ---------------------------------------------------------------------------
// Launch wrappers (C-linkage for Python binding)
// ---------------------------------------------------------------------------

extern "C" void launch_givens_fwd(
    const void* x, const float* angles, const float* alpha,
    const float* scales, void* out, const int* shifts,
    int N, int H, int D, int K, int use_half
) {
    int blocks = N * H;
    int threads = D / 2;
    if (threads > 1024) threads = 1024;

    // Select kernel variant based on problem size
    // Priority: 1) Large D (multi-block), 2) Small N (persistent), 3) Standard
    //
    // NOTE: the tiled forward kernel silently drops butterfly rotations whose
    // partner crosses the D_TILE boundary (givens_forward.h:118-119), so its
    // forward != backward (backward uses the standard kernel which handles all
    // pairs). Disabled until tile-shared global-memory accumulation is added.
    // d_ff=4096 fits the standard path: 16.5 KB shared memory per block.
    if (false && D > 2048) {
        // Multi-block tiled for large D (highest priority for performance)
        int tiles_per_th = (D + D_TILE - 1) / D_TILE;
        blocks = N * H * tiles_per_th;
        threads = min(D_TILE / 2, 1024);
        int smem_bytes = (D_TILE + (D_TILE / SHMEM_BANK_SIZE) * 4) * sizeof(float);
        if (use_half) {
            givens_fwd_tiled_kernel<__half><<<blocks, threads, smem_bytes>>>(
                (const __half*)x, angles, alpha, scales, (__half*)out, shifts, N, H, D, K, tiles_per_th
            );
        } else {
            givens_fwd_tiled_kernel<float><<<blocks, threads, smem_bytes>>>(
                (const float*)x, angles, alpha, scales, (float*)out, shifts, N, H, D, K, tiles_per_th
            );
        }
    } else if (N < PERSISTENT_BATCH_THRESHOLD) {
        // Persistent thread block for small batches
        int smem_bytes = (D + (D / SHMEM_BANK_SIZE) * 4) * sizeof(float);
        if (use_half) {
            givens_fwd_persistent_kernel<__half><<<blocks, threads, smem_bytes>>>(
                (const __half*)x, angles, alpha, scales, (__half*)out, shifts, N, H, D, K
            );
        } else {
            givens_fwd_persistent_kernel<float><<<blocks, threads, smem_bytes>>>(
                (const float*)x, angles, alpha, scales, (float*)out, shifts, N, H, D, K
            );
        }
    } else {
        // Standard kernel for normal cases
        int smem_bytes = (D + (D / SHMEM_BANK_SIZE) * 4) * sizeof(float);
        if (use_half) {
            givens_fwd_kernel<__half><<<blocks, threads, smem_bytes>>>(
                (const __half*)x, angles, alpha, scales, (__half*)out, shifts, N, H, D, K
            );
        } else {
            givens_fwd_kernel<float><<<blocks, threads, smem_bytes>>>(
                (const float*)x, angles, alpha, scales, (float*)out, shifts, N, H, D, K
            );
        }
    }
}

extern "C" void launch_givens_bwd(
    const void* grad_out, const void* x_in,
    const float* angles, const float* alpha, const float* scales,
    void* grad_x, float* grad_angles, float* grad_alpha, float* grad_scales,
    const int* shifts, float* save_buf,
    int N, int H, int D, int K, int use_half
) {
    int blocks = N * H;
    int threads = D / 2;
    if (threads > 1024) threads = 1024;

    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int smem_bytes = padded_D * sizeof(float);  // only work buffer needed

    if (use_half) {
        givens_bwd_kernel<__half><<<blocks, threads, smem_bytes>>>(
            (const __half*)grad_out, (const __half*)x_in, angles, alpha, scales,
            (__half*)grad_x, grad_angles, grad_alpha, grad_scales, shifts, save_buf,
            N, H, D, K
        );
    } else {
        givens_bwd_kernel<float><<<blocks, threads, smem_bytes>>>(
            (const float*)grad_out, (const float*)x_in, angles, alpha, scales,
            (float*)grad_x, grad_angles, grad_alpha, grad_scales, shifts, save_buf,
            N, H, D, K
        );
    }
}

// ---------------------------------------------------------------------------
// Precomputed trig launch wrappers (alpha=0 FFN fast path)
// ---------------------------------------------------------------------------

extern "C" void launch_givens_precomp_fwd(
    const void* x, const float* cos_angles, const float* sin_angles,
    const float* scales, void* out, const int* shifts,
    int N, int H, int D, int K, int use_half
) {
    int blocks = N * H;
    int threads = D / 2;
    if (threads > 1024) threads = 1024;
    int smem_bytes = (D + (D / SHMEM_BANK_SIZE) * 4) * sizeof(float);

    if (use_half) {
        givens_fwd_precomp_kernel<__half><<<blocks, threads, smem_bytes>>>(
            (const __half*)x, cos_angles, sin_angles, scales, (__half*)out, shifts, N, H, D, K
        );
    } else {
        givens_fwd_precomp_kernel<float><<<blocks, threads, smem_bytes>>>(
            (const float*)x, cos_angles, sin_angles, scales, (float*)out, shifts, N, H, D, K
        );
    }
}

extern "C" void launch_givens_precomp_bwd(
    const void* grad_out, const void* x_in,
    const float* cos_angles, const float* sin_angles, const float* scales,
    void* grad_x, float* grad_cos, float* grad_sin, float* grad_scales,
    const int* shifts, float* save_buf,
    int N, int H, int D, int K, int use_half
) {
    int blocks = N * H;
    int threads = D / 2;
    if (threads > 1024) threads = 1024;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int smem_bytes = padded_D * sizeof(float);

    if (use_half) {
        givens_bwd_precomp_kernel<__half><<<blocks, threads, smem_bytes>>>(
            (const __half*)grad_out, (const __half*)x_in, cos_angles, sin_angles, scales,
            (__half*)grad_x, grad_cos, grad_sin, grad_scales, shifts, save_buf,
            N, H, D, K
        );
    } else {
        givens_bwd_precomp_kernel<float><<<blocks, threads, smem_bytes>>>(
            (const float*)grad_out, (const float*)x_in, cos_angles, sin_angles, scales,
            (float*)grad_x, grad_cos, grad_sin, grad_scales, shifts, save_buf,
            N, H, D, K
        );
    }
}

extern "C" void launch_givens_radix4_precomp_fwd(
    const void* x, const float* cos_angles, const float* sin_angles,
    const float* scales, void* out, const int* shifts,
    int N, int H, int D, int K, int use_half
) {
    int blocks = N * H;
    // For large D (>=2048), use fewer threads to give Radix-4's heavier
    // per-thread work more register headroom.  For small D the kernel is
    // memory-bound and needs maximum thread-level parallelism.
    int threads = D / 2;
    if (D >= 2048) threads = min(threads, 256);
    if (threads > 1024) threads = 1024;
    int smem_bytes = (D + (D / SHMEM_BANK_SIZE) * 4) * sizeof(float);

    if (use_half) {
        givens_fwd_radix4_precomp_kernel<__half><<<blocks, threads, smem_bytes>>>(
            (const __half*)x, cos_angles, sin_angles, scales, (__half*)out, shifts, N, H, D, K
        );
    } else {
        givens_fwd_radix4_precomp_kernel<float><<<blocks, threads, smem_bytes>>>(
            (const float*)x, cos_angles, sin_angles, scales, (float*)out, shifts, N, H, D, K
        );
    }
}

extern "C" void launch_givens_radix4_precomp_bwd(
    const void* grad_out, const void* x_in,
    const float* cos_angles, const float* sin_angles, const float* scales,
    void* grad_x, float* grad_cos, float* grad_sin, float* grad_scales,
    const int* shifts, float* save_buf,
    int N, int H, int D, int K, int use_half
) {
    int blocks = N * H;
    int threads = D / 2;
    if (D >= 2048) threads = min(threads, 256);
    if (threads > 1024) threads = 1024;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int smem_bytes = padded_D * sizeof(float);

    if (use_half) {
        givens_bwd_radix4_precomp_kernel<__half><<<blocks, threads, smem_bytes>>>(
            (const __half*)grad_out, (const __half*)x_in, cos_angles, sin_angles, scales,
            (__half*)grad_x, grad_cos, grad_sin, grad_scales, shifts, save_buf,
            N, H, D, K
        );
    } else {
        givens_bwd_radix4_precomp_kernel<float><<<blocks, threads, smem_bytes>>>(
            (const float*)grad_out, (const float*)x_in, cos_angles, sin_angles, scales,
            (float*)grad_x, grad_cos, grad_sin, grad_scales, shifts, save_buf,
            N, H, D, K
        );
    }
}

extern "C" void launch_givens_mor_fwd(
    const void* x, const float* expert_angles, const float* expert_scales,
    const float* router_weights, void* out, const int* shifts,
    int N, int H, int D, int K, int E, int use_half
) {
    int blocks = N * H;
    int threads = D / 2;
    if (threads > 1024) threads = 1024;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int smem_bytes = (padded_D + E) * sizeof(float);

    if (use_half) {
        givens_mor_fwd_kernel<__half><<<blocks, threads, smem_bytes>>>(
            (const __half*)x, expert_angles, expert_scales, router_weights, (__half*)out, shifts,
            N, H, D, K, E
        );
    } else {
        givens_mor_fwd_kernel<float><<<blocks, threads, smem_bytes>>>(
            (const float*)x, expert_angles, expert_scales, router_weights, (float*)out, shifts,
            N, H, D, K, E
        );
    }
}

extern "C" void launch_givens_mor_bwd(
    const void* grad_out, const void* x_in,
    const float* expert_angles, const float* expert_scales,
    const float* router_weights,
    void* grad_x, float* grad_expert_angles, float* grad_expert_scales,
    float* grad_router, const int* shifts, float* save_buf,
    int N, int H, int D, int K, int E, int use_half
) {
    int blocks = N * H;
    int threads = D / 2;
    if (threads > 1024) threads = 1024;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int smem_bytes = (padded_D + 2 * E) * sizeof(float);

    if (use_half) {
        givens_mor_bwd_kernel<__half><<<blocks, threads, smem_bytes>>>(
            (const __half*)grad_out, (const __half*)x_in, expert_angles, expert_scales, router_weights,
            (__half*)grad_x, grad_expert_angles, grad_expert_scales, grad_router,
            shifts, save_buf, N, H, D, K, E
        );
    } else {
        givens_mor_bwd_kernel<float><<<blocks, threads, smem_bytes>>>(
            (const float*)grad_out, (const float*)x_in, expert_angles, expert_scales, router_weights,
            (float*)grad_x, grad_expert_angles, grad_expert_scales, grad_router,
            shifts, save_buf, N, H, D, K, E
        );
    }
}

// ---------------------------------------------------------------------------
// Coupling forward/backward launch wrappers (for attention)
// ---------------------------------------------------------------------------

extern "C" void launch_givens_coupling_fwd(
    const float* x, const float* angles,
    const float* coupling, const float* scales, const int* shifts, float* out,
    int N, int H, int D, int K, int B
) {
    int half_D = D / 2;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int f_feat = 4;

    // Warp-shuffle path: when half_D fits in one warp, keep the entire
    // D-vector in registers and use __shfl for butterfly permutations.
    // Shared memory holds the coupling tile + per-token feature spill.
    if (half_D <= WARP_SIZE) {
        int coupling_size = f_feat * half_D * half_D;

        // TPB: how many tokens per block. Each token = one warp (half_D threads).
        // Maximize TPB while staying under 1024 threads total.
        int TPB = 1024 / half_D;
        if (TPB < 1) TPB = 1;
        // Cap to 32 tokens per block — beyond that, diminishing returns
        // from coupling tile amortization vs launch overhead.
        if (TPB > 32) TPB = 32;

        // smem: coupling tile (shared across tokens) + per-token feature spill
        // (4*P floats/token, written by the warp and read back by the einsum).
        int feat_smem = TPB * f_feat * half_D;
        int smem_bytes = (coupling_size + feat_smem) * (int)sizeof(float);

        int n_tiles = (N + TPB - 1) / TPB;
        int blocks = B * n_tiles * H;
        dim3 threads(half_D, TPB);

        givens_coupling_fwd_warp_kernel<<<blocks, threads, smem_bytes>>>(
            x, angles, coupling, scales, shifts, out,
            N, H, D, K, B
        );
        return;
    }

    // Dispatch: if half_D * COUPLING_TPB <= 1024 AND coupling tile fits in 64 KiB,
    // use the original tiled kernel. Otherwise fall back to the chunked variant.
    int total_threads_orig = half_D * COUPLING_TPB;
    int orig_coupling_size = f_feat * half_D * half_D;
    int orig_smem_bytes = (orig_coupling_size + COUPLING_TPB * (padded_D + (2 + f_feat) * half_D)) * (int)sizeof(float);

    bool use_original = (total_threads_orig <= 1024) && (orig_smem_bytes <= 65536);

    if (use_original) {
        int TPB = COUPLING_TPB;
        int n_tiles = (N + TPB - 1) / TPB;
        int blocks = B * n_tiles * H;
        dim3 threads(half_D, TPB);

        givens_coupling_fwd_tiled_kernel<<<blocks, threads, orig_smem_bytes>>>(
            x, angles, coupling, scales, shifts, out,
            N, H, D, K, B
        );
    } else {
        // Chunked path. Pick TPB so half_D*TPB<=1024, then choose largest
        // chunk_rows (power of 2) that keeps total smem under 60 KiB
        // (64 KiB hardware limit minus 4 KiB margin for static shmem).
        int TPB = 1024 / half_D;
        if (TPB < 1) TPB = 1;
        int per_token_smem = padded_D + (2 + f_feat) * half_D;
        int budget = 60 * 1024;
        int per_token_bytes = TPB * per_token_smem * (int)sizeof(float);
        int avail_for_chunk = budget - per_token_bytes;
        if (avail_for_chunk < (int)sizeof(float) * half_D) {
            TPB = 1;
            per_token_bytes = per_token_smem * (int)sizeof(float);
            avail_for_chunk = budget - per_token_bytes;
        }
        int max_chunk_rows = avail_for_chunk / ((int)sizeof(float) * half_D);
        if (max_chunk_rows < 1) max_chunk_rows = 1;
        int chunk_rows = 1;
        while (chunk_rows * 2 <= max_chunk_rows) chunk_rows *= 2;

        int n_tiles = (N + TPB - 1) / TPB;
        int blocks = B * n_tiles * H;
        dim3 threads(half_D, TPB);
        int smem_bytes = (chunk_rows * half_D + TPB * per_token_smem) * (int)sizeof(float);

        givens_coupling_fwd_chunked_kernel<<<blocks, threads, smem_bytes>>>(
            x, angles, coupling, scales, shifts, out,
            N, H, D, K, B, chunk_rows
        );
    }
}

extern "C" void launch_givens_coupling_bwd(
    const float* grad_out, const float* x_in,
    const float* angles,
    const float* coupling, const float* scales,
    const int* shifts,
    float* grad_x, float* grad_angles, float* grad_scales,
    float* feat_buf, float* grad_mod_buf,
    float* save_buf,
    int N, int H, int D, int K, int B
) {
    int half_D = D / 2;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int f_feat = 4;

    // Warp-shuffle backward path (matches forward warp dispatch)
    if (half_D <= WARP_SIZE) {
        int coupling_stride = half_D + 1;  // padded for transpose access
        int coupling_tile_size = f_feat * half_D * coupling_stride;
        int TPB = 1024 / half_D;
        if (TPB < 1) TPB = 1;
        if (TPB > 32) TPB = 32;

        // smem: padded coupling tile + per-token features + per-token grad_mod
        int feat_smem = TPB * f_feat * half_D;
        int gmod_smem = TPB * half_D;
        int smem_bytes = (coupling_tile_size + feat_smem + gmod_smem) * (int)sizeof(float);

        int n_tiles = (N + TPB - 1) / TPB;
        int blocks = B * n_tiles * H;
        dim3 threads(half_D, TPB);

        givens_coupling_bwd_warp_kernel<<<blocks, threads, smem_bytes>>>(
            grad_out, x_in, angles, coupling, scales, shifts,
            grad_x, grad_angles, grad_scales,
            feat_buf, grad_mod_buf, save_buf,
            N, H, D, K, B
        );
        return;
    }

    int coupling_stride = half_D + 1;
    int orig_coupling_tile_size = f_feat * half_D * coupling_stride;

    // Same dispatch logic as forward, mirrored. Backward smem layout is bigger
    // (extra grad_mod, gx0_rot, gx1_rot, accums) so the threshold's a bit tighter.
    int total_threads_orig = half_D * COUPLING_TPB;
    int orig_smem = (orig_coupling_tile_size
                     + COUPLING_TPB * (padded_D + (2 + f_feat + 3) * half_D)
                     + half_D + D) * (int)sizeof(float);
    bool use_original = (total_threads_orig <= 1024) && (orig_smem <= 65536);

    if (use_original) {
        int TPB = COUPLING_TPB;
        int n_tiles = (N + TPB - 1) / TPB;
        int blocks = B * n_tiles * H;
        dim3 threads(half_D, TPB);

        givens_coupling_bwd_tiled_kernel<<<blocks, threads, orig_smem>>>(
            grad_out, x_in, angles, coupling, scales, shifts,
            grad_x, grad_angles, grad_scales,
            feat_buf, grad_mod_buf, save_buf,
            N, H, D, K, B
        );
    } else {
        // Chunked path: pick chunk_rows so total smem stays under 60 KiB
        // (64 KiB hardware limit, leaving 4 KiB margin for static shmem).
        int TPB = 1024 / half_D;
        if (TPB < 1) TPB = 1;
        int per_token_smem = padded_D + (2 + f_feat + 3) * half_D;
        int extras = half_D + D;
        int budget = 60 * 1024;   // hard cap, in bytes
        int per_token_bytes = TPB * per_token_smem * (int)sizeof(float);
        int extras_bytes = extras * (int)sizeof(float);
        int avail_for_chunk = budget - per_token_bytes - extras_bytes;
        if (avail_for_chunk < (int)sizeof(float) * half_D) {
            // Even one row doesn't fit; reduce TPB further (extreme cases)
            TPB = 1;
            per_token_bytes = per_token_smem * (int)sizeof(float);
            avail_for_chunk = budget - per_token_bytes - extras_bytes;
        }
        int max_chunk_rows = avail_for_chunk / ((int)sizeof(float) * half_D);
        if (max_chunk_rows < 1) max_chunk_rows = 1;
        // Round down to power of 2 for cleaner indexing
        int chunk_rows = 1;
        while (chunk_rows * 2 <= max_chunk_rows) chunk_rows *= 2;

        int n_tiles = (N + TPB - 1) / TPB;
        int blocks = B * n_tiles * H;
        dim3 threads(half_D, TPB);
        int smem_bytes = (chunk_rows * half_D + TPB * per_token_smem + extras) * (int)sizeof(float);

        givens_coupling_bwd_chunked_kernel<<<blocks, threads, smem_bytes>>>(
            grad_out, x_in, angles, coupling, scales, shifts,
            grad_x, grad_angles, grad_scales,
            feat_buf, grad_mod_buf, save_buf,
            N, H, D, K, B, chunk_rows
        );
    }
}
