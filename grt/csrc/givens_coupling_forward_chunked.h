/*
 * Chunked Coupling Forward Kernel (handles large D where the coupling matrix
 * doesn't fit in shared memory all at once).
 *
 * The original givens_coupling_fwd_tiled_kernel loads the entire coupling
 * matrix (size 4*P*P) into shared memory per stage. At D=64 (P=32) that's
 * 16 KiB; fine. At D=256 (P=128) it's 256 KiB; the typical 64 KiB shmem
 * budget can't hold it.
 *
 * This chunked variant processes the coupling matrix in `Q` column-chunks
 * per stage (each chunk holds 4*P*(P/Q) floats). For each chunk:
 *   1. Cooperative load of the chunk into shared memory
 *   2. Each thread (responsible for output dim p) accumulates partial
 *      modulation contributions from this chunk's columns
 *   3. After all Q chunks, full modulation is in registers per thread
 *
 * Math: modulation[p] = sum_j features[j] * coupling[j, p]
 *   We split the inner sum-over-j into `Q` chunks. Each chunk loads a
 *   contiguous slice of rows (same number of cols per row).
 *
 * NOTE: Output dim p is one per thread. Coupling is laid out as [4P rows, P cols].
 *   - "Column" axis is p (output dim); thread fixes p and reads down a column.
 *   - "Row" axis is j (feature index). Chunking ALONG ROWS (the j-axis) keeps
 *     each thread's read pattern contiguous within the chunk.
 *
 * Shared memory layout (per block):
 *   coupling_chunk[CHUNK_ROWS * half_D]   -- one chunk of coupling rows
 *   work[TPB * padded_D]
 *   smem_x0[TPB * P]
 *   smem_x1[TPB * P]
 *   features[TPB * 4*P]
 *
 * blockDim:
 *   threadIdx.x = p        (range [0, half_D))
 *   threadIdx.y = token    (range [0, TPB))
 *   total = half_D * TPB   (must be <= 1024)
 *
 * For D=256 (half_D=128), TPB=8 -> 1024 threads. CHUNK_ROWS=64 -> coupling
 * chunk size = 64*128 = 8192 floats = 32 KiB. Total smem ~ 80 KiB.
 *
 * For D=512 (half_D=256), TPB=4 -> 1024 threads. CHUNK_ROWS=32 -> 32*256
 * = 8192 floats = 32 KiB. Total smem ~ 96 KiB.
 */

#ifndef GIVENS_COUPLING_FORWARD_CHUNKED_H
#define GIVENS_COUPLING_FORWARD_CHUNKED_H

#include "givens_common.h"

LAUNCH_BOUNDS_FWD  // 1024 thread max
__global__ void givens_coupling_fwd_chunked_kernel(
    const float* __restrict__ x,           // [N, H, D]
    const float* __restrict__ angles,       // [B, K, H, P]
    const float* __restrict__ coupling,    // [B, K, H, 4*P, P]
    const float* __restrict__ scales,      // [B, K, H, D]
    const int* __restrict__ shifts,
    float* __restrict__ out,               // [B, N, H, D]
    int N, int H, int D, int K, int B,
    int chunk_rows                         // CHUNK_ROWS (compile-time-friendly)
) {
    int half_D = D / 2;
    int f_feat = 4;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int total_rows = f_feat * half_D;       // 4P
    int chunk_size = chunk_rows * half_D;   // floats per coupling chunk

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
    float* coupling_chunk = smem;
    float* work_base      = smem + chunk_size;
    float* smem_x0_base   = work_base + TPB * padded_D;
    float* smem_x1_base   = smem_x0_base + TPB * half_D;
    float* feat_base      = smem_x1_base + TPB * half_D;

    float* my_work     = work_base + token_slot * padded_D;
    float* my_x0       = smem_x0_base + token_slot * half_D;
    float* my_x1       = smem_x1_base + token_slot * half_D;
    float* my_features = feat_base + token_slot * f_feat * half_D;

    // Load input
    if (valid) {
        int x_offset = (my_n * H + h) * D;
        my_work[SHMEM_IDX(2 * p)]     = x[x_offset + 2 * p];
        my_work[SHMEM_IDX(2 * p + 1)] = x[x_offset + 2 * p + 1];
    }
    __syncthreads();

    int tid_linear = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;

    for (int k = 0; k < K; k++) {
        // --- Extract pairs ---
        int p0_perm, p1_perm;
        compute_pair_indices(p, shifts[k], D, p0_perm, p1_perm);
        if (valid) {
            my_x0[p] = my_work[SHMEM_IDX(p0_perm)];
            my_x1[p] = my_work[SHMEM_IDX(p1_perm)];
        }
        __syncthreads();

        // --- Compute features ---
        if (valid) {
            float x0 = my_x0[p];
            float x1 = my_x1[p];
            int next = (p + 1) & (half_D - 1);
            float x0n = my_x0[next];
            float x1n = my_x1[next];

            my_features[p]                  = x0 * x0 + x1 * x1;
            my_features[half_D + p]         = x0;
            my_features[2 * half_D + p]     = x1;
            my_features[3 * half_D + p]     = x0 * x1n - x1 * x0n;
        }
        __syncthreads();

        // --- L2-normalize features ---
        if (valid) {
            float sq = 0.0f;
            sq += my_features[p]              * my_features[p];
            sq += my_features[half_D + p]     * my_features[half_D + p];
            sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
            sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];

            // Cross-warp reduction via shared memory (half_D > WARP_SIZE possible).
            // Use the feature buffer's first slot as a temporary reduce area.
            // Alternative: per-warp shfl + cross-warp shmem combine.
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                sq += SHFL_XOR(sq, offset);
            // If half_D > WARP_SIZE, sq now holds per-warp partial; combine via shmem
            if (half_D > WARP_SIZE) {
                __shared__ float warp_sums[16];   // up to 16 warps per token (max half_D=512)
                int lane = p & (WARP_SIZE - 1);
                int warp = p / WARP_SIZE;
                int n_warps = half_D / WARP_SIZE;
                // First lane of each warp writes
                if (lane == 0) warp_sums[token_slot * n_warps + warp] = sq;
                __syncthreads();
                // All threads read and sum
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

        // --- Coupling einsum across Q chunks ---
        // modulation[p] = sum_{j=0..total_rows-1} features[j] * coupling[j, p]
        // Split j into chunks of chunk_rows.
        float modulation = 0.0f;
        int cp_base = ((b * K + k) * H + h) * total_rows * half_D;
        int n_chunks = (total_rows + chunk_rows - 1) / chunk_rows;
        for (int chunk = 0; chunk < n_chunks; chunk++) {
            int row_start = chunk * chunk_rows;
            int rows_this = (row_start + chunk_rows <= total_rows)
                            ? chunk_rows : (total_rows - row_start);
            int elems_this = rows_this * half_D;

            // Cooperative load
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

        // --- Effective angle, rotation, scale ---
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

    if (valid) {
        int out_offset = ((b * N + my_n) * H + h) * D;
        out[out_offset + 2 * p]     = my_work[SHMEM_IDX(2 * p)];
        out[out_offset + 2 * p + 1] = my_work[SHMEM_IDX(2 * p + 1)];
    }
}

#endif // GIVENS_COUPLING_FORWARD_CHUNKED_H
