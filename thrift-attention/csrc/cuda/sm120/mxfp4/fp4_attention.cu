// SM120 MXFP4 attention baseline

#include <cstdint>
#include <float.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "thriftattention/sm120/cuda_common.cuh"

template<typename T, bool CAUSAL, int BLOCK_Q, int BLOCK_KV, int HEAD_DIM,
         int HEAD_DIM_2, int SCALE_DIM,
         int NUM_WARPS, int WARP_Q>
__launch_bounds__(NUM_WARPS * TA_WARP_SIZE)
__global__
void fp4_attention_kernel(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e8m0* S_Q,
    const __nv_fp8_e8m0* S_K,
    const __nv_fp8_e8m0* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / TA_WARP_SIZE;
    const int lane_id = tid % TA_WARP_SIZE;

    const int num_q_blocks = ta_cdiv(q_len, BLOCK_Q);
    const int q_bid = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int batch_id = q_bid / num_q_heads;
    const int q_head = q_bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;
    const int v_kv = ta_cdiv(kv_capacity, 128) * 128;

    Q += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM_2;
    K += kv_bid * kv_capacity * HEAD_DIM_2;
    V += kv_bid * HEAD_DIM * (v_kv / 2);

    S_Q += (q_bid * q_len + q_block_id * BLOCK_Q) * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM;
    S_V += kv_bid * v_kv * SCALE_DIM;
    O += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM;

    extern __shared__ uint8_t smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);
    const uint32_t V_smem = Q_sf_smem + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e8m0);
    const uint32_t V_sf_smem = V_smem + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4];
    uint32_t K_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t V_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
    uint32_t sfQ_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K];
    uint32_t sfK_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K];
    uint32_t sfV_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N];

    float rowmax[WARP_Q/MMA_M][2];
    float rowsum[WARP_Q/MMA_M][2] = {};
    float O_rmem[WARP_Q/MMA_M][HEAD_DIM/MMA_N][4] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q/MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    // ---- load Q data global -> shared (swizzled) -> registers ----
    ta_gmem_to_smem<BLOCK_Q, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(Q_smem, Q, tid, HEAD_DIM_2);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // Pre-compute swizzled base address for Q ldmatrix.
    // Row offsets that are multiples of 8 rows can be added (swizzle repeats every 8 rows).
    // Column tile offsets use XOR: swizzle(base + col_delta) == swizzle(base) ^ col_delta.
    uint32_t Q_ld_base;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = ta_swizzle<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            uint32_t addr = Q_ld_base;
            addr += mma_id_q * MMA_M * HEAD_DIM_2;  // row: MMA_M=16, 16%8==0
            addr ^= mma_id_d * (MMA_K / 2);          // col via XOR
            ta_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }

    // ---- load Q scales global -> shared -> registers ----
    ta_load_scales<BLOCK_Q, SCALE_DIM, TB_SIZE, __nv_fp8_e8m0>(Q_sf_smem, S_Q, SCALE_DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + sf_row_q;
            const uint32_t offset = (row * SCALE_DIM + mma_id_d * 2) * (uint32_t)sizeof(__nv_fp8_e8m0);
            sfQ_rmem[mma_id_q][mma_id_d] = ta_ld_shared_u16(Q_sf_smem + offset);
        }
    }

    // ---- KV loop ----
    __syncthreads();
    const uint32_t K_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_sf_smem = K_smem + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    const int total_kv_iters = ta_cdiv(kv_len, BLOCK_KV);
    const int max_kv_pos = q_block_id * BLOCK_Q + BLOCK_Q - 1;
    const int num_kv_iters = CAUSAL
        ? min(max_kv_pos / BLOCK_KV + 1, total_kv_iters)
        : total_kv_iters;

    // Pre-compute base address for K ldmatrix.
    uint32_t K_ld_base;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base = ta_swizzle<HEAD_DIM_2>(K_smem + row_off * HEAD_DIM_2 + col_off);
    }

    // Precompute per-thread query row offsets within the block.
    const int q_block_start = q_block_id * BLOCK_Q;
    const int q_row_upper = warp_id * WARP_Q + (lane_id / 4);       // rows 0-7 of MMA tile
    const int q_row_lower = q_row_upper + 8;                         // rows 8-15 of MMA tile
    const int q_row_upper_global = q_block_start + q_row_upper;
    const int q_row_lower_global = q_block_start + q_row_lower;
    const int k_col_base = (lane_id % 4) * 2;                        // base key col within MMA_N tile

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
        uint32_t S_fp4_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K];

        const int k_block_start = kv_iter * BLOCK_KV;
        const bool needs_causal_mask = CAUSAL && ((k_block_start + BLOCK_KV - 1) > q_block_start);

        // Load K first; issue V in a second async group so QK can overlap
        // with the V transfer.
        ta_gmem_to_smem<BLOCK_KV, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_smem, K, tid, HEAD_DIM_2);;
        ta_load_scales<BLOCK_KV, SCALE_DIM, TB_SIZE, __nv_fp8_e8m0>(K_sf_smem, S_K, SCALE_DIM, tid);
        asm volatile("cp.async.commit_group;");
        ta_gmem_to_smem<HEAD_DIM, BLOCK_KV / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_smem, V, tid, v_kv / 2);
        ta_load_scales<HEAD_DIM, BLOCK_KV / 32, TB_SIZE, __nv_fp8_e8m0>(V_sf_smem, S_V, v_kv / 32, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        if constexpr (HEAD_DIM / MMA_K >= 2) {
            // ldmatrix_x4: lanes 0-15 cover mma_id_d=0, lanes 16-31 cover mma_id_d=1
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                uint32_t addr = K_ld_base + mma_id_kv * MMA_N * HEAD_DIM_2;
                ta_ldmatrix_x4(K_rmem[mma_id_kv][0], addr);
            }
        } else {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                    uint32_t addr = K_ld_base;
                    addr += mma_id_kv * MMA_N * HEAD_DIM_2;
                    addr ^= mma_id_d * (MMA_K / 2);
                    ta_ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
                }
        }

        const int sf_row_k = (lane_id / 4);
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                const int row = mma_id_kv * MMA_N + sf_row_k;
                const uint32_t offset = (row * SCALE_DIM + mma_id_d * 2) * (uint32_t)sizeof(__nv_fp8_e8m0);
                sfK_rmem[mma_id_kv][mma_id_d] = ta_ld_shared_u16(K_sf_smem + offset);
            }
        }

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++)
                    ta_mma_m16n8k64_mxfp4(
                        Q_rmem[mma_id_q][mma_id_d],
                        K_rmem[mma_id_kv][mma_id_d],
                        sfQ_rmem[mma_id_q][mma_id_d],
                        sfK_rmem[mma_id_kv][mma_id_d],
                        S_rmem[mma_id_q][mma_id_kv]);

        // ---- Apply causal mask on blocks overlapping the query tile ----
        if constexpr (CAUSAL) {
            if (needs_causal_mask) {
                for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                        const int k_phys_0 = mma_id_kv * MMA_N + k_col_base;
                        const int k_phys_1 = k_phys_0 + 1;
                        const int k_col_0 = k_block_start + ta_kv_logical_from_physical(k_phys_0);
                        const int k_col_1 = k_block_start + ta_kv_logical_from_physical(k_phys_1);

                        // regs[0]: (q_row_upper, k_col_0), regs[1]: (q_row_upper, k_col_1)
                        // regs[2]: (q_row_lower, k_col_0), regs[3]: (q_row_lower, k_col_1)
                        if (k_col_0 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][0] = -INFINITY;
                        if (k_col_1 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][1] = -INFINITY;
                        if (k_col_0 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][2] = -INFINITY;
                        if (k_col_1 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][3] = -INFINITY;
                    }
                }
            }
        }

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            // apply softmax scale
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for (int reg_id = 0; reg_id < 4; reg_id++)
                    S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

            // rowmax over this KV block
            float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
            }
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

            // new rowmax
            this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

            // rescale previous O for new rowmax
            float rescale[2];
            rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
            rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
            }

            // save new rowmax
            rowmax[mma_id_q][0] = this_rowmax[0];
            rowmax[mma_id_q][1] = this_rowmax[1];

            // softmax numerator and rowsumexp
            float this_rowsumexp[2] = {};
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
                regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
                regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
                regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

                this_rowsumexp[0] += regs[0] + regs[1];
                this_rowsumexp[1] += regs[2] + regs[3];
            }

            // butterfly reduction within 4 threads
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

            // accumulate to running rowsum
            rowsum[mma_id_q][0] = rowsum[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
            rowsum[mma_id_q][1] = rowsum[mma_id_q][1] * rescale[1] + this_rowsumexp[1];

            constexpr float FP4_RANGE = 6.0f;
            constexpr float FP4_MAX = 6.0f;
            float sf_P_upper[BLOCK_KV / MMA_N / 4];
            float sf_P_lower[BLOCK_KV / MMA_N / 4];

            for (int blk = 0; blk < BLOCK_KV / MMA_N / 4; blk++) {
                const int t0 = 4 * blk;
                const int t1 = t0 + 1;
                const int t2 = t0 + 2;
                const int t3 = t0 + 3;
                const float gmax_upper = rowmax[mma_id_q][0];
                const float gmax_lower = rowmax[mma_id_q][1];
                const bool valid_upper = gmax_upper > -FLT_MAX * 0.5f;
                const bool valid_lower = gmax_lower > -FLT_MAX * 0.5f;
                const float inv_upper = valid_upper ? (FP4_MAX * __expf(rowmax[mma_id_q][0] - gmax_upper)) : 0.0f;
                const float inv_lower = valid_lower ? (FP4_MAX * __expf(rowmax[mma_id_q][1] - gmax_lower)) : 0.0f;
                sf_P_upper[blk] = valid_upper ? (FP4_RANGE / FP4_MAX * __expf(gmax_upper - rowmax[mma_id_q][0])) : 1.0f;
                sf_P_lower[blk] = valid_lower ? (FP4_RANGE / FP4_MAX * __expf(gmax_lower - rowmax[mma_id_q][1])) : 1.0f;

                S_rmem[mma_id_q][t0][0] *= inv_upper;
                S_rmem[mma_id_q][t0][1] *= inv_upper;
                S_rmem[mma_id_q][t1][0] *= inv_upper;
                S_rmem[mma_id_q][t1][1] *= inv_upper;
                S_rmem[mma_id_q][t0][2] *= inv_lower;
                S_rmem[mma_id_q][t0][3] *= inv_lower;
                S_rmem[mma_id_q][t1][2] *= inv_lower;
                S_rmem[mma_id_q][t1][3] *= inv_lower;
                S_rmem[mma_id_q][t2][0] *= inv_upper;
                S_rmem[mma_id_q][t2][1] *= inv_upper;
                S_rmem[mma_id_q][t3][0] *= inv_upper;
                S_rmem[mma_id_q][t3][1] *= inv_upper;
                S_rmem[mma_id_q][t2][2] *= inv_lower;
                S_rmem[mma_id_q][t2][3] *= inv_lower;
                S_rmem[mma_id_q][t3][2] *= inv_lower;
                S_rmem[mma_id_q][t3][3] *= inv_lower;
            }

            for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
                int blk = 4 * g;
                float *r0 = S_rmem[mma_id_q][blk];
                float *r1 = S_rmem[mma_id_q][blk + 1];
                float *r2 = S_rmem[mma_id_q][blk + 2];
                float *r3 = S_rmem[mma_id_q][blk + 3];

                S_fp4_rmem[mma_id_q][0][2 * g] = ta_cvt_8xf32_to_e2m1_packed(
                    r0[1], r0[0], r1[1], r1[0],
                    r2[1], r2[0], r3[1], r3[0]);
                S_fp4_rmem[mma_id_q][0][2 * g + 1] = ta_cvt_8xf32_to_e2m1_packed(
                    r0[3], r0[2], r1[3], r1[2],
                    r2[3], r2[2], r3[3], r3[2]);
            }

            for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV / MMA_K; mma_sc_id++) {
                int base = mma_sc_id * 2;
                uint32_t sfP_upper_packed = ta_cvt_2xf32_to_e8m0_packed(
                    sf_P_upper[base + 1], sf_P_upper[base + 0]);

                uint32_t sfP_lower_packed = ta_cvt_2xf32_to_e8m0_packed(
                    sf_P_lower[base + 1], sf_P_lower[base + 0]);

                S_fp4_s_rmem[mma_id_q][mma_sc_id] =
                    (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
            }
        }

        // V layout in smem: [HEAD_DIM, BLOCK_KV/2], stride = BLOCK_KV/2 bytes.
        // ldmatrix_x4: lanes 0-15 address tile mma_id_d, lanes 16-31 address tile mma_id_d+1.
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d += 2) {
                const int n_idx = mma_id_d * MMA_N + (lane_id / 16) * MMA_N + (lane_id % 8);
                const int k_byte_offset = mma_id_kv * 32 + ((lane_id % 16) / 8) * 16;
                uint32_t addr = ta_swizzle<BLOCK_KV / 2>(
                    V_smem + n_idx * (BLOCK_KV / 2) + k_byte_offset);

                ta_ldmatrix_x4(V_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        constexpr int V_SF_STRIDE = BLOCK_KV / 32;
        const int sf_col_v = lane_id / 4;
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                const int hd_col = mma_id_d * MMA_N + sf_col_v;
                const uint32_t offset = (hd_col * V_SF_STRIDE + mma_id_kv * 2) * (uint32_t)sizeof(__nv_fp8_e8m0);
                sfV_rmem[mma_id_kv][mma_id_d] = ta_ld_shared_u16(V_sf_smem + offset);
            }

        // MMA O += P @ V
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
                    ta_mma_m16n8k64_mxfp4(
                        S_fp4_rmem[mma_id_q][mma_id_kv],
                        V_rmem[mma_id_kv][mma_id_d],
                        S_fp4_s_rmem[mma_id_q][mma_id_kv],
                        sfV_rmem[mma_id_kv][mma_id_d],
                        O_rmem[mma_id_q][mma_id_d]);

        K += BLOCK_KV * HEAD_DIM_2;
        S_K += BLOCK_KV * SCALE_DIM;

        V  += BLOCK_KV / 2;
        S_V += BLOCK_KV / 32;
    }

    constexpr float FP4_RANGE_INV = 1.0f / 6.0f;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];

            float norm0 = FP4_RANGE_INV / rowsum[mma_id_q][0];
            float norm1 = FP4_RANGE_INV / rowsum[mma_id_q][1];

            regs[0] *= norm0;
            regs[1] *= norm0;
            regs[2] *= norm1;
            regs[3] *= norm1;

            reinterpret_cast<typename Traits::vec2*>(O + (row + 0) * HEAD_DIM + col)[0] =
                Traits::pack2(regs[0], regs[1]);
            reinterpret_cast<typename Traits::vec2*>(O + (row + 8) * HEAD_DIM + col)[0] =
                Traits::pack2(regs[2], regs[3]);
        }
}

template<typename T, bool CAUSAL, int HEAD_DIM>
static void launch_fp4_attention(
    const __nv_fp4x2_e2m1* Q, const __nv_fp4x2_e2m1* K, const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e8m0* S_Q, const __nv_fp8_e8m0* S_K, const __nv_fp8_e8m0* S_V,
    T* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads) {

    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 32;
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int WARP_Q = 16;
    constexpr int NUM_WARPS = BLOCK_Q / WARP_Q;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;

    const int num_blocks = bs * ta_cdiv(q_len, BLOCK_Q);

    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e8m0);
    constexpr int v_phase_smem = HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1)
                                + HEAD_DIM * (BLOCK_KV / 32) * sizeof(__nv_fp8_e8m0);
    constexpr int k_phase_smem = BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                                + BLOCK_KV * SCALE_DIM * sizeof(__nv_fp8_e8m0);
    constexpr int smem_size = q_phase_smem + v_phase_smem > k_phase_smem
                            ? q_phase_smem + v_phase_smem : k_phase_smem;

    auto kernel = fp4_attention_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM,
                                       HEAD_DIM_2, SCALE_DIM,
                                       NUM_WARPS, WARP_Q>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<num_blocks, TB_SIZE, smem_size>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

template<typename T, bool CAUSAL>
static void dispatch_fp4_attention(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e8m0* S_Q,
    const __nv_fp8_e8m0* S_K,
    const __nv_fp8_e8m0* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
    if (head_dim == 64) {
        launch_fp4_attention<T, CAUSAL, 64>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    } else {
        launch_fp4_attention<T, CAUSAL, 128>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    }
}

template<typename T, bool CAUSAL>
static void fp4_attention_mxfp4_typed(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
    auto Q = reinterpret_cast<const __nv_fp4x2_e2m1*>(Q_raw);
    auto K = reinterpret_cast<const __nv_fp4x2_e2m1*>(K_raw);
    auto V = reinterpret_cast<const __nv_fp4x2_e2m1*>(V_raw);
    auto S_Q = reinterpret_cast<const __nv_fp8_e8m0*>(S_Q_raw);
    auto S_K = reinterpret_cast<const __nv_fp8_e8m0*>(S_K_raw);
    auto S_V = reinterpret_cast<const __nv_fp8_e8m0*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    dispatch_fp4_attention<T, CAUSAL>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
        kv_capacity, num_q_heads, num_kv_heads, head_dim);
}

void fp4_attention_causal_mxfp4(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        fp4_attention_mxfp4_typed<__nv_bfloat16, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    } else {
        fp4_attention_mxfp4_typed<half, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}

void fp4_attention_noncausal_mxfp4(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        fp4_attention_mxfp4_typed<__nv_bfloat16, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    } else {
        fp4_attention_mxfp4_typed<half, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}
