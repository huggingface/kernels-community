// SM120 NVFP4 single-query attention baseline.

#include <cstdint>
#include <cstdio>
#include <float.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "thriftattention/sm120/cuda_common.cuh"

__device__ inline
void ldmatrix_x4_sqfp4(uint32_t reg[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
         : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
         : "r"(addr));
}

__device__ inline
void ldmatrix_x2_sqfp4(uint32_t reg[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
         : "=r"(reg[0]), "=r"(reg[1])
         : "r"(addr));
}

// Swizzle shared memory addresses to avoid bank conflicts during ldmatrix.
// XORs bits 4-6 of the byte address based on (row_index % 8), so that
// consecutive rows land on different bank groups.
// STRIDE = row width in bytes.
template <int STRIDE>
__device__
uint32_t swizzle_sqfp4(uint32_t index) {
    if constexpr (STRIDE == 16)
        return index;
    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void gmem_to_smem_sqfp4(uint32_t dst, const T *src, int tid, int src_stride)
{
    constexpr int num_elements = 16 / sizeof(T);
    constexpr int num_iters = (HEIGHT * WIDTH) / (num_elements * TB_SIZE);

    for (int iter = 0; iter < num_iters; iter++) {
        const int index = (iter * TB_SIZE + tid) * num_elements;
        const int row = index / WIDTH;
        const int col = index % WIDTH;

        uint32_t dst_addr = swizzle_sqfp4<WIDTH * (int)sizeof(T)>(
            dst + (row * WIDTH + col) * sizeof(T));
        const T *src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(dst_addr), "l"(src_addr));
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void load_scales_sqfp4(uint32_t dst, const T *src, int src_stride, int tid) {
  constexpr int cp_size = WIDTH * sizeof(T);
  static_assert(cp_size < 16);

  auto load_row = [&](int row) {
    const uint32_t dst_addr = dst + row * WIDTH * sizeof(T);
    const T *src_addr = src + row * src_stride;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(dst_addr), "l"(src_addr), "n"(cp_size));
  };

  for (int iter = 0; iter < HEIGHT / TB_SIZE; iter++)
    load_row(iter * TB_SIZE + tid);

  if constexpr (HEIGHT % TB_SIZE != 0) {
    const int row = HEIGHT / TB_SIZE * TB_SIZE + tid;
    if (row < HEIGHT)
      load_row(row);
  }
}

__device__ inline
uint32_t cvt_8xf32_to_e2m1_packed_sqfp4(float f0, float f1, float f2, float f3,
                                     float f4, float f5, float f6, float f7) {
    uint32_t packed;
    asm volatile(
        "{\n\t"
        ".reg .b8 a0, a1, a2, a3;\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a0, %1, %2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a1, %3, %4;\n\t"
        "mov.b16 lo, {a0, a1};\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a2, %5, %6;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a3, %7, %8;\n\t"
        "mov.b16 hi, {a2, a3};\n\t"
        "mov.b32 %0, {lo, hi};\n\t"
        "}"
        : "=r"(packed)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3),
          "f"(f4), "f"(f5), "f"(f6), "f"(f7)
    );
    return packed;
}

__device__ inline
uint32_t cvt_4xf32_to_e4m3_packed_sqfp4(float f0, float f1, float f2, float f3) {
    uint32_t packed;
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, %1, %2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, %3, %4;\n\t"
        "mov.b32 %0, {lo, hi};\n\t"
        "}"
        : "=r"(packed)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3)
    );
    return packed;
}

__device__ inline
void mma_m16n8k64_nvfp4_sqfp4(uint32_t A[4], uint32_t B[2], uint32_t sf_A, uint32_t sf_B, float D[4]) {
    const uint16_t byte_id = 0;
    const uint16_t thread_id = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4"
        ".block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, "
        "{%17}, {%18, %19};"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]),
          "r"(sf_A), "h"(byte_id), "h"(thread_id),
          "r"(sf_B), "h"(byte_id), "h"(thread_id));
}

constexpr int WARP_SIZE_sqfp4 = 32;

__host__ __device__ inline
int cdiv_sqfp4(int a, int b) { return (a + b - 1) / b; }

__host__ __device__ inline
int sage_perm32_sqfp4(int x) {
    return (x / 8) * 2 + ((x % 8) / 2) * 8 + (x % 2);
}

__host__ __device__ inline
int sage_perm_seq_sqfp4(int x) {
    const int base = (x / 32) * 32;
    return base + sage_perm32_sqfp4(x & 31);
}

template<typename T, int BLOCK_KV, int HEAD_DIM, int HEAD_DIM_2, int SCALE_DIM>
__launch_bounds__(WARP_SIZE_sqfp4)
__global__
void fp4_attention_single_query_kernel(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    float* O_partial,       // [bs, num_kv_splits, 16, HEAD_DIM]
    float* rowmax_partial,  // [bs, num_kv_splits, 16]
    float* rowsum_partial,  // [bs, num_kv_splits, 16]
    T* O_final,             // [bs, q_len, HEAD_DIM]
    int* split_counter,     // [bs] — zeroed before launch
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_kv_splits,
    int num_q_heads,
    int num_kv_heads)
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = WARP_SIZE_sqfp4;
    constexpr int BLOCK_Q = 16;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    // Size of one FP4 KV buffer in smem (bytes)
    constexpr int FP4_BUF_BYTES =
        BLOCK_KV * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
      + BLOCK_KV * SCALE_DIM  * (int)sizeof(__nv_fp8_e4m3)
      + HEAD_DIM * (BLOCK_KV / 2)  * (int)sizeof(__nv_fp4x2_e2m1)
      + HEAD_DIM * (BLOCK_KV / 16) * (int)sizeof(__nv_fp8_e4m3);

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // V/S_V are transposed; per-row stride uses capacity, not logical kv_len.
    const int v_kv = cdiv_sqfp4(kv_capacity, 128) * 128;

    const int split_id = blockIdx.x;
    const int bid = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = tid;
    const int batch_id = bid / num_q_heads;
    const int q_head = bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;

    // Compute KV range for this split
    const int total_kv_blocks = cdiv_sqfp4(kv_len, BLOCK_KV);
    const int kv_block_start = (split_id * total_kv_blocks) / num_kv_splits;
    const int kv_block_end = ((split_id + 1) * total_kv_blocks) / num_kv_splits;

    Q   += bid * q_len       * HEAD_DIM_2;
    K   += kv_bid * kv_capacity * HEAD_DIM_2 + kv_block_start * BLOCK_KV * HEAD_DIM_2;
    V   += kv_bid * HEAD_DIM    * (v_kv / 2) + kv_block_start * BLOCK_KV / 2;
    S_Q += bid * q_len       * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM  + kv_block_start * BLOCK_KV * SCALE_DIM;
    S_V += kv_bid * v_kv        * SCALE_DIM  + kv_block_start * BLOCK_KV / 16;

    const int partial_offset = bid * num_kv_splits + split_id;
    O_partial      += partial_offset * BLOCK_Q * HEAD_DIM;
    rowmax_partial += partial_offset * BLOCK_Q;
    rowsum_partial += partial_offset * BLOCK_Q;

    extern __shared__ uint8_t smem[];

    // ---- Phase 1: load Q data + scales into smem → registers ----
    const uint32_t Q_smem    = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[HEAD_DIM / MMA_K][4];
    uint32_t sfQ_rmem[HEAD_DIM / MMA_K];

    float rowmax[2];
    float rowsum[2] = {};
    float O_rmem[HEAD_DIM / MMA_N][4] = {};

    rowmax[0] = -FLT_MAX;
    rowmax[1] = -FLT_MAX;

    // Zero Q data + Q scale smem, then load only q_len valid rows
    {
        constexpr int TOTAL_WORDS = (BLOCK_Q * HEAD_DIM_2 + BLOCK_Q * SCALE_DIM) / 4;
        uint32_t* base = reinterpret_cast<uint32_t*>(smem);
        for (int i = tid; i < TOTAL_WORDS; i += TB_SIZE)
            base[i] = 0;
    }
    __syncwarp();
    {
        constexpr int LOAD_SIZE = 16;
        const int total_loads = q_len * HEAD_DIM_2 / LOAD_SIZE;
        for (int load_id = tid; load_id < total_loads; load_id += TB_SIZE) {
            const int byte_offset = load_id * LOAD_SIZE;
            const int row = byte_offset / HEAD_DIM_2;
            const int col = byte_offset % HEAD_DIM_2;
            uint32_t dst_addr = swizzle_sqfp4<HEAD_DIM_2>(Q_smem + row * HEAD_DIM_2 + col);
            const auto *src_addr = Q + row * HEAD_DIM_2 + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(dst_addr), "l"(src_addr));
        }
    }
    {
        constexpr int SF_ROW_BYTES = SCALE_DIM * (int)sizeof(__nv_fp8_e4m3);
        for (int row = tid; row < q_len; row += TB_SIZE) {
            const uint32_t dst_addr = Q_sf_smem + row * SF_ROW_BYTES;
            const auto *src_addr = S_Q + row * SCALE_DIM;
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                :: "r"(dst_addr), "l"(src_addr), "n"(SF_ROW_BYTES));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncwarp();

    uint32_t Q_ld_base;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = swizzle_sqfp4<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
        uint32_t addr = Q_ld_base;
        addr ^= mma_id_d * (MMA_K / 2);
        ldmatrix_x4_sqfp4(Q_rmem[mma_id_d], addr);
    }

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
        const uint32_t offset = (sf_row_q * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
        asm volatile("ld.shared.u32 %0, [%1];"
            : "=r"(sfQ_rmem[mma_id_d])
            : "r"(Q_sf_smem + offset));
    }

    // ---- Phase 2: Double-buffered KV loop ----
    __syncwarp();

    const uint32_t smem_base = __cvta_generic_to_shared(smem);

    // Two smem buffers for double-buffering
    uint32_t K_smem_buf[2], K_sf_buf[2], V_smem_buf[2], V_sf_buf[2], K_ld_buf[2];
    for (int b = 0; b < 2; b++) {
        K_smem_buf[b] = smem_base + b * FP4_BUF_BYTES;
        K_sf_buf[b]   = K_smem_buf[b] + BLOCK_KV * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);
        V_smem_buf[b] = K_sf_buf[b]   + BLOCK_KV * SCALE_DIM  * (int)sizeof(__nv_fp8_e4m3);
        V_sf_buf[b]   = V_smem_buf[b] + HEAD_DIM * (BLOCK_KV / 2) * (int)sizeof(__nv_fp4x2_e2m1);

        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_buf[b] = swizzle_sqfp4<HEAD_DIM_2>(K_smem_buf[b] + row_off * HEAD_DIM_2 + col_off);
    }

    uint32_t K_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t V_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
    uint32_t sfK_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K];
    uint32_t sfV_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N];

    const int num_kv_iters = kv_block_end - kv_block_start;

    // Helper: start async load of KV block at offset iter into buffer buf
    auto start_kv_load = [&](int buf, int iter) {
        const auto* Kp  = K   + iter * BLOCK_KV * HEAD_DIM_2;
        const auto* Vp  = V   + iter * BLOCK_KV / 2;
        const auto* SKp = S_K + iter * BLOCK_KV * SCALE_DIM;
        const auto* SVp = S_V + iter * BLOCK_KV / 16;
        gmem_to_smem_sqfp4<BLOCK_KV, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_smem_buf[buf], Kp, tid, HEAD_DIM_2);
        load_scales_sqfp4<BLOCK_KV, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(K_sf_buf[buf], SKp, SCALE_DIM, tid);
        gmem_to_smem_sqfp4<HEAD_DIM, BLOCK_KV / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_smem_buf[buf], Vp, tid, v_kv / 2);
        load_scales_sqfp4<HEAD_DIM, BLOCK_KV / 16, TB_SIZE, __nv_fp8_e4m3>(V_sf_buf[buf], SVp, v_kv / 16, tid);
        asm volatile("cp.async.commit_group;");
    };

    // Preload first block
    if (num_kv_iters > 0)
        start_kv_load(0, 0);

    int cur_buf = 0;
    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        float S_rmem[BLOCK_KV / MMA_N][4] = {};
        uint32_t S_fp4_rmem[BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[BLOCK_KV / MMA_K];

        // Preload next block into alternate buffer
        if (kv_iter + 1 < num_kv_iters)
            start_kv_load(1 - cur_buf, kv_iter + 1);

        // Wait for current buffer
        if (kv_iter + 1 < num_kv_iters) {
            asm volatile("cp.async.wait_group %0;" :: "n"(1));
        } else {
            asm volatile("cp.async.wait_all;");
        }
        __syncwarp();

        // Aliases for current buffer
        const uint32_t cur_K_sf  = K_sf_buf[cur_buf];
        const uint32_t cur_V     = V_smem_buf[cur_buf];
        const uint32_t cur_V_sf  = V_sf_buf[cur_buf];
        const uint32_t cur_K_ld  = K_ld_buf[cur_buf];

        // K → registers
        if constexpr (HEAD_DIM / MMA_K >= 2) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                uint32_t addr = cur_K_ld + mma_id_kv * MMA_N * HEAD_DIM_2;
                ldmatrix_x4_sqfp4(K_rmem[mma_id_kv][0], addr);
            }
        } else {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                    uint32_t addr = cur_K_ld;
                    addr += mma_id_kv * MMA_N * HEAD_DIM_2;
                    addr ^= mma_id_d * (MMA_K / 2);
                    ldmatrix_x2_sqfp4(K_rmem[mma_id_kv][mma_id_d], addr);
                }
        }

        // K scales → registers
        const int sf_row_k = lane_id / 4;
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                const int row = mma_id_kv * MMA_N + sf_row_k;
                const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(sfK_rmem[mma_id_kv][mma_id_d])
                    : "r"(cur_K_sf + offset));
            }

        // QK^T MMA
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++)
                mma_m16n8k64_nvfp4_sqfp4(
                    Q_rmem[mma_id_d],
                    K_rmem[mma_id_kv][mma_id_d],
                    sfQ_rmem[mma_id_d],
                    sfK_rmem[mma_id_kv][mma_id_d],
                    S_rmem[mma_id_kv]);

        // ---- online softmax (no causal mask) ----
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int reg_id = 0; reg_id < 4; reg_id++)
                S_rmem[mma_id_kv][reg_id] *= softmax_scale;

        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        for (int blk = 0; blk < BLOCK_KV / MMA_N / 2; blk++) {
            const int t0 = 2 * blk;
            const int t1 = t0 + 1;
            float gmax_upper = max(
                max(S_rmem[t0][0], S_rmem[t0][1]),
                max(S_rmem[t1][0], S_rmem[t1][1])
            );
            float gmax_lower = max(
                max(S_rmem[t0][2], S_rmem[t0][3]),
                max(S_rmem[t1][2], S_rmem[t1][3])
            );
            gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 1));
            gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 2));
            gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 1));
            gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 2));
            this_rowmax[0] = max(this_rowmax[0], gmax_upper);
            this_rowmax[1] = max(this_rowmax[1], gmax_lower);
        }

        this_rowmax[0] = max(this_rowmax[0], rowmax[0]);
        this_rowmax[1] = max(this_rowmax[1], rowmax[1]);

        float rescale[2];
        rescale[0] = __expf(rowmax[0] - this_rowmax[0]);
        rescale[1] = __expf(rowmax[1] - this_rowmax[1]);
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            O_rmem[mma_id_d][0] *= rescale[0];
            O_rmem[mma_id_d][1] *= rescale[0];
            O_rmem[mma_id_d][2] *= rescale[1];
            O_rmem[mma_id_d][3] *= rescale[1];
        }

        rowmax[0] = this_rowmax[0];
        rowmax[1] = this_rowmax[1];

        float this_rowsumexp[2] = {};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_kv];
            regs[0] = __expf(regs[0] - rowmax[0]);
            regs[1] = __expf(regs[1] - rowmax[0]);
            regs[2] = __expf(regs[2] - rowmax[1]);
            regs[3] = __expf(regs[3] - rowmax[1]);

            this_rowsumexp[0] += regs[0] + regs[1];
            this_rowsumexp[1] += regs[2] + regs[3];
        }

        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

        rowsum[0] = rowsum[0] * rescale[0] + this_rowsumexp[0];
        rowsum[1] = rowsum[1] * rescale[1] + this_rowsumexp[1];

        constexpr float FP4_RANGE = 448.0f * 6.0f;
        constexpr float FP4_MAX = 6.0f;

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            S_rmem[mma_id_kv][0] *= FP4_RANGE;
            S_rmem[mma_id_kv][1] *= FP4_RANGE;
            S_rmem[mma_id_kv][2] *= FP4_RANGE;
            S_rmem[mma_id_kv][3] *= FP4_RANGE;
        }

        float sf_P_upper[BLOCK_KV / MMA_N / 2];
        float sf_P_lower[BLOCK_KV / MMA_N / 2];

        for (int blk = 0; blk < BLOCK_KV / MMA_N / 2; blk++) {
            int t0 = 2 * blk;
            int t1 = t0 + 1;

            float amax_upper = max(
                max(S_rmem[t0][0], S_rmem[t0][1]),
                max(S_rmem[t1][0], S_rmem[t1][1])
            );
            amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 1));
            amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 2));

            float amax_lower = max(
                max(S_rmem[t0][2], S_rmem[t0][3]),
                max(S_rmem[t1][2], S_rmem[t1][3])
            );
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 1));
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 2));

            sf_P_upper[blk] = amax_upper / FP4_MAX;
            sf_P_lower[blk] = amax_lower / FP4_MAX;

            float inv_upper = 1.0f / sf_P_upper[blk];
            float inv_lower = 1.0f / sf_P_lower[blk];

            S_rmem[t0][0] *= inv_upper;
            S_rmem[t0][1] *= inv_upper;
            S_rmem[t1][0] *= inv_upper;
            S_rmem[t1][1] *= inv_upper;
            S_rmem[t0][2] *= inv_lower;
            S_rmem[t0][3] *= inv_lower;
            S_rmem[t1][2] *= inv_lower;
            S_rmem[t1][3] *= inv_lower;
        }

        // Reorder lanes so the following MMA consumes contiguous probability fragments.
        const int qid = lane_id & 3;

        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            for (int r = 0; r < 4; r++) {
                float send = (qid & 1) ? S_rmem[g*4 + 0][r] : S_rmem[g*4 + 1][r];
                float recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                if (qid & 1) S_rmem[g*4 + 0][r] = recv;
                else         S_rmem[g*4 + 1][r] = recv;

                send = (qid & 1) ? S_rmem[g*4 + 2][r] : S_rmem[g*4 + 3][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                if (qid & 1) S_rmem[g*4 + 2][r] = recv;
                else         S_rmem[g*4 + 3][r] = recv;

                send = (qid & 2) ? S_rmem[g*4 + 0][r] : S_rmem[g*4 + 2][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                if (qid & 2) S_rmem[g*4 + 0][r] = recv;
                else         S_rmem[g*4 + 2][r] = recv;

                send = (qid & 2) ? S_rmem[g*4 + 1][r] : S_rmem[g*4 + 3][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                if (qid & 2) S_rmem[g*4 + 1][r] = recv;
                else         S_rmem[g*4 + 3][r] = recv;
            }
        }

        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            for (int r = 0; r < 4; r++) {
                float send = (qid & 1) ? S_rmem[g*4 + 0][r] : S_rmem[g*4 + 1][r];
                float recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                if (qid & 1) S_rmem[g*4 + 0][r] = recv;
                else         S_rmem[g*4 + 1][r] = recv;

                send = (qid & 1) ? S_rmem[g*4 + 2][r] : S_rmem[g*4 + 3][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                if (qid & 1) S_rmem[g*4 + 2][r] = recv;
                else         S_rmem[g*4 + 3][r] = recv;

                send = (qid & 2) ? S_rmem[g*4 + 0][r] : S_rmem[g*4 + 2][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                if (qid & 2) S_rmem[g*4 + 0][r] = recv;
                else         S_rmem[g*4 + 2][r] = recv;

                send = (qid & 2) ? S_rmem[g*4 + 1][r] : S_rmem[g*4 + 3][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                if (qid & 2) S_rmem[g*4 + 1][r] = recv;
                else         S_rmem[g*4 + 3][r] = recv;
            }
        }

        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            float *r0 = S_rmem[g*4];
            float *r1 = S_rmem[g*4 + 1];
            float *r2 = S_rmem[g*4 + 2];
            float *r3 = S_rmem[g*4 + 3];

            S_fp4_rmem[0][2 * g] = cvt_8xf32_to_e2m1_packed_sqfp4(
                r0[1], r0[0], r1[1], r1[0],
                r2[1], r2[0], r3[1], r3[0]);

            S_fp4_rmem[0][2 * g + 1] = cvt_8xf32_to_e2m1_packed_sqfp4(
                r0[3], r0[2], r1[3], r1[2],
                r2[3], r2[2], r3[3], r3[2]);
        }

        for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV / MMA_K; mma_sc_id++) {
            int base = mma_sc_id * 4;
            uint32_t sfP_upper_packed = cvt_4xf32_to_e4m3_packed_sqfp4(
                sf_P_upper[base + 1], sf_P_upper[base + 0],
                sf_P_upper[base + 3], sf_P_upper[base + 2]);

            uint32_t sfP_lower_packed = cvt_4xf32_to_e4m3_packed_sqfp4(
                sf_P_lower[base + 1], sf_P_lower[base + 0],
                sf_P_lower[base + 3], sf_P_lower[base + 2]);

            S_fp4_s_rmem[mma_sc_id] =
                (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
        }

        // V → registers
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d += 2) {
                const int n_idx = mma_id_d * MMA_N + (lane_id / 16) * MMA_N + (lane_id % 8);
                const int k_byte_offset = mma_id_kv * 32 + ((lane_id % 16) / 8) * 16;
                uint32_t addr = swizzle_sqfp4<BLOCK_KV / 2>(
                    cur_V + n_idx * (BLOCK_KV / 2) + k_byte_offset);

                ldmatrix_x4_sqfp4(V_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        // V scales → registers
        constexpr int V_SF_STRIDE = BLOCK_KV / 16;
        const int sf_col_v = lane_id / 4;
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                const int hd_col = mma_id_d * MMA_N + sf_col_v;
                const uint32_t offset = (hd_col * V_SF_STRIDE + mma_id_kv * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(sfV_rmem[mma_id_kv][mma_id_d])
                    : "r"(cur_V_sf + offset));
            }

        // O += P @ V
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
                mma_m16n8k64_nvfp4_sqfp4(
                    S_fp4_rmem[mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    S_fp4_s_rmem[mma_id_kv],
                    sfV_rmem[mma_id_kv][mma_id_d],
                    O_rmem[mma_id_d]);

        cur_buf = 1 - cur_buf;
    }

    // ---- Phase 3: Write partials — float2 vectorised stores ----
    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
        const int row = lane_id / 4;
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
        float *regs = O_rmem[mma_id_d];

        *reinterpret_cast<float2*>(&O_partial[(row + 0) * HEAD_DIM + col]) =
            make_float2(regs[0], regs[1]);
        *reinterpret_cast<float2*>(&O_partial[(row + 8) * HEAD_DIM + col]) =
            make_float2(regs[2], regs[3]);
    }

    if (lane_id % 4 == 0) {
        const int row = lane_id / 4;
        rowmax_partial[row + 0] = rowmax[0];
        rowmax_partial[row + 8] = rowmax[1];
        rowsum_partial[row + 0] = rowsum[0];
        rowsum_partial[row + 8] = rowsum[1];
    }

    // ---- Fused reduction: last block to finish reduces all partials ----
    __threadfence();

    int completed = 0;
    if (lane_id == 0)
        completed = atomicAdd(&split_counter[bid], 1);
    completed = __shfl_sync(0xFFFFFFFF, completed, 0);

    if (completed != num_kv_splits - 1)
        return;

    // ---- Two-pass smem-assisted reduction ----
    constexpr float FP4_RANGE_INV = 1.0f / (448.0f * 6.0f);
    constexpr int ELEMS_PER_THREAD = (BLOCK_Q * HEAD_DIM) / TB_SIZE;

    const float* batch_O = O_partial - split_id * BLOCK_Q * HEAD_DIM;
    const float* batch_rowmax = rowmax_partial - split_id * BLOCK_Q;
    const float* batch_rowsum = rowsum_partial - split_id * BLOCK_Q;

    // Pass 1: 16 threads compute per-row rescale factors + norms → smem
    float* rsc_smem  = reinterpret_cast<float*>(smem);
    float* norm_smem = rsc_smem + num_kv_splits * BLOCK_Q;

    if (lane_id < BLOCK_Q) {
        const int row = lane_id;
        float global_max = -FLT_MAX;
        for (int s = 0; s < num_kv_splits; s++)
            global_max = max(global_max, batch_rowmax[s * BLOCK_Q + row]);

        float sum_acc = 0.0f;
        for (int s = 0; s < num_kv_splits; s++) {
            float rsc = __expf(batch_rowmax[s * BLOCK_Q + row] - global_max);
            rsc_smem[s * BLOCK_Q + row] = rsc;
            sum_acc += batch_rowsum[s * BLOCK_Q + row] * rsc;
        }
        norm_smem[row] = FP4_RANGE_INV / sum_acc;
    }
    __syncwarp();

    // Pass 2: all 32 threads reduce O elements using precomputed factors
    for (int elem = 0; elem < ELEMS_PER_THREAD; elem++) {
        const int flat_idx = lane_id + elem * TB_SIZE;
        const int row = flat_idx / HEAD_DIM;
        const int col = flat_idx % HEAD_DIM;

        float acc = 0.0f;
        for (int s = 0; s < num_kv_splits; s++) {
            float rsc = rsc_smem[s * BLOCK_Q + row];
            acc += batch_O[s * BLOCK_Q * HEAD_DIM + row * HEAD_DIM + col] * rsc;
        }

        if (row < q_len) {
            float norm = norm_smem[row];
            O_final[bid * q_len * HEAD_DIM + row * HEAD_DIM + col] =
                Traits::from_float(acc * norm);
        }
    }
}

// ---------------------------------------------------------------------------
// Decode kernel: no-split fast path. One block per batch element, processes
// all KV blocks, writes FP16 output directly. No workspace, no atomics.
// Handles unpadded Q (q_len rows, typically 1 for single-query).
// Grid: (bs,)
// ---------------------------------------------------------------------------
template<typename T, int BLOCK_KV, int HEAD_DIM, int HEAD_DIM_2, int SCALE_DIM>
__launch_bounds__(WARP_SIZE_sqfp4)
__global__
void fp4_attention_single_query_nosplit_kernel(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = WARP_SIZE_sqfp4;
    constexpr int BLOCK_Q = 16;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int v_kv = cdiv_sqfp4(kv_capacity, 128) * 128;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid;
    const int batch_id = bid / num_q_heads;
    const int q_head = bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;

    Q   += bid * q_len       * HEAD_DIM_2;
    K   += kv_bid * kv_capacity * HEAD_DIM_2;
    V   += kv_bid * HEAD_DIM    * (v_kv / 2);
    S_Q += bid * q_len       * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM;
    S_V += kv_bid * v_kv        * SCALE_DIM;
    O   += bid * q_len       * HEAD_DIM;

    extern __shared__ uint8_t smem[];

    // ---- Phase 1: load Q data + scales into smem → registers ----
    const uint32_t Q_smem    = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[HEAD_DIM / MMA_K][4];
    uint32_t sfQ_rmem[HEAD_DIM / MMA_K];

    float rowmax[2];
    float rowsum[2] = {};
    float O_rmem[HEAD_DIM / MMA_N][4] = {};

    rowmax[0] = -FLT_MAX;
    rowmax[1] = -FLT_MAX;

    // Zero Q data + Q scale smem, then load only q_len valid rows
    {
        constexpr int TOTAL_WORDS = (BLOCK_Q * HEAD_DIM_2 + BLOCK_Q * SCALE_DIM) / 4;
        uint32_t* base = reinterpret_cast<uint32_t*>(smem);
        for (int i = tid; i < TOTAL_WORDS; i += TB_SIZE)
            base[i] = 0;
    }
    __syncwarp();
    {
        constexpr int LOAD_SIZE = 16;
        const int total_loads = q_len * HEAD_DIM_2 / LOAD_SIZE;
        for (int load_id = tid; load_id < total_loads; load_id += TB_SIZE) {
            const int byte_offset = load_id * LOAD_SIZE;
            const int row = byte_offset / HEAD_DIM_2;
            const int col = byte_offset % HEAD_DIM_2;
            uint32_t dst_addr = swizzle_sqfp4<HEAD_DIM_2>(Q_smem + row * HEAD_DIM_2 + col);
            const auto *src_addr = Q + row * HEAD_DIM_2 + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(dst_addr), "l"(src_addr));
        }
    }
    {
        constexpr int SF_ROW_BYTES = SCALE_DIM * (int)sizeof(__nv_fp8_e4m3);
        for (int row = tid; row < q_len; row += TB_SIZE) {
            const uint32_t dst_addr = Q_sf_smem + row * SF_ROW_BYTES;
            const auto *src_addr = S_Q + row * SCALE_DIM;
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                :: "r"(dst_addr), "l"(src_addr), "n"(SF_ROW_BYTES));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncwarp();

    uint32_t Q_ld_base;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = swizzle_sqfp4<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
        uint32_t addr = Q_ld_base;
        addr ^= mma_id_d * (MMA_K / 2);
        ldmatrix_x4_sqfp4(Q_rmem[mma_id_d], addr);
    }

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
        const uint32_t offset = (sf_row_q * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
        asm volatile("ld.shared.u32 %0, [%1];"
            : "=r"(sfQ_rmem[mma_id_d])
            : "r"(Q_sf_smem + offset));
    }

    // ---- Phase 2: KV loop (all blocks) ----
    __syncwarp();

    const uint32_t K_smem    = __cvta_generic_to_shared(smem);
    const uint32_t K_sf_smem = K_smem    + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);
    const uint32_t V_smem    = K_sf_smem + BLOCK_KV * SCALE_DIM  * sizeof(__nv_fp8_e4m3);
    const uint32_t V_sf_smem = V_smem    + HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1);

    uint32_t K_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t V_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
    uint32_t sfK_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K];
    uint32_t sfV_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N];

    uint32_t K_ld_base;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base = swizzle_sqfp4<HEAD_DIM_2>(K_smem + row_off * HEAD_DIM_2 + col_off);
    }

    const int total_kv_blocks = cdiv_sqfp4(kv_len, BLOCK_KV);

    for (int kv_iter = 0; kv_iter < total_kv_blocks; kv_iter++) {
        float S_rmem[BLOCK_KV / MMA_N][4] = {};
        uint32_t S_fp4_rmem[BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[BLOCK_KV / MMA_K];

        gmem_to_smem_sqfp4<BLOCK_KV, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_smem, K, tid, HEAD_DIM_2);
        load_scales_sqfp4<BLOCK_KV, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(K_sf_smem, S_K, SCALE_DIM, tid);
        gmem_to_smem_sqfp4<HEAD_DIM, BLOCK_KV / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_smem, V, tid, v_kv / 2);
        load_scales_sqfp4<HEAD_DIM, BLOCK_KV / 16, TB_SIZE, __nv_fp8_e4m3>(V_sf_smem, S_V, v_kv / 16, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncwarp();

        if constexpr (HEAD_DIM / MMA_K >= 2) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                uint32_t addr = K_ld_base + mma_id_kv * MMA_N * HEAD_DIM_2;
                ldmatrix_x4_sqfp4(K_rmem[mma_id_kv][0], addr);
            }
        } else {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                    uint32_t addr = K_ld_base;
                    addr += mma_id_kv * MMA_N * HEAD_DIM_2;
                    addr ^= mma_id_d * (MMA_K / 2);
                    ldmatrix_x2_sqfp4(K_rmem[mma_id_kv][mma_id_d], addr);
                }
        }

        const int sf_row_k = lane_id / 4;
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                const int row = mma_id_kv * MMA_N + sf_row_k;
                const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(sfK_rmem[mma_id_kv][mma_id_d])
                    : "r"(K_sf_smem + offset));
            }

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++)
                mma_m16n8k64_nvfp4_sqfp4(
                    Q_rmem[mma_id_d],
                    K_rmem[mma_id_kv][mma_id_d],
                    sfQ_rmem[mma_id_d],
                    sfK_rmem[mma_id_kv][mma_id_d],
                    S_rmem[mma_id_kv]);

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int reg_id = 0; reg_id < 4; reg_id++)
                S_rmem[mma_id_kv][reg_id] *= softmax_scale;

        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        for (int blk = 0; blk < BLOCK_KV / MMA_N / 2; blk++) {
            const int t0 = 2 * blk;
            const int t1 = t0 + 1;
            float gmax_upper = max(
                max(S_rmem[t0][0], S_rmem[t0][1]),
                max(S_rmem[t1][0], S_rmem[t1][1])
            );
            float gmax_lower = max(
                max(S_rmem[t0][2], S_rmem[t0][3]),
                max(S_rmem[t1][2], S_rmem[t1][3])
            );
            gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 1));
            gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 2));
            gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 1));
            gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 2));
            this_rowmax[0] = max(this_rowmax[0], gmax_upper);
            this_rowmax[1] = max(this_rowmax[1], gmax_lower);
        }

        this_rowmax[0] = max(this_rowmax[0], rowmax[0]);
        this_rowmax[1] = max(this_rowmax[1], rowmax[1]);

        float rescale[2];
        rescale[0] = __expf(rowmax[0] - this_rowmax[0]);
        rescale[1] = __expf(rowmax[1] - this_rowmax[1]);
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            O_rmem[mma_id_d][0] *= rescale[0];
            O_rmem[mma_id_d][1] *= rescale[0];
            O_rmem[mma_id_d][2] *= rescale[1];
            O_rmem[mma_id_d][3] *= rescale[1];
        }

        rowmax[0] = this_rowmax[0];
        rowmax[1] = this_rowmax[1];

        float this_rowsumexp[2] = {};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_kv];
            regs[0] = __expf(regs[0] - rowmax[0]);
            regs[1] = __expf(regs[1] - rowmax[0]);
            regs[2] = __expf(regs[2] - rowmax[1]);
            regs[3] = __expf(regs[3] - rowmax[1]);

            this_rowsumexp[0] += regs[0] + regs[1];
            this_rowsumexp[1] += regs[2] + regs[3];
        }

        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

        rowsum[0] = rowsum[0] * rescale[0] + this_rowsumexp[0];
        rowsum[1] = rowsum[1] * rescale[1] + this_rowsumexp[1];

        constexpr float FP4_RANGE = 448.0f * 6.0f;
        constexpr float inv_sP1 = FP4_RANGE;

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            S_rmem[mma_id_kv][0] *= inv_sP1;
            S_rmem[mma_id_kv][1] *= inv_sP1;
            S_rmem[mma_id_kv][2] *= inv_sP1;
            S_rmem[mma_id_kv][3] *= inv_sP1;
        }

        constexpr float FP4_MAX = 6.0f;
        float sf_P_upper[BLOCK_KV / MMA_N / 2];
        float sf_P_lower[BLOCK_KV / MMA_N / 2];

        for (int blk = 0; blk < BLOCK_KV / MMA_N / 2; blk++) {
            int t0 = 2 * blk;
            int t1 = t0 + 1;

            float amax_upper = max(
                max(S_rmem[t0][0], S_rmem[t0][1]),
                max(S_rmem[t1][0], S_rmem[t1][1])
            );
            amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 1));
            amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 2));

            float amax_lower = max(
                max(S_rmem[t0][2], S_rmem[t0][3]),
                max(S_rmem[t1][2], S_rmem[t1][3])
            );
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 1));
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 2));

            sf_P_upper[blk] = amax_upper / FP4_MAX;
            sf_P_lower[blk] = amax_lower / FP4_MAX;
        }

        for (int blk = 0; blk < BLOCK_KV / MMA_N / 2; blk++) {
            float inv_upper = 1.0f / sf_P_upper[blk];
            float inv_lower = 1.0f / sf_P_lower[blk];
            int t0 = 2 * blk;
            int t1 = t0 + 1;

            S_rmem[t0][0] *= inv_upper;
            S_rmem[t0][1] *= inv_upper;
            S_rmem[t1][0] *= inv_upper;
            S_rmem[t1][1] *= inv_upper;
            S_rmem[t0][2] *= inv_lower;
            S_rmem[t0][3] *= inv_lower;
            S_rmem[t1][2] *= inv_lower;
            S_rmem[t1][3] *= inv_lower;
        }

        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            float *r0 = S_rmem[g*4];
            float *r1 = S_rmem[g*4 + 1];
            float *r2 = S_rmem[g*4 + 2];
            float *r3 = S_rmem[g*4 + 3];

            S_fp4_rmem[0][2 * g] = cvt_8xf32_to_e2m1_packed_sqfp4(
                r0[1], r0[0], r1[1], r1[0],
                r2[1], r2[0], r3[1], r3[0]);

            S_fp4_rmem[0][2 * g + 1] = cvt_8xf32_to_e2m1_packed_sqfp4(
                r0[3], r0[2], r1[3], r1[2],
                r2[3], r2[2], r3[3], r3[2]);
        }

        for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV / MMA_K; mma_sc_id++) {
            int base = mma_sc_id * 4;
            uint32_t sfP_upper_packed = cvt_4xf32_to_e4m3_packed_sqfp4(
                sf_P_upper[base + 1], sf_P_upper[base + 0],
                sf_P_upper[base + 3], sf_P_upper[base + 2]);

            uint32_t sfP_lower_packed = cvt_4xf32_to_e4m3_packed_sqfp4(
                sf_P_lower[base + 1], sf_P_lower[base + 0],
                sf_P_lower[base + 3], sf_P_lower[base + 2]);

            S_fp4_s_rmem[mma_sc_id] =
                (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
        }

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d += 2) {
                const int n_idx = mma_id_d * MMA_N + (lane_id / 16) * MMA_N + (lane_id % 8);
                const int k_byte_offset = mma_id_kv * 32 + ((lane_id % 16) / 8) * 16;
                uint32_t addr = swizzle_sqfp4<BLOCK_KV / 2>(
                    V_smem + n_idx * (BLOCK_KV / 2) + k_byte_offset);

                ldmatrix_x4_sqfp4(V_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        constexpr int V_SF_STRIDE = BLOCK_KV / 16;
        const int sf_col_v = lane_id / 4;
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                const int hd_col = mma_id_d * MMA_N + sf_col_v;
                const uint32_t offset = (hd_col * V_SF_STRIDE + mma_id_kv * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(sfV_rmem[mma_id_kv][mma_id_d])
                    : "r"(V_sf_smem + offset));
            }

        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
                mma_m16n8k64_nvfp4_sqfp4(
                    S_fp4_rmem[mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    S_fp4_s_rmem[mma_id_kv],
                    sfV_rmem[mma_id_kv][mma_id_d],
                    O_rmem[mma_id_d]);

        K   += BLOCK_KV * HEAD_DIM_2;
        S_K += BLOCK_KV * SCALE_DIM;
        V   += BLOCK_KV / 2;
        S_V += BLOCK_KV / 16;
    }

    // ---- Phase 3: normalize and write FP16 output directly ----
    constexpr float FP4_RANGE_INV = 1.0f / (448.0f * 6.0f);

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
        const int row = lane_id / 4;
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

        float *regs = O_rmem[mma_id_d];

        float norm0 = FP4_RANGE_INV / rowsum[0];
        float norm1 = FP4_RANGE_INV / rowsum[1];

        regs[0] *= norm0;
        regs[1] *= norm0;
        regs[2] *= norm1;
        regs[3] *= norm1;

        if (row < q_len)
            reinterpret_cast<typename Traits::vec2*>(O + row * HEAD_DIM + col)[0] =
                Traits::pack2(regs[0], regs[1]);
        if (row + 8 < q_len)
            reinterpret_cast<typename Traits::vec2*>(O + (row + 8) * HEAD_DIM + col)[0] =
                Traits::pack2(regs[2], regs[3]);
    }
}

// ---------------------------------------------------------------------------
// Decode kernel: CTA-cooperative version. NUM_WARPS warps per block, each
// warp processes a disjoint chunk of KV blocks independently using its own
// smem region, then all warps reduce (O, rowmax, rowsum) via shared memory.
// No workspace allocation needed — reduction is entirely intra-CTA.
// Grid: (bs,)
// ---------------------------------------------------------------------------
template<typename T, int BLOCK_KV, int HEAD_DIM, int HEAD_DIM_2, int SCALE_DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE_sqfp4, 1)
__global__
void fp4_attention_single_query_cta_kernel(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE_sqfp4;
    constexpr int BLOCK_Q = 16;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int v_kv = cdiv_sqfp4(kv_capacity, 128) * 128;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE_sqfp4;
    const int lane_id = tid % WARP_SIZE_sqfp4;
    const int batch_id = bid / num_q_heads;
    const int q_head = bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;

    Q   += bid * q_len       * HEAD_DIM_2;
    K   += kv_bid * kv_capacity * HEAD_DIM_2;
    V   += kv_bid * HEAD_DIM    * (v_kv / 2);
    S_Q += bid * q_len       * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM;
    S_V += kv_bid * v_kv        * SCALE_DIM;
    O   += bid * q_len       * HEAD_DIM;

    extern __shared__ uint8_t smem[];

    // ---- Phase 1: cooperatively load Q data + scales into smem ----
    const uint32_t Q_smem    = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[HEAD_DIM / MMA_K][4];
    uint32_t sfQ_rmem[HEAD_DIM / MMA_K];

    float rowmax[2];
    float rowsum[2] = {};
    float O_rmem[HEAD_DIM / MMA_N][4] = {};
    rowmax[0] = -FLT_MAX;
    rowmax[1] = -FLT_MAX;

    // Zero Q smem, then load valid rows (all threads cooperate)
    {
        constexpr int TOTAL_WORDS = (BLOCK_Q * HEAD_DIM_2 + BLOCK_Q * SCALE_DIM) / 4;
        uint32_t* base = reinterpret_cast<uint32_t*>(smem);
        for (int i = tid; i < TOTAL_WORDS; i += TB_SIZE)
            base[i] = 0;
    }
    __syncthreads();
    {
        constexpr int LOAD_SIZE = 16;
        const int total_loads = q_len * HEAD_DIM_2 / LOAD_SIZE;
        for (int load_id = tid; load_id < total_loads; load_id += TB_SIZE) {
            const int byte_offset = load_id * LOAD_SIZE;
            const int row = byte_offset / HEAD_DIM_2;
            const int col = byte_offset % HEAD_DIM_2;
            uint32_t dst_addr = swizzle_sqfp4<HEAD_DIM_2>(Q_smem + row * HEAD_DIM_2 + col);
            const auto *src_addr = Q + row * HEAD_DIM_2 + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(dst_addr), "l"(src_addr));
        }
    }
    {
        constexpr int SF_ROW_BYTES = SCALE_DIM * (int)sizeof(__nv_fp8_e4m3);
        for (int row = tid; row < q_len; row += TB_SIZE) {
            const uint32_t dst_addr = Q_sf_smem + row * SF_ROW_BYTES;
            const auto *src_addr = S_Q + row * SCALE_DIM;
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                :: "r"(dst_addr), "l"(src_addr), "n"(SF_ROW_BYTES));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // Each warp reads Q into registers (same smem, only lane_id matters)
    uint32_t Q_ld_base;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = swizzle_sqfp4<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
        uint32_t addr = Q_ld_base;
        addr ^= mma_id_d * (MMA_K / 2);
        ldmatrix_x4_sqfp4(Q_rmem[mma_id_d], addr);
    }

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
        const uint32_t offset = (sf_row_q * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
        asm volatile("ld.shared.u32 %0, [%1];"
            : "=r"(sfQ_rmem[mma_id_d])
            : "r"(Q_sf_smem + offset));
    }

    // ---- Phase 2: each warp processes its KV chunk independently ----
    __syncthreads();

    // Per-warp KV smem: [warp0: K|Ksf|V|Vsf] [warp1: ...] ...
    constexpr int KV_SMEM_PER_WARP = BLOCK_KV * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
                                    + BLOCK_KV * SCALE_DIM * (int)sizeof(__nv_fp8_e4m3)
                                    + HEAD_DIM * (BLOCK_KV / 2) * (int)sizeof(__nv_fp4x2_e2m1)
                                    + HEAD_DIM * (BLOCK_KV / 16) * (int)sizeof(__nv_fp8_e4m3);

    const uint32_t warp_kv_base = __cvta_generic_to_shared(smem) + warp_id * KV_SMEM_PER_WARP;
    const uint32_t K_smem    = warp_kv_base;
    const uint32_t K_sf_smem = K_smem    + BLOCK_KV * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);
    const uint32_t V_smem    = K_sf_smem + BLOCK_KV * SCALE_DIM  * (int)sizeof(__nv_fp8_e4m3);
    const uint32_t V_sf_smem = V_smem    + HEAD_DIM * (BLOCK_KV / 2) * (int)sizeof(__nv_fp4x2_e2m1);

    // Divide KV blocks evenly across warps
    const int total_kv_blocks = cdiv_sqfp4(kv_len, BLOCK_KV);
    const int blocks_per_warp = cdiv_sqfp4(total_kv_blocks, NUM_WARPS);
    const int kv_block_start = warp_id * blocks_per_warp;
    const int kv_block_end = min(kv_block_start + blocks_per_warp, total_kv_blocks);
    const int num_kv_iters = (kv_block_start < total_kv_blocks)
                           ? (kv_block_end - kv_block_start) : 0;

    // Per-warp pointers offset to this warp's starting KV block
    const __nv_fp4x2_e2m1* K_ptr = K + kv_block_start * BLOCK_KV * HEAD_DIM_2;
    const __nv_fp4x2_e2m1* V_ptr = V + kv_block_start * BLOCK_KV / 2;
    const __nv_fp8_e4m3* SK_ptr = S_K + kv_block_start * BLOCK_KV * SCALE_DIM;
    const __nv_fp8_e4m3* SV_ptr = S_V + kv_block_start * BLOCK_KV / 16;

    uint32_t K_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t V_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
    uint32_t sfK_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K];
    uint32_t sfV_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N];

    uint32_t K_ld_base;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base = swizzle_sqfp4<HEAD_DIM_2>(K_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        float S_rmem[BLOCK_KV / MMA_N][4] = {};
        uint32_t S_fp4_rmem[BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[BLOCK_KV / MMA_K];

        // Each warp loads its own KV block into its own smem region
        gmem_to_smem_sqfp4<BLOCK_KV, HEAD_DIM_2, WARP_SIZE_sqfp4, __nv_fp4x2_e2m1>(K_smem, K_ptr, lane_id, HEAD_DIM_2);
        load_scales_sqfp4<BLOCK_KV, SCALE_DIM, WARP_SIZE_sqfp4, __nv_fp8_e4m3>(K_sf_smem, SK_ptr, SCALE_DIM, lane_id);
        gmem_to_smem_sqfp4<HEAD_DIM, BLOCK_KV / 2, WARP_SIZE_sqfp4, __nv_fp4x2_e2m1>(V_smem, V_ptr, lane_id, v_kv / 2);
        load_scales_sqfp4<HEAD_DIM, BLOCK_KV / 16, WARP_SIZE_sqfp4, __nv_fp8_e4m3>(V_sf_smem, SV_ptr, v_kv / 16, lane_id);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncwarp();

        // K → registers
        if constexpr (HEAD_DIM / MMA_K >= 2) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                uint32_t addr = K_ld_base + mma_id_kv * MMA_N * HEAD_DIM_2;
                ldmatrix_x4_sqfp4(K_rmem[mma_id_kv][0], addr);
            }
        } else {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                    uint32_t addr = K_ld_base;
                    addr += mma_id_kv * MMA_N * HEAD_DIM_2;
                    addr ^= mma_id_d * (MMA_K / 2);
                    ldmatrix_x2_sqfp4(K_rmem[mma_id_kv][mma_id_d], addr);
                }
        }

        // K scales → registers
        const int sf_row_k = lane_id / 4;
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                const int row = mma_id_kv * MMA_N + sf_row_k;
                const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(sfK_rmem[mma_id_kv][mma_id_d])
                    : "r"(K_sf_smem + offset));
            }

        // QK^T MMA
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++)
                mma_m16n8k64_nvfp4_sqfp4(
                    Q_rmem[mma_id_d],
                    K_rmem[mma_id_kv][mma_id_d],
                    sfQ_rmem[mma_id_d],
                    sfK_rmem[mma_id_kv][mma_id_d],
                    S_rmem[mma_id_kv]);

        // ---- online softmax ----
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int reg_id = 0; reg_id < 4; reg_id++)
                S_rmem[mma_id_kv][reg_id] *= softmax_scale;

        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        float p_group_max_upper[BLOCK_KV / MMA_N / 2];
        float p_group_max_lower[BLOCK_KV / MMA_N / 2];
        for (int blk = 0; blk < BLOCK_KV / MMA_N / 2; blk++) {
            const int t0 = 2 * blk;
            const int t1 = t0 + 1;
            float gmax_upper = max(
                max(S_rmem[t0][0], S_rmem[t0][1]),
                max(S_rmem[t1][0], S_rmem[t1][1])
            );
            float gmax_lower = max(
                max(S_rmem[t0][2], S_rmem[t0][3]),
                max(S_rmem[t1][2], S_rmem[t1][3])
            );
            gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 1));
            gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 2));
            gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 1));
            gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 2));
            p_group_max_upper[blk] = gmax_upper;
            p_group_max_lower[blk] = gmax_lower;
            this_rowmax[0] = max(this_rowmax[0], gmax_upper);
            this_rowmax[1] = max(this_rowmax[1], gmax_lower);
        }

        this_rowmax[0] = max(this_rowmax[0], rowmax[0]);
        this_rowmax[1] = max(this_rowmax[1], rowmax[1]);

        float rescale[2];
        rescale[0] = __expf(rowmax[0] - this_rowmax[0]);
        rescale[1] = __expf(rowmax[1] - this_rowmax[1]);
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            O_rmem[mma_id_d][0] *= rescale[0];
            O_rmem[mma_id_d][1] *= rescale[0];
            O_rmem[mma_id_d][2] *= rescale[1];
            O_rmem[mma_id_d][3] *= rescale[1];
        }

        rowmax[0] = this_rowmax[0];
        rowmax[1] = this_rowmax[1];

        float this_rowsumexp[2] = {};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_kv];
            regs[0] = __expf(regs[0] - rowmax[0]);
            regs[1] = __expf(regs[1] - rowmax[0]);
            regs[2] = __expf(regs[2] - rowmax[1]);
            regs[3] = __expf(regs[3] - rowmax[1]);

            this_rowsumexp[0] += regs[0] + regs[1];
            this_rowsumexp[1] += regs[2] + regs[3];
        }

        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

        rowsum[0] = rowsum[0] * rescale[0] + this_rowsumexp[0];
        rowsum[1] = rowsum[1] * rescale[1] + this_rowsumexp[1];

        constexpr float FP4_RANGE = 448.0f * 6.0f;
        constexpr float FP4_MAX = 6.0f;

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            S_rmem[mma_id_kv][0] *= FP4_RANGE;
            S_rmem[mma_id_kv][1] *= FP4_RANGE;
            S_rmem[mma_id_kv][2] *= FP4_RANGE;
            S_rmem[mma_id_kv][3] *= FP4_RANGE;
        }

        float sf_P_upper[BLOCK_KV / MMA_N / 2];
        float sf_P_lower[BLOCK_KV / MMA_N / 2];

        for (int blk = 0; blk < BLOCK_KV / MMA_N / 2; blk++) {
            int t0 = 2 * blk;
            int t1 = t0 + 1;

            float amax_upper = max(
                max(S_rmem[t0][0], S_rmem[t0][1]),
                max(S_rmem[t1][0], S_rmem[t1][1])
            );
            amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 1));
            amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 2));

            float amax_lower = max(
                max(S_rmem[t0][2], S_rmem[t0][3]),
                max(S_rmem[t1][2], S_rmem[t1][3])
            );
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 1));
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 2));

            sf_P_upper[blk] = amax_upper / FP4_MAX;
            sf_P_lower[blk] = amax_lower / FP4_MAX;

            float inv_upper = 1.0f / sf_P_upper[blk];
            float inv_lower = 1.0f / sf_P_lower[blk];

            S_rmem[t0][0] *= inv_upper;
            S_rmem[t0][1] *= inv_upper;
            S_rmem[t1][0] *= inv_upper;
            S_rmem[t1][1] *= inv_upper;
            S_rmem[t0][2] *= inv_lower;
            S_rmem[t0][3] *= inv_lower;
            S_rmem[t1][2] *= inv_lower;
            S_rmem[t1][3] *= inv_lower;
        }

        const int qid = lane_id & 3;

        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            for (int r = 0; r < 4; r++) {
                float send = (qid & 1) ? S_rmem[g*4 + 0][r] : S_rmem[g*4 + 1][r];
                float recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                if (qid & 1) S_rmem[g*4 + 0][r] = recv;
                else         S_rmem[g*4 + 1][r] = recv;

                send = (qid & 1) ? S_rmem[g*4 + 2][r] : S_rmem[g*4 + 3][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                if (qid & 1) S_rmem[g*4 + 2][r] = recv;
                else         S_rmem[g*4 + 3][r] = recv;

                send = (qid & 2) ? S_rmem[g*4 + 0][r] : S_rmem[g*4 + 2][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                if (qid & 2) S_rmem[g*4 + 0][r] = recv;
                else         S_rmem[g*4 + 2][r] = recv;

                send = (qid & 2) ? S_rmem[g*4 + 1][r] : S_rmem[g*4 + 3][r];
                recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                if (qid & 2) S_rmem[g*4 + 1][r] = recv;
                else         S_rmem[g*4 + 3][r] = recv;
            }
        }

        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            float *r0 = S_rmem[g*4];
            float *r1 = S_rmem[g*4 + 1];
            float *r2 = S_rmem[g*4 + 2];
            float *r3 = S_rmem[g*4 + 3];

            S_fp4_rmem[0][2 * g] = cvt_8xf32_to_e2m1_packed_sqfp4(
                r0[1], r0[0], r1[1], r1[0],
                r2[1], r2[0], r3[1], r3[0]);

            S_fp4_rmem[0][2 * g + 1] = cvt_8xf32_to_e2m1_packed_sqfp4(
                r0[3], r0[2], r1[3], r1[2],
                r2[3], r2[2], r3[3], r3[2]);
        }

        for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV / MMA_K; mma_sc_id++) {
            int base = mma_sc_id * 4;
            uint32_t sfP_upper_packed = cvt_4xf32_to_e4m3_packed_sqfp4(
                sf_P_upper[base + 1], sf_P_upper[base + 0],
                sf_P_upper[base + 3], sf_P_upper[base + 2]);

            uint32_t sfP_lower_packed = cvt_4xf32_to_e4m3_packed_sqfp4(
                sf_P_lower[base + 1], sf_P_lower[base + 0],
                sf_P_lower[base + 3], sf_P_lower[base + 2]);

            S_fp4_s_rmem[mma_sc_id] =
                (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
        }

        // V → registers
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d += 2) {
                const int n_idx = mma_id_d * MMA_N + (lane_id / 16) * MMA_N + (lane_id % 8);
                const int k_byte_offset = mma_id_kv * 32 + ((lane_id % 16) / 8) * 16;
                uint32_t addr = swizzle_sqfp4<BLOCK_KV / 2>(
                    V_smem + n_idx * (BLOCK_KV / 2) + k_byte_offset);

                ldmatrix_x4_sqfp4(V_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        // V scales → registers
        constexpr int V_SF_STRIDE = BLOCK_KV / 16;
        const int sf_col_v = lane_id / 4;
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                const int hd_col = mma_id_d * MMA_N + sf_col_v;
                const uint32_t offset = (hd_col * V_SF_STRIDE + mma_id_kv * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(sfV_rmem[mma_id_kv][mma_id_d])
                    : "r"(V_sf_smem + offset));
            }

        // O += P @ V
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
                mma_m16n8k64_nvfp4_sqfp4(
                    S_fp4_rmem[mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    S_fp4_s_rmem[mma_id_kv],
                    sfV_rmem[mma_id_kv][mma_id_d],
                    O_rmem[mma_id_d]);

        K_ptr  += BLOCK_KV * HEAD_DIM_2;
        SK_ptr += BLOCK_KV * SCALE_DIM;
        V_ptr  += BLOCK_KV / 2;
        SV_ptr += BLOCK_KV / 16;
    }

    // ---- Phase 3: CTA-level reduction via shared memory ----
    __syncthreads();

    // Reduction smem layout:
    //   rowmax [NUM_WARPS][BLOCK_Q]           — float
    //   rowsum [NUM_WARPS][BLOCK_Q]           — float
    //   O_part [NUM_WARPS][BLOCK_Q][HEAD_DIM] — float
    float* reduce_base = reinterpret_cast<float*>(smem);
    float* rowmax_smem = reduce_base;
    float* rowsum_smem = rowmax_smem + NUM_WARPS * BLOCK_Q;
    float* O_part_smem = rowsum_smem + NUM_WARPS * BLOCK_Q;

    // Each warp writes its per-row rowmax/rowsum
    if (lane_id % 4 == 0) {
        const int row = lane_id / 4;  // 0..7
        rowmax_smem[warp_id * BLOCK_Q + row]     = rowmax[0];
        rowmax_smem[warp_id * BLOCK_Q + row + 8] = rowmax[1];
        rowsum_smem[warp_id * BLOCK_Q + row]     = rowsum[0];
        rowsum_smem[warp_id * BLOCK_Q + row + 8] = rowsum[1];
    }

    // Each warp writes its partial O
    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
        const int row = lane_id / 4;
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
        float *regs = O_rmem[mma_id_d];

        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + row * HEAD_DIM + col]           = regs[0];
        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + row * HEAD_DIM + col + 1]       = regs[1];
        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + (row + 8) * HEAD_DIM + col]     = regs[2];
        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + (row + 8) * HEAD_DIM + col + 1] = regs[3];
    }

    __syncthreads();

    // Precompute per-row rescale weights and normalization factor.
    // One thread per row: computes global_max, per-warp rescale, and final norm.
    // Reuses rowmax_smem for weights, rowsum_smem[0..BLOCK_Q-1] for norm.
    constexpr float FP4_RANGE_INV = 1.0f / (448.0f * 6.0f);

    if (tid < BLOCK_Q) {
        const int row = tid;
        float global_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            global_max = max(global_max, rowmax_smem[w * BLOCK_Q + row]);

        float sum_acc = 0.0f;
        float rsc[NUM_WARPS];
        for (int w = 0; w < NUM_WARPS; w++) {
            rsc[w] = __expf(rowmax_smem[w * BLOCK_Q + row] - global_max);
            sum_acc += rowsum_smem[w * BLOCK_Q + row] * rsc[w];
        }

        float norm = FP4_RANGE_INV / sum_acc;
        for (int w = 0; w < NUM_WARPS; w++)
            rowmax_smem[w * BLOCK_Q + row] = rsc[w];
        rowsum_smem[row] = norm;
    }

    __syncthreads();

    // All threads: weighted sum of warp partials + vectorised FP16 output
    constexpr int TOTAL_PAIRS = BLOCK_Q * (HEAD_DIM / 2);

    for (int pair = tid; pair < TOTAL_PAIRS; pair += TB_SIZE) {
        const int elem = pair * 2;
        const int row = elem / HEAD_DIM;
        const int col = elem % HEAD_DIM;

        float norm = rowsum_smem[row];
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) {
            float rsc = rowmax_smem[w * BLOCK_Q + row];
            const float* src = &O_part_smem[w * BLOCK_Q * HEAD_DIM + row * HEAD_DIM + col];
            acc0 += src[0] * rsc;
            acc1 += src[1] * rsc;
        }

        if (row < q_len) {
            typename Traits::vec2 result = Traits::pack2(acc0 * norm, acc1 * norm);
            *reinterpret_cast<typename Traits::vec2*>(&O[row * HEAD_DIM + col]) = result;
        }
    }
}

template<typename T, int HEAD_DIM>
static void fp4_attention_single_query_nosplit_launch(
    const __nv_fp4x2_e2m1* Q, const __nv_fp4x2_e2m1* K, const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q, const __nv_fp8_e4m3* S_K, const __nv_fp8_e4m3* S_V,
    T* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads) {

    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int BLOCK_KV = 64;
    constexpr int BLOCK_Q = 16;
    constexpr int TB_SIZE = WARP_SIZE_sqfp4;

    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int kv_phase_smem = BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                                + BLOCK_KV * SCALE_DIM * sizeof(__nv_fp8_e4m3)
                                + HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1)
                                + HEAD_DIM * (BLOCK_KV / 16) * sizeof(__nv_fp8_e4m3);
    constexpr int smem_size = q_phase_smem > kv_phase_smem ? q_phase_smem : kv_phase_smem;

    auto kernel = fp4_attention_single_query_nosplit_kernel<T, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<bs, TB_SIZE, smem_size>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);

}

// CTA-cooperative launch: multiple warps per block, shared-memory reduction
template<typename T, int HEAD_DIM>
static void fp4_attention_single_query_cta_launch(
    const __nv_fp4x2_e2m1* Q, const __nv_fp4x2_e2m1* K, const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q, const __nv_fp8_e4m3* S_K, const __nv_fp8_e4m3* S_V,
    T* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads) {

    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int BLOCK_KV = 64;
    constexpr int BLOCK_Q = 16;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE_sqfp4;

    // Smem: max of Q phase, KV phase (per-warp regions), reduction phase
    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * (int)sizeof(__nv_fp8_e4m3);
    constexpr int kv_smem_per_warp = BLOCK_KV * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
                                    + BLOCK_KV * SCALE_DIM * (int)sizeof(__nv_fp8_e4m3)
                                    + HEAD_DIM * (BLOCK_KV / 2) * (int)sizeof(__nv_fp4x2_e2m1)
                                    + HEAD_DIM * (BLOCK_KV / 16) * (int)sizeof(__nv_fp8_e4m3);
    constexpr int kv_phase_smem = NUM_WARPS * kv_smem_per_warp;
    constexpr int reduce_smem = (int)sizeof(float) * (NUM_WARPS * BLOCK_Q * 2
                              + NUM_WARPS * BLOCK_Q * HEAD_DIM);
    constexpr int smem_12 = q_phase_smem > kv_phase_smem ? q_phase_smem : kv_phase_smem;
    constexpr int smem_size = smem_12 > reduce_smem ? smem_12 : reduce_smem;

    auto kernel = fp4_attention_single_query_cta_kernel<T, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, NUM_WARPS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<bs, TB_SIZE, smem_size>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);

}

// Split-KV launch: multiple blocks per batch element, requires workspace
template<typename T, int HEAD_DIM>
static void fp4_attention_single_query_split_launch(
    const __nv_fp4x2_e2m1* Q, const __nv_fp4x2_e2m1* K, const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q, const __nv_fp8_e4m3* S_K, const __nv_fp8_e4m3* S_V,
    T* O, float* workspace, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads) {

    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int BLOCK_KV = 64;
    constexpr int BLOCK_Q = 16;
    constexpr int TB_SIZE = WARP_SIZE_sqfp4;

    const int total_kv_blocks = cdiv_sqfp4(kv_len, BLOCK_KV);
    int num_kv_splits = max(1, min(total_kv_blocks, cdiv_sqfp4(256, bs)));
    num_kv_splits = min(num_kv_splits, total_kv_blocks);

    float* O_partial      = workspace;
    float* rowmax_partial = O_partial + bs * num_kv_splits * BLOCK_Q * HEAD_DIM;
    float* rowsum_partial = rowmax_partial + bs * num_kv_splits * BLOCK_Q;
    int* split_counter    = reinterpret_cast<int*>(rowsum_partial + bs * num_kv_splits * BLOCK_Q);

    cudaMemsetAsync(split_counter, 0, bs * sizeof(int));

    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int kv_phase_smem_single = BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                                + BLOCK_KV * SCALE_DIM * sizeof(__nv_fp8_e4m3)
                                + HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1)
                                + HEAD_DIM * (BLOCK_KV / 16) * sizeof(__nv_fp8_e4m3);
    constexpr int kv_phase_smem = kv_phase_smem_single * 2;  // double-buffer
    constexpr int smem_size = q_phase_smem > kv_phase_smem ? q_phase_smem : kv_phase_smem;

    auto kernel = fp4_attention_single_query_kernel<T, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 grid(num_kv_splits, bs);
    kernel<<<grid, TB_SIZE, smem_size>>>(
        Q, K, V, S_Q, S_K, S_V,
        O_partial, rowmax_partial, rowsum_partial,
        O, split_counter,
        bs, q_len, kv_len, kv_capacity, num_kv_splits, num_q_heads, num_kv_heads);
}

template<typename T>
static void fp4_attention_single_query_nvfp4_typed(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    void* workspace_raw,
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
    auto S_Q = reinterpret_cast<const __nv_fp8_e4m3*>(S_Q_raw);
    auto S_K = reinterpret_cast<const __nv_fp8_e4m3*>(S_K_raw);
    auto S_V = reinterpret_cast<const __nv_fp8_e4m3*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    constexpr int BLOCK_KV = 64;
    const int total_kv_blocks = cdiv_sqfp4(kv_len, BLOCK_KV);

    if (total_kv_blocks <= 4) {
        if (head_dim == 64)
            fp4_attention_single_query_nosplit_launch<T, 64>(Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
        else
            fp4_attention_single_query_nosplit_launch<T, 128>(Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
    } else {
        if (head_dim == 64)
            fp4_attention_single_query_cta_launch<T, 64>(Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
        else
            fp4_attention_single_query_cta_launch<T, 128>(Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
    }
}

void fp4_attention_single_query_nvfp4(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    void* workspace_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        fp4_attention_single_query_nvfp4_typed<__nv_bfloat16>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw, workspace_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    } else {
        fp4_attention_single_query_nvfp4_typed<half>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw, workspace_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}
