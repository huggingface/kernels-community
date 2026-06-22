// SM120 mixed-precision single-query attention.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <float.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "thriftattention/sm120/cuda_common.cuh"

__device__ inline
void ldmatrix_x4_sqmxfp4mix(uint32_t reg[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
         : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
         : "r"(addr));
}

__device__ inline
void ldmatrix_x2_sqmxfp4mix(uint32_t reg[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
         : "=r"(reg[0]), "=r"(reg[1])
         : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans_sqmxfp4mix(uint32_t reg[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(addr));
}

__device__ inline
uint32_t ld_shared_e8m0x2_sqmxfp4mix(uint32_t addr) {
    uint16_t value;
    asm volatile("ld.shared.u16 %0, [%1];"
        : "=h"(value)
        : "r"(addr));
    return static_cast<uint32_t>(value);
}

__device__ __forceinline__
void mma_m16n8k16_f16_sqmxfp4mix(
    const uint32_t (&A)[4],
    const uint32_t (&B)[2],
    float (&D)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// keep your existing mma_m16n8k64_mxfp4_sqmxfp4mix(...) and conversion helpers

template <int STRIDE>
__device__
uint32_t swizzle_sqmxfp4mix(uint32_t index) {
    if constexpr (STRIDE == 16)
        return index;
    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void gmem_to_smem_sqmxfp4mix(uint32_t dst, const T *src, int tid, int src_stride)
{
    constexpr int num_elements = 16 / sizeof(T);
    constexpr int num_iters = (HEIGHT * WIDTH) / (num_elements * TB_SIZE);

    for (int iter = 0; iter < num_iters; iter++) {
        const int index = (iter * TB_SIZE + tid) * num_elements;
        const int row = index / WIDTH;
        const int col = index % WIDTH;

        uint32_t dst_addr = swizzle_sqmxfp4mix<WIDTH * (int)sizeof(T)>(
            dst + (row * WIDTH + col) * sizeof(T));
        const T *src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(dst_addr), "l"(src_addr));
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void load_scales_sqmxfp4mix(uint32_t dst, const T *src, int src_stride, int tid) {
    constexpr int cp_size = WIDTH * sizeof(T);
    static_assert(cp_size < 16);

    auto load_row = [&](int row) {
        const uint32_t dst_addr = dst + row * WIDTH * sizeof(T);
        const T *src_addr = src + row * src_stride;
        if constexpr (cp_size == 2) {
            uint16_t value;
            asm volatile("ld.global.u16 %0, [%1];"
                : "=h"(value)
                : "l"(src_addr));
            asm volatile("st.shared.u16 [%0], %1;"
                :: "r"(dst_addr), "h"(value));
        } else {
            asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                :: "r"(dst_addr), "l"(src_addr), "n"(cp_size));
        }
    };

    for (int iter = 0; iter < HEIGHT / TB_SIZE; iter++)
        load_row(iter * TB_SIZE + tid);

    if constexpr (HEIGHT % TB_SIZE != 0) {
        const int row = HEIGHT / TB_SIZE * TB_SIZE + tid;
        if (row < HEIGHT)
            load_row(row);
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void gmem_to_smem_linear_sqmxfp4mix(uint32_t dst, const T *src, int tid, int src_stride)
{
    constexpr int num_elements = 16 / sizeof(T);
    constexpr int total_vectors = (HEIGHT * WIDTH) / num_elements;

    for (int vec = tid; vec < total_vectors; vec += TB_SIZE) {
        const int index = vec * num_elements;
        const int row = index / WIDTH;
        const int col = index % WIDTH;
        const uint32_t dst_addr = dst + (row * WIDTH + col) * (int)sizeof(T);
        const T *src_addr = src + row * src_stride + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(dst_addr), "l"(src_addr));
    }
}

constexpr int WARP_SIZE_sqmxfp4mix = 32;

static int single_query_target_split_ctas_sqmxfp4mix(int total_kv_blocks) {
    if (const char* env = std::getenv("SAGE_MIXED_SPLIT_CTAS")) {
        const int v = std::atoi(env);
        if (v > 0) return v;
    }
    return (total_kv_blocks <= 512) ? 384
         : (total_kv_blocks <= 1024) ? 384
         : 896;
}

__host__ __device__ inline
int cdiv_sqmxfp4mix(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__
bool block_in_topk_sqmxfp4mix(const int32_t* topk_row, int topk_count, int block_id)
{
    bool hit = false;
    for (int i = 0; i < topk_count; i++)
        hit |= (topk_row[i] == block_id);
    return hit;
}

__device__ __forceinline__
bool range_has_topk_sqmxfp4mix(const int32_t* topk_row, int topk_count, int block_begin, int block_end)
{
    bool hit = false;
    for (int i = 0; i < topk_count; i++) {
        const int b = topk_row[i];
        hit |= (b >= block_begin) && (b < block_end);
    }
    return hit;
}

__device__ inline
uint32_t cvt_8xf32_to_e2m1_packed_sqmxfp4mix(float f0, float f1, float f2, float f3,
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
uint32_t cvt_2xf32_to_e8m0_packed_sqmxfp4mix(float f0, float f1) {
    return ta_cvt_2xf32_to_e8m0_packed(f0, f1);
}

__device__ inline
void mma_m16n8k64_mxfp4_sqmxfp4mix(uint32_t A[4], uint32_t B[2], uint32_t sf_A, uint32_t sf_B, float D[4]) {
    const uint16_t byte_id = 0;
    const uint16_t thread_id = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4"
        ".block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
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


template<typename T, int HEAD_DIM, int BLOCK_Q = 16>
__device__ __forceinline__
void load_q_fp16_to_regs_single_query_sqmxfp4mix(
    uint32_t (&Q_rmem)[HEAD_DIM / 16][4],
    uint32_t Q_smem,
    const T* Q_ptr,
    int lane_id,
    int q_len)
{
    constexpr int MMA_K = 16;
    constexpr int ROW_BYTES = HEAD_DIM * (int)sizeof(T);
    constexpr int TOTAL_BYTES = BLOCK_Q * ROW_BYTES;
    constexpr int LOAD_UNIT = 16;  // cp.async.cg copies 16 bytes
    constexpr int TOTAL_CHUNKS = TOTAL_BYTES / LOAD_UNIT;

    static_assert(HEAD_DIM % MMA_K == 0);
    static_assert(ROW_BYTES % LOAD_UNIT == 0);

    // Zero the entire BLOCK_Q × HEAD_DIM staging area, then load valid rows.
    // 32 lanes cooperate; each lane handles multiple 16-byte chunks.
    for (int chunk = lane_id; chunk < TOTAL_CHUNKS; chunk += WARP_SIZE_sqmxfp4mix) {
        const int byte_off = chunk * LOAD_UNIT;
        const int row = byte_off / ROW_BYTES;
        const int col = byte_off % ROW_BYTES;
        uint32_t dst = swizzle_sqmxfp4mix<ROW_BYTES>(Q_smem + row * ROW_BYTES + col);

        if (row < q_len) {
            const char* src = reinterpret_cast<const char*>(Q_ptr + row * HEAD_DIM) + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(dst), "l"(src));
        } else {
            // Zero-fill rows beyond q_len so the MMA sees zeros
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "r"(dst), "r"(0), "r"(0), "r"(0), "r"(0));
        }
    }

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncwarp();

    // ldmatrix layout for A operand of mma.m16n8k16.row.col
    const uint32_t Q_ld_base = swizzle_sqmxfp4mix<ROW_BYTES>(
        Q_smem + ((lane_id % 16) * HEAD_DIM + (lane_id / 16) * 8) * (int)sizeof(T));

    #pragma unroll
    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
        uint32_t addr = Q_ld_base;
        addr ^= mma_id_d * MMA_K * (int)sizeof(T);
        ldmatrix_x4_sqmxfp4mix(Q_rmem[mma_id_d], addr);
    }
}


template<typename T, int BLOCK_KV, int BLOCK_KV_STEP, int HEAD_DIM,
         bool USE_LOWER_ROWS = true, int O_REGS = 4>
__device__ __forceinline__
void attention_inner_loop_fp16_mxfp4(
    float rowmax[2],
    float rowsum[2],
    float O_rmem[HEAD_DIM / 8][O_REGS],
    uint32_t Q_rmem[HEAD_DIM / 16][4],
    uint32_t K_smem,
    uint32_t V_smem,
    const T*& K_ptr,
    const T*& V_ptr,
    const int lane_id,
    const float softmax_scale)
{
    using Traits = PrecisionTraits<T>;
    constexpr int MMA_K = 16;
    constexpr int MMA_N = 8;
    constexpr float FP4_RANGE = 6.0f;  // keep final normalization compatible

    static_assert(BLOCK_KV % BLOCK_KV_STEP == 0);
    static_assert(BLOCK_KV_STEP % MMA_K == 0);
    static_assert(HEAD_DIM % MMA_K == 0);
    static_assert(HEAD_DIM % MMA_N == 0);

    const uint32_t K_ld_base = swizzle_sqmxfp4mix<HEAD_DIM * (int)sizeof(T)>(
        K_smem + ((lane_id % 8) * HEAD_DIM + (lane_id / 8) * 8) * (int)sizeof(T));

    const uint32_t V_ld_base = swizzle_sqmxfp4mix<HEAD_DIM * (int)sizeof(T)>(
        V_smem + ((lane_id % 16) * HEAD_DIM + (lane_id / 16) * 8) * (int)sizeof(T));

    for (int kv_sub = 0; kv_sub < BLOCK_KV; kv_sub += BLOCK_KV_STEP) {
        gmem_to_smem_sqmxfp4mix<BLOCK_KV_STEP, HEAD_DIM, WARP_SIZE_sqmxfp4mix, T>(
            K_smem, K_ptr, lane_id, HEAD_DIM);
        gmem_to_smem_sqmxfp4mix<BLOCK_KV_STEP, HEAD_DIM, WARP_SIZE_sqmxfp4mix, T>(
            V_smem, V_ptr, lane_id, HEAD_DIM);

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncwarp();

        uint32_t K_rmem[BLOCK_KV_STEP / MMA_N][HEAD_DIM / MMA_K][2];

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_STEP / MMA_N; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                uint32_t addr = K_ld_base;
                addr += mma_id_kv * MMA_N * HEAD_DIM * (int)sizeof(T);
                addr ^= mma_id_d * MMA_K * (int)sizeof(T);
                ldmatrix_x2_sqmxfp4mix(K_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        float S_rmem[BLOCK_KV_STEP / MMA_N][4] = {};

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_STEP / MMA_N; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                Traits::mma(
                    Q_rmem[mma_id_d],
                    K_rmem[mma_id_kv][mma_id_d],
                    S_rmem[mma_id_kv]);
            }
        }

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_STEP / MMA_N; mma_id_kv++) {
            for (int reg_id = 0; reg_id < 4; reg_id++) {
                S_rmem[mma_id_kv][reg_id] *= softmax_scale;
            }
        }

        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_STEP / MMA_N; mma_id_kv++) {
            float* r = S_rmem[mma_id_kv];
            this_rowmax[0] = max(this_rowmax[0], max(r[0], r[1]));
            if constexpr (USE_LOWER_ROWS)
                this_rowmax[1] = max(this_rowmax[1], max(r[2], r[3]));
        }

        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
        if constexpr (USE_LOWER_ROWS) {
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));
        }

        this_rowmax[0] = max(this_rowmax[0], rowmax[0]);
        if constexpr (USE_LOWER_ROWS)
            this_rowmax[1] = max(this_rowmax[1], rowmax[1]);

        float rescale0 = __expf(rowmax[0] - this_rowmax[0]);
        float rescale1 = 1.0f;
        if constexpr (USE_LOWER_ROWS)
            rescale1 = __expf(rowmax[1] - this_rowmax[1]);

        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            O_rmem[mma_id_d][0] *= rescale0;
            O_rmem[mma_id_d][1] *= rescale0;
            if constexpr (USE_LOWER_ROWS && O_REGS == 4) {
                O_rmem[mma_id_d][2] *= rescale1;
                O_rmem[mma_id_d][3] *= rescale1;
            }
        }

        rowmax[0] = this_rowmax[0];
        if constexpr (USE_LOWER_ROWS)
            rowmax[1] = this_rowmax[1];

        float this_rowsumexp[2] = {};
        uint32_t P_rmem[BLOCK_KV_STEP / MMA_K][4];

        for (int mma_blk = 0; mma_blk < BLOCK_KV_STEP / MMA_K; mma_blk++) {
            const int t0 = 2 * mma_blk + 0;
            const int t1 = 2 * mma_blk + 1;

            float* r0 = S_rmem[t0];
            float* r1 = S_rmem[t1];

            r0[0] = __expf(r0[0] - rowmax[0]);
            r0[1] = __expf(r0[1] - rowmax[0]);

            r1[0] = __expf(r1[0] - rowmax[0]);
            r1[1] = __expf(r1[1] - rowmax[0]);

            this_rowsumexp[0] += r0[0] + r0[1] + r1[0] + r1[1];
            if constexpr (USE_LOWER_ROWS) {
                r0[2] = __expf(r0[2] - rowmax[1]);
                r0[3] = __expf(r0[3] - rowmax[1]);
                r1[2] = __expf(r1[2] - rowmax[1]);
                r1[3] = __expf(r1[3] - rowmax[1]);
                this_rowsumexp[1] += r0[2] + r0[3] + r1[2] + r1[3];
            } else {
                r0[2] = 0.0f; r0[3] = 0.0f;
                r1[2] = 0.0f; r1[3] = 0.0f;
            }

            typename Traits::vec2* p = reinterpret_cast<typename Traits::vec2*>(P_rmem[mma_blk]);
            p[0] = Traits::pack2(r0[0] * FP4_RANGE, r0[1] * FP4_RANGE);
            p[1] = Traits::pack2(r0[2] * FP4_RANGE, r0[3] * FP4_RANGE);
            p[2] = Traits::pack2(r1[0] * FP4_RANGE, r1[1] * FP4_RANGE);
            p[3] = Traits::pack2(r1[2] * FP4_RANGE, r1[3] * FP4_RANGE);
        }

        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);

        rowsum[0] = rowsum[0] * rescale0 + this_rowsumexp[0];
        if constexpr (USE_LOWER_ROWS) {
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);
            rowsum[1] = rowsum[1] * rescale1 + this_rowsumexp[1];
        }

        uint32_t V_rmem[BLOCK_KV_STEP / MMA_K][HEAD_DIM / MMA_N][2];

        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_STEP / MMA_K; mma_id_kv++) {
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                uint32_t addr = V_ld_base;
                addr += mma_id_kv * MMA_K * HEAD_DIM * (int)sizeof(T);
                addr ^= mma_id_d * MMA_N * (int)sizeof(T);
                ldmatrix_x2_trans_sqmxfp4mix(V_rmem[mma_id_kv][mma_id_d], addr);
            }
        }

        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_STEP / MMA_K; mma_id_kv++) {
                if constexpr (O_REGS == 4) {
                    Traits::mma(
                        P_rmem[mma_id_kv],
                        V_rmem[mma_id_kv][mma_id_d],
                        O_rmem[mma_id_d]);
                } else {
                    float d[4] = {O_rmem[mma_id_d][0], O_rmem[mma_id_d][1], 0.0f, 0.0f};
                    Traits::mma(
                        P_rmem[mma_id_kv],
                        V_rmem[mma_id_kv][mma_id_d],
                        d);
                    O_rmem[mma_id_d][0] = d[0];
                    O_rmem[mma_id_d][1] = d[1];
                }
            }
        }

        K_ptr += BLOCK_KV_STEP * HEAD_DIM;
        V_ptr += BLOCK_KV_STEP * HEAD_DIM;
    }
}

// ---------------------------------------------------------------------------
// Separated load / compute helpers for FP4 attention.
// Used by the split kernel for double-buffering; the combined
// attention_inner_loop_fp4_mxfp4 below calls them sequentially.
// ---------------------------------------------------------------------------

template<int BLOCK_KV, int HEAD_DIM, int HEAD_DIM_2, int SCALE_DIM>
__device__ __forceinline__
void load_kv_fp4_async_sqmxfp4mix(
    uint32_t K_smem, uint32_t K_sf_smem,
    uint32_t V_smem, uint32_t V_sf_smem,
    const __nv_fp4x2_e2m1* K_ptr,
    const __nv_fp4x2_e2m1* V_ptr,
    const __nv_fp8_e8m0* SK_ptr,
    const __nv_fp8_e8m0* SV_ptr,
    int lane_id, int v_kv)
{
    gmem_to_smem_sqmxfp4mix<BLOCK_KV, HEAD_DIM_2, WARP_SIZE_sqmxfp4mix, __nv_fp4x2_e2m1>(K_smem,    K_ptr,  lane_id, HEAD_DIM_2);
    load_scales_sqmxfp4mix <BLOCK_KV, SCALE_DIM,  WARP_SIZE_sqmxfp4mix, __nv_fp8_e8m0>  (K_sf_smem, SK_ptr, SCALE_DIM, lane_id);
    gmem_to_smem_sqmxfp4mix<HEAD_DIM, BLOCK_KV/2, WARP_SIZE_sqmxfp4mix, __nv_fp4x2_e2m1>(V_smem,    V_ptr,  lane_id, v_kv / 2);
    load_scales_sqmxfp4mix <HEAD_DIM, BLOCK_KV / 32,WARP_SIZE_sqmxfp4mix, __nv_fp8_e8m0>  (V_sf_smem, SV_ptr, v_kv / 32, lane_id);
    asm volatile("cp.async.commit_group;");
}

template<int BLOCK_KV, int HEAD_DIM, int HEAD_DIM_2, int SCALE_DIM,
         bool USE_LOWER_ROWS = true, int O_REGS = 4>
__device__ __forceinline__
void compute_kv_fp4_sqmxfp4mix(
    float rowmax[2],
    float rowsum[2],
    float O_rmem[HEAD_DIM / 8][O_REGS],
    uint32_t Q_rmem[HEAD_DIM / 64][4],
    uint32_t sfQ_rmem[HEAD_DIM / 64],
    uint32_t K_sf_smem,
    uint32_t V_smem, uint32_t V_sf_smem,
    uint32_t K_ld_base,
    const int lane_id,
    const float softmax_scale)
{
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    // ---- K → registers ----
    uint32_t K_rmem [BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t sfK_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K];

    if constexpr (HEAD_DIM / MMA_K >= 2) {
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            uint32_t addr = K_ld_base + mma_id_kv * MMA_N * HEAD_DIM_2;
            ldmatrix_x4_sqmxfp4mix(K_rmem[mma_id_kv][0], addr);
        }
    } else {
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                uint32_t addr = K_ld_base;
                addr += mma_id_kv * MMA_N * HEAD_DIM_2;
                addr ^= mma_id_d * (MMA_K / 2);
                ldmatrix_x2_sqmxfp4mix(K_rmem[mma_id_kv][mma_id_d], addr);
            }
    }

    // ---- K scales → registers ----
    const int sf_row_k = lane_id / 4;
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            const int row = mma_id_kv * MMA_N + sf_row_k;
            const uint32_t offset = (row * SCALE_DIM + mma_id_d * 2) * (uint32_t)sizeof(__nv_fp8_e8m0);
            sfK_rmem[mma_id_kv][mma_id_d] = ld_shared_e8m0x2_sqmxfp4mix(K_sf_smem + offset);
        }

    // ---- QK^T MMA → S_rmem (float) ----
    float S_rmem[BLOCK_KV / MMA_N][4] = {};
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++)
            mma_m16n8k64_mxfp4_sqmxfp4mix(
                Q_rmem[mma_id_d],
                K_rmem[mma_id_kv][mma_id_d],
                sfQ_rmem[mma_id_d],
                sfK_rmem[mma_id_kv][mma_id_d],
                S_rmem[mma_id_kv]);

    // ---- Online softmax ----
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int reg_id = 0; reg_id < 4; reg_id++)
            S_rmem[mma_id_kv][reg_id] *= softmax_scale;

    float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *r = S_rmem[mma_id_kv];
        this_rowmax[0] = max(this_rowmax[0], max(r[0], r[1]));
        if constexpr (USE_LOWER_ROWS)
            this_rowmax[1] = max(this_rowmax[1], max(r[2], r[3]));
    }
    this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
    this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
    if constexpr (USE_LOWER_ROWS) {
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));
    }

    this_rowmax[0] = max(this_rowmax[0], rowmax[0]);
    if constexpr (USE_LOWER_ROWS)
        this_rowmax[1] = max(this_rowmax[1], rowmax[1]);

    float rescale0 = __expf(rowmax[0] - this_rowmax[0]);
    float rescale1 = 1.0f;
    if constexpr (USE_LOWER_ROWS)
        rescale1 = __expf(rowmax[1] - this_rowmax[1]);
    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_d][0] *= rescale0;  O_rmem[mma_id_d][1] *= rescale0;
        if constexpr (USE_LOWER_ROWS && O_REGS == 4) {
            O_rmem[mma_id_d][2] *= rescale1;  O_rmem[mma_id_d][3] *= rescale1;
        }
    }
    rowmax[0] = this_rowmax[0];
    if constexpr (USE_LOWER_ROWS)
        rowmax[1] = this_rowmax[1];

    float this_rowsumexp[2] = {};
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *r = S_rmem[mma_id_kv];
        r[0] = __expf(r[0] - rowmax[0]);  r[1] = __expf(r[1] - rowmax[0]);
        this_rowsumexp[0] += r[0] + r[1];
        if constexpr (USE_LOWER_ROWS) {
            r[2] = __expf(r[2] - rowmax[1]);  r[3] = __expf(r[3] - rowmax[1]);
            this_rowsumexp[1] += r[2] + r[3];
        } else {
            r[2] = 0.0f;  r[3] = 0.0f;
        }
    }
    this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
    this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
    rowsum[0] = rowsum[0] * rescale0 + this_rowsumexp[0];
    if constexpr (USE_LOWER_ROWS) {
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);
        rowsum[1] = rowsum[1] * rescale1 + this_rowsumexp[1];
    }

    // ---- Quantise S → FP4 for P@V ----
    constexpr float FP4_RANGE = 6.0f;
    constexpr float FP4_MAX   = 6.0f;

    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        S_rmem[mma_id_kv][0] *= FP4_RANGE;
        S_rmem[mma_id_kv][1] *= FP4_RANGE;
        if constexpr (USE_LOWER_ROWS) {
            S_rmem[mma_id_kv][2] *= FP4_RANGE;
            S_rmem[mma_id_kv][3] *= FP4_RANGE;
        } else {
            S_rmem[mma_id_kv][2] = 0.0f;
            S_rmem[mma_id_kv][3] = 0.0f;
        }
    }

    float sf_P_upper[BLOCK_KV / MMA_N / 4];
    float sf_P_lower[BLOCK_KV / MMA_N / 4];
    for (int blk = 0; blk < BLOCK_KV / MMA_N / 4; blk++) {
        int t0 = 4 * blk, t1 = t0 + 1, t2 = t0 + 2, t3 = t0 + 3;
        float amax_upper = max(
            max(max(S_rmem[t0][0], S_rmem[t0][1]),
                max(S_rmem[t1][0], S_rmem[t1][1])),
            max(max(S_rmem[t2][0], S_rmem[t2][1]),
                max(S_rmem[t3][0], S_rmem[t3][1])));
        amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 1));
        amax_upper = max(amax_upper, __shfl_xor_sync(0xFFFFFFFF, amax_upper, 2));
        sf_P_upper[blk] = amax_upper / FP4_MAX;
        if constexpr (USE_LOWER_ROWS) {
            float amax_lower = max(
                max(max(S_rmem[t0][2], S_rmem[t0][3]),
                    max(S_rmem[t1][2], S_rmem[t1][3])),
                max(max(S_rmem[t2][2], S_rmem[t2][3]),
                    max(S_rmem[t3][2], S_rmem[t3][3])));
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 1));
            amax_lower = max(amax_lower, __shfl_xor_sync(0xFFFFFFFF, amax_lower, 2));
            sf_P_lower[blk] = amax_lower / FP4_MAX;
        } else {
            sf_P_lower[blk] = 1.0f;
        }
    }

    for (int blk = 0; blk < BLOCK_KV / MMA_N / 4; blk++) {
        float inv_upper = 1.0f / sf_P_upper[blk];
        int t0 = 4 * blk, t1 = t0 + 1, t2 = t0 + 2, t3 = t0 + 3;
        S_rmem[t0][0] *= inv_upper;  S_rmem[t0][1] *= inv_upper;
        S_rmem[t1][0] *= inv_upper;  S_rmem[t1][1] *= inv_upper;
        S_rmem[t2][0] *= inv_upper;  S_rmem[t2][1] *= inv_upper;
        S_rmem[t3][0] *= inv_upper;  S_rmem[t3][1] *= inv_upper;
        if constexpr (USE_LOWER_ROWS) {
            float inv_lower = 1.0f / sf_P_lower[blk];
            S_rmem[t0][2] *= inv_lower;  S_rmem[t0][3] *= inv_lower;
            S_rmem[t1][2] *= inv_lower;  S_rmem[t1][3] *= inv_lower;
            S_rmem[t2][2] *= inv_lower;  S_rmem[t2][3] *= inv_lower;
            S_rmem[t3][2] *= inv_lower;  S_rmem[t3][3] *= inv_lower;
        }
    }

    // Shuffle S into MMA-compatible layout
    uint32_t S_fp4_rmem[BLOCK_KV / MMA_K][4];
    const int qid = lane_id & 3;
    for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
        for (int r = 0; r < 4; r++) {
            float send, recv;
            send = (qid & 1) ? S_rmem[g*4+0][r] : S_rmem[g*4+1][r];
            recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
            if (qid & 1) S_rmem[g*4+0][r] = recv; else S_rmem[g*4+1][r] = recv;

            send = (qid & 1) ? S_rmem[g*4+2][r] : S_rmem[g*4+3][r];
            recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
            if (qid & 1) S_rmem[g*4+2][r] = recv; else S_rmem[g*4+3][r] = recv;

            send = (qid & 2) ? S_rmem[g*4+0][r] : S_rmem[g*4+2][r];
            recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
            if (qid & 2) S_rmem[g*4+0][r] = recv; else S_rmem[g*4+2][r] = recv;

            send = (qid & 2) ? S_rmem[g*4+1][r] : S_rmem[g*4+3][r];
            recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
            if (qid & 2) S_rmem[g*4+1][r] = recv; else S_rmem[g*4+3][r] = recv;
        }

        float *r0 = S_rmem[g*4], *r1 = S_rmem[g*4+1],
              *r2 = S_rmem[g*4+2], *r3 = S_rmem[g*4+3];
        S_fp4_rmem[0][2*g]   = cvt_8xf32_to_e2m1_packed_sqmxfp4mix(r0[1],r0[0],r1[1],r1[0], r2[1],r2[0],r3[1],r3[0]);
        S_fp4_rmem[0][2*g+1] = cvt_8xf32_to_e2m1_packed_sqmxfp4mix(r0[3],r0[2],r1[3],r1[2], r2[3],r2[2],r3[3],r3[2]);
    }

    uint32_t S_fp4_s_rmem[BLOCK_KV / MMA_K];
    for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV / MMA_K; mma_sc_id++) {
        int base = mma_sc_id * 2;
        uint32_t sfP_upper_packed = cvt_2xf32_to_e8m0_packed_sqmxfp4mix(
            sf_P_upper[base+1], sf_P_upper[base+0]);
        uint32_t sfP_lower_packed = cvt_2xf32_to_e8m0_packed_sqmxfp4mix(
            sf_P_lower[base+1], sf_P_lower[base+0]);
        S_fp4_s_rmem[mma_sc_id] = (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
    }

    // ---- V → registers ----
    uint32_t V_rmem [BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
    uint32_t sfV_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N];

    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d += 2) {
            const int n_idx = mma_id_d * MMA_N + (lane_id / 16) * MMA_N + (lane_id % 8);
            const int k_byte_offset = mma_id_kv * 32 + ((lane_id % 16) / 8) * 16;
            uint32_t addr = swizzle_sqmxfp4mix<BLOCK_KV / 2>(
                V_smem + n_idx * (BLOCK_KV / 2) + k_byte_offset);
            ldmatrix_x4_sqmxfp4mix(V_rmem[mma_id_kv][mma_id_d], addr);
        }

    constexpr int V_SF_STRIDE = BLOCK_KV / 32;
    const int sf_col_v = lane_id / 4;
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int hd_col = mma_id_d * MMA_N + sf_col_v;
            const uint32_t offset = (hd_col * V_SF_STRIDE + mma_id_kv * 2) * (uint32_t)sizeof(__nv_fp8_e8m0);
            sfV_rmem[mma_id_kv][mma_id_d] = ld_shared_e8m0x2_sqmxfp4mix(V_sf_smem + offset);
        }

    // ---- O += P @ V ----
    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
            if constexpr (O_REGS == 4) {
                mma_m16n8k64_mxfp4_sqmxfp4mix(
                    S_fp4_rmem[mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    S_fp4_s_rmem[mma_id_kv],
                    sfV_rmem[mma_id_kv][mma_id_d],
                    O_rmem[mma_id_d]);
            } else {
                float d[4] = {O_rmem[mma_id_d][0], O_rmem[mma_id_d][1], 0.0f, 0.0f};
                mma_m16n8k64_mxfp4_sqmxfp4mix(
                    S_fp4_rmem[mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    S_fp4_s_rmem[mma_id_kv],
                    sfV_rmem[mma_id_kv][mma_id_d],
                    d);
                O_rmem[mma_id_d][0] = d[0];
                O_rmem[mma_id_d][1] = d[1];
            }
        }
}

template<int BLOCK_KV, int HEAD_DIM, int HEAD_DIM_2, int SCALE_DIM>
__device__ __forceinline__
void attention_inner_loop_fp4_mxfp4(
    // Persistent accumulator state (read-modify-write across iterations)
    float rowmax[2],
    float rowsum[2],
    float O_rmem[HEAD_DIM / 8][4],  // HEAD_DIM / MMA_N, MMA_N=8

    // Q register tiles (loaded once before the loop)
    uint32_t Q_rmem[HEAD_DIM / 64][4],   // HEAD_DIM / MMA_K
    uint32_t sfQ_rmem[HEAD_DIM / 64],

    // KV smem base addresses (warp-private regions)
    uint32_t K_smem, uint32_t K_sf_smem,
    uint32_t V_smem, uint32_t V_sf_smem,
    uint32_t K_ld_base,

    // KV global-memory pointers (advanced at end of each iteration)
    const __nv_fp4x2_e2m1*& K_ptr,
    const __nv_fp4x2_e2m1*& V_ptr,
    const __nv_fp8_e8m0*&   SK_ptr,
    const __nv_fp8_e8m0*&   SV_ptr,

    // Per-call constants
    const int lane_id,
    const float softmax_scale,
    const int v_kv)
{
    load_kv_fp4_async_sqmxfp4mix<BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM>(
        K_smem, K_sf_smem, V_smem, V_sf_smem,
        K_ptr, V_ptr, SK_ptr, SV_ptr, lane_id, v_kv);
    asm volatile("cp.async.wait_all;");
    __syncwarp();

    compute_kv_fp4_sqmxfp4mix<BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM>(
        rowmax, rowsum, O_rmem, Q_rmem, sfQ_rmem,
        K_sf_smem, V_smem, V_sf_smem, K_ld_base,
        lane_id, softmax_scale);

    K_ptr  += BLOCK_KV * HEAD_DIM_2;
    SK_ptr += BLOCK_KV * SCALE_DIM;
    V_ptr  += BLOCK_KV / 2;
    SV_ptr += BLOCK_KV / 32;
}

template<
    typename T,
    int BLOCK_KV_FP16,
    int BLOCK_KV_FP4,
    int HEAD_DIM,
    int HEAD_DIM_2,
    int SCALE_DIM,
    int NUM_WARPS,
    int KV_SMEM_PER_WARP>
__launch_bounds__(NUM_WARPS * WARP_SIZE_sqmxfp4mix, 1)
__global__
void thrift_attention_single_query_cta_kernel(
    const T* Q_fp16,
    const T* K_fp16,
    const T* V_fp16,
    const int32_t* top_k,
    int topk_count,
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
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE_sqmxfp4mix;
    constexpr int BLOCK_Q = 16;
    constexpr int MMA_K_FP4 = 64;
    constexpr int MMA_N = 8;

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    // Stride uses capacity (allocation), loop bounds use kv_len (logical).
    const int v_kv = cdiv_sqmxfp4mix(kv_capacity, 128) * 128;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE_sqmxfp4mix;
    const int lane_id = tid % WARP_SIZE_sqmxfp4mix;
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

    Q_fp16 += bid * q_len       * HEAD_DIM;
    K_fp16 += kv_bid * kv_capacity * HEAD_DIM;
    V_fp16 += kv_bid * kv_capacity * HEAD_DIM;

    const int32_t* topk_row = top_k + bid * topk_count;

    extern __shared__ uint8_t smem[];

    // ---- Phase 1: cooperatively load Q_fp4 + scales into smem ----
    const uint32_t Q_smem    = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[HEAD_DIM / MMA_K_FP4][4];
    uint32_t sfQ_rmem[HEAD_DIM / MMA_K_FP4];

    float rowmax[2];
    float rowsum[2] = {};
    float O_rmem[HEAD_DIM / MMA_N][4] = {};
    rowmax[0] = -FLT_MAX;
    rowmax[1] = -FLT_MAX;

    // Zero Q data + Q scale smem so unused rows (q_len..15) are zero
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
            uint32_t dst_addr = swizzle_sqmxfp4mix<HEAD_DIM_2>(Q_smem + row * HEAD_DIM_2 + col);
            const auto* src_addr = Q + row * HEAD_DIM_2 + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(dst_addr), "l"(src_addr));
        }
    }

    {
        constexpr int SF_ROW_BYTES = SCALE_DIM * (int)sizeof(__nv_fp8_e8m0);
        for (int row = tid; row < q_len; row += TB_SIZE) {
            const uint32_t dst_addr = Q_sf_smem + row * SF_ROW_BYTES;
            const auto* src_addr = S_Q + row * SCALE_DIM;
            if constexpr (SF_ROW_BYTES == 2) {
                uint16_t value;
                asm volatile("ld.global.u16 %0, [%1];"
                    : "=h"(value)
                    : "l"(src_addr));
                asm volatile("st.shared.u16 [%0], %1;"
                    :: "r"(dst_addr), "h"(value));
            } else {
                asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                    :: "r"(dst_addr), "l"(src_addr), "n"(SF_ROW_BYTES));
            }
        }
    }

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    uint32_t Q_ld_base;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = swizzle_sqmxfp4mix<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
        uint32_t addr = Q_ld_base;
        addr ^= mma_id_d * (MMA_K_FP4 / 2);
        ldmatrix_x4_sqmxfp4mix(Q_rmem[mma_id_d], addr);
    }

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
        const uint32_t offset =
            (sf_row_q * SCALE_DIM + mma_id_d * 2) * (uint32_t)sizeof(__nv_fp8_e8m0);
        sfQ_rmem[mma_id_d] = ld_shared_e8m0x2_sqmxfp4mix(Q_sf_smem + offset);
    }

    __syncthreads();

    // ---- Phase 2: warp-private KV processing ----
    const uint32_t warp_kv_base = __cvta_generic_to_shared(smem) + warp_id * KV_SMEM_PER_WARP;

    // FP4 layout in warp-private smem
    const uint32_t K_smem_fp4    = warp_kv_base;
    const uint32_t K_sf_smem     = K_smem_fp4 + BLOCK_KV_FP4 * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);
    const uint32_t V_smem_fp4    = K_sf_smem + BLOCK_KV_FP4 * SCALE_DIM * (int)sizeof(__nv_fp8_e8m0);
    const uint32_t V_sf_smem     = V_smem_fp4 + HEAD_DIM * (BLOCK_KV_FP4 / 2) * (int)sizeof(__nv_fp4x2_e2m1);

    // FP16 layout in warp-private smem
    const uint32_t K_smem_fp16   = warp_kv_base;
    const uint32_t V_smem_fp16   = K_smem_fp16 + BLOCK_KV_FP16 * HEAD_DIM * (int)sizeof(T);

    const int total_kv_blocks = cdiv_sqmxfp4mix(kv_len, BLOCK_KV_FP4);
    const int blocks_per_warp = cdiv_sqmxfp4mix(total_kv_blocks, NUM_WARPS);
    const int kv_block_start = warp_id * blocks_per_warp;
    const int kv_block_end   = (kv_block_start + blocks_per_warp < total_kv_blocks)
                             ? (kv_block_start + blocks_per_warp)
                             : total_kv_blocks;
    const int num_kv_iters   = (kv_block_start < total_kv_blocks)
                             ? (kv_block_end - kv_block_start)
                             : 0;

    const bool warp_has_fp16 = range_has_topk_sqmxfp4mix(topk_row, topk_count, kv_block_start, kv_block_end);

    uint32_t K_ld_base_fp4;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base_fp4 = swizzle_sqmxfp4mix<HEAD_DIM_2>(K_smem_fp4 + row_off * HEAD_DIM_2 + col_off);
    }

    // Phase 2a: process all FP4 blocks first (no FP16 registers live)
    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        const int global_block_id = kv_block_start + kv_iter;
        if (warp_has_fp16 && block_in_topk_sqmxfp4mix(topk_row, topk_count, global_block_id))
            continue;

        const __nv_fp4x2_e2m1* K_ptr_iter = K + global_block_id * BLOCK_KV_FP4 * HEAD_DIM_2;
        const __nv_fp4x2_e2m1* V_ptr_iter = V + global_block_id * BLOCK_KV_FP4 / 2;
        const __nv_fp8_e8m0*   SK_ptr_iter = S_K + global_block_id * BLOCK_KV_FP4 * SCALE_DIM;
        const __nv_fp8_e8m0*   SV_ptr_iter = S_V + global_block_id * BLOCK_KV_FP4 / 32;

        attention_inner_loop_fp4_mxfp4<BLOCK_KV_FP4, HEAD_DIM, HEAD_DIM_2, SCALE_DIM>(
            rowmax, rowsum, O_rmem,
            Q_rmem, sfQ_rmem,
            K_smem_fp4, K_sf_smem,
            V_smem_fp4, V_sf_smem,
            K_ld_base_fp4,
            K_ptr_iter, V_ptr_iter,
            SK_ptr_iter, SV_ptr_iter,
            lane_id, softmax_scale, v_kv);
    }

    // Phase 2b: process FP16 (top-k) blocks — Q FP16 loaded once
    if (warp_has_fp16) {
        uint32_t Q_fp16_rmem[HEAD_DIM / 16][4];
        load_q_fp16_to_regs_single_query_sqmxfp4mix<T, HEAD_DIM, BLOCK_Q>(
            Q_fp16_rmem, K_smem_fp16, Q_fp16, lane_id, q_len);

        for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
            const int global_block_id = kv_block_start + kv_iter;
            if (!block_in_topk_sqmxfp4mix(topk_row, topk_count, global_block_id))
                continue;

            const T* K_fp16_ptr_iter = K_fp16 + global_block_id * BLOCK_KV_FP4 * HEAD_DIM;
            const T* V_fp16_ptr_iter = V_fp16 + global_block_id * BLOCK_KV_FP4 * HEAD_DIM;

            attention_inner_loop_fp16_mxfp4<T, BLOCK_KV_FP4, BLOCK_KV_FP16, HEAD_DIM>(
                rowmax, rowsum, O_rmem,
                Q_fp16_rmem,
                K_smem_fp16, V_smem_fp16,
                K_fp16_ptr_iter, V_fp16_ptr_iter,
                lane_id, softmax_scale);
        }
    }

    // ---- Phase 3: CTA reduction ----
    __syncthreads();

    float* reduce_base = reinterpret_cast<float*>(smem);
    float* rowmax_smem = reduce_base;
    float* rowsum_smem = rowmax_smem + NUM_WARPS * BLOCK_Q;
    float* O_part_smem = rowsum_smem + NUM_WARPS * BLOCK_Q;

    if (lane_id % 4 == 0) {
        const int row = lane_id / 4;
        rowmax_smem[warp_id * BLOCK_Q + row]     = rowmax[0];
        rowmax_smem[warp_id * BLOCK_Q + row + 8] = rowmax[1];
        rowsum_smem[warp_id * BLOCK_Q + row]     = rowsum[0];
        rowsum_smem[warp_id * BLOCK_Q + row + 8] = rowsum[1];
    }

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
        const int row = lane_id / 4;
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
        float* regs = O_rmem[mma_id_d];

        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + row * HEAD_DIM + col]           = regs[0];
        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + row * HEAD_DIM + col + 1]       = regs[1];
        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + (row + 8) * HEAD_DIM + col]     = regs[2];
        O_part_smem[warp_id * BLOCK_Q * HEAD_DIM + (row + 8) * HEAD_DIM + col + 1] = regs[3];
    }

    __syncthreads();

    constexpr float FP4_RANGE_INV = 1.0f / (6.0f);

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
static void thrift_attention_single_query_cta_launch(
    const T* Q_fp16,
    const T* K_fp16,
    const T* V_fp16,
    const int32_t* top_k,
    int topk_count,
    const __nv_fp4x2_e2m1* Q_fp4,
    const __nv_fp4x2_e2m1* K_fp4,
    const __nv_fp4x2_e2m1* V_fp4,
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
    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 32;
    constexpr int BLOCK_KV_FP4 = 64;
    constexpr int BLOCK_KV_FP16 = 32;
    constexpr int BLOCK_Q = 16;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE_sqmxfp4mix;

    constexpr int q_phase_smem_fp4 =
        BLOCK_Q * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1) +
        BLOCK_Q * SCALE_DIM * (int)sizeof(__nv_fp8_e8m0);

    constexpr int q_phase_smem_fp16 =
        BLOCK_Q * HEAD_DIM * (int)sizeof(T);

    constexpr int q_phase_smem =
        (q_phase_smem_fp16 > q_phase_smem_fp4) ? q_phase_smem_fp16 : q_phase_smem_fp4;

    constexpr int kv_smem_per_warp_fp4 =
        BLOCK_KV_FP4 * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1) +
        BLOCK_KV_FP4 * SCALE_DIM * (int)sizeof(__nv_fp8_e8m0) +
        HEAD_DIM * (BLOCK_KV_FP4 / 2) * (int)sizeof(__nv_fp4x2_e2m1) +
        HEAD_DIM * (BLOCK_KV_FP4 / 32) * (int)sizeof(__nv_fp8_e8m0);

    constexpr int kv_smem_per_warp_fp16 =
        BLOCK_KV_FP16 * HEAD_DIM * (int)sizeof(T) * 2;

    constexpr int kv_smem_per_warp =
        (kv_smem_per_warp_fp4 > kv_smem_per_warp_fp16) ? kv_smem_per_warp_fp4 : kv_smem_per_warp_fp16;

    constexpr int kv_phase_smem = NUM_WARPS * kv_smem_per_warp;

    constexpr int reduce_smem =
        (int)sizeof(float) * (NUM_WARPS * BLOCK_Q * 2 + NUM_WARPS * BLOCK_Q * HEAD_DIM);

    constexpr int smem_12 = (q_phase_smem > kv_phase_smem) ? q_phase_smem : kv_phase_smem;
    constexpr int smem_size = (smem_12 > reduce_smem) ? smem_12 : reduce_smem;

    auto kernel =
        thrift_attention_single_query_cta_kernel<
            T,
            BLOCK_KV_FP16,
            BLOCK_KV_FP4,
            HEAD_DIM,
            HEAD_DIM_2,
            SCALE_DIM,
            NUM_WARPS,
            kv_smem_per_warp>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<bs, TB_SIZE, smem_size>>>(
        Q_fp16,
        K_fp16,
        V_fp16,
        top_k,
        topk_count,
        Q_fp4,
        K_fp4,
        V_fp4,
        S_Q,
        S_K,
        S_V,
        O,
        bs,
        q_len,
        kv_len,
        kv_capacity,
        num_q_heads,
        num_kv_heads);

}

// ---------------------------------------------------------------------------
// Split-KV mixed-precision single-query kernel.
// Grid: (num_kv_splits, bs).  Each block is a single warp that processes a
// contiguous range of KV blocks (FP16 for top-k, FP4 for the rest), writes
// partial (O, rowmax, rowsum) to a global workspace, then the last block to
// finish reduces all partials and writes the final FP16 output.
//
// Optimisations over the naive version:
//   1. Bitmask for O(1) top-k lookup instead of O(topk_count) linear scan
//   2. Double-buffered FP4 KV loads (overlaps gmem latency with MMA compute)
//   3. float2 vectorised partial-O stores
//   4. Two-pass smem-assisted reduction (precompute per-row rescale once)
// ---------------------------------------------------------------------------
template<
    typename T,
    int BLOCK_KV_FP16,
    int BLOCK_KV_FP4,
    int HEAD_DIM,
    int HEAD_DIM_2,
    int SCALE_DIM,
    bool UPPER_ONLY = false>
__launch_bounds__(WARP_SIZE_sqmxfp4mix)
__global__
void thrift_attention_single_query_split_kernel(
    const T* Q_fp16,
    const T* K_fp16,
    const T* V_fp16,
    const int32_t* top_k,
    int topk_count,
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e8m0* S_Q,
    const __nv_fp8_e8m0* S_K,
    const __nv_fp8_e8m0* S_V,
    float* O_partial,
    float* rowmax_partial,
    float* rowsum_partial,
    T* O_final,
    int* split_counter,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_kv_splits,
    int num_q_heads,
    int num_kv_heads)
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = WARP_SIZE_sqmxfp4mix;
    constexpr int BLOCK_Q = 16;
    constexpr int MMA_K_FP4 = 64;
    constexpr int MMA_N = 8;

    // Size of one FP4 KV buffer in smem (bytes)
    constexpr int FP4_BUF_BYTES =
        BLOCK_KV_FP4 * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
      + BLOCK_KV_FP4 * SCALE_DIM  * (int)sizeof(__nv_fp8_e8m0)
      + HEAD_DIM * (BLOCK_KV_FP4 / 2)  * (int)sizeof(__nv_fp4x2_e2m1)
      + HEAD_DIM * (BLOCK_KV_FP4 / 32) * (int)sizeof(__nv_fp8_e8m0);

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int v_kv = cdiv_sqmxfp4mix(kv_capacity, 128) * 128;

    const int split_id = blockIdx.x;
    const int bid = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = tid;
    const int batch_id = bid / num_q_heads;
    const int q_head = bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;

    const int total_kv_blocks = cdiv_sqmxfp4mix(kv_len, BLOCK_KV_FP4);
    const int kv_block_start = (split_id * total_kv_blocks) / num_kv_splits;
    const int kv_block_end = ((split_id + 1) * total_kv_blocks) / num_kv_splits;

    Q   += bid * q_len       * HEAD_DIM_2;
    K   += kv_bid * kv_capacity * HEAD_DIM_2;
    V   += kv_bid * HEAD_DIM    * (v_kv / 2);
    S_Q += bid * q_len       * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM;
    S_V += kv_bid * v_kv        * SCALE_DIM;

    Q_fp16 += bid * q_len       * HEAD_DIM;
    K_fp16 += kv_bid * kv_capacity * HEAD_DIM;
    V_fp16 += kv_bid * kv_capacity * HEAD_DIM;

    const int partial_offset = bid * num_kv_splits + split_id;
    O_partial      += partial_offset * q_len * HEAD_DIM;
    rowmax_partial += partial_offset * q_len;
    rowsum_partial += partial_offset * q_len;

    const int32_t* topk_row = top_k + bid * topk_count;

    // ---- Optimisation 1: build local bitmask for O(1) top-k lookup ----
    constexpr int TOPK_MASK_WORDS = 4;  // 128 bits — covers up to 128 blocks/split
    constexpr int TOPK_MASK_BITS  = TOPK_MASK_WORDS * 32;
    uint32_t topk_local_mask[TOPK_MASK_WORDS] = {};

    const int num_kv_iters = kv_block_end - kv_block_start;

    for (int i = lane_id; i < topk_count; i += WARP_SIZE_sqmxfp4mix) {
        int b = topk_row[i] - kv_block_start;
        if (b >= 0 && b < min(num_kv_iters, TOPK_MASK_BITS))
            topk_local_mask[b >> 5] |= (1u << (b & 31));
    }
    #pragma unroll
    for (int w = 0; w < TOPK_MASK_WORDS; w++) {
        #pragma unroll
        for (int delta = WARP_SIZE_sqmxfp4mix / 2; delta > 0; delta >>= 1)
            topk_local_mask[w] |= __shfl_down_sync(0xffffffffu, topk_local_mask[w], delta);
        topk_local_mask[w] = __shfl_sync(0xffffffffu, topk_local_mask[w], 0);
    }

    const bool range_has_fp16 = (num_kv_iters <= TOPK_MASK_BITS)
        ? ((topk_local_mask[0] | topk_local_mask[1] |
            topk_local_mask[2] | topk_local_mask[3]) != 0)
        : range_has_topk_sqmxfp4mix(topk_row, topk_count, kv_block_start, kv_block_end);

    // Helper lambda: O(1) top-k test for local block index
    auto is_topk = [&](int local_idx) -> bool {
        if (local_idx < TOPK_MASK_BITS)
            return (topk_local_mask[local_idx >> 5] >> (local_idx & 31)) & 1;
        return block_in_topk_sqmxfp4mix(topk_row, topk_count, kv_block_start + local_idx);
    };

    extern __shared__ uint8_t smem[];

    // ---- Phase 1: load Q_fp4 + scales → registers ----
    const uint32_t Q_smem    = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[HEAD_DIM / MMA_K_FP4][4];
    uint32_t sfQ_rmem[HEAD_DIM / MMA_K_FP4];

    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {};
    constexpr int O_REGS = UPPER_ONLY ? 2 : 4;
    float O_rmem[HEAD_DIM / MMA_N][O_REGS] = {};

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
            uint32_t dst_addr = swizzle_sqmxfp4mix<HEAD_DIM_2>(Q_smem + row * HEAD_DIM_2 + col);
            const auto *src_addr = Q + row * HEAD_DIM_2 + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"(dst_addr), "l"(src_addr));
        }
    }
    {
        constexpr int SF_ROW_BYTES = SCALE_DIM * (int)sizeof(__nv_fp8_e8m0);
        for (int row = tid; row < q_len; row += TB_SIZE) {
            const uint32_t dst_addr = Q_sf_smem + row * SF_ROW_BYTES;
            const auto *src_addr = S_Q + row * SCALE_DIM;
            if constexpr (SF_ROW_BYTES == 2) {
                uint16_t value;
                asm volatile("ld.global.u16 %0, [%1];"
                    : "=h"(value)
                    : "l"(src_addr));
                asm volatile("st.shared.u16 [%0], %1;"
                    :: "r"(dst_addr), "h"(value));
            } else {
                asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                    :: "r"(dst_addr), "l"(src_addr), "n"(SF_ROW_BYTES));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncwarp();

    uint32_t Q_ld_base;
    {
        const int row_off = lane_id % 16;
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = swizzle_sqmxfp4mix<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
        uint32_t addr = Q_ld_base;
        addr ^= mma_id_d * (MMA_K_FP4 / 2);
        ldmatrix_x4_sqmxfp4mix(Q_rmem[mma_id_d], addr);
    }
    __syncwarp();

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
        const uint32_t offset =
            (sf_row_q * SCALE_DIM + mma_id_d * 2) * (uint32_t)sizeof(__nv_fp8_e8m0);
        sfQ_rmem[mma_id_d] = ld_shared_e8m0x2_sqmxfp4mix(Q_sf_smem + offset);
    }

    // ---- Phase 2: KV loop ----
    __syncwarp();

    // ---- Optimisation 2: double-buffered FP4 smem layout ----
    const uint32_t smem_base = __cvta_generic_to_shared(smem);

    uint32_t K_smem_buf[2], K_sf_buf[2], V_smem_buf[2], V_sf_buf[2], K_ld_buf[2];
    for (int b = 0; b < 2; b++) {
        K_smem_buf[b] = smem_base + b * FP4_BUF_BYTES;
        K_sf_buf[b]   = K_smem_buf[b] + BLOCK_KV_FP4 * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);
        V_smem_buf[b] = K_sf_buf[b]   + BLOCK_KV_FP4 * SCALE_DIM  * (int)sizeof(__nv_fp8_e8m0);
        V_sf_buf[b]   = V_smem_buf[b] + HEAD_DIM * (BLOCK_KV_FP4 / 2) * (int)sizeof(__nv_fp4x2_e2m1);

        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_buf[b] = swizzle_sqmxfp4mix<HEAD_DIM_2>(K_smem_buf[b] + row_off * HEAD_DIM_2 + col_off);
    }

    // FP16 smem layout (overlaps buffer region — used only after FP4 is done)
    const uint32_t K_smem_fp16 = smem_base;
    const uint32_t V_smem_fp16 = K_smem_fp16 + BLOCK_KV_FP16 * HEAD_DIM * (int)sizeof(T);

    // ---- Double-buffered FP4 loop ----
    {
        // Find first non-topk block
        int cur_iter = -1;
        if (range_has_fp16) {
            for (int i = 0; i < num_kv_iters; i++) {
                if (!is_topk(i)) { cur_iter = i; break; }
            }
        } else {
            if (num_kv_iters > 0) cur_iter = 0;
        }

        if (cur_iter >= 0) {
            int cur_buf = 0;
            int gid = kv_block_start + cur_iter;

            // Preload first FP4 block
            load_kv_fp4_async_sqmxfp4mix<BLOCK_KV_FP4, HEAD_DIM, HEAD_DIM_2, SCALE_DIM>(
                K_smem_buf[0], K_sf_buf[0], V_smem_buf[0], V_sf_buf[0],
                K + gid * BLOCK_KV_FP4 * HEAD_DIM_2,
                V + gid * BLOCK_KV_FP4 / 2,
                S_K + gid * BLOCK_KV_FP4 * SCALE_DIM,
                S_V + gid * BLOCK_KV_FP4 / 32,
                lane_id, v_kv);

            while (cur_iter >= 0) {
                // Find next non-topk block
                int next_iter = -1;
                for (int j = cur_iter + 1; j < num_kv_iters; j++) {
                    if (!range_has_fp16 || !is_topk(j)) {
                        next_iter = j;
                        break;
                    }
                }

                // Preload next block into alternate buffer
                if (next_iter >= 0) {
                    int next_gid = kv_block_start + next_iter;
                    load_kv_fp4_async_sqmxfp4mix<BLOCK_KV_FP4, HEAD_DIM, HEAD_DIM_2, SCALE_DIM>(
                        K_smem_buf[1 - cur_buf], K_sf_buf[1 - cur_buf],
                        V_smem_buf[1 - cur_buf], V_sf_buf[1 - cur_buf],
                        K + next_gid * BLOCK_KV_FP4 * HEAD_DIM_2,
                        V + next_gid * BLOCK_KV_FP4 / 2,
                        S_K + next_gid * BLOCK_KV_FP4 * SCALE_DIM,
                        S_V + next_gid * BLOCK_KV_FP4 / 32,
                        lane_id, v_kv);
                }

                // Wait for current buffer's load to complete
                if (next_iter >= 0) {
                    asm volatile("cp.async.wait_group %0;" :: "n"(1));
                } else {
                    asm volatile("cp.async.wait_all;");
                }
                __syncwarp();

                // Compute attention on current buffer.  GQA single-query usually
                // has q_len=4, so instantiate a variant that skips lower-half
                // scalar softmax/packing work for rows 8..15.
                if constexpr (UPPER_ONLY) {
                    compute_kv_fp4_sqmxfp4mix<BLOCK_KV_FP4, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, false, 2>(
                        rowmax, rowsum, O_rmem, Q_rmem, sfQ_rmem,
                        K_sf_buf[cur_buf], V_smem_buf[cur_buf], V_sf_buf[cur_buf],
                        K_ld_buf[cur_buf],
                        lane_id, softmax_scale);
                } else {
                    compute_kv_fp4_sqmxfp4mix<BLOCK_KV_FP4, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, true, 4>(
                        rowmax, rowsum, O_rmem, Q_rmem, sfQ_rmem,
                        K_sf_buf[cur_buf], V_smem_buf[cur_buf], V_sf_buf[cur_buf],
                        K_ld_buf[cur_buf],
                        lane_id, softmax_scale);
                }

                cur_buf = 1 - cur_buf;
                cur_iter = next_iter;
            }
        }
    }

    // FP16 (top-k) blocks — Q FP16 loaded once
    if (range_has_fp16) {
        uint32_t Q_fp16_rmem[HEAD_DIM / 16][4];
        load_q_fp16_to_regs_single_query_sqmxfp4mix<T, HEAD_DIM, BLOCK_Q>(
            Q_fp16_rmem, K_smem_fp16, Q_fp16, lane_id, q_len);

        for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
            if (!is_topk(kv_iter))
                continue;

            const int global_block_id = kv_block_start + kv_iter;
            const T* K_fp16_ptr_iter = K_fp16 + global_block_id * BLOCK_KV_FP4 * HEAD_DIM;
            const T* V_fp16_ptr_iter = V_fp16 + global_block_id * BLOCK_KV_FP4 * HEAD_DIM;

            if constexpr (UPPER_ONLY) {
                attention_inner_loop_fp16_mxfp4<T, BLOCK_KV_FP4, BLOCK_KV_FP16, HEAD_DIM, false, 2>(
                    rowmax, rowsum, O_rmem,
                    Q_fp16_rmem,
                    K_smem_fp16, V_smem_fp16,
                    K_fp16_ptr_iter, V_fp16_ptr_iter,
                    lane_id, softmax_scale);
            } else {
                attention_inner_loop_fp16_mxfp4<T, BLOCK_KV_FP4, BLOCK_KV_FP16, HEAD_DIM, true, 4>(
                    rowmax, rowsum, O_rmem,
                    Q_fp16_rmem,
                    K_smem_fp16, V_smem_fp16,
                    K_fp16_ptr_iter, V_fp16_ptr_iter,
                lane_id, softmax_scale);
            }
        }
    }

    // ---- Phase 3: Write partials to global memory ----
    // Optimisation 3: float2 vectorised stores
    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
        const int row = lane_id / 4;
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
        float *regs = O_rmem[mma_id_d];

        if (row < q_len) {
            *reinterpret_cast<float2*>(&O_partial[row * HEAD_DIM + col]) =
                make_float2(regs[0], regs[1]);
        }
        if constexpr (!UPPER_ONLY) {
        if (row + 8 < q_len) {
            *reinterpret_cast<float2*>(&O_partial[(row + 8) * HEAD_DIM + col]) =
                make_float2(regs[2], regs[3]);
        }
        }
    }

    if (lane_id % 4 == 0) {
        const int row = lane_id / 4;
        if (row < q_len) {
            rowmax_partial[row] = rowmax[0];
            rowsum_partial[row] = rowsum[0];
        }
        if constexpr (!UPPER_ONLY) {
        if (row + 8 < q_len) {
            rowmax_partial[row + 8] = rowmax[1];
            rowsum_partial[row + 8] = rowsum[1];
        }
        }
    }

    // ---- Fused reduction: last split to finish reduces all partials ----
    __threadfence();

    int completed = 0;
    if (lane_id == 0)
        completed = atomicAdd(&split_counter[bid], 1);
    completed = __shfl_sync(0xFFFFFFFF, completed, 0);

    if (completed != num_kv_splits - 1)
        return;

    // ---- Optimisation 4: two-pass smem-assisted reduction ----
    constexpr float FP4_RANGE_INV = 1.0f / (6.0f);
    const float* batch_O = O_partial - split_id * q_len * HEAD_DIM;
    const float* batch_rowmax = rowmax_partial - split_id * q_len;
    const float* batch_rowsum = rowsum_partial - split_id * q_len;

    // Pass 1: 16 threads compute per-row rescale factors + norms → smem
    // Layout: rsc_smem[s * q_len + row], then norm_smem[row]
    float* rsc_smem  = reinterpret_cast<float*>(smem);
    float* norm_smem = rsc_smem + num_kv_splits * q_len;

    if (lane_id < q_len) {
        const int row = lane_id;
        float global_max = -FLT_MAX;
        for (int s = 0; s < num_kv_splits; s++)
            global_max = max(global_max, batch_rowmax[s * q_len + row]);

        float sum_acc = 0.0f;
        for (int s = 0; s < num_kv_splits; s++) {
            float rsc = __expf(batch_rowmax[s * q_len + row] - global_max);
            rsc_smem[s * q_len + row] = rsc;
            sum_acc += batch_rowsum[s * q_len + row] * rsc;
        }
        norm_smem[row] = FP4_RANGE_INV / sum_acc;
    }
    __syncwarp();

    // Pass 2: reduce only the valid q_len rows.  Qwen-style grouped single-query
    // passes q_len=4, so reducing the full m16 tile mostly moves dead data.
    const int valid_elems = q_len * HEAD_DIM;
    for (int flat_idx = lane_id; flat_idx < valid_elems; flat_idx += TB_SIZE) {
        const int row = flat_idx / HEAD_DIM;
        const int col = flat_idx % HEAD_DIM;

        float acc = 0.0f;
        for (int s = 0; s < num_kv_splits; s++) {
            float rsc = rsc_smem[s * q_len + row];
            acc += batch_O[s * q_len * HEAD_DIM + row * HEAD_DIM + col] * rsc;
        }

        float norm = norm_smem[row];
        O_final[bid * q_len * HEAD_DIM + row * HEAD_DIM + col] =
            Traits::from_float(acc * norm);
    }
}

template<typename T, int HEAD_DIM>
static void thrift_attention_single_query_split_launch(
    const T* Q_fp16,
    const T* K_fp16,
    const T* V_fp16,
    const int32_t* top_k,
    int topk_count,
    const __nv_fp4x2_e2m1* Q_fp4,
    const __nv_fp4x2_e2m1* K_fp4,
    const __nv_fp4x2_e2m1* V_fp4,
    const __nv_fp8_e8m0* S_Q,
    const __nv_fp8_e8m0* S_K,
    const __nv_fp8_e8m0* S_V,
    T* O,
    float* workspace,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 32;
    constexpr int BLOCK_KV_FP4 = 64;
    constexpr int BLOCK_KV_FP16 = 64;
    constexpr int BLOCK_Q = 16;
    constexpr int TB_SIZE = WARP_SIZE_sqmxfp4mix;

    const int total_kv_blocks = cdiv_sqmxfp4mix(kv_len, BLOCK_KV_FP4);
    const int target_split_ctas = single_query_target_split_ctas_sqmxfp4mix(total_kv_blocks);
    int num_kv_splits = max(1, min(total_kv_blocks, cdiv_sqmxfp4mix(target_split_ctas, bs)));
    num_kv_splits = min(num_kv_splits, total_kv_blocks);

    float* O_partial      = workspace;
    float* rowmax_partial = O_partial + bs * num_kv_splits * q_len * HEAD_DIM;
    float* rowsum_partial = rowmax_partial + bs * num_kv_splits * q_len;
    int* split_counter    = reinterpret_cast<int*>(rowsum_partial + bs * num_kv_splits * q_len);

    cudaMemsetAsync(split_counter, 0, bs * sizeof(int));

    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * (int)sizeof(__nv_fp8_e8m0);

    // Single FP4 buffer size (K + K_scales + V_transposed + V_scales)
    constexpr int kv_smem_fp4_single =
        BLOCK_KV_FP4 * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
      + BLOCK_KV_FP4 * SCALE_DIM  * (int)sizeof(__nv_fp8_e8m0)
      + HEAD_DIM * (BLOCK_KV_FP4 / 2)  * (int)sizeof(__nv_fp4x2_e2m1)
      + HEAD_DIM * (BLOCK_KV_FP4 / 32) * (int)sizeof(__nv_fp8_e8m0);

    // Double-buffered FP4
    constexpr int kv_smem_fp4 = kv_smem_fp4_single * 2;

    constexpr int kv_smem_fp16 = BLOCK_KV_FP16 * HEAD_DIM * (int)sizeof(T) * 2;

    constexpr int kv_smem = (kv_smem_fp4 > kv_smem_fp16) ? kv_smem_fp4 : kv_smem_fp16;
    constexpr int smem_size = (q_phase_smem > kv_smem) ? q_phase_smem : kv_smem;

    dim3 grid(num_kv_splits, bs);
    if (q_len <= 8) {
        auto kernel = thrift_attention_single_query_split_kernel<
            T, BLOCK_KV_FP16, BLOCK_KV_FP4, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, TB_SIZE, smem_size>>>(
            Q_fp16, K_fp16, V_fp16,
            top_k, topk_count,
            Q_fp4, K_fp4, V_fp4,
            S_Q, S_K, S_V,
            O_partial, rowmax_partial, rowsum_partial,
            O, split_counter,
            bs, q_len, kv_len, kv_capacity, num_kv_splits,
            num_q_heads, num_kv_heads);
    } else {
        auto kernel = thrift_attention_single_query_split_kernel<
            T, BLOCK_KV_FP16, BLOCK_KV_FP4, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, TB_SIZE, smem_size>>>(
            Q_fp16, K_fp16, V_fp16,
            top_k, topk_count,
            Q_fp4, K_fp4, V_fp4,
            S_Q, S_K, S_V,
            O_partial, rowmax_partial, rowsum_partial,
            O, split_counter,
            bs, q_len, kv_len, kv_capacity, num_kv_splits,
            num_q_heads, num_kv_heads);
    }

}

template<typename T>
static void thrift_attention_single_query_mxfp4_typed(
    const void* Q_fp16_raw,
    const void* K_fp16_raw,
    const void* V_fp16_raw,
    const int32_t* top_k,
    int topk_count,
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
    int head_dim)
{
    auto Q_fp16 = reinterpret_cast<const T*>(Q_fp16_raw);
    auto K_fp16 = reinterpret_cast<const T*>(K_fp16_raw);
    auto V_fp16 = reinterpret_cast<const T*>(V_fp16_raw);

    auto Q = reinterpret_cast<const __nv_fp4x2_e2m1*>(Q_raw);
    auto K = reinterpret_cast<const __nv_fp4x2_e2m1*>(K_raw);
    auto V = reinterpret_cast<const __nv_fp4x2_e2m1*>(V_raw);

    auto S_Q = reinterpret_cast<const __nv_fp8_e8m0*>(S_Q_raw);
    auto S_K = reinterpret_cast<const __nv_fp8_e8m0*>(S_K_raw);
    auto S_V = reinterpret_cast<const __nv_fp8_e8m0*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    constexpr int BLOCK_KV_FP4 = 64;
    const int total_kv_blocks = cdiv_sqmxfp4mix(kv_len, BLOCK_KV_FP4);

    if (total_kv_blocks < 128) {
        // CTA-cooperative: up to kv_len=8192
        if (head_dim == 64) {
            thrift_attention_single_query_cta_launch<T, 64>(
                Q_fp16, K_fp16, V_fp16,
                top_k, topk_count,
                Q, K, V, S_Q, S_K, S_V, O,
                bs, q_len, kv_len, kv_capacity,
                num_q_heads, num_kv_heads);
        } else {
            thrift_attention_single_query_cta_launch<T, 128>(
                Q_fp16, K_fp16, V_fp16,
                top_k, topk_count,
                Q, K, V, S_Q, S_K, S_V, O,
                bs, q_len, kv_len, kv_capacity,
                num_q_heads, num_kv_heads);
        }
    } else {
        // Split-KV: kv_len > 4096
        auto workspace = reinterpret_cast<float*>(workspace_raw);
        if (head_dim == 64) {
            thrift_attention_single_query_split_launch<T, 64>(
                Q_fp16, K_fp16, V_fp16,
                top_k, topk_count,
                Q, K, V, S_Q, S_K, S_V, O, workspace,
                bs, q_len, kv_len, kv_capacity,
                num_q_heads, num_kv_heads);
        } else {
            thrift_attention_single_query_split_launch<T, 128>(
                Q_fp16, K_fp16, V_fp16,
                top_k, topk_count,
                Q, K, V, S_Q, S_K, S_V, O, workspace,
                bs, q_len, kv_len, kv_capacity,
                num_q_heads, num_kv_heads);
        }
    }
}

void thrift_attention_single_query_mxfp4(
    const void* Q_fp16_raw,
    const void* K_fp16_raw,
    const void* V_fp16_raw,
    const int32_t* top_k,
    int topk_count,
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
    bool is_bf16)
{
    if (is_bf16) {
        thrift_attention_single_query_mxfp4_typed<__nv_bfloat16>(
            Q_fp16_raw, K_fp16_raw, V_fp16_raw, top_k, topk_count,
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            workspace_raw, bs, q_len, kv_len, kv_capacity, num_q_heads,
            num_kv_heads, head_dim);
    } else {
        thrift_attention_single_query_mxfp4_typed<half>(
            Q_fp16_raw, K_fp16_raw, V_fp16_raw, top_k, topk_count,
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            workspace_raw, bs, q_len, kv_len, kv_capacity, num_q_heads,
            num_kv_heads, head_dim);
    }
}
