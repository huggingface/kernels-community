#pragma once

#include <cstdint>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int TA_WARP_SIZE = 32;

__host__ __device__ inline int ta_cdiv(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int ta_sage_perm32(int x) {
    return (x / 8) * 2 + ((x % 8) / 2) * 8 + (x % 2);
}

__host__ __device__ inline int ta_sage_perm32_inv(int x) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (ta_sage_perm32(i) == x) {
            return i;
        }
    }
    return x;
}

__host__ __device__ inline int ta_sage_perm_seq(int x, bool inverse = false) {
    const int base = (x / 32) * 32;
    const int local = x & 31;
    return base + (inverse ? ta_sage_perm32_inv(local) : ta_sage_perm32(local));
}

// K/V are stored in the physical order consumed by the shuffle-free P operand.
// This names the direction used by the quantizers and causal masks explicitly.
__host__ __device__ inline int ta_kv_logical_from_physical(int physical) {
    return ta_sage_perm_seq(physical, false);
}

__host__ __device__ inline int ta_kv_physical_from_logical(int logical) {
    return ta_sage_perm_seq(logical, true);
}

template <int STRIDE>
__device__ inline uint32_t ta_swizzle(uint32_t index) {
    if constexpr (STRIDE == 16) {
        return index;
    }
    const uint32_t row_idx = (index / STRIDE) % 8;
    const uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}

__device__ inline void ta_ldmatrix_x4(uint32_t reg[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
         : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
         : "r"(addr));
}

__device__ inline void ta_ldmatrix_x2(uint32_t reg[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
         : "=r"(reg[0]), "=r"(reg[1])
         : "r"(addr));
}

__device__ inline void ta_ldmatrix_x2_trans(uint32_t reg[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
         : "=r"(reg[0]), "=r"(reg[1])
         : "r"(addr));
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline void ta_gmem_to_smem(uint32_t dst, const T* src, int tid, int src_stride) {
    constexpr int num_elements = 16 / sizeof(T);
    constexpr int num_iters = (HEIGHT * WIDTH) / (num_elements * TB_SIZE);

    for (int iter = 0; iter < num_iters; iter++) {
        const int index = (iter * TB_SIZE + tid) * num_elements;
        const int row = index / WIDTH;
        const int col = index % WIDTH;
        const uint32_t dst_addr = ta_swizzle<WIDTH * static_cast<int>(sizeof(T))>(
            dst + (row * WIDTH + col) * static_cast<int>(sizeof(T)));
        const T* src_addr = src + row * src_stride + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(dst_addr), "l"(src_addr));
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline void ta_gmem_to_smem_linear(uint32_t dst, const T* src, int tid, int src_stride) {
    constexpr int num_elements = 16 / sizeof(T);
    constexpr int total_vectors = (HEIGHT * WIDTH) / num_elements;

    for (int vec = tid; vec < total_vectors; vec += TB_SIZE) {
        const int index = vec * num_elements;
        const int row = index / WIDTH;
        const int col = index % WIDTH;
        const uint32_t dst_addr = dst + (row * WIDTH + col) * static_cast<int>(sizeof(T));
        const T* src_addr = src + row * src_stride + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(dst_addr), "l"(src_addr));
    }
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline void ta_load_scales(uint32_t dst, const T* src, int src_stride, int tid) {
    constexpr int cp_size = WIDTH * sizeof(T);
    static_assert(cp_size < 16);

    auto load_row = [&](int row) {
        const uint32_t dst_addr = dst + row * WIDTH * sizeof(T);
        const T* src_addr = src + row * src_stride;
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

    for (int iter = 0; iter < HEIGHT / TB_SIZE; iter++) {
        load_row(iter * TB_SIZE + tid);
    }
    if constexpr (HEIGHT % TB_SIZE != 0) {
        const int row = HEIGHT / TB_SIZE * TB_SIZE + tid;
        if (row < HEIGHT) {
            load_row(row);
        }
    }
}

__device__ inline uint32_t ta_ld_shared_u16(uint32_t addr) {
    uint16_t value;
    asm volatile("ld.shared.u16 %0, [%1];"
        : "=h"(value)
        : "r"(addr));
    return static_cast<uint32_t>(value);
}

__device__ inline uint32_t ta_cvt_8xf32_to_e2m1_packed(
    float f0, float f1, float f2, float f3,
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
          "f"(f4), "f"(f5), "f"(f6), "f"(f7));
    return packed;
}

__device__ inline uint32_t ta_cvt_2xf32_to_e8m0_packed(float f0, float f1) {
    uint16_t packed;
    asm volatile(
        "{cvt.rp.satfinite.ue8m0x2.f32 %0, %2, %1;}"
        : "=h"(packed)
        : "f"(f0), "f"(f1));
    return static_cast<uint32_t>(packed);
}

__device__ inline uint32_t ta_cvt_4xf32_to_e4m3_packed(
    float f0, float f1, float f2, float f3) {
    uint32_t packed;
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, %1, %2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, %3, %4;\n\t"
        "mov.b32 %0, {lo, hi};\n\t"
        "}"
        : "=r"(packed)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3));
    return packed;
}

__device__ inline void ta_mma_m16n8k64_nvfp4(
    uint32_t a[4],
    uint32_t b[2],
    uint32_t scale_a,
    uint32_t scale_b,
    float acc[4]) {
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
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
          "r"(scale_a), "h"(byte_id), "h"(thread_id),
          "r"(scale_b), "h"(byte_id), "h"(thread_id));
}

__device__ inline void ta_mma_m16n8k64_mxfp4(
    uint32_t a[4],
    uint32_t b[2],
    uint32_t scale_a,
    uint32_t scale_b,
    float acc[4]) {
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
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
          "r"(scale_a), "h"(byte_id), "h"(thread_id),
          "r"(scale_b), "h"(byte_id), "h"(thread_id));
}

__device__ __forceinline__ void ta_mma_m16n8k16_f16(
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    float (&acc)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

__device__ __forceinline__ void ta_mma_m16n8k16_bf16(
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    float (&acc)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

template <typename T>
struct PrecisionTraits;

template <>
struct PrecisionTraits<half> {
    using scalar = half;
    using vec2 = __half2;

    static __device__ __forceinline__ float to_float(scalar x) {
        return __half2float(x);
    }

    static __device__ __forceinline__ scalar from_float(float x) {
        return __float2half_rn(x);
    }

    static __device__ __forceinline__ vec2 pack2(float a, float b) {
        return __floats2half2_rn(a, b);
    }

    static __device__ __forceinline__ vec2 make_vec2(scalar a, scalar b) {
        return __halves2half2(a, b);
    }

    static __device__ __forceinline__ vec2 abs2(vec2 x) {
        return __habs2(x);
    }

    static __device__ __forceinline__ vec2 max2(vec2 a, vec2 b) {
        return __hmax2(a, b);
    }

    static __device__ __forceinline__ scalar low(vec2 x) {
        return __low2half(x);
    }

    static __device__ __forceinline__ scalar high(vec2 x) {
        return __high2half(x);
    }

    static __device__ __forceinline__ float2 to_float2(vec2 x) {
        return __half22float2(x);
    }

    static __device__ __forceinline__ void mma(
        const uint32_t (&a)[4],
        const uint32_t (&b)[2],
        float (&acc)[4]) {
        ta_mma_m16n8k16_f16(a, b, acc);
    }
};

template <>
struct PrecisionTraits<__nv_bfloat16> {
    using scalar = __nv_bfloat16;
    using vec2 = __nv_bfloat162;

    static __device__ __forceinline__ float to_float(scalar x) {
        return __bfloat162float(x);
    }

    static __device__ __forceinline__ scalar from_float(float x) {
        return __float2bfloat16_rn(x);
    }

    static __device__ __forceinline__ vec2 pack2(float a, float b) {
        return __floats2bfloat162_rn(a, b);
    }

    static __device__ __forceinline__ vec2 make_vec2(scalar a, scalar b) {
        return __halves2bfloat162(a, b);
    }

    static __device__ __forceinline__ vec2 abs2(vec2 x) {
        return __habs2(x);
    }

    static __device__ __forceinline__ vec2 max2(vec2 a, vec2 b) {
        return __hmax2(a, b);
    }

    static __device__ __forceinline__ scalar low(vec2 x) {
        return __low2bfloat16(x);
    }

    static __device__ __forceinline__ scalar high(vec2 x) {
        return __high2bfloat16(x);
    }

    static __device__ __forceinline__ float2 to_float2(vec2 x) {
        return __bfloat1622float2(x);
    }

    static __device__ __forceinline__ void mma(
        const uint32_t (&a)[4],
        const uint32_t (&b)[2],
        float (&acc)[4]) {
        ta_mma_m16n8k16_bf16(a, b, acc);
    }
};
