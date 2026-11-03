/*
 * Givens Rotation Kernel - Common Definitions
 * Shared constants, macros, and utility functions
 */

#ifndef GIVENS_COMMON_H
#define GIVENS_COMMON_H

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

// FP16 support
#ifdef USE_ROCM
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

// ---------------------------------------------------------------------------
// Optimization constants
// ---------------------------------------------------------------------------
#define SHMEM_BANK_SIZE 32
#define WARP_SIZE 32
#define D_TILE 1024
#define BLOCK_SIZE 8
#define PERSISTENT_BATCH_THRESHOLD 64
#define COUPLING_TPB 16

// Bank-conflict-free shared memory indexing
#define SHMEM_IDX(idx) ((idx) + ((idx) / SHMEM_BANK_SIZE) * 4)

// Launch bounds: split by kernel family.
// FFN/MoR kernels launch up to D/2 threads (max 1024 for D=2048).
// Coupling kernels use half_D threads (typically 32 for D=64).
// Higher min_blocks = better occupancy = better latency hiding on
// memory-bound backward (save_buf reads) and forward (param loads).
#define LAUNCH_BOUNDS_FWD __launch_bounds__(1024, 4)
#define LAUNCH_BOUNDS_BWD __launch_bounds__(1024, 2)
#define LAUNCH_BOUNDS_MOR_BWD __launch_bounds__(1024, 2)
// Radix-4 fuses 2 stages per iteration — needs ~2x registers per thread.
// Fewer threads (256) + relaxed min_blocks (2) = ~128 regs/thread budget.
#define LAUNCH_BOUNDS_R4_FWD __launch_bounds__(256, 2)
#define LAUNCH_BOUNDS_R4_BWD __launch_bounds__(256, 2)
// Coupling kernels are memory-bandwidth-bound (tile loads dominate).
// High occupancy (min_blocks=8) is critical to hide global memory latency,
// even at the cost of register spilling. This matches the original perf.
// Tiled coupling kernels use dim3(half_D, TPB) = (32, 16) = 512 threads.
// Tighter bounds allow better register allocation.
#define LAUNCH_BOUNDS_COUPLING_FWD __launch_bounds__(512, 8)
#define LAUNCH_BOUNDS_COUPLING_BWD __launch_bounds__(512, 8)
// Warp-shuffle coupling: data lives in registers, only smem is coupling tile.
// Can run up to 1024 threads (32 lanes × 32 tokens) with lower occupancy
// target since the kernel is compute-bound on shfl + FMA, not memory-bound.
#define LAUNCH_BOUNDS_COUPLING_WARP_FWD __launch_bounds__(1024, 2)

// Warp-level or block-level barrier depending on problem size.
// When half_D <= WARP_SIZE, all pair threads fit in one warp so
// __syncwarp() suffices (cheaper than full-block __syncthreads()).
//
// The test is the WAVEFRONT SIZE, not a list of chips. RDNA (gfx10/11/12) is
// wave32 like NVIDIA and can use __syncwarp(); CDNA and GCN (gfx9xx, MI250/
// MI300) are wave64, where the 32 pair threads do NOT span a whole wavefront
// and only __syncthreads() is correct.
//
// This replaced an allowlist of __gfx1100__..__gfx1103__. An allowlist is
// wrong by default: every chip not named took the slow branch even when it was
// wave32. gfx1151 (RDNA3.5, Strix Halo — the sparks) is exactly that case, so
// every coupling stage with half_D <= 32 was paying a full block barrier it did
// not need. gfx1150/1151/1152/1153, gfx12xx and anything later would each have
// needed another edit; a wave-size test needs none.
//
// __GFX9__ etc. are clang's AMDGPU family macros and cover every member of a
// family (gfx1151 is GFX11). They are only defined during device compilation,
// so the host pass falls through to the portable branch — correct either way,
// since STAGE_BARRIER is only ever expanded inside __global__/__device__ code.
// The explicit gfx9 target macros are belt and braces in case a toolchain
// defines only the per-target macro.
#if defined(USE_ROCM)
#  if defined(__GFX8__) || defined(__GFX9__) || defined(__gfx900__) || \
      defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
      defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || \
      defined(__gfx950__)
// wave64: a 32-thread pair group is a partial wavefront, so sync the block.
#    define STAGE_BARRIER(half_D) __syncthreads()
#  else
// wave32 (RDNA: gfx10xx/11xx/12xx, including gfx1151), or host pass.
#    define STAGE_BARRIER(half_D) \
        do { if ((half_D) <= WARP_SIZE) __syncwarp(); else __syncthreads(); } while(0)
#  endif
#else
#define STAGE_BARRIER(half_D) \
    do { if ((half_D) <= WARP_SIZE) __syncwarp(); else __syncthreads(); } while(0)
#endif

// ---------------------------------------------------------------------------
// Fast atan2 approximation (optional)
// ---------------------------------------------------------------------------
#ifdef USE_FAST_ATAN2

__device__ inline float fast_atan2f(float y, float x) {
    const float M_PIf_4 = 0.78539816339744830962f;
    float ax = fabsf(x), ay = fabsf(y);
    float a = fminf(ax, ay) / fmaxf(ax, ay + 1e-12f);
    float angle = M_PIf_4 * a + a * (1.0f - a) * (0.2447f + 0.0663f * a);
    
    if (x < 0.0f && y >= 0.0f) angle = M_PIf - angle;
    else if (x < 0.0f && y < 0.0f) angle = -M_PIf + angle;
    else if (x >= 0.0f && y < 0.0f) angle = -angle;
    
    return (y < 0.0f) ? -angle : angle;
}

#define ATAN2F(y, x) fast_atan2f((y), (x))
#else
#define ATAN2F(y, x) atan2f((y), (x))
#endif

// ---------------------------------------------------------------------------
// Fast tanh approximation — Pade (3,3): max error < 0.005
// tanhf costs ~40-50 cycles; this costs ~8-10 cycles.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fast_tanhf(float x) {
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// T-109: exact tanhf. fast_tanhf is unbounded for |x| > 3 and the trained couplings drive
// |modulation| past that; the ~30 extra cycles per pair-stage are noise next to the einsum.
#define FAST_TANHF(x) tanhf(x)

// ---------------------------------------------------------------------------
// Warp shuffle abstractions — ROCm 7.2 requires 64-bit masks for _sync variants.
// On ROCm we use the non-_sync versions (implicit wavefront sync).
// ---------------------------------------------------------------------------
#ifdef USE_ROCM
#define SHFL_DOWN(val, offset)      __shfl_down((val), (offset))
#define SHFL_BROADCAST(val, lane)   __shfl((val), (lane))
#define SHFL_XOR(val, lanemask)     __shfl_xor((val), (lanemask))
#else
#define SHFL_DOWN(val, offset)      __shfl_down_sync(0xFFFFFFFF, (val), (offset), WARP_SIZE)
#define SHFL_BROADCAST(val, lane)   __shfl_sync(0xFFFFFFFF, (val), (lane))
#define SHFL_XOR(val, lanemask)     __shfl_xor_sync(0xFFFFFFFF, (val), (lanemask), WARP_SIZE)
#endif

// ---------------------------------------------------------------------------
// Warp-level reduction (sum across WARP_SIZE lanes)
// ---------------------------------------------------------------------------
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += SHFL_DOWN(val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// FP16 data I/O helpers — load as float, store from float
// Zero-cost: compiler inlines completely. Float specialization is identity.
// ---------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ float load_f(const T* ptr, int idx);

template<>
__device__ __forceinline__ float load_f<float>(const float* ptr, int idx) {
    return ptr[idx];
}

template<>
__device__ __forceinline__ float load_f<__half>(const __half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

template<typename T>
__device__ __forceinline__ void store_f(T* ptr, int idx, float val);

template<>
__device__ __forceinline__ void store_f<float>(float* ptr, int idx, float val) {
    ptr[idx] = val;
}

template<>
__device__ __forceinline__ void store_f<__half>(__half* ptr, int idx, float val) {
    ptr[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Vectorized pair load/store — float2 for 2x bandwidth on aligned data
// ---------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ void load_pair(const T* ptr, int pair_offset, float& x0, float& x1);

template<>
__device__ __forceinline__ void load_pair<float>(const float* ptr, int pair_offset, float& x0, float& x1) {
    float2 v = *reinterpret_cast<const float2*>(&ptr[pair_offset]);
    x0 = v.x;
    x1 = v.y;
}

template<>
__device__ __forceinline__ void load_pair<__half>(const __half* ptr, int pair_offset, float& x0, float& x1) {
    __half2 v = *reinterpret_cast<const __half2*>(&ptr[pair_offset]);
    x0 = __half2float(v.x);
    x1 = __half2float(v.y);
}

template<typename T>
__device__ __forceinline__ void store_pair(T* ptr, int pair_offset, float x0, float x1);

template<>
__device__ __forceinline__ void store_pair<float>(float* ptr, int pair_offset, float x0, float x1) {
    *reinterpret_cast<float2*>(&ptr[pair_offset]) = make_float2(x0, x1);
}

template<>
__device__ __forceinline__ void store_pair<__half>(__half* ptr, int pair_offset, float x0, float x1) {
    *reinterpret_cast<__half2*>(&ptr[pair_offset]) = __halves2half2(__float2half(x0), __float2half(x1));
}

// ---------------------------------------------------------------------------
// Pair index computation for butterfly/cyclic permutations
// ---------------------------------------------------------------------------
__device__ inline void compute_pair_indices(
    int pair_idx, int shift_val, int D,
    int& p0, int& p1) {
    // D and stride are always powers of 2: bitwise AND replaces
    // expensive integer modulo/division (~30 cycles -> 1 cycle)
    if (shift_val >= 0) {
        int d_mask = D - 1;
        p0 = (2 * pair_idx - shift_val + D) & d_mask;
        p1 = (2 * pair_idx + 1 - shift_val + D) & d_mask;
    } else {
        int stride = -shift_val;
        int stride_mask = stride - 1;
        int block_start = (pair_idx & ~stride_mask) << 1;
        int in_block = pair_idx & stride_mask;
        p0 = block_start + in_block;
        p1 = block_start + in_block + stride;
    }
}

#endif // GIVENS_COMMON_H
