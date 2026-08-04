// Original to this repo; not derived from vLLM (the vllm:: namespace is used
// only for consistency with the vendored kernels in this tree).
//
// NVFP4 W4A16 decode GEMV (batch-1 specialized).
//
// y[m, n] = sum_k x[m, k] * dequant(W[n, k]),  W stored as e2m1 packed uint8
// [N, K/2] with row-major FP8-E4M3 block scales [N, K/16] and a per-tensor
// global scale (alpha = 1/global_scale).

#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace vllm {
namespace nvfp4_gemv {

constexpr int WARPS_PER_CTA = 8;
constexpr int ROWS_PER_WARP = 2;
constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
constexpr int THREADS = WARPS_PER_CTA * 32;
constexpr int XV_STRIDE = 5;  // uint4 per 32-elem x chunk (4 data + 1 pad)

enum class Mode { kPlain, kGatedIn, kSwiGLUOut };

__device__ __forceinline__ void decode_scales(const uint8_t* __restrict__ srow,
                                              int c, float& s0, float& s1) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  const unsigned short sboth =
      *reinterpret_cast<const unsigned short*>(srow + 2 * c);
  s0 = __half2float(
      __half(__nv_cvt_fp8_to_halfraw((uint8_t)(sboth & 0xffu), __NV_E4M3)));
  s1 = __half2float(
      __half(__nv_cvt_fp8_to_halfraw((uint8_t)(sboth >> 8), __NV_E4M3)));
#endif
}

__device__ __forceinline__ float chunk_dot(const uint4& w4, float s0, float s1,
                                           const __half2* __restrict__ xr) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  __half2 h0a = __float2half2_rn(0.f), h0b = __float2half2_rn(0.f);
  uint32_t wa = w4.x, wb = w4.y;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2_raw pa =
        __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(wa & 0xffu), __NV_E2M1);
    const __half2_raw pb =
        __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(wb & 0xffu), __NV_E2M1);
    h0a = __hfma2(__half2(pa), xr[i], h0a);
    h0b = __hfma2(__half2(pb), xr[i + 4], h0b);
    wa >>= 8;
    wb >>= 8;
  }
  __half2 h1a = __float2half2_rn(0.f), h1b = __float2half2_rn(0.f);
  wa = w4.z;
  wb = w4.w;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2_raw pa =
        __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(wa & 0xffu), __NV_E2M1);
    const __half2_raw pb =
        __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(wb & 0xffu), __NV_E2M1);
    h1a = __hfma2(__half2(pa), xr[i + 8], h1a);
    h1b = __hfma2(__half2(pb), xr[i + 12], h1b);
    wa >>= 8;
    wb >>= 8;
  }
  const __half2 h0 = __hadd2(h0a, h0b);
  const __half2 h1 = __hadd2(h1a, h1b);
  return (__half2float(h0.x) + __half2float(h0.y)) * s0 +
         (__half2float(h1.x) + __half2float(h1.y)) * s1;
#else
  return 0.f;
#endif
}

template <Mode MODE>
__global__ void __launch_bounds__(THREADS) gemv_kernel(
    const __nv_bfloat16* __restrict__ A,  // [M, K] activations
    const __nv_bfloat16* __restrict__ G,  // [M, K] gate for kGatedIn, else unused
    const uint4* __restrict__ B,          // [N, K/32] 16B chunks (K/2 bytes/row)
    const uint8_t* __restrict__ Bsf,      // [N, K/16] fp8-e4m3 bits, row-major
    const float* __restrict__ alpha,      // [1] scalar or [N] per-row = 1/global_scale
    __nv_bfloat16* __restrict__ D,        // [M, N]
    int N, int K, int alpha_per_row) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  // x cached as 16B vectors with a 5-uint4 stride per 16-half2 chunk: keeps
  // ld.shared.128 alignment while the odd word-phase across lanes stays
  // bank-conflict-free.
  extern __shared__ uint4 xsv[];
  const int m = blockIdx.y;
  const __nv_bfloat16* a_row = A + (size_t)m * K;
  const __nv_bfloat16* g_row =
      (MODE == Mode::kGatedIn) ? G + (size_t)m * K : nullptr;
  __half2* xh = reinterpret_cast<__half2*>(xsv);
  for (int p = threadIdx.x; p < (K >> 1); p += blockDim.x) {
    const int c = p >> 4, i = p & 15;
    float v0 = __bfloat162float(a_row[2 * p]);
    float v1 = __bfloat162float(a_row[2 * p + 1]);
    if (MODE == Mode::kGatedIn) {
      v0 *= 1.f / (1.f + expf(-__bfloat162float(g_row[2 * p])));
      v1 *= 1.f / (1.f + expf(-__bfloat162float(g_row[2 * p + 1])));
    }
    xh[c * (XV_STRIDE * 4) + i] = __floats2half2_rn(v0, v1);
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // SwiGLU fused weights are row-concatenated as [gate, up].  For that mode
  // each warp computes one output row and pairs gate row r with up row r+N/2.
  // Plain/gated modes retain the two-rows-per-warp path.
  const int logical_n = MODE == Mode::kSwiGLUOut ? N / 2 : N;
  const int row0 = (blockIdx.x * WARPS_PER_CTA + warp) *
                   (MODE == Mode::kSwiGLUOut ? 1 : ROWS_PER_WARP);
  if (row0 >= logical_n) return;
  const int row1 = MODE == Mode::kSwiGLUOut ? row0 + logical_n : row0 + 1;
  const bool has2 = MODE != Mode::kSwiGLUOut && row1 < N;

  const int kchunks = K >> 5;  // 16B chunks per row (32 elements each)
  const uint4* brow0 = B + (size_t)row0 * kchunks;
  const uint4* brow1 = B + (size_t)row1 * kchunks;
  const uint8_t* srow0 = Bsf + (size_t)row0 * (K >> 4);
  const uint8_t* srow1 = Bsf + (size_t)row1 * (K >> 4);

  float acc0 = 0.f, acc1 = 0.f;
  for (int c = lane; c < kchunks; c += 32) {
    const uint4* xv = xsv + c * XV_STRIDE;
    __half2 xr[16];
    *reinterpret_cast<uint4*>(&xr[0]) = xv[0];
    *reinterpret_cast<uint4*>(&xr[4]) = xv[1];
    *reinterpret_cast<uint4*>(&xr[8]) = xv[2];
    *reinterpret_cast<uint4*>(&xr[12]) = xv[3];
    float s0, s1;
    decode_scales(srow0, c, s0, s1);
    acc0 += chunk_dot(brow0[c], s0, s1, xr);
    decode_scales(srow1, c, s0, s1);
    acc1 += chunk_dot(brow1[c], s0, s1, xr);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
    acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
  }
  if (lane == 0) {
    // alpha is a scalar, or per-row when fused matrices with different
    // global scales are row-concatenated (alpha_per_row selects branchlessly).
    if (MODE == Mode::kSwiGLUOut) {
      const float g = acc0 * alpha[alpha_per_row * row0];
      const float u = acc1 * alpha[alpha_per_row * row1];
      D[(size_t)m * logical_n + row0] =
          __float2bfloat16(g / (1.f + expf(-g)) * u);
    } else {
      D[(size_t)m * N + row0] =
          __float2bfloat16(acc0 * alpha[alpha_per_row * row0]);
      if (has2) {
        D[(size_t)m * N + row0 + 1] =
            __float2bfloat16(acc1 * alpha[alpha_per_row * (row0 + 1)]);
      }
    }
  }
#endif
}

}  // namespace nvfp4_gemv
}  // namespace vllm

namespace {

using vllm::nvfp4_gemv::Mode;

template <Mode MODE>
nvfp4_tensor gemv_run(nvfp4_tensor const& A, nvfp4_tensor const* G,
                      nvfp4_tensor const& B, nvfp4_tensor const& B_sf,
                      nvfp4_tensor const& alpha) {
  STD_TORCH_CHECK(A.is_cuda() && B.is_cuda() && B_sf.is_cuda(),
                  "inputs must be CUDA");
  STD_TORCH_CHECK(A.dim() == 2, "A must be 2-D");
  STD_TORCH_CHECK(A.scalar_type() == ScalarType::BFloat16, "A must be bf16");
  STD_TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && B_sf.is_contiguous(),
                  "inputs must be contiguous");
  STD_TORCH_CHECK(B.scalar_type() == ScalarType::Byte,
                  "B must be uint8 packed");
  STD_TORCH_CHECK(B_sf.scalar_type() == ScalarType::Byte,
                  "B_sf must be uint8 (fp8-e4m3 bits, row-major [N, K/16])");
  STD_TORCH_CHECK(alpha.scalar_type() == ScalarType::Float,
                  "alpha must be fp32");
  STD_TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");

  const int64_t M = A.size(0);
  const int64_t K = B.size(1) * 2;
  const int64_t N = B.size(0);
  STD_TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32, got ", K);
  STD_TORCH_CHECK(A.size(1) == K, "A must have ", K, " columns, got ",
                  A.size(1));
  if (MODE == Mode::kSwiGLUOut) {
    STD_TORCH_CHECK(N % 2 == 0, "concatenated gate/up weight needs even N");
  }
  if (MODE == Mode::kGatedIn) {
    STD_TORCH_CHECK(G != nullptr && G->is_cuda() && G->is_contiguous() &&
                        G->scalar_type() == ScalarType::BFloat16 &&
                        G->dim() == 2 && G->size(0) == M && G->size(1) == K,
                    "gate must be a contiguous bf16 [M, K] CUDA tensor");
  }
  STD_TORCH_CHECK(B_sf.size(0) == N && B_sf.size(1) == K / 16,
                  "B_sf must be [N, K/16] row-major");
  STD_TORCH_CHECK(alpha.numel() == 1 || alpha.numel() == N,
                  "alpha must have 1 or N elements, got ", alpha.numel());
  const int32_t sm = get_sm_version_num();
  STD_TORCH_CHECK(sm >= 100, "nvfp4_gemv requires sm100+, got sm", sm);

  const tsa::DeviceGuard device_guard(A.get_device_index());
  const int64_t n_out = MODE == Mode::kSwiGLUOut ? N / 2 : N;
  auto D = torch::stable::new_empty(A, {M, n_out}, ScalarType::BFloat16);
  const cudaStream_t stream = get_current_cuda_stream();

  auto kfn = vllm::nvfp4_gemv::gemv_kernel<MODE>;
  const int smem = (int)((K / 32) * vllm::nvfp4_gemv::XV_STRIDE * sizeof(uint4));
  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
  }
  // SwiGLU pairs two physical weight rows into one logical output row, so
  // each warp produces one output instead of the two produced by the plain
  // and gated modes. Size the grid by the number of rows each CTA actually
  // writes; using ROWS_PER_CTA here leaves half of the SwiGLU output
  // uninitialized.
  constexpr int rows_per_cta =
      MODE == Mode::kSwiGLUOut ? vllm::nvfp4_gemv::WARPS_PER_CTA
                               : vllm::nvfp4_gemv::ROWS_PER_CTA;
  dim3 grid((unsigned)((n_out + rows_per_cta - 1) / rows_per_cta),
            (unsigned)M);
  kfn<<<grid, vllm::nvfp4_gemv::THREADS, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      G ? reinterpret_cast<const __nv_bfloat16*>(G->data_ptr()) : nullptr,
      reinterpret_cast<const uint4*>(B.data_ptr()),
      reinterpret_cast<const uint8_t*>(B_sf.data_ptr()),
      reinterpret_cast<const float*>(alpha.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(D.data_ptr()), (int)N, (int)K,
      alpha.numel() == N ? 1 : 0);
  return D;
}

}  // namespace

nvfp4_tensor nvfp4_gemv(nvfp4_tensor const& A, nvfp4_tensor const& B,
                        nvfp4_tensor const& B_sf, nvfp4_tensor const& alpha) {
  return gemv_run<Mode::kPlain>(A, nullptr, B, B_sf, alpha);
}

nvfp4_tensor nvfp4_gemv_swiglu(nvfp4_tensor const& A, nvfp4_tensor const& B,
                               nvfp4_tensor const& B_sf,
                               nvfp4_tensor const& alpha) {
  return gemv_run<Mode::kSwiGLUOut>(A, nullptr, B, B_sf, alpha);
}

nvfp4_tensor nvfp4_gemv_gated(nvfp4_tensor const& A, nvfp4_tensor const& G,
                              nvfp4_tensor const& B, nvfp4_tensor const& B_sf,
                              nvfp4_tensor const& alpha) {
  return gemv_run<Mode::kGatedIn>(A, &G, B, B_sf, alpha);
}
