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

constexpr int ROWS_PER_CTA = 8;   // one warp per output row
constexpr int THREADS = ROWS_PER_CTA * 32;

__device__ __forceinline__ float chunk_dot(const uint4* __restrict__ brow,
                                           const uint8_t* __restrict__ srow,
                                           const __half2* __restrict__ xs2,
                                           int c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  const uint4 w4 = brow[c];  // 32 codes, coalesced 512B/warp
  const unsigned short sboth =
      *reinterpret_cast<const unsigned short*>(srow + 2 * c);
  const float s0 = __half2float(
      __half(__nv_cvt_fp8_to_halfraw((uint8_t)(sboth & 0xffu), __NV_E4M3)));
  const float s1 = __half2float(
      __half(__nv_cvt_fp8_to_halfraw((uint8_t)(sboth >> 8), __NV_E4M3)));
  const __half2* x2 = xs2 + c * 17;  // padded stride (see fill loop)

  // Split accumulators per scale block (measured NEUTRAL vs a single 8-deep
  // chain -- warp occupancy already hides FMA latency; kept for clarity).
  __half2 h0a = __float2half2_rn(0.f), h0b = __float2half2_rn(0.f);
  uint32_t wa = w4.x, wb = w4.y;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2_raw pa =
        __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(wa & 0xffu), __NV_E2M1);
    const __half2_raw pb =
        __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(wb & 0xffu), __NV_E2M1);
    h0a = __hfma2(__half2(pa), x2[i], h0a);
    h0b = __hfma2(__half2(pb), x2[i + 4], h0b);
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
    h1a = __hfma2(__half2(pa), x2[i + 8], h1a);
    h1b = __hfma2(__half2(pb), x2[i + 12], h1b);
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

__global__ void __launch_bounds__(THREADS) gemv_kernel(
    const __nv_bfloat16* __restrict__ A,  // [M, K] bf16 activations
    const uint4* __restrict__ B,          // [N, K/32] 16B chunks (K/2 bytes/row)
    const uint8_t* __restrict__ Bsf,      // [N, K/16] fp8-e4m3 bits, row-major
    const float* __restrict__ alpha,      // [1] scalar or [N] per-row = 1/global_scale
    __nv_bfloat16* __restrict__ D,        // [M, N]
    int N, int K, int alpha_per_row) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  // x cached as half2 with a 17-slot stride per 16-half2 chunk: bank-conflict-free
  // (lane l reads chunk c=l; stride 16 would put all lanes on 2 banks = 16-way conflict).
  extern __shared__ __half2 xs2[];
  const int m = blockIdx.y;
  const __nv_bfloat16* a_row = A + (size_t)m * K;
  for (int p = threadIdx.x; p < (K >> 1); p += blockDim.x) {
    const int c = p >> 4, i = p & 15;
    xs2[c * 17 + i] = __floats2half2_rn(__bfloat162float(a_row[2 * p]),
                                        __bfloat162float(a_row[2 * p + 1]));
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * ROWS_PER_CTA + warp;
  if (row >= N) return;

  const int kchunks = K >> 5;  // 16B chunks per row (32 elements each)
  const uint4* brow = B + (size_t)row * kchunks;
  const uint8_t* srow = Bsf + (size_t)row * (K >> 4);

  float acc0 = 0.f, acc1 = 0.f;
  int c = lane;
  for (; c + 32 < kchunks; c += 64) {
    acc0 += chunk_dot(brow, srow, xs2, c);
    acc1 += chunk_dot(brow, srow, xs2, c + 32);
  }
  if (c < kchunks) acc0 += chunk_dot(brow, srow, xs2, c);
  float acc = acc0 + acc1;

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) {
    // alpha is a scalar, or per-row when fused matrices with different
    // global scales are row-concatenated (alpha_per_row selects branchlessly).
    D[(size_t)m * N + row] = __float2bfloat16(acc * alpha[alpha_per_row * row]);
  }
#endif
}

}  // namespace nvfp4_gemv
}  // namespace vllm

torch::Tensor nvfp4_gemv(torch::Tensor const& A, torch::Tensor const& B,
                         torch::Tensor const& B_sf,
                         torch::Tensor const& alpha) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && B_sf.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(A.dim() == 2, "A must be [M, K]");
  TORCH_CHECK(A.scalar_type() == at::ScalarType::BFloat16, "A must be bf16");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && B_sf.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(B.scalar_type() == at::ScalarType::Byte, "B must be uint8 packed");
  TORCH_CHECK(B_sf.scalar_type() == at::ScalarType::Byte,
              "B_sf must be uint8 (fp8-e4m3 bits, row-major [N, K/16])");
  TORCH_CHECK(alpha.scalar_type() == at::ScalarType::Float, "alpha must be fp32");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");

  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(0);
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32, got ", K);
  TORCH_CHECK(B.size(1) == K / 2, "B must be [N, K/2]");
  TORCH_CHECK(B_sf.size(0) == N && B_sf.size(1) == K / 16,
              "B_sf must be [N, K/16] row-major");
  TORCH_CHECK(alpha.numel() == 1 || alpha.numel() == N,
              "alpha must have 1 or N elements, got ", alpha.numel());
  const int32_t sm = get_sm_version_num();
  TORCH_CHECK(sm >= 100, "nvfp4_gemv requires sm100+, got sm", sm);

  const at::cuda::CUDAGuard device_guard(A.device());
  auto D = torch::empty({M, N}, torch::TensorOptions()
                                    .dtype(at::ScalarType::BFloat16)
                                    .device(A.device()));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int smem = (int)((K / 32) * 17 * sizeof(__half2));  // padded x2 layout
  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(vllm::nvfp4_gemv::gemv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  dim3 grid((unsigned)((N + vllm::nvfp4_gemv::ROWS_PER_CTA - 1) /
                       vllm::nvfp4_gemv::ROWS_PER_CTA),
            (unsigned)M);
  vllm::nvfp4_gemv::gemv_kernel<<<grid, vllm::nvfp4_gemv::THREADS, smem,
                                  stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      reinterpret_cast<const uint4*>(B.data_ptr()),
      reinterpret_cast<const uint8_t*>(B_sf.data_ptr()),
      reinterpret_cast<const float*>(alpha.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(D.data_ptr()), (int)N, (int)K,
      alpha.numel() == N ? 1 : 0);
  return D;
}
