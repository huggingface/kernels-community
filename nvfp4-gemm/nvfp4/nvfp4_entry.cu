// Dense-only entry points for the vendored vLLM NVFP4 kernels
// (classic torch API; MoE/silu-fusion variants are intentionally omitted).
// Rewritten from vLLM's nvfp4_quant_entry.cu and
// nvfp4_scaled_mm_entry.cu (csrc/libtorch_stable/quantization/fp4/,
// https://github.com/vllm-project/vllm, Apache-2.0; MoE/silu variants omitted).

#include "common.cuh"
#include "nvfp4_utils.cuh"

// Implemented in nvfp4_quant_kernels.cu
void scaled_fp4_quant_sm1xxa(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf,
                             bool is_sf_swizzled_layout);

// Implemented in nvfp4_scaled_mm_kernels.cu / nvfp4_scaled_mm_sm120_kernels.cu
#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
void cutlass_scaled_fp4_mm_sm100a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha);
#endif
#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
void cutlass_scaled_fp4_mm_sm120a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha);
#endif

static bool nvfp4_sm_supported() {
  const int32_t sm = get_sm_version_num();
#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  if (sm >= 100 && sm < 120) return true;
#endif
#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  if (sm >= 120 && sm < 130) return true;
#endif
  return false;
}

std::tuple<torch::Tensor, torch::Tensor> scaled_fp4_quant(
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    bool is_sf_swizzled_layout) {
  TORCH_CHECK(nvfp4_sm_supported(),
              "No compiled nvfp4 quantization kernel for SM ",
              get_sm_version_num());
  const at::cuda::CUDAGuard device_guard(input.device());
  int64_t n = input.size(-1);
  int64_t m = input.numel() / n;
  auto opts = torch::TensorOptions().device(input.device());

  // Two fp4 values packed into a uint8.
  auto output = torch::empty({m, n / 2}, opts.dtype(at::ScalarType::Byte));

  torch::Tensor output_sf;
  if (is_sf_swizzled_layout) {
    auto [sf_m, sf_n] = vllm::computeSwizzledSFShape(m, n);
    output_sf = torch::empty({sf_m, sf_n}, opts.dtype(at::ScalarType::Int));
  } else {
    output_sf = torch::empty({m, n / CVT_FP4_SF_VEC_SIZE},
                             opts.dtype(at::ScalarType::Byte));
  }

  scaled_fp4_quant_sm1xxa(output, input, output_sf, input_global_scale,
                          is_sf_swizzled_layout);
  return {output, output_sf};
}

torch::Tensor cutlass_scaled_fp4_mm(torch::Tensor const& A,
                                    torch::Tensor const& B,
                                    torch::Tensor const& A_sf,
                                    torch::Tensor const& B_sf,
                                    torch::Tensor const& alpha) {
  const at::cuda::CUDAGuard device_guard(A.device());
  const int32_t sm = get_sm_version_num();
  auto D = torch::empty(
      {A.size(0), B.size(0)},
      torch::TensorOptions().dtype(at::ScalarType::BFloat16).device(A.device()));

#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  if (sm >= 100 && sm < 120) {
    cutlass_scaled_fp4_mm_sm100a(D, A, B, A_sf, B_sf, alpha);
    return D;
  }
#endif
#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  if (sm >= 120 && sm < 130) {
    cutlass_scaled_fp4_mm_sm120a(D, A, B, A_sf, B_sf, alpha);
    return D;
  }
#endif
  TORCH_CHECK(false, "No compiled nvfp4 GEMM for SM ", sm);
}
