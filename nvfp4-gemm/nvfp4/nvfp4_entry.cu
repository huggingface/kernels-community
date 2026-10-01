// Dense-only entry points for the vendored vLLM NVFP4 kernels
// (stable-ABI torch API; MoE/silu-fusion variants are intentionally omitted).
// Rewritten from vLLM's nvfp4_quant_entry.cu and
// nvfp4_scaled_mm_entry.cu (csrc/libtorch_stable/quantization/fp4/,
// https://github.com/vllm-project/vllm, Apache-2.0; MoE/silu variants omitted).

#include <tuple>

#include "common.cuh"
#include "nvfp4_utils.cuh"

// Implemented in nvfp4_quant_kernels.cu
void scaled_fp4_quant_sm1xxa(nvfp4_tensor const& output,
                             nvfp4_tensor const& input,
                             nvfp4_tensor const& output_sf,
                             nvfp4_tensor const& input_sf,
                             bool is_sf_swizzled_layout);

// Implemented in nvfp4_scaled_mm_kernels.cu / nvfp4_scaled_mm_sm120_kernels.cu
#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
void cutlass_scaled_fp4_mm_sm100a(nvfp4_tensor& D, nvfp4_tensor const& A,
                                  nvfp4_tensor const& B,
                                  nvfp4_tensor const& A_sf,
                                  nvfp4_tensor const& B_sf,
                                  nvfp4_tensor const& alpha);
#endif
#if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
void cutlass_scaled_fp4_mm_sm120a(nvfp4_tensor& D, nvfp4_tensor const& A,
                                  nvfp4_tensor const& B,
                                  nvfp4_tensor const& A_sf,
                                  nvfp4_tensor const& B_sf,
                                  nvfp4_tensor const& alpha);
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

std::tuple<nvfp4_tensor, nvfp4_tensor> scaled_fp4_quant(
    nvfp4_tensor const& input, nvfp4_tensor const& input_global_scale,
    bool is_sf_swizzled_layout) {
  STD_TORCH_CHECK(nvfp4_sm_supported(),
                  "No compiled nvfp4 quantization kernel for SM ",
                  get_sm_version_num());
  const tsa::DeviceGuard device_guard(input.get_device_index());
  int64_t n = input.size(input.dim() - 1);
  int64_t m = input.numel() / n;

  // Two fp4 values packed into a uint8.
  auto output = torch::stable::new_empty(input, {m, n / 2}, ScalarType::Byte);

  nvfp4_tensor output_sf;
  if (is_sf_swizzled_layout) {
    auto [sf_m, sf_n] = vllm::computeSwizzledSFShape(m, n);
    output_sf = torch::stable::new_empty(input, {sf_m, sf_n}, ScalarType::Int);
  } else {
    output_sf = torch::stable::new_empty(
        input, {m, n / CVT_FP4_SF_VEC_SIZE}, ScalarType::Byte);
  }

  scaled_fp4_quant_sm1xxa(output, input, output_sf, input_global_scale,
                          is_sf_swizzled_layout);
  return {output, output_sf};
}

nvfp4_tensor cutlass_scaled_fp4_mm(nvfp4_tensor const& A,
                                   nvfp4_tensor const& B,
                                   nvfp4_tensor const& A_sf,
                                   nvfp4_tensor const& B_sf,
                                   nvfp4_tensor const& alpha) {
  const tsa::DeviceGuard device_guard(A.get_device_index());
  const int32_t sm = get_sm_version_num();
  auto D = torch::stable::new_empty(A, {A.size(0), B.size(0)},
                                    ScalarType::BFloat16);

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
  STD_TORCH_CHECK(false, "No compiled nvfp4 GEMM for SM ", sm);
}
