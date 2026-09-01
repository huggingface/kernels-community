#include "grouped_gemm_interface.h"
#include <stdio.h>
#include "../xpu_features.hpp"
#include "xe_2/grouped_gemm_xe2.h"

using megablocks::xpu::XPUFeatures;

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4,
    bool is_B_mxfp8) {
  // The grouped GEMMs are built twice: an Xe20 image for pvc/bmg and an Xe35
  // image for CRI. Both cover the same dtypes, so pick the variant by device.
  auto gemm = XPUFeatures::isXe35(ptr_A.device().index())
      ? MoE::cutlass_grouped_gemm_xe2<35>
      : MoE::cutlass_grouped_gemm_xe2<20>;
  return gemm(
      ptr_A,
      ptr_B,
      ptr_scales,
      ptr_bias,
      ptr_D,
      expert_first_token_offset,
      N,
      K,
      num_experts,
      is_B_int4,
      is_B_mxfp4,
      is_B_mxfp8);
}
