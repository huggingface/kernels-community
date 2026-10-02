#include "../utils.h"

torch::stable::Tensor cutlass_grouped_gemm_interface(
    torch::stable::Tensor ptr_A,
    torch::stable::Tensor ptr_B,
    const std::optional<torch::stable::Tensor>& ptr_scales,
    const std::optional<torch::stable::Tensor>& ptr_bias,
    torch::stable::Tensor ptr_D,
    torch::stable::Tensor expert_first_token_offset,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4,
    bool is_B_mxfp8);