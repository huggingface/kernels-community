#include <torch/all.h>

namespace MoE {

// Instantiated once per target architecture: 20 for pvc/bmg, 35 for CRI. The
// two instantiations live in separate translation units with different flags.
template <int Arch>
torch::Tensor cutlass_grouped_gemm_xe2(
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
    bool is_B_mxfp8);

}  // namespace MoE
