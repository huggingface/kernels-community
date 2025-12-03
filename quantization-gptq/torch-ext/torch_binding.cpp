#include <torch/all.h>
#include "registration.h"


#if defined(CPU_KERNEL)
torch::Tensor gemm_int4_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &absmax,
    int64_t blocksize,
    int64_t quant_type);
#endif
// Unified dispatcher for both CPU and XPU
torch::Tensor gemm_int4_forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &absmax,
    int64_t blocksize,
    int64_t quant_type) {
#if defined(CPU_KERNEL)
    if (input.device().type() == torch::kCPU) {
        TORCH_CHECK(input.device().type() == torch::kCPU, "input must be on CPU");
        TORCH_CHECK(weight.device().type() == torch::kCPU, "weight must be on CPU");
        TORCH_CHECK(absmax.device().type() == torch::kCPU, "absmax must be on CPU");
        TORCH_CHECK(blocksize > 0, "blocksize must be > 0");
        return gemm_int4_cpu_forward(input, weight, absmax, blocksize, quant_type);
    }
#endif
    else {
        TORCH_CHECK(false, "Unsupported device type: ", input.device().type());
    }
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemm_int4_forward(Tensor input, Tensor weight, Tensor absmax, int blocksize, int quant_type) -> Tensor");
#if defined(CPU_KERNEL)
  // Register CPU implementation
  ops.impl("gemm_int4_forward", torch::kCPU, &gemm_int4_forward);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
