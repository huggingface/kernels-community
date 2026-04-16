#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "sdpa_int4("
        "Tensor queries, "
        "Tensor k_quant, "
        "Tensor k_scales, "
        "Tensor k_biases, "
        "Tensor v_quant, "
        "Tensor v_scales, "
        "Tensor v_biases, "
        "int gqa_factor, "
        "int N, "
        "float scale, "
        "int sliding_window"
        ") -> Tensor");
#if defined(METAL_KERNEL)
    ops.impl("sdpa_int4", torch::kMPS, &sdpa_int4);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
