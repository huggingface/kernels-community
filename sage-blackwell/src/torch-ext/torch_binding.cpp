#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("fwd(Tensor q, Tensor k, Tensor v, Tensor sfq, Tensor sfk, Tensor sfv, Tensor delta_s, int unpadded_k, Tensor? out_, float softmax_scale, bool is_causal, bool per_block_mean, bool is_bf16) -> Tensor[]");
    ops.impl("fwd", torch::kCUDA, &fwd_wrap);

    ops.def("scaled_fp4_quant(Tensor input, Tensor! output, Tensor! output_sf, int tensor_layout) -> ()");
    ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant_wrap);

    ops.def("scaled_fp4_quant_permute(Tensor input, Tensor! output, Tensor! output_sf, int tensor_layout) -> ()");
    ops.impl("scaled_fp4_quant_permute", torch::kCUDA, &scaled_fp4_quant_permute_wrap);

    ops.def("scaled_fp4_quant_trans(Tensor input, Tensor! output, Tensor! output_sf, int tensor_layout) -> ()");
    ops.impl("scaled_fp4_quant_trans", torch::kCUDA, &scaled_fp4_quant_trans_wrap);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);
