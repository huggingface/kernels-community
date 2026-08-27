#include <torch/library.h>

#include "registration.h"

#include "pytorch_shim.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "causal_conv1d_fwd("
        "    Tensor x, Tensor weight, Tensor? bias, Tensor? seq_idx,"
        "    Tensor? initial_states, Tensor! out, Tensor!? final_states_out,"
        "    bool silu_activation) -> ()");
    ops.impl("causal_conv1d_fwd", torch::kCUDA, make_pytorch_shim(&causal_conv1d_fwd));
    
    ops.def(
        "causal_conv1d_bwd("
        "    Tensor x, Tensor weight, Tensor? bias, Tensor! dout,"
        "    Tensor? seq_idx, Tensor? initial_states, Tensor? dfinal_states,"
        "    Tensor! dx, Tensor! dweight, Tensor!? dbias,"
        "    Tensor!? dinitial_states, bool silu_activation) -> ()");
    ops.impl("causal_conv1d_bwd", torch::kCUDA, make_pytorch_shim(&causal_conv1d_bwd));
    
    ops.def(
        "causal_conv1d_update("
        "    Tensor x, Tensor conv_state, Tensor weight, Tensor? bias,"
        "    Tensor! out, bool silu_activation, Tensor? cache_seqlens,"
        "    Tensor? conv_state_indices) -> ()");
    ops.impl("causal_conv1d_update", torch::kCUDA, make_pytorch_shim(&causal_conv1d_update));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
