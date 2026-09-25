#include <torch/csrc/stable/library.h>

#include "registration.h"

#include "torch_binding.h"

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "causal_conv1d_fwd("
        "    Tensor x, Tensor weight, Tensor? bias, Tensor? seq_idx,"
        "    Tensor? initial_states, Tensor! out, Tensor!? final_states_out,"
        "    bool silu_activation) -> ()");

    ops.def(
        "causal_conv1d_bwd("
        "    Tensor x, Tensor weight, Tensor? bias, Tensor! dout,"
        "    Tensor? seq_idx, Tensor? initial_states, Tensor? dfinal_states,"
        "    Tensor! dx, Tensor! dweight, Tensor!? dbias,"
        "    Tensor!? dinitial_states, bool silu_activation) -> ()");

    ops.def(
        "causal_conv1d_update("
        "    Tensor x, Tensor conv_state, Tensor weight, Tensor? bias,"
        "    Tensor! out, bool silu_activation, Tensor? cache_seqlens,"
        "    Tensor? conv_state_indices) -> ()");
}

#if defined(CUDA_KERNEL)
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("causal_conv1d_fwd", TORCH_BOX(&causal_conv1d_fwd));
    ops.impl("causal_conv1d_bwd", TORCH_BOX(&causal_conv1d_bwd));
    ops.impl("causal_conv1d_update", TORCH_BOX(&causal_conv1d_update));
}
#elif defined(XPU_KERNEL)
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, XPU, ops) {
    ops.impl("causal_conv1d_fwd", TORCH_BOX(&causal_conv1d_fwd));
    ops.impl("causal_conv1d_bwd", TORCH_BOX(&causal_conv1d_bwd));
    ops.impl("causal_conv1d_update", TORCH_BOX(&causal_conv1d_update));
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
