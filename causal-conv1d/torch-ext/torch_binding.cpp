#if defined(CUDA_KERNEL)
#include <torch/csrc/stable/library.h>
#else
#include <torch/library.h>
#endif

#include "registration.h"

#include "torch_binding.h"

#if defined(CUDA_KERNEL)

// Stable-ABI registration
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

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("causal_conv1d_fwd", TORCH_BOX(&causal_conv1d_fwd));
    ops.impl("causal_conv1d_bwd", TORCH_BOX(&causal_conv1d_bwd));
    ops.impl("causal_conv1d_update", TORCH_BOX(&causal_conv1d_update));
}

#else

#include "pytorch_shim.h"

// Non-stable registration - XPU
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "causal_conv1d_fwd("
        "    Tensor x, Tensor weight, Tensor? bias, Tensor? seq_idx,"
        "    Tensor? initial_states, Tensor! out, Tensor!? final_states_out,"
        "    bool silu_activation) -> ()");
    ops.impl("causal_conv1d_fwd", torch::kXPU, make_pytorch_shim(&causal_conv1d_fwd));

    ops.def(
        "causal_conv1d_bwd("
        "    Tensor x, Tensor weight, Tensor? bias, Tensor! dout,"
        "    Tensor? seq_idx, Tensor? initial_states, Tensor? dfinal_states,"
        "    Tensor! dx, Tensor! dweight, Tensor!? dbias,"
        "    Tensor!? dinitial_states, bool silu_activation) -> ()");
    ops.impl("causal_conv1d_bwd", torch::kXPU, make_pytorch_shim(&causal_conv1d_bwd));

    ops.def(
        "causal_conv1d_update("
        "    Tensor x, Tensor conv_state, Tensor weight, Tensor? bias,"
        "    Tensor! out, bool silu_activation, Tensor? cache_seqlens,"
        "    Tensor? conv_state_indices) -> ()");
    ops.impl("causal_conv1d_update", torch::kXPU, make_pytorch_shim(&causal_conv1d_update));
}

#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
