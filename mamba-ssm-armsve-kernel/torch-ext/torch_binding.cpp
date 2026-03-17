#include <torch/library.h>
#include "registration.h"
#include "torch_binding.h"


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("mamba_selective_scan(Tensor A, Tensor B, Tensor C, Tensor hidden_states, Tensor discrete_time_step, Tensor ssm_state, Tensor scan_output, int B_size, int D_size, int L_size, int N_size) -> ()");
    ops.impl("mamba_selective_scan", torch::kCPU, mamba_selective_scan);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)