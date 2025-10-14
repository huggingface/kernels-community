#pragma once

// // Set default if not already defined
// #ifndef GROUPED_GEMM_CUTLASS
// #define GROUPED_GEMM_CUTLASS 0
// #endif

// #include <torch/extension.h>
#include <torch/torch.h>

namespace grouped_gemm {

void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b);

}  // namespace grouped_gemm

