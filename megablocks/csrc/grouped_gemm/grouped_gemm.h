#pragma once

// // Set default if not already defined
// #ifndef GROUPED_GEMM_CUTLASS
// #define GROUPED_GEMM_CUTLASS 0
// #endif

#include "../stable_utils.h"

namespace grouped_gemm {

void GroupedGemm(torch::stable::Tensor a,
		 torch::stable::Tensor b,
		 torch::stable::Tensor c,
		 torch::stable::Tensor batch_sizes,
		 bool trans_a, bool trans_b);

}  // namespace grouped_gemm
