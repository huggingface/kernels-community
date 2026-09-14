#include "grouped_gemm_xe2.h"
#include "grouped_gemm_xe2_interface.hpp"

// Xe20 image, used by pvc and bmg.
template decltype(MoE::cutlass_grouped_gemm_xe2<20>) MoE::cutlass_grouped_gemm_xe2<20>;
