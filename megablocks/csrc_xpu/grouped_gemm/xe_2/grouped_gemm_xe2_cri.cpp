#include "grouped_gemm_xe2.h"
#include "grouped_gemm_xe2_interface.hpp"

// Xe35 image. Built with -D__SYCL_TARGET_INTEL_GPU_CRI__=1 and
// -fsycl-targets=spir64_gen so sycl-tla emits the CRI block-scaled DPAS path.
template decltype(MoE::cutlass_grouped_gemm_xe2<35>) MoE::cutlass_grouped_gemm_xe2<35>;
