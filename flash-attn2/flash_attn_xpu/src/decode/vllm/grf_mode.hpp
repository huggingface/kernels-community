// vllm-xpu-kernels csrc/xpu/grf_mode.hpp.
#pragma once

// Compiler version guard for maximum_registers vs grf_size.
// maximum_registers is available starting from DPC++ 2026.2.
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20260200
  #include <sycl/ext/intel/experimental/maximum_registers_properties.hpp>
  #define VLLM_GRF_PROP(N) sycl::ext::intel::experimental::maximum_registers<N>
#else
  #include <sycl/ext/intel/experimental/grf_size_properties.hpp>
  #define VLLM_GRF_PROP(N) sycl::ext::intel::experimental::grf_size<N>
#endif
