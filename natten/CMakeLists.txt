cmake_minimum_required(VERSION 3.26)

# Set Intel SYCL compiler before project() call
find_program(ICX_COMPILER icx)
find_program(ICPX_COMPILER icpx)

if(ICX_COMPILER OR ICPX_COMPILER)
  set(CMAKE_C_COMPILER ${ICX_COMPILER})

 if(WIN32)
    set(CMAKE_CXX_COMPILER ${ICX_COMPILER})
  else()
    set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
  endif()
endif()

project( LANGUAGES CXX)

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201")

# Make Torch CMake machinery happy. Whatever we use does not matter, since
# we set the arches per-file later anyway.
set(ENV{PYTORCH_ROCM_ARCH} "${HIP_SUPPORTED_ARCHS}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/kernel.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/get_gpu_lang.cmake)

if(DEFINED Python3_EXECUTABLE)
  # Allow passing through the interpreter (e.g. from setup.py).
  find_package(Python3 COMPONENTS Development Development.SABIModule Interpreter)
  if (NOT Python3_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
else()
  find_package(Python3 REQUIRED COMPONENTS Development Development.SABIModule Interpreter)
endif()

get_gpu_lang(DETECTED_GPU_LANG)
set(GPU_LANG "${DETECTED_GPU_LANG}" CACHE STRING "GPU language")
gpu_lang_to_backend(BACKEND "${GPU_LANG}")
message(STATUS "Using backend: ${BACKEND}, GPU language: ${GPU_LANG}")

set(OPS_NAME "_natten_${BACKEND}_28fa1dd")

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(Torch REQUIRED)

run_python(TORCH_VERSION "import torch; print(torch.__version__.split('+')[0])" "Failed to get Torch version")




option(BUILD_ALL_SUPPORTED_ARCHS "Build all supported architectures" off)

if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
 set(CUDA_DEFAULT_KERNEL_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1+PTX")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
 set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX")
else()
  set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX")
endif()

# Basic checks for each GPU language.
if(GPU_LANG STREQUAL "CUDA")
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "GPU language is set to CUDA, but cannot find CUDA toolkit")
  endif()

  
  
  # This clears out -gencode arguments from `CMAKE_CUDA_FLAGS`, which we need
  # to set our own set of capabilities.
  clear_gencode_flags()

  # Get the capabilities without +PTX suffixes, so that we can use them as
  # the target archs in the loose intersection with a kernel's capabilities.
  cuda_remove_ptx_suffixes(CUDA_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  message(STATUS "CUDA base archs used for intersection with kernel archs: ${CUDA_ARCHS}")

  if(BUILD_ALL_SUPPORTED_ARCHS)
    set(CUDA_KERNEL_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  else()
    try_run_python(CUDA_KERNEL_ARCHS SUCCESS "import torch; cc=torch.cuda.get_device_capability(); print(f\"{cc[0]}.{cc[1]}\")" "Failed to get CUDA capability")
    if(NOT SUCCESS)
      message(WARNING "Failed to detect CUDA capability, using default capabilities.")
      set(CUDA_KERNEL_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
    endif()
  endif()

  message(STATUS "Default CUDA kernel architectures: ${CUDA_KERNEL_ARCHS}")

  if(NVCC_THREADS AND GPU_LANG STREQUAL "CUDA")
    message(STATUS "Using nvcc with: -threads=${NVCC_THREADS}")
    list(APPEND GPU_FLAGS "--threads=${NVCC_THREADS}")
  endif()

  # TODO: deprecate one of these settings.
  add_compile_definitions(USE_CUDA=1)
  add_compile_definitions(CUDA_KERNEL)
elseif(GPU_LANG STREQUAL "HIP")
  if(NOT HIP_FOUND AND NOT PYTORCH_FOUND_HIP)
    message(FATAL_ERROR "GPU language is set to HIP, but cannot find ROCm toolkit")
  endif()

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  set(ROCM_ARCHS ${HIP_SUPPORTED_ARCHS})
  message(STATUS "ROCM supported target architectures: ${ROCM_ARCHS}")

  # TODO: deprecate one of these settings.
  add_compile_definitions(USE_ROCM=1)
  add_compile_definitions(ROCM_KERNEL)
elseif(GPU_LANG STREQUAL "CPU")
  add_compile_definitions(CPU_KERNEL)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS deployment version")
elseif(GPU_LANG STREQUAL "METAL")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS deployment version")
  enable_language(C OBJC OBJCXX)

  add_compile_definitions(METAL_KERNEL)

  # Initialize lists for Metal shader sources and their include directories
  set(ALL_METAL_SOURCES)
  set(METAL_INCLUDE_DIRS)
elseif(GPU_LANG STREQUAL "SYCL")
  if(NOT ICX_COMPILER AND NOT ICPX_COMPILER)
    message(FATAL_ERROR "Intel SYCL C++ compiler (icpx) and/or C compiler (icx) not found. Please install Intel oneAPI toolkit.")
  endif()

  execute_process(
    COMMAND ${ICPX_COMPILER} --version
    OUTPUT_VARIABLE ICPX_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "[0-9]+\\.[0-9]+" DPCPP_VERSION "${ICPX_VERSION_OUTPUT}")
  set(DPCPP_VERSION "${DPCPP_VERSION}" CACHE STRING "DPCPP major.minor version")

  # On Windows, use icx (MSVC-compatible) for C++ to work with Ninja generator
  # On Linux, use icpx (GNU-compatible) for C++
  if(WIN32)
    message(STATUS "Using Intel SYCL C++ compiler: ${ICX_COMPILER} and C compiler: ${ICX_COMPILER} Version: ${DPCPP_VERSION} (Windows MSVC-compatible mode)")
  else()
    message(STATUS "Using Intel SYCL C++ compiler: ${ICPX_COMPILER} and C compiler: ${ICX_COMPILER} Version: ${DPCPP_VERSION}")
  endif()


  set(sycl_link_flags "-Wl,-z,noexecstack;-fsycl;--offload-compress;-fsycl-targets=spir64_gen,spir64;-Xs;-device pvc,xe-lpg,ats-m150 -options ' -cl-intel-enable-auto-large-GRF-mode -cl-poison-unsupported-fp64-kernels -cl-intel-greater-than-4GB-buffer-required';")
  set(sycl_flags "-fPIC;-fsycl;-fhonor-nans;-fhonor-infinities;-fno-associative-math;-fno-approx-func;-fno-sycl-instrument-device-code;--offload-compress;-fsycl-targets=spir64_gen,spir64;")
  set(GPU_FLAGS "${sycl_flags}")


  add_compile_definitions(XPU_KERNEL)
  add_compile_definitions(USE_XPU)
else()
  message(FATAL_ERROR "Unsupported GPU language: ${GPU_LANG}")
endif()

# Initialize SRC list for kernel and binding sources
set(SRC "")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/build-variants.cmake)

# Generate build variant name.
if(GPU_LANG STREQUAL "CUDA")
  generate_build_name(BUILD_VARIANT_NAME "${TORCH_VERSION}" "cuda" "${CUDA_VERSION}" ${_STABLE_ABI_ARG})
elseif(GPU_LANG STREQUAL "HIP")
  run_python(ROCM_VERSION "import torch.version; print(torch.version.hip.split('.')[0] + '.' + torch.version.hip.split('.')[1])" "Failed to get ROCm version")
  generate_build_name(BUILD_VARIANT_NAME "${TORCH_VERSION}" "rocm" "${ROCM_VERSION}" ${_STABLE_ABI_ARG})
elseif(GPU_LANG STREQUAL "SYCL")
  generate_build_name(BUILD_VARIANT_NAME "${TORCH_VERSION}" "xpu" "${DPCPP_VERSION}" ${_STABLE_ABI_ARG})
elseif(GPU_LANG STREQUAL "METAL")
  generate_build_name(BUILD_VARIANT_NAME "${TORCH_VERSION}" "metal" "" ${_STABLE_ABI_ARG})
elseif(GPU_LANG STREQUAL "CPU")
  generate_build_name(BUILD_VARIANT_NAME "${TORCH_VERSION}" "cpu" "" ${_STABLE_ABI_ARG})
else()
  message(FATAL_ERROR "Cannot generate build name for unknown GPU_LANG: ${GPU_LANG}")
endif()

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/_ops.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/torch-ext/natten/_ops.py
  @ONLY
)

if(GPU_LANG STREQUAL "CUDA")
  message(STATUS "Including CUTLASS dependency")

find_package(NvidiaCutlass)

if (NOT NvidiaCutlass_FOUND)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

# Set CUTLASS_REVISION manually -- its revision detection doesn't work in this case.
  set(CUTLASS_REVISION "v4.5.2" CACHE STRING "CUTLASS revision to use")


# Use the specified CUTLASS source directory for compilation if CUTLASS_SRC_DIR is provided
  if (DEFINED ENV{CUTLASS_SRC_DIR})
    set(CUTLASS_SRC_DIR $ENV{CUTLASS_SRC_DIR})
  endif()

  if(CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE CUTLASS_SRC_DIR)
      get_filename_component(CUTLASS_SRC_DIR "${CUTLASS_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The CUTLASS_SRC_DIR is set, using ${CUTLASS_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${CUTLASS_SRC_DIR})
  else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/nvidia/cutlass.git
        GIT_TAG ${CUTLASS_REVISION}
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW TRUE
    )
  endif()
  FetchContent_MakeAvailable(cutlass)

  include_directories(${CUTLASS_INCLUDE_DIR})
else()
  message(STATUS "Using system cutlass with version: ${NvidiaCutlass_VERSION}")
endif(NOT NvidiaCutlass_FOUND)

endif(GPU_LANG STREQUAL "CUDA")
if(GPU_LANG STREQUAL "CUDA")
  get_torch_gpu_compiler_flags(TORCH_GPU_FLAGS ${GPU_LANG})
  list(APPEND GPU_FLAGS ${TORCH_GPU_FLAGS})
endif()

set(TORCH_natten_SRC
  torch-ext/torch_binding.cpp
)

# TODO: check if CLion support this:
# https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
set_source_files_properties(
  ${TORCH_natten_SRC}
  PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/csrc/include")


list(APPEND SRC "${TORCH_natten_SRC}")
if(GPU_LANG STREQUAL "CUDA")
cuda_kernel_component(SRC
  NAME natten_blackwell
  SOURCES "csrc/autogen/include/natten_autogen/cuda/blackwell_fna/dispatch_cm.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna/dispatch_dtype.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna/dispatch_tile_shape.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna/interface.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna/kernels.h"
"csrc/autogen/src/cuda/blackwell_fna/source_0.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_1.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_10.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_11.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_12.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_13.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_14.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_15.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_16.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_17.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_18.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_19.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_2.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_20.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_21.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_22.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_23.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_24.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_25.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_26.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_27.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_3.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_4.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_5.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_6.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_7.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_8.cu"
"csrc/autogen/src/cuda/blackwell_fna/source_9.cu"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna_bwd/dispatch_cm.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna_bwd/dispatch_dtype.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna_bwd/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna_bwd/dispatch_tile_shape.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna_bwd/interface.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fna_bwd/kernels.h"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_0.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_1.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_10.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_11.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_12.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_13.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_2.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_3.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_4.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_5.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_6.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_7.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_8.cu"
"csrc/autogen/src/cuda/blackwell_fna_bwd/source_9.cu"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha/dispatch_tile_size.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha/interface.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha/kernels.h"
"csrc/autogen/src/cuda/blackwell_fmha/source_0.cu"
"csrc/autogen/src/cuda/blackwell_fmha/source_1.cu"
"csrc/autogen/src/cuda/blackwell_fmha/source_2.cu"
"csrc/autogen/src/cuda/blackwell_fmha/source_3.cu"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha_bwd/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha_bwd/dispatch_tile_size.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha_bwd/interface.h"
"csrc/autogen/include/natten_autogen/cuda/blackwell_fmha_bwd/kernels.h"
"csrc/autogen/src/cuda/blackwell_fmha_bwd/source_0.cu"
"csrc/autogen/src/cuda/blackwell_fmha_bwd/source_1.cu"
"csrc/autogen/src/cuda/blackwell_fmha_bwd/source_2.cu"
"csrc/autogen/src/cuda/blackwell_fmha_bwd/source_3.cu"
  CUDA_MINVER 12.8  INCLUDES "${CMAKE_SOURCE_DIR}/csrc/include;${CMAKE_SOURCE_DIR}/csrc/autogen/include"  CUDA_CAPABILITIES 10.0a  CUDA_FLAGS "-O3;-std=c++17;--expt-relaxed-constexpr;--extended-lambda;--use_fast_math;--ftemplate-backtrace-limit=0;-Xfatbin=-compress-all;-DNDEBUG;-DNATTEN_WITH_CUTLASS;-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1;-DNATTEN_WITH_BLACKWELL_FNA=1"  )
endif()
if(GPU_LANG STREQUAL "CUDA")
cuda_kernel_component(SRC
  NAME natten
  SOURCES "csrc/include/natten/blackwell_fmha.h"
"csrc/include/natten/blackwell_fna.h"
"csrc/include/natten/compute_delta.h"
"csrc/include/natten/cuda/fmha/epilogue/epilogue_pipelined.h"
"csrc/include/natten/cuda/fmha/epilogue/epilogue_rescale_output.h"
"csrc/include/natten/cuda/fmha/epilogue/epilogue_thread_apply_logsumexp.h"
"csrc/include/natten/cuda/fmha/fmha_backward.cuh"
"csrc/include/natten/cuda/fmha/fmha_forward.cuh"
"csrc/include/natten/cuda/fmha/gemm/custom_mma.h"
"csrc/include/natten/cuda/fmha/gemm/custom_mma_base.h"
"csrc/include/natten/cuda/fmha/gemm/custom_mma_multistage.h"
"csrc/include/natten/cuda/fmha/gemm/custom_mma_pipelined.h"
"csrc/include/natten/cuda/fmha/gemm/find_default_mma.h"
"csrc/include/natten/cuda/fmha/gemm/mma_accum_lambda_iterator.h"
"csrc/include/natten/cuda/fmha/gemm/mma_from_smem.h"
"csrc/include/natten/cuda/fmha/gemm_kernel_utils.h"
"csrc/include/natten/cuda/fmha/iterators/default_warp_iterator_from_smem.h"
"csrc/include/natten/cuda/fmha/iterators/epilogue_predicated_tile_iterator.h"
"csrc/include/natten/cuda/fmha/iterators/make_residual_last.h"
"csrc/include/natten/cuda/fmha/iterators/predicated_tile_access_iterator_residual_last.h"
"csrc/include/natten/cuda/fmha/iterators/predicated_tile_iterator_residual_last.h"
"csrc/include/natten/cuda/fmha/iterators/transpose_warp_iterator.h"
"csrc/include/natten/cuda/fmha/iterators/warp_iterator_from_smem.h"
"csrc/include/natten/cuda/fmha/kernel_backward.h"
"csrc/include/natten/cuda/fmha/kernel_forward.h"
"csrc/include/natten/cuda/fmha/transform/tile_smem_loader.h"
"csrc/include/natten/cuda/fmha_blackwell/collective/fmha_common.hpp"
"csrc/include/natten/cuda/fmha_blackwell/collective/fmha_fusion.hpp"
"csrc/include/natten/cuda/fmha_blackwell/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_blackwell/collective/sm100_fmha_load_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_blackwell/common/pow_2.hpp"
"csrc/include/natten/cuda/fmha_blackwell/device/fmha_bwd_sm100.hpp"
"csrc/include/natten/cuda/fmha_blackwell/device/fmha_sm100.hpp"
"csrc/include/natten/cuda/fmha_blackwell/fmha_backward.cuh"
"csrc/include/natten/cuda/fmha_blackwell/fmha_forward.cuh"
"csrc/include/natten/cuda/fmha_blackwell/kernel/fmha_causal_tile_scheduler.hpp"
"csrc/include/natten/cuda/fmha_blackwell/kernel/fmha_kernel_bwd_convert.hpp"
"csrc/include/natten/cuda/fmha_blackwell/kernel/fmha_kernel_bwd_sum_OdO.hpp"
"csrc/include/natten/cuda/fmha_blackwell/kernel/fmha_tile_scheduler.hpp"
"csrc/include/natten/cuda/fmha_blackwell/kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_blackwell/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_collective_bwd_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_collective_load.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_collective_softmax.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_collective_tma.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_collective_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_common.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_epilogue.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_epilogue_bwd.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_fusion.hpp"
"csrc/include/natten/cuda/fmha_hopper/collective/fmha_varlen.hpp"
"csrc/include/natten/cuda/fmha_hopper/device/fmha_bwd_sm90.hpp"
"csrc/include/natten/cuda/fmha_hopper/device/fmha_sm90.hpp"
"csrc/include/natten/cuda/fmha_hopper/fmha_backward.cuh"
"csrc/include/natten/cuda/fmha_hopper/fmha_forward.cuh"
"csrc/include/natten/cuda/fmha_hopper/kernel/fmha_kernel_builder.hpp"
"csrc/include/natten/cuda/fmha_hopper/kernel/fmha_kernel_bwd_convert.hpp"
"csrc/include/natten/cuda/fmha_hopper/kernel/fmha_kernel_bwd_sum_OdO.hpp"
"csrc/include/natten/cuda/fmha_hopper/kernel/fmha_kernel_tma.hpp"
"csrc/include/natten/cuda/fmha_hopper/kernel/fmha_kernel_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fmha_hopper/kernel/fmha_options.hpp"
"csrc/include/natten/cuda/fmha_hopper/kernel/fmha_tile_scheduler.hpp"
"csrc/include/natten/cuda/fna/epilogue/epilogue_pipelined.h"
"csrc/include/natten/cuda/fna/epilogue/epilogue_rescale_output.h"
"csrc/include/natten/cuda/fna/epilogue/epilogue_thread_apply_logsumexp.h"
"csrc/include/natten/cuda/fna/epilogue/predicated_tile_iterator.h"
"csrc/include/natten/cuda/fna/epilogue/predicated_tile_iterator_params.h"
"csrc/include/natten/cuda/fna/fna_backward.cuh"
"csrc/include/natten/cuda/fna/fna_forward.cuh"
"csrc/include/natten/cuda/fna/gemm/custom_mma.h"
"csrc/include/natten/cuda/fna/gemm/custom_mma_base.h"
"csrc/include/natten/cuda/fna/gemm/custom_mma_multistage.h"
"csrc/include/natten/cuda/fna/gemm/custom_mma_pipelined.h"
"csrc/include/natten/cuda/fna/gemm/find_default_mma.h"
"csrc/include/natten/cuda/fna/gemm/mma_accum_lambda_iterator.h"
"csrc/include/natten/cuda/fna/gemm/mma_from_smem.h"
"csrc/include/natten/cuda/fna/gemm/replace_mma_iterators.h"
"csrc/include/natten/cuda/fna/gemm_kernel_utils.h"
"csrc/include/natten/cuda/fna/iterators/default_warp_iterator_from_smem.h"
"csrc/include/natten/cuda/fna/iterators/epilogue_predicated_tile_iterator.h"
"csrc/include/natten/cuda/fna/iterators/make_residual_last.h"
"csrc/include/natten/cuda/fna/iterators/predicated_tile_access_iterator.h"
"csrc/include/natten/cuda/fna/iterators/predicated_tile_access_iterator_residual_last.h"
"csrc/include/natten/cuda/fna/iterators/predicated_tile_iterator.h"
"csrc/include/natten/cuda/fna/iterators/predicated_tile_iterator_residual_last.h"
"csrc/include/natten/cuda/fna/iterators/transpose_warp_iterator.h"
"csrc/include/natten/cuda/fna/iterators/warp_iterator_from_smem.h"
"csrc/include/natten/cuda/fna/kernel_backward.h"
"csrc/include/natten/cuda/fna/kernel_forward.h"
"csrc/include/natten/cuda/fna/na_utils.cuh"
"csrc/include/natten/cuda/fna_blackwell/collective/fna_common.hpp"
"csrc/include/natten/cuda/fna_blackwell/collective/fna_fusion.hpp"
"csrc/include/natten/cuda/fna_blackwell/collective/fna_fusion_bwd.hpp"
"csrc/include/natten/cuda/fna_blackwell/collective/sm100_fna_fwd_epilogue_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fna_blackwell/collective/sm100_fna_fwd_mainloop_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fna_blackwell/collective/sm100_fna_load_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fna_blackwell/common/pow_2.hpp"
"csrc/include/natten/cuda/fna_blackwell/device/fna_bwd_sm100.hpp"
"csrc/include/natten/cuda/fna_blackwell/device/fna_sm100.hpp"
"csrc/include/natten/cuda/fna_blackwell/fna_backward.cuh"
"csrc/include/natten/cuda/fna_blackwell/fna_forward.cuh"
"csrc/include/natten/cuda/fna_blackwell/kernel/sm100_fna_bwd_kernel_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fna_blackwell/kernel/sm100_fna_fwd_kernel_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_collective_bwd_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_collective_load.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_collective_softmax.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_collective_tma.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_collective_tma_warpspecialized.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_common.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_fusion.hpp"
"csrc/include/natten/cuda/fna_hopper/collective/fna_fusion_bwd.hpp"
"csrc/include/natten/cuda/fna_hopper/device/fna_bwd_sm90.hpp"
"csrc/include/natten/cuda/fna_hopper/device/fna_sm90.hpp"
"csrc/include/natten/cuda/fna_hopper/fna_backward.cuh"
"csrc/include/natten/cuda/fna_hopper/fna_forward.cuh"
"csrc/include/natten/cuda/fna_hopper/kernel/fna_kernel_builder.hpp"
"csrc/include/natten/cuda/hopper_fmha_fna.h"
"csrc/include/natten/cuda/reduction/compute_delta.cuh"
"csrc/include/natten/cuda/reduction/fmha_kernel_bwd_sum_OdO.hpp"
"csrc/include/natten/cuda/reference/fna_reference_backward.hpp"
"csrc/include/natten/cuda/reference/fna_reference_forward.hpp"
"csrc/include/natten/cuda/reference/mask.hpp"
"csrc/include/natten/cuda/reference/utils.hpp"
"csrc/include/natten/cuda/tokperm/layouts.hpp"
"csrc/include/natten/cuda/tokperm/token_permute_kernel.cuh"
"csrc/include/natten/cuda/tokperm/tokperm.hpp"
"csrc/include/natten/cuda/tokperm/utils/permute.cuh"
"csrc/include/natten/cuda/tokperm/utils/stride.cuh"
"csrc/include/natten/cuda/tokperm/utils/tuple.cuh"
"csrc/include/natten/cuda/utils/cuda.h"
"csrc/include/natten/cuda/utils/cutlass.cuh"
"csrc/include/natten/cuda/utils/generic_cutlass_device.hpp"
"csrc/include/natten/fmha.h"
"csrc/include/natten/fna.h"
"csrc/include/natten/helpers.h"
"csrc/include/natten/hopper_fmha.h"
"csrc/include/natten/hopper_fna.h"
"csrc/include/natten/natten.h"
"csrc/include/natten/reference.h"
"csrc/include/natten/token_permute.h"
"csrc/src/blackwell_fmha.cu"
"csrc/src/blackwell_fna_backward.cu"
"csrc/src/blackwell_fna_forward.cu"
"csrc/src/compute_delta.cu"
"csrc/src/fmha.cu"
"csrc/src/fna_backward.cu"
"csrc/src/fna_forward.cu"
"csrc/src/hopper_fmha.cu"
"csrc/src/hopper_fna_backward.cu"
"csrc/src/hopper_fna_forward.cu"
"csrc/src/reference_backward.cu"
"csrc/src/reference_forward.cu"
"csrc/src/token_permute.cu"
"csrc/src/token_unpermute.cu"
"csrc/autogen/include/natten_autogen/cuda/reference/dispatch_cm.h"
"csrc/autogen/include/natten_autogen/cuda/reference/dispatch_dtype.h"
"csrc/autogen/include/natten_autogen/cuda/reference/interface.h"
"csrc/autogen/include/natten_autogen/cuda/reference/kernels.h"
"csrc/autogen/src/cuda/reference/source_0.cu"
"csrc/autogen/src/cuda/reference/source_1.cu"
"csrc/autogen/include/natten_autogen/cuda/fna/dispatch_cm.h"
"csrc/autogen/include/natten_autogen/cuda/fna/dispatch_device.h"
"csrc/autogen/include/natten_autogen/cuda/fna/dispatch_dtype.h"
"csrc/autogen/include/natten_autogen/cuda/fna/interface.h"
"csrc/autogen/include/natten_autogen/cuda/fna/kernels.h"
"csrc/autogen/src/cuda/fna/source_0.cu"
"csrc/autogen/src/cuda/fna/source_1.cu"
"csrc/autogen/src/cuda/fna/source_10.cu"
"csrc/autogen/src/cuda/fna/source_11.cu"
"csrc/autogen/src/cuda/fna/source_12.cu"
"csrc/autogen/src/cuda/fna/source_13.cu"
"csrc/autogen/src/cuda/fna/source_14.cu"
"csrc/autogen/src/cuda/fna/source_15.cu"
"csrc/autogen/src/cuda/fna/source_16.cu"
"csrc/autogen/src/cuda/fna/source_17.cu"
"csrc/autogen/src/cuda/fna/source_18.cu"
"csrc/autogen/src/cuda/fna/source_19.cu"
"csrc/autogen/src/cuda/fna/source_2.cu"
"csrc/autogen/src/cuda/fna/source_20.cu"
"csrc/autogen/src/cuda/fna/source_21.cu"
"csrc/autogen/src/cuda/fna/source_22.cu"
"csrc/autogen/src/cuda/fna/source_23.cu"
"csrc/autogen/src/cuda/fna/source_24.cu"
"csrc/autogen/src/cuda/fna/source_25.cu"
"csrc/autogen/src/cuda/fna/source_26.cu"
"csrc/autogen/src/cuda/fna/source_27.cu"
"csrc/autogen/src/cuda/fna/source_28.cu"
"csrc/autogen/src/cuda/fna/source_29.cu"
"csrc/autogen/src/cuda/fna/source_3.cu"
"csrc/autogen/src/cuda/fna/source_30.cu"
"csrc/autogen/src/cuda/fna/source_31.cu"
"csrc/autogen/src/cuda/fna/source_32.cu"
"csrc/autogen/src/cuda/fna/source_33.cu"
"csrc/autogen/src/cuda/fna/source_34.cu"
"csrc/autogen/src/cuda/fna/source_35.cu"
"csrc/autogen/src/cuda/fna/source_36.cu"
"csrc/autogen/src/cuda/fna/source_37.cu"
"csrc/autogen/src/cuda/fna/source_38.cu"
"csrc/autogen/src/cuda/fna/source_39.cu"
"csrc/autogen/src/cuda/fna/source_4.cu"
"csrc/autogen/src/cuda/fna/source_40.cu"
"csrc/autogen/src/cuda/fna/source_41.cu"
"csrc/autogen/src/cuda/fna/source_42.cu"
"csrc/autogen/src/cuda/fna/source_43.cu"
"csrc/autogen/src/cuda/fna/source_44.cu"
"csrc/autogen/src/cuda/fna/source_45.cu"
"csrc/autogen/src/cuda/fna/source_46.cu"
"csrc/autogen/src/cuda/fna/source_47.cu"
"csrc/autogen/src/cuda/fna/source_48.cu"
"csrc/autogen/src/cuda/fna/source_49.cu"
"csrc/autogen/src/cuda/fna/source_5.cu"
"csrc/autogen/src/cuda/fna/source_50.cu"
"csrc/autogen/src/cuda/fna/source_51.cu"
"csrc/autogen/src/cuda/fna/source_52.cu"
"csrc/autogen/src/cuda/fna/source_53.cu"
"csrc/autogen/src/cuda/fna/source_54.cu"
"csrc/autogen/src/cuda/fna/source_55.cu"
"csrc/autogen/src/cuda/fna/source_56.cu"
"csrc/autogen/src/cuda/fna/source_57.cu"
"csrc/autogen/src/cuda/fna/source_58.cu"
"csrc/autogen/src/cuda/fna/source_59.cu"
"csrc/autogen/src/cuda/fna/source_6.cu"
"csrc/autogen/src/cuda/fna/source_60.cu"
"csrc/autogen/src/cuda/fna/source_61.cu"
"csrc/autogen/src/cuda/fna/source_62.cu"
"csrc/autogen/src/cuda/fna/source_63.cu"
"csrc/autogen/src/cuda/fna/source_7.cu"
"csrc/autogen/src/cuda/fna/source_8.cu"
"csrc/autogen/src/cuda/fna/source_9.cu"
"csrc/autogen/include/natten_autogen/cuda/fmha/dispatch_dtype.h"
"csrc/autogen/include/natten_autogen/cuda/fmha/interface.h"
"csrc/autogen/include/natten_autogen/cuda/fmha/kernels.h"
"csrc/autogen/src/cuda/fmha/source_0.cu"
"csrc/autogen/src/cuda/fmha/source_1.cu"
"csrc/autogen/src/cuda/fmha/source_2.cu"
"csrc/autogen/src/cuda/fmha/source_3.cu"
"csrc/autogen/src/cuda/fmha/source_4.cu"
"csrc/autogen/src/cuda/fmha/source_5.cu"
    INCLUDES "${CMAKE_SOURCE_DIR}/csrc/include;${CMAKE_SOURCE_DIR}/csrc/autogen/include"  CUDA_CAPABILITIES 7.0 7.5 8.0 8.6 8.9 9.0 10.0 12.0  CUDA_FLAGS "-O3;-std=c++17;--expt-relaxed-constexpr;--extended-lambda;--use_fast_math;--ftemplate-backtrace-limit=0;-Xfatbin=-compress-all;-DNDEBUG;-DNATTEN_WITH_CUTLASS;-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"  )
endif()
if(GPU_LANG STREQUAL "CUDA")
cuda_kernel_component(SRC
  NAME natten_hopper
  SOURCES "csrc/autogen/include/natten_autogen/cuda/hopper_fna/dispatch_cm.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna/dispatch_dtype.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna/dispatch_tile_shape.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna/interface.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna/kernels.h"
"csrc/autogen/src/cuda/hopper_fna/source_0.cu"
"csrc/autogen/src/cuda/hopper_fna/source_1.cu"
"csrc/autogen/src/cuda/hopper_fna/source_2.cu"
"csrc/autogen/src/cuda/hopper_fna/source_3.cu"
"csrc/autogen/src/cuda/hopper_fna/source_4.cu"
"csrc/autogen/src/cuda/hopper_fna/source_5.cu"
"csrc/autogen/src/cuda/hopper_fna/source_6.cu"
"csrc/autogen/src/cuda/hopper_fna/source_7.cu"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna_bwd/dispatch_cm.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna_bwd/dispatch_dtype.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna_bwd/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna_bwd/dispatch_tile_shape.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna_bwd/interface.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fna_bwd/kernels.h"
"csrc/autogen/src/cuda/hopper_fna_bwd/source_0.cu"
"csrc/autogen/src/cuda/hopper_fna_bwd/source_1.cu"
"csrc/autogen/src/cuda/hopper_fna_bwd/source_2.cu"
"csrc/autogen/src/cuda/hopper_fna_bwd/source_3.cu"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha/dispatch_tile_size.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha/interface.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha/kernels.h"
"csrc/autogen/src/cuda/hopper_fmha/source_0.cu"
"csrc/autogen/src/cuda/hopper_fmha/source_1.cu"
"csrc/autogen/src/cuda/hopper_fmha/source_2.cu"
"csrc/autogen/src/cuda/hopper_fmha/source_3.cu"
"csrc/autogen/src/cuda/hopper_fmha/source_4.cu"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha_bwd/dispatch_head_dim.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha_bwd/dispatch_tile_size.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha_bwd/interface.h"
"csrc/autogen/include/natten_autogen/cuda/hopper_fmha_bwd/kernels.h"
"csrc/autogen/src/cuda/hopper_fmha_bwd/source_0.cu"
"csrc/autogen/src/cuda/hopper_fmha_bwd/source_1.cu"
"csrc/autogen/src/cuda/hopper_fmha_bwd/source_2.cu"
"csrc/autogen/src/cuda/hopper_fmha_bwd/source_3.cu"
"csrc/autogen/src/cuda/hopper_fmha_bwd/source_4.cu"
    INCLUDES "${CMAKE_SOURCE_DIR}/csrc/include;${CMAKE_SOURCE_DIR}/csrc/autogen/include"  CUDA_CAPABILITIES 9.0a  CUDA_FLAGS "-O3;-std=c++17;--expt-relaxed-constexpr;--extended-lambda;--use_fast_math;--ftemplate-backtrace-limit=0;-Xfatbin=-compress-all;-DNDEBUG;-DNATTEN_WITH_CUTLASS;-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1;-DNATTEN_WITH_HOPPER_FNA=1"  )
endif()
# Include Metal shader compilation utilities if needed
if(GPU_LANG STREQUAL "METAL")
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/compile-metal.cmake)
endif()

# Define the extension target with unified parameters
define_gpu_extension_target(
  ${OPS_NAME}
  ${OPS_NAME}
  DESTINATION ${OPS_NAME}
  LANGUAGE ${GPU_LANG}
  SOURCES ${SRC}
  COMPILE_FLAGS ${GPU_FLAGS}
  USE_SABI 3
  WITH_SOABI)

if(GPU_LANG STREQUAL "SYCL")
    target_link_options(${OPS_NAME} PRIVATE ${sycl_link_flags})
    target_link_libraries(${OPS_NAME} PRIVATE dnnl)
endif()

# Compile Metal shaders if any were found
if(GPU_LANG STREQUAL "METAL")
    if(ALL_METAL_SOURCES)
        compile_metal_shaders(${OPS_NAME} "${ALL_METAL_SOURCES}" "${METAL_INCLUDE_DIRS}")
    endif()
endif()


# Add kernels_install target for huggingface/kernels library layout
add_kernels_install_target(${OPS_NAME} "natten" "${BUILD_VARIANT_NAME}"
    DATA_EXTENSIONS ""
    GPU_ARCHS "${ALL_GPU_ARCHS}")

# Add local_install target for local development with get_local_kernel()
add_local_install_target(${OPS_NAME} "natten" "${BUILD_VARIANT_NAME}"
    DATA_EXTENSIONS ""
    GPU_ARCHS "${ALL_GPU_ARCHS}")
