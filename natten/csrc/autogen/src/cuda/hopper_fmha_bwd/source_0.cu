#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_HOPPER_FNA
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <natten/natten.h>
#include <natten/helpers.h>
#include <natten/cuda/fmha_hopper/fmha_backward.cuh>
#include <natten_autogen/cuda/hopper_fmha_bwd/kernels.h>
namespace natten { 
namespace cuda { 
namespace fmha_hopper { 




void hopper_fmha_backward_float16_64x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      void* ptr_dQ,
      void* ptr_dK,
      void* ptr_dV,
      void* ptr_dO,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t, GemmShape, cutlass::fmha::collective::DefaultFusion, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t, GemmShape, cutlass::fmha::collective::ResidualFusion, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t, GemmShape, cutlass::fmha::collective::CausalFusion, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t,
    GemmShape,
    cutlass::fmha::collective::ResidualFusion,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t,
    GemmShape,
    cutlass::fmha::collective::CausalFusion,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
        ptr_dQ,
        ptr_dK,
        ptr_dV,
        ptr_dO,
        batch_size,
        seqlen_q,
        seqlen_k,
        heads,
        dim,
        attn_scale,
        // varlen
        max_seqlen_Q,
        max_seqlen_KV,
        ptr_cumulative_seqlen_Q,
        ptr_cumulative_seqlen_KV,
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_q % get<0>(GemmShape{}) == 0 && seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
    Kernel kernel;
    launch_kernel(kernel);
  }
  else if (is_varlen && is_causal) {
    VarlenKernelWithCausalMask kernel;
    launch_kernel(kernel);
  }
  else if (is_varlen && not is_causal) {
    VarlenKernelWithResidualMask kernel;
    launch_kernel(kernel);
  }
  else if (not is_varlen && is_causal) {
    KernelWithCausalMask kernel;
    launch_kernel(kernel);
  }
  else {
    KernelWithResidualMask kernel;
    launch_kernel(kernel);
  }
}





void hopper_fmha_backward_float16_128x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      void* ptr_dQ,
      void* ptr_dK,
      void* ptr_dV,
      void* ptr_dO,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t, GemmShape, cutlass::fmha::collective::DefaultFusion, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t, GemmShape, cutlass::fmha::collective::ResidualFusion, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t, GemmShape, cutlass::fmha::collective::CausalFusion, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t,
    GemmShape,
    cutlass::fmha::collective::ResidualFusion,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_hopper::KernelBackward<
    cutlass::half_t,
    GemmShape,
    cutlass::fmha::collective::CausalFusion,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
        ptr_dQ,
        ptr_dK,
        ptr_dV,
        ptr_dO,
        batch_size,
        seqlen_q,
        seqlen_k,
        heads,
        dim,
        attn_scale,
        // varlen
        max_seqlen_Q,
        max_seqlen_KV,
        ptr_cumulative_seqlen_Q,
        ptr_cumulative_seqlen_KV,
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_q % get<0>(GemmShape{}) == 0 && seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
    Kernel kernel;
    launch_kernel(kernel);
  }
  else if (is_varlen && is_causal) {
    VarlenKernelWithCausalMask kernel;
    launch_kernel(kernel);
  }
  else if (is_varlen && not is_causal) {
    VarlenKernelWithResidualMask kernel;
    launch_kernel(kernel);
  }
  else if (not is_varlen && is_causal) {
    KernelWithCausalMask kernel;
    launch_kernel(kernel);
  }
  else {
    KernelWithResidualMask kernel;
    launch_kernel(kernel);
  }
}


} // namespace fmha_hopper 
} // namespace cuda 
} // namespace natten 
#endif 
#endif 

