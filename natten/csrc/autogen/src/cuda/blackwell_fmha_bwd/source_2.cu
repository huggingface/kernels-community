#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <natten/natten.h>
#include <natten/helpers.h>
#include <natten/cuda/fmha_blackwell/fmha_backward.cuh>
#include <natten_autogen/cuda/blackwell_fmha_bwd/kernels.h>
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 




void blackwell_fmha_backward_bfloat16_128x128x64(
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
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      bool deterministic,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  auto run = [&](auto kIsDeterministic) {
    constexpr bool IsDeterministic = decltype(kIsDeterministic)::value;

    using GemmShape = cute::tuple<cute::Int<128>, cute::Int<128>, cute::Int<64>>;
    using Kernel = natten::cuda::fmha_blackwell::KernelBackward<
      cutlass::bfloat16_t, GemmShape, cutlass::fmha::collective::NoMask, /* kIsVarlen= */ false, IsDeterministic>;
    using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelBackward<
      cutlass::bfloat16_t, GemmShape, cutlass::fmha::collective::ResidualMaskForBackward, /* kIsVarlen= */ false, IsDeterministic>;
    using KernelWithCausalMask = natten::cuda::fmha_blackwell::KernelBackward<
      cutlass::bfloat16_t, GemmShape, cutlass::fmha::collective::CausalForBackwardMask<true>, /* kIsVarlen= */ false, IsDeterministic>;

    // Varlen kernels
    using VarlenKernelWithResidualMask = natten::cuda::fmha_blackwell::KernelBackward<
      cutlass::bfloat16_t,
      GemmShape,
      cutlass::fmha::collective::ResidualMaskForBackward,
      /* kIsVarlen= */ true, IsDeterministic>;
    using VarlenKernelWithCausalMask = natten::cuda::fmha_blackwell::KernelBackward<
      cutlass::bfloat16_t,
      GemmShape,
      cutlass::fmha::collective::CausalForBackwardMask<true>,
      /* kIsVarlen= */ true, IsDeterministic>;

    int* dq_semaphore_ptr = nullptr;
    at::Tensor dq_semaphore;
    if constexpr (IsDeterministic) {
      auto kBlockM = cute::get<0>(GemmShape{});
      int effective_seqlen_q = is_varlen ? max_seqlen_Q : seqlen_q;
      dq_semaphore = at::zeros(
          {(effective_seqlen_q + kBlockM - 1) / kBlockM, batch_size, heads_q},
          tensor_options.dtype(at::ScalarType::Int));
      dq_semaphore_ptr = static_cast<int*>(dq_semaphore.data_ptr());
    }

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
          heads_q,
          heads_kv,
          dim,
          attn_scale,
          // varlen
          max_seqlen_Q,
          max_seqlen_KV,
          ptr_cumulative_seqlen_Q,
          ptr_cumulative_seqlen_KV,
          dq_semaphore_ptr,
          //
          device_id);

      auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
      auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
      auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
      kernel.run(args, workspace_ptr, stream);
    };

    bool no_mask_required = not is_varlen && not is_causal &&
        seqlen_q % get<0>(GemmShape{}) == 0 && seqlen_k % get<1>(GemmShape{}) == 0;

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
  };

  if (deterministic) {
    run(std::true_type{});
  } else {
    run(std::false_type{});
  }
}


} // namespace fmha_blackwell 
} // namespace cuda 
} // namespace natten 
#endif 
#endif 

