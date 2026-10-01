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
#include <natten/cuda/fmha_blackwell/fmha_forward.cuh>
#include <natten_autogen/cuda/blackwell_fmha/kernels.h>
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 




void blackwell_fmha_e5m2_256x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
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
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::NoMask, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::ResidualMask, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::CausalMask<true>, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ false,
    cutlass::fmha::collective::ResidualMask,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ false,
    cutlass::fmha::collective::CausalMask<true>,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
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
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_k % get<1>(GemmShape{}) == 0;
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





void blackwell_fmha_e5m2_256x128x32_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
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
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::NoMask, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::ResidualMask, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::CausalMask<true>, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ true,
    cutlass::fmha::collective::ResidualMask,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ true,
    cutlass::fmha::collective::CausalMask<true>,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
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
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_k % get<1>(GemmShape{}) == 0;
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





void blackwell_fmha_e5m2_256x128x64(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
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
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
  using Kernel = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::NoMask, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::ResidualMask, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::CausalMask<true>, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ false,
    cutlass::fmha::collective::ResidualMask,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ false,
    cutlass::fmha::collective::CausalMask<true>,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
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
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_k % get<1>(GemmShape{}) == 0;
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





void blackwell_fmha_e5m2_256x128x64_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
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
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
  using Kernel = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::NoMask, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::ResidualMask, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::CausalMask<true>, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ true,
    cutlass::fmha::collective::ResidualMask,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ true,
    cutlass::fmha::collective::CausalMask<true>,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
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
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_k % get<1>(GemmShape{}) == 0;
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





void blackwell_fmha_e5m2_256x128x128(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
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
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::NoMask, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::ResidualMask, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, false, cutlass::fmha::collective::CausalMask<true>, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ false,
    cutlass::fmha::collective::ResidualMask,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ false,
    cutlass::fmha::collective::CausalMask<true>,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
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
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_k % get<1>(GemmShape{}) == 0;
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





void blackwell_fmha_e5m2_256x128x128_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
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
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::NoMask, /* kIsVarlen= */ false>;
  using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::ResidualMask, /* kIsVarlen= */ false>;
  using KernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t, GemmShape, true, cutlass::fmha::collective::CausalMask<true>, /* kIsVarlen= */ false>;

  // Varlen kernels
  using VarlenKernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ true,
    cutlass::fmha::collective::ResidualMask,
    /* kIsVarlen= */ true>;
  using VarlenKernelWithCausalMask = natten::cuda::fmha_blackwell::KernelForward<
    cutlass::float_e5m2_t,
    GemmShape,
    /* kIsPersistent= */ true,
    cutlass::fmha::collective::CausalMask<true>,
    /* kIsVarlen= */ true>;

  auto launch_kernel = [&](auto& kernel) {
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
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
        //
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  };

  bool no_mask_required = not is_varlen && not is_causal && seqlen_k % get<1>(GemmShape{}) == 0;
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


} // namespace fmha_blackwell 
} // namespace cuda 
} // namespace natten 
#endif 
#endif 

