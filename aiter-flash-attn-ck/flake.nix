{
  description = "Flake for AITER Flash Attention (AMD ROCm Composable Kernel FMHA) kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/rocm-archs-fix";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
