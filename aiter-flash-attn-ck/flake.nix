{
  description = "Flake for AITER Flash Attention (AMD ROCm Composable Kernel FMHA) kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.15.0-rc5";
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
