{
  description = "Flake for AITER Flash Attention (AMD ROCm Triton MHA) kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.12";
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
