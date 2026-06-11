{
  description = "Flake for AITER Flash Attention (AMD ROCm Triton MHA) kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-noarch-pyext";
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
