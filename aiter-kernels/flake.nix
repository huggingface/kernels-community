{
  description = "Flake for AITER Triton kernels (AMD ROCm)";

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
