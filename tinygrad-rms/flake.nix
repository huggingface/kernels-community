{
  description = "Flake for tinygrad-style RMSNorm kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.15.0-final";
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
