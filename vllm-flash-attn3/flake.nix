{
  description = "Flake for Hopper Flash Attention kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/framework-cxx-flags";
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
