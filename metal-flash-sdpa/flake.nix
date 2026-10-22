{
  description = "Flake for SDPA kernel";

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
