{
  description = "Flake for activation kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/flake-ci-test";
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
