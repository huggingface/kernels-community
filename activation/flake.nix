{
  description = "Flake for activation kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/backend-scoped-stable-abi";
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
