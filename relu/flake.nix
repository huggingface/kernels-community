{
  description = "Flake for ReLU kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/ops-backend-name";
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
