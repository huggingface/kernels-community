{
  description = "Flake for ReLU kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/kernels-use-kernels-data";
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
