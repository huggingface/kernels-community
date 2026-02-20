{
  description = "Flake for DeepGEMM kernel";

  inputs = {
    # kernel-builder.url = "github:huggingface/kernels";
    kernel-builder.url = "path:/home/drbh/Projects/kernels";
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
