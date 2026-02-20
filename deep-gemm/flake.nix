{
  description = "Flake for DeepGEMM kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/support-bundling-deps";
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
