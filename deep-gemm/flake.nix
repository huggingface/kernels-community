{
  description = "Flake for DeepGEMM kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/b079fd8c66612177cc8edd13292613abb4de994c";
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
