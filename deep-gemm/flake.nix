{
  description = "Flake for DeepGEMM kernel";

  inputs = {
    self.submodules = true;
    kernel-builder.url = "github:huggingface/kernels/cxx11-abi-tag";
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
