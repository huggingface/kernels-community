{
  description = "Flake for DeepGEMM kernel";

  inputs = {
    self.submodules = true;
    kernel-builder.url = "github:huggingface/kernels/nix-pass-gpu-lang";
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
