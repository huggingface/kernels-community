{
  description = "Flake for trimul_gpumode kernels";

  inputs = {
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
