{
  description = "Flake for attention kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      pythonCheckInputs = pkgs: with pkgs; [ einops ];
    };
}
