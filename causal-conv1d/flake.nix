{
  description = "Flake for attention kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/build-non-stable-abi-versions";
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
