{
  description = "Flake for attention kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/version-option";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
      pythonCheckInputs = pkgs: with pkgs; [ einops ];
    };
}
