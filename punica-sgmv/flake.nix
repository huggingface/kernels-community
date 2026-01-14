{
  description = "Flake for Punica SGMV kernel";

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
    };
}
