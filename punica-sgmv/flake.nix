{
  description = "Flake for Punica SGMV kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/torch-2.10";
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
