{
  description = "Flake for tinygrad-style RMSNorm kernel";

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
