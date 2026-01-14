{
  description = "Flake for deformable_detr kernels";

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
