{
  description = "Flake for Torch kernel extension";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/metal_kernels";
  };

  outputs = { self, kernel-builder, }:
    kernel-builder.lib.genFlakeOutputs {
        inherit self;
        path = ./.;
    };
}
