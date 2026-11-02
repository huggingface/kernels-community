{
  description = "Flake for Torch kernel extension";
  inputs = {
    kernel-builder.url = "github:huggingface/kernels/flatten-before-kernel-flake";
  };
  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      # This is a workaround, we should be able to specify flags per arch in
      # kernel-builder.
      torchVersions =
        allVersions: builtins.filter (version: version.system == "x86_64-linux") allVersions;
    };
}
