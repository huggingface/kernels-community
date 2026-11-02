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

      torchVersions =
        allVersions:
        # For CPU builds, only x86_64-linux is currently supported.
        builtins.filter (version: !(version ? "cpu") || version.system == "x86_64-linux") allVersions;
    };
}
