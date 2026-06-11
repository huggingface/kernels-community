{
  description = "Flake for Torch kernel extension";
  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };
  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      # This is a workaround, we should be able to specify flags per arch in
      # kernel-builder.
      torchVersions =
        allVersions:
        builtins.map (
          version:
          version // { systems = builtins.filter (system: system == "x86_64-linux") version.systems; }
        ) allVersions;
    };
}
