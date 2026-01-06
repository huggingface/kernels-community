{
  description = "Flake for Torch kernel extension";
  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/torch-2.10";
  };
  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
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
