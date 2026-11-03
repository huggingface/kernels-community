{
  description = "Flake for megablocks_moe kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/flatten-before-kernel-flake";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      pythonCheckInputs =
        pkgs: with pkgs; [
          tqdm
          py-cpuinfo
          importlib-metadata
          torchmetrics
        ];

      torchVersions =
        allVersions:
        # For CPU builds, only x86_64-linux is currently supported.
        builtins.filter (version: !(version ? "cpu") || version.system == "x86_64-linux") allVersions;
    };

}
