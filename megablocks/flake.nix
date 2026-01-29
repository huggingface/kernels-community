{
  description = "Flake for megablocks_moe kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
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
        let
          # For CPU builds, only x86_64-linux is currently supported.
          cpuSupported = version: system: !(version ? "cpu") || system == "x86_64-linux";
        in
        allVersions:
        builtins.map (
          version: version // { systems = builtins.filter (cpuSupported version) version.systems; }
        ) allVersions;
    };

}
