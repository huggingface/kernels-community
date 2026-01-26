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

      torchVersions = builtins.filter (
        version: !(version ? xpuVersion) || builtins.compareVersions version.torchVersion "2.9" >= 0
      );
    };

}
