{
  description = "Flake for megablocks_moe kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;

      pythonCheckInputs =
        pkgs: with pkgs; [
          tqdm
          py-cpuinfo
          importlib-metadata
          torchmetrics
        ];
    };
}
