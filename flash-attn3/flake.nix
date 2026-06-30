{
  description = "Flake for Hopper Flash Attention kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/make-ptxas-overridable";
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
        ps: with ps; [
          einops
        ];

      torchVersions =
        allVersions:
        builtins.map (
          version:
          if (version.cudaVersion or null) == "12.6" then version // { ptxasVersion = "12.8"; } else version
        ) allVersions;

    };
}
