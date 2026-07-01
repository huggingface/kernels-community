{
  description = "Flake for Hopper Flash Attention kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/framework-cxx-flags";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      # Use ptxas from CUDA 12.8 for CUDA 12.6, since ptxas 12.6 crashes when building FA3.
      torchVersions =
        allVersions:
        builtins.map (
          version:
          if (version.cudaVersion or null) == "12.6" then version // { ptxasVersion = "12.8"; } else version
        ) allVersions;
    };
}
