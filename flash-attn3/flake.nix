{
  description = "Flake for Hopper Flash Attention kernel";

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
        ps: with ps; [
          einops
        ];

      # Per system: CUDA version -> ptxas version to use instead.
      #
      # - ptxas 13.0 segfaults on aarch64-linux compiling the sm90 hdimall
      #   bf16 instantiations.
      # - ptxas 13.2 fails on x86_64-linux.
      torchVersions =
        allVersions:
        let
          ptxasExceptions = {
            x86_64-linux = {
              "12.6" = "12.8";
              "13.2" = "13.4";
            };
            aarch64-linux = {
              "12.6" = "12.8";
              "13.0" = "13.4";
              "13.2" = "13.4";
            };
          };
        in
        builtins.map (
          version:
          let
            exceptions = ptxasExceptions.${version.system} or { };
            cudaVersion = version.cudaVersion or null;
          in
          if cudaVersion != null && exceptions ? ${cudaVersion} then
            version // { ptxasVersion = exceptions.${cudaVersion}; }
          else
            version
        ) allVersions;

    };
}
