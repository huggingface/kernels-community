{
  description = "Flake for Hopper Flash Attention kernel";

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
      # Building with CUDA later than 12.4 fails with:
      #
      # error: 'ptxas' died due to signal 11 (Invalid memory reference)
      #
      # So, build for 12.4 only and copy to all the other build variants
      # by hand (which works fine thanks to backward compat).
      torchVersions = _: [
        {
          torchVersion = "2.9";
          cudaVersion = "12.4";
          cxx11Abi = true;
          systems = [
            "x86_64-linux"
            "aarch64-linux"
          ];
          bundleBuild = true;
        }
      ];
    };
}
