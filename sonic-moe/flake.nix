{
  description = "Flake for sonic-moe kernels";

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

      # aarch64 get-kernel fails because it cannot load libcuda.so when
      # loading cutlass. We have to look into this, but we can filter out
      # aarch64 in the meanwhile. Since this is a noarch kernel, the
      # resulting build will work on aarch64 as well.
      torchVersions =
        allVersions:
        builtins.map (
          version:
          version // { systems = builtins.filter (system: system == "x86_64-linux") version.systems; }
        ) allVersions;

    };
}
