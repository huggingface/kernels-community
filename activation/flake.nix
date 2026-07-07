{
  description = "Flake for activation kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/backend-scoped-stable-abi";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

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
