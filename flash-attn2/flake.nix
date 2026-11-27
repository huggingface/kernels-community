{
  description = "Flake for flash-attn kernel";

  inputs = {
    kernel-builder.url = "github:jiqing-feng/kernels/xpu-cri-grf-size";
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
          einops
        ];

      torchVersions =
        allVersions:
        builtins.filter (
          version:
          # For CPU builds, only x86_64-linux is currently supported.
          (!(version ? "cpu") || version.system == "x86_64-linux")
          # For XPU, require Torch >= 2.9.
          && (!(version ? xpuVersion) || builtins.compareVersions version.torchVersion "2.9" >= 0)
        ) allVersions;
    };
}
