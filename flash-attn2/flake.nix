{
  description = "Flake for flash-attn kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/torch-2.10";
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
          einops
        ];

      torchVersions =
        allVersions:
        let
          # For CPU builds, only x86_64-linux is currently supported.
          supported = version: system: !(version ? "cpu") || system == "x86_64-linux";
          filteredSystems = builtins.map (
            version: version // { systems = builtins.filter (supported version) version.systems; }
          ) allVersions;
        in
        # For XPU, require Torch >= 2.9.
        builtins.filter (
          version: !(version ? xpuVersion) || builtins.compareVersions version.torchVersion "2.9" >= 0
        ) filteredSystems;
    };
}
