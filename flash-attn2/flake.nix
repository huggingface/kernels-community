{
  description = "Flake for flash-attn kernel";

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
          einops
        ];

      torchVersions =
        let
          supported =
            # Only x86_64-linux CPU builds are supported currently.
            version: system: !(version ? "cpu") || system == "x86_64-linux";
        in
        allVersions:
        builtins.map (
          version: version // { systems = builtins.filter (supported version) version.systems; }
        ) allVersions;
    };
}
