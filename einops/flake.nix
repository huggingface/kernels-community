{
  description = "Flake for the einops SDK";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/sdk-mk-tests";
  };

  outputs =
    { self, kernel-builder }:
    let
      inherit (kernel-builder.inputs.nixpkgs) lib;

      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];

      version = "0.8.2";

      einopsSrc = pkgs: pkgs.callPackage ./src.nix { inherit version; };

      mkVariant =
        {
          buildSet,
          variant,
          rev,
          buildSets,
        }:
        let
          inherit (buildSet) pkgs;
          sdkId = "_einops_${lib.replaceStrings [ "-" "." ] [ "_" "_" ] variant}_${rev}";
        in
        pkgs.callPackage ./sdk.nix {
          inherit sdkId variant version;
          einopsSrc = einopsSrc pkgs;
          # Backend-agnostic SDK. The metadata requires a backend.
          backend = "cpu";
        };
    in
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self mkVariant;

      name = "einops";
      repoId = "kernels-sdk/einops";

      python = null;
      useBackend = false;

      mkTests =
        { buildSet, ... }:
        buildSet.pkgs.callPackage ./tests.nix { einopsSrc = einopsSrc buildSet.pkgs; };
    }
    // {
      formatter = lib.genAttrs systems (
        system: kernel-builder.inputs.nixpkgs.legacyPackages.${system}.nixfmt-tree
      );
    };
}
