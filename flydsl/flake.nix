{
  description = "Flake for the FlyDSL SDK (repackaged upstream wheel)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/gen-sdk-outputs";
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

      version = "0.5.0";

      # Torch/ROCm versions that the SDK is built for (and that are used to
      # load-check and test it, Torch is only needed for that). One entry for
      # each ROCm version: the one with the newest Torch.
      #
      # ROCm 7.0 and later.
      # NOTE: ROCm 7.0 up to 7.2 provide libamdhip64.so.7. Newer versions are
      # packaged differently, check that this works for them.
      torchVersions =
        versions:
        let
          candidates = lib.filter (
            version:
            version.system == "x86_64-linux"
            && version ? rocmVersion
            && lib.versionAtLeast version.rocmVersion "7.0"
          ) versions;

          # Prefer newer Torch versions over older, older ROCm versions over newer.
          preferred = lib.sort (
            a: b:
            if a.torchVersion != b.torchVersion then
              builtins.compareVersions a.torchVersion b.torchVersion > 0
            else
              builtins.compareVersions a.rocmVersion b.rocmVersion < 0
          ) candidates;
        in
        map (
          rocmVersion: lib.findFirst (version: version.rocmVersion == rocmVersion) null preferred
        ) (lib.unique (map (version: version.rocmVersion) preferred));

      # The same LLVM/MLIR is used for all variants, it does not depend on ROCm.
      mkShared =
        buildSets:
        let
          pkgs = (builtins.head buildSets).pkgs;
          flydslSrc = pkgs.callPackage ./src.nix { inherit version; };
        in
        {
          inherit flydslSrc;
          llvm = pkgs.callPackage ./llvm.nix { inherit flydslSrc; };
        };

      mkVariant =
        {
          buildSet,
          variant,
          rev,
          buildSets,
        }:
        let
          inherit (buildSet) pkgs;
          inherit (mkShared buildSets) flydslSrc llvm;

          native = pkgs.callPackage ./flydsl.nix { inherit flydslSrc llvm; };
        in
        {
          inherit native llvm;
          kernel = pkgs.callPackage ./sdk.nix {
            inherit
              rev
              variant
              version
              native
              llvm
              ;
          };
        };
    in
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self torchVersions mkVariant;

      name = "flydsl";
      repoId = "kernels-community/flydsl";

      # Upstream ships an abi3 wheel for Python 3.12.
      python = {
        version = "3.12";
        abi = "abi3";
      };

      tests = ./tests;

      extraPackages =
        { variantSets, buildSets, ... }:
        {
          native = lib.mapAttrs (_: variantSet: variantSet.native) variantSets;
          inherit (mkShared buildSets) llvm;
        };
    }
    // {
      formatter = lib.genAttrs systems (
        system: kernel-builder.inputs.nixpkgs.legacyPackages.${system}.nixfmt-tree
      );
    };
}
