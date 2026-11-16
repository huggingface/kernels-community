{
  description = "Flake for the UCCL SDK";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/gen-sdk-outputs";
  };

  outputs =
    { self, kernel-builder }:
    let
      inherit (kernel-builder.inputs.nixpkgs) lib;

      version = "0.1.1";

      # The ROCm architectures that upstream builds for in its releases.
      rocmArchs = [
        "gfx942"
        "gfx950"
      ];

      # CUDA capabilities, similar to kernels, but only >= 8.0. The ep component
      # is only built for 9.0+PTX (so that it can build optimized device code).
      cudaArchsFor =
        cudaVersion:
        if lib.versionAtLeast cudaVersion "13.0" then
          [
            "8.0"
            "8.6"
            "8.7"
            "8.9"
            "9.0"
            "10.0"
            "10.3"
            "11.0"
            "12.0"
            "12.1+PTX"
          ]
        else if lib.versionAtLeast cudaVersion "12.9" then
          [
            "8.0"
            "8.6"
            "8.7"
            "8.9"
            "9.0"
            "10.0"
            "10.1"
            "10.3"
            "12.0"
            "12.1+PTX"
          ]
        else if lib.versionAtLeast cudaVersion "12.8" then
          [
            "8.0"
            "8.6"
            "8.7"
            "8.9"
            "9.0"
            "10.0"
            "10.1"
            "12.0+PTX"
          ]
        else
          [
            "8.0"
            "8.6"
            "8.7"
            "8.9"
            "9.0+PTX"
          ];

      # Build one variant for each supported backend version.
      torchVersions =
        versions:
        let
          backendVersion = version: version.cudaVersion or version.rocmVersion;

          supported = lib.filter (
            version: version.system == "x86_64-linux" && (version ? cudaVersion || version ? rocmVersion)
          ) versions;
        in
        map (backend: lib.findFirst (version: backendVersion version == backend) null supported) (
          lib.unique (map backendVersion supported)
        );

      mkVariant =
        {
          buildSet,
          variant,
          rev,
          ...
        }:
        let
          inherit (buildSet)
            pkgs
            python3
            buildConfig
            torch
            ;
          inherit (buildConfig) backend;

          cuda = backend == "cuda";

          src = pkgs.fetchFromGitHub {
            owner = "uccl-project";
            repo = "uccl";
            rev = "v${version}";
            hash = "sha256-ujSaz6NQEZWQjAT2GC1OdosmIm5D5zkVjR4TQDoz4u8=";
          };
        in
        pkgs.callPackage ./uccl-sdk.nix {
          inherit
            python3
            rev
            variant
            version
            src
            backend
            ;
          archs = if cuda then cudaArchsFor buildConfig.cudaVersion else rocmArchs;

          ep = pkgs.callPackage ./uccl-ep.nix {
            inherit
              backend
              version
              src
              torch
              ;
            offloadArchs = rocmArchs;
          };

          p2p = pkgs.callPackage ./uccl-p2p.nix { inherit backend version src; };

          ncclPlugin = pkgs.callPackage ./uccl-nccl-plugin.nix { inherit backend version src; };

        };
    in
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self torchVersions mkVariant;

      name = "uccl";
      repoId = "kernels-community/uccl";

      # The extensions use the stable ABI of Python 3.12 and later (abi3).
      python = {
        version = "3.12";
        abi = "abi3";
      };

      tests = ./tests;
    };
}
