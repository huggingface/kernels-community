{
  description = "Flake for the UCCL SDK (repackaged upstream wheels)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/gen-sdk-outputs";
  };

  outputs =
    { self, kernel-builder }:
    let
      inherit (kernel-builder.inputs.nixpkgs) lib;

      version = "0.1.1";
      releaseUrl = "https://github.com/uccl-project/uccl/releases/download/v${version}";

      # The upstream wheel for each backend (and CUDA major version). The CUDA 12
      # wheel is built with CUDA 12.8, but all CUDA runtime symbols that it uses
      # are in the CUDA 12.6 runtime.
      wheelFor = {
        cu12 = {
          name = "uccl-${version}-cp312-abi3-manylinux_2_35_x86_64.whl";
          hash = "sha256-OQwTIJGJciBlRuRNebEymI8oGOwH4jr80FlfcYORbOw=";
        };
        cu13 = {
          name = "uccl-${version}+cu13-cp312-abi3-manylinux_2_35_x86_64.whl";
          hash = "sha256-wlHJ0sLM25Omm+DwQzW5JLSnLcjApqyEusQznzQMzHw=";
        };
        rocm = {
          name = "uccl-${version}+rocm-cp312-abi3-manylinux_2_35_x86_64.whl";
          hash = "sha256-+f+pTq35ZeWsQbEUBSRysjfSWJ56QG7uw8l+Nu5dqw8=";
        };
      };

      wheelKeyOf =
        buildConfig:
        if buildConfig ? cudaVersion then "cu${lib.versions.major buildConfig.cudaVersion}" else "rocm";

      # The EP component requires 9.0 or later (9.0+PTX), but the other components
      # seem compatible with any capability, so let's add 8.x capabilities as well.
      cudaArchs = [
        "8.0"
        "8.6"
        "8.7"
        "8.9"
        "9.0+PTX"
      ];

      # Backend versions that the SDK is built for. The variants are named after
      # the backend version, so we build one variant for each wheel, with the
      # oldest backend version that the wheel supports:
      #
      # - CUDA 12 wheel: CUDA 12.6 (compatible with all later CUDA 12 versions).
      # - CUDA 13 wheel: CUDA 13.0.
      # - ROCm wheel: ROCm 7.1 (the version that the wheel is built with).
      #
      # Torch is only needed to load-check and test the SDK, so any Torch version
      # that has the backend version is fine (the first one is used).
      torchVersions =
        versions:
        let
          backendVersion = version: version.cudaVersion or version.rocmVersion;

          candidates = lib.filter (
            version:
            version.system == "x86_64-linux"
            && (
              (version ? cudaVersion && lib.versionAtLeast version.cudaVersion "12.6")
              # Newer ROCm versions are packaged differently.
              || (
                version ? rocmVersion
                && lib.versionAtLeast version.rocmVersion "7.1"
                && lib.versionOlder version.rocmVersion "7.3"
              )
            )
          ) versions;

          # The version with the oldest backend version for a wheel.
          oldestFor =
            key:
            lib.foldl' (
              best: version:
              if best == null || builtins.compareVersions (backendVersion version) (backendVersion best) < 0 then
                version
              else
                best
            ) null (lib.filter (version: wheelKeyOf version == key) candidates);
        in
        lib.filter (version: version != null) (map oldestFor (builtins.attrNames wheelFor));

      mkVariant =
        {
          buildSet,
          variant,
          rev,
          ...
        }:
        let
          inherit (buildSet) pkgs python3 buildConfig;

          wheelInfo = wheelFor.${wheelKeyOf buildConfig};
          wheel = pkgs.fetchurl {
            inherit (wheelInfo) name hash;
            url = "${releaseUrl}/${builtins.replaceStrings [ "+" ] [ "%2B" ] wheelInfo.name}";
          };
        in
        pkgs.callPackage ./uccl-kernel.nix {
          inherit
            python3
            rev
            variant
            version
            wheel
            cudaArchs
            ;
          backend = buildConfig.backend;
          cudaMajorVersion = lib.versions.major (buildConfig.cudaVersion or "0");
        };
    in
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self torchVersions mkVariant;

      name = "uccl";
      repoId = "kernels-community/uccl";

      # The upstream wheels are for Python 3.12 and later (abi3).
      python = {
        version = "3.12";
        abi = "abi3";
      };

      tests = ./tests;
    };
}
