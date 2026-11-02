{
  description = "Flake for the UCCL kernel (repackaged upstream wheels)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
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
      forAllSystems = lib.genAttrs systems;

      rev = lib.replaceStrings [ "-" ] [ "_" ] (
        self.shortRev or self.dirtyShortRev or self.lastModifiedDate
      );

      repoId = "kernels-community/uccl";
      version = "0.1.1";
      releaseUrl = "https://github.com/uccl-project/uccl/releases/download/v${version}";

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

      # The EP component requires 9.0 or later (9.0+PTX), but the other components
      # seem compatible with any capability, so let's add 8.x capabilities as well.
      cudaArchs = [
        "8.0"
        "8.6"
        "8.7"
        "8.9"
        "9.0+PTX"
      ];

      builderLib = kernel-builder + "/nix-builder/lib";
      builderProvenance = import (builderLib + "/flake-provenance.nix") { inherit lib; } kernel-builder;
      inherit
        (import (builderLib + "/build-sets.nix") {
          inherit (kernel-builder.inputs) nixpkgs rust-overlay;
          inherit builderProvenance;
        })
        mkBuildSets
        ;
      torchVersions = import (kernel-builder + "/nix-builder/versions.nix");

      variantName = buildSet: buildSet.variants.torch.arch;

      # Order build sets by how suitable they are as the single variant that is
      # built/tested by default (same logic as `genKernelFlakeOutputs`):
      #
      # - Prefer CUDA over ROCm.
      # - Prefer newer Torch versions over older.
      # - Prefer older backend versions over newer (best compatibility).
      backendOrder = {
        cuda = 0;
        rocm = 1;
      };
      backendVersion = buildConfig: buildConfig.cudaVersion or buildConfig.rocmVersion;
      buildSetCompare =
        setA: setB:
        let
          a = setA.buildConfig;
          b = setB.buildConfig;
        in
        if a.backend != b.backend then
          backendOrder.${a.backend} < backendOrder.${b.backend}
        else if a.torchVersion != b.torchVersion then
          builtins.compareVersions a.torchVersion b.torchVersion > 0
        else
          builtins.compareVersions (backendVersion a) (backendVersion b) < 0;

      mkSystemOutputs =
        system:
        let
          # Only x86_64-linux CUDA and ROCm build sets get a UCCL variant (the
          # upstream wheels are x86_64-linux only). Other systems get empty
          # outputs.
          buildSets = lib.optionals (system == "x86_64-linux") (
            lib.sort buildSetCompare (
              builtins.filter (
                buildSet: builtins.elem buildSet.buildConfig.backend (builtins.attrNames backendOrder)
              ) (mkBuildSets [ ] torchVersions [ system ])
            )
          );

          mkVariant =
            buildSet:
            let
              inherit (buildSet)
                pkgs
                python3
                torch
                overrideTorch
                ;
              variant = variantName buildSet;

              wheelKey = if torch.cudaSupport then "cu${torch.cudaPackages.cudaMajorVersion}" else "rocm";
              wheelInfo = wheelFor.${wheelKey} or (throw "No UCCL wheel for ${wheelKey}");
              wheel = pkgs.fetchurl {
                inherit (wheelInfo) name hash;
                url = "${releaseUrl}/${builtins.replaceStrings [ "+" ] [ "%2B" ] wheelInfo.name}";
              };

              kernel = pkgs.callPackage ./uccl-kernel.nix {
                inherit
                  python3
                  torch
                  overrideTorch
                  rev
                  variant
                  version
                  wheel
                  cudaArchs
                  ;
              };

              testPython = python3.withPackages (ps: [
                torch
                (overrideTorch python3.pkgs.kernels)
                ps.pytest
              ]);
            in
            {
              inherit
                kernel
                pkgs
                python3
                testPython
                ;
            };

          variantSets = builtins.listToAttrs (
            map (buildSet: {
              name = variantName buildSet;
              value = mkVariant buildSet;
            }) buildSets
          );
          variantDrvs = lib.mapAttrs (_: set: set.kernel) variantSets;

          bundleBuildSets = builtins.filter (buildSet: buildSet.bundleBuild) buildSets;

          backends = lib.unique (map (buildSet: buildSet.buildConfig.backend) bundleBuildSets);
          buildSetsOfBackend =
            backend: builtins.filter (buildSet: buildSet.buildConfig.backend == backend) bundleBuildSets;
          bestVariant = variantName (builtins.head bundleBuildSets);

          pkgs = (builtins.head buildSets).pkgs;

          mkBundle =
            buildSets:
            pkgs.runCommand "torch-ext-bundle"
              {
                preferLocalBuild = true;
                allowSubstitutes = false;
              }
              ''
                mkdir -p $out
                ${lib.concatMapStringsSep "\n" (
                  buildSet: "cp -r ${variantDrvs.${variantName buildSet}}/* $out/"
                ) buildSets}

                ${pkgs.kernel-builder}/bin/kernel-builder fill-card ${cardSource} $out/CARD.md
              '';

          cardSource = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./build.toml
              ./CARD.md
            ];
          };

          bundle = mkBundle bundleBuildSets;

          backendCi = lib.genAttrs backends (
            backend: mkBundle [ (builtins.head (buildSetsOfBackend backend)) ]
          );

          ciTests = lib.mapAttrs (
            variant: set:
            pkgs.writeShellScriptBin "ci-test" ''
              if [ -d ${./tests} ]; then
                unset LD_LIBRARY_PATH
                export LOCAL_KERNELS="${repoId}=${set.kernel}"
                # Accept exit code 5: no tests are selected
                ${set.testPython}/bin/python3 -m pytest ${./tests} -m kernels_ci -p no:cacheprovider || test $? -eq 5
              fi
            ''
          ) variantSets;

          testShells = lib.mapAttrs (
            variant: set:
            set.pkgs.mkShell {
              buildInputs = [ set.testPython ];
              shellHook = ''
                # This is run from `nix develop`, which provides the existing
                # environment. We clear the LD_LIBRARY_PATH to make testing as
                # pure as possible.
                unset LD_LIBRARY_PATH
                export PYTHONPATH=${set.kernel}/${variant}
                export LOCAL_KERNELS="${repoId}=${set.kernel}"
              '';
            }
          ) variantSets;
        in
        {
          packages = {
            backendBundle = lib.genAttrs backends (backend: mkBundle (buildSetsOfBackend backend));
            inherit backendCi ciTests;
            redistributable = variantDrvs;
            variants = builtins.attrNames variantDrvs;
            archVariants = builtins.attrNames variantDrvs;
          }
          // lib.optionalAttrs (variantDrvs != { }) {
            inherit bundle;
            default = bundle;

            # One variant per backend.
            ci = mkBundle (map (backend: builtins.head (buildSetsOfBackend backend)) backends);

            ci-test = ciTests.${bestVariant};

            build-and-copy = pkgs.writeShellScriptBin "build-and-copy" ''
              set -euo pipefail

              if [ ! -d build ]; then
                mkdir build
              fi

              for build_variant in ${bundle}/*; do
                build_variant=$(basename $build_variant)
                if [ -e build/$build_variant ]; then
                  rm -rf build/$build_variant
                fi

                cp -r ${bundle}/$build_variant build/
              done

              chmod -R +w build
            '';

            build-and-upload = pkgs.writeShellScriptBin "build-and-upload" ''
              set -euo pipefail
              ${pkgs.kernel-builder}/bin/kernel-builder upload --repo-id ${repoId} ${bundle}
            '';
          };

          # There is nothing to compile, so there are no development shells,
          # only test shells.
          devShells = lib.optionalAttrs (variantDrvs != { }) {
            inherit testShells;
            test = testShells.${bestVariant};
          };
        };
    in
    {
      packages = forAllSystems (system: (mkSystemOutputs system).packages);
      devShells = forAllSystems (system: (mkSystemOutputs system).devShells);
      formatter = forAllSystems (
        system: kernel-builder.inputs.nixpkgs.legacyPackages.${system}.nixfmt-tree
      );
    };
}
