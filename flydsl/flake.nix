{
  description = "Flake for the FlyDSL SDK (repackaged upstream wheel)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/sdk-variants";
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

      repoId = "kernels-community/flydsl";
      version = "0.5.0";

      # FlyDSL only supports ROCm. The extensions are built for a specific Python
      # version (cp314, not the stable ABI), but the SDK variant format does not
      # express that yet, so the variant is (incorrectly) marked as abi3 for now.
      # The ROCm version is the lowest one that provides libamdhip64.so.7.
      wheelFor = {
        rocm = {
          variant = "sdk-rocm-7.0-abi3-x86_64-linux";
          name = "flydsl-${version}-cp314-cp314-manylinux_2_27_x86_64.whl";
          url = "https://files.pythonhosted.org/packages/e9/97/44f972c047a68f842a5a815592896787bb0c2812a7c5d80c4cda4f7e4e53/flydsl-${version}-cp314-cp314-manylinux_2_27_x86_64.whl";
          hash = "sha256-NlS6Vqo22eIWj/UYTsn/2Gga/qDukNHrv4fEYDUpSGo=";
        };
      };

      builderLib = kernel-builder + "/nix-builder/lib";
      builderProvenance = import (builderLib + "/flake-provenance.nix") { inherit lib; } kernel-builder;
      inherit
        (import (builderLib + "/build-sets.nix") {
          inherit (kernel-builder.inputs) nixpkgs rust-overlay;
          inherit builderProvenance;
        })
        mkBuildSets
        ;
      inherit (import (builderLib + "/torch-version-utils.nix") { inherit lib; }) flattenSystems;
      torchVersions = flattenSystems (import (kernel-builder + "/nix-builder/versions.nix"));

      wheelKeyOf = _: "rocm";

      # Order build sets by how suitable they are as the build set that is used
      # to load-check and test an SDK variant (Torch is only needed for that,
      # the first build set for each wheel is used):
      #
      # - Prefer newer Torch versions over older.
      # - Prefer older backend versions over newer (best compatibility).
      backendOrder = {
        rocm = 0;
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
          # Only x86_64-linux CUDA and ROCm build sets get a FlyDSL variant (the
          # upstream wheels are x86_64-linux only). Other systems get empty
          # outputs.
          allBuildSets = lib.optionals (system == "x86_64-linux") (
            lib.sort buildSetCompare (
              builtins.filter (
                buildSet: builtins.elem buildSet.buildConfig.backend (builtins.attrNames backendOrder)
              ) (mkBuildSets [ ] torchVersions [ system ])
            )
          );

          # The most suitable build set for each wheel, i.e. one per SDK variant.
          buildSets = lib.filter (buildSet: buildSet != null) (
            lib.mapAttrsToList (
              key: _: lib.findFirst (buildSet: wheelKeyOf buildSet == key) null allBuildSets
            ) wheelFor
          );

          variantOf = buildSet: wheelFor.${wheelKeyOf buildSet}.variant;

          mkVariant =
            buildSet:
            let
              inherit (buildSet)
                pkgs
                python3
                torch
                overrideTorch
                ;
              wheelInfo = wheelFor.${wheelKeyOf buildSet};
              inherit (wheelInfo) variant;
              wheel = pkgs.fetchurl {
                inherit (wheelInfo) name hash url;
              };

              kernel = pkgs.callPackage ./flydsl-kernel.nix {
                inherit
                  python3
                  torch
                  overrideTorch
                  rev
                  variant
                  version
                  wheel
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
              name = variantOf buildSet;
              value = mkVariant buildSet;
            }) buildSets
          );
          variantDrvs = lib.mapAttrs (_: set: set.kernel) variantSets;

          bundleBuildSets = buildSets;

          backends = lib.unique (map (buildSet: buildSet.buildConfig.backend) bundleBuildSets);
          buildSetsOfBackend =
            backend: builtins.filter (buildSet: buildSet.buildConfig.backend == backend) bundleBuildSets;
          bestVariant = variantOf (builtins.head bundleBuildSets);

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
                  buildSet: "cp -r ${variantDrvs.${variantOf buildSet}}/* $out/"
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
