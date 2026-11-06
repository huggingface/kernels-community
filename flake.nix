{
  description = "kernels-community tooling";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.follows = "kernels/nixpkgs";
    kernels.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      flake-utils,
      kernels,
      nixpkgs,
    }:
    let
      systems = with flake-utils.lib.system; [
        aarch64-darwin
        aarch64-linux
        x86_64-linux
      ];
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        kernel-status = pkgs.rustPlatform.buildRustPackage {
          pname = "kernel-status";
          version = "0.1.0";
          src = ./scripts/kernel-status;
          cargoLock.lockFile = ./scripts/kernel-status/Cargo.lock;
          meta.mainProgram = "kernel-status";
        };
      in
      {
        packages = {
          inherit kernel-status;
          default = kernel-status;
        };

        devShell =
          with pkgs;
          mkShell {
            name = "kernels-community-dev-shell";
            nativeBuildInputs = [
              kernel-status
              pinact
            ];
          };

        formatter = pkgs.nixfmt-tree;
      }
    );
}
