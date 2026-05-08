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
      in
      {
        devShell =
          with pkgs;
          mkShell {
            name = "kernels-community-dev-shell";
            nativeBuildInputs = [
              pinact
            ];
          };

        formatter = pkgs.nixfmt-tree;
      }
    );
}
