{
  description = "Flake for Flash Attention 4 kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
    nixpkgs.follows = "kernel-builder/nixpkgs";
  };

  outputs =
    {
      self,
      kernel-builder,
      nixpkgs,
    }:
    {
    packages = builtins.mapAttrs (system: builderPkgs: 
      builtins.mapAttrs (variant: torch: torch.pkgs.callPackage ./cutlass-dsl.nix { inherit variant; })
      builderPkgs.torch)
     kernel-builder.packages;
   };
}
