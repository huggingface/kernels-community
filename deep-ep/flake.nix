{
  description = "Flake for DeepEP kernel";

  inputs = {
    self.submodules = true;
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./src;

      # DeepEP 2.5 uses NCCL 2.31/2.32 APIs (GIN, EFA-GDA), newer than the NCCL that
      # kernel-builder's Torch versions ship. Build against Torch with NCCL 2.32.3.
      overlays = [
        (
          final: prev:
          let
            inherit (prev) lib;
          in
          {
            nccl_2_32_3 = final.mkNccl {
              version = "2.32.3";
              wheels = {
                x86_64-linux.cuda13 = {
                  url = "https://files.pythonhosted.org/packages/5b/29/6b277e63c92d91f9cb4d1a3a554e148983de39d54baa652bb52c798af78e/nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_x86_64.whl";
                  sha256 = "1459723080ac889d73a26edfa3e04383a7928ab31ac8f0ec43b3ea9548b04ff3";
                };
                aarch64-linux.cuda13 = {
                  url = "https://files.pythonhosted.org/packages/58/0a/c29c302036a06d27dd732588f3fd8ee89b1f7b0087e38c668ae7be8ff7b2/nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_aarch64.whl";
                  sha256 = "a5bee92b2f4af218c109f8d221c3c9adcc752b94ae0ffcb0ed5abf9341a7724c";
                };
              };
            };

            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (
                python-self: python-super:
                lib.mapAttrs (_: torch: torch.override { nccl = final.nccl_2_32_3; }) (
                  lib.filterAttrs (name: _: lib.hasPrefix "torch-bin_" name) python-super
                )
              )
            ];
          }
        )
      ];
    };
}
