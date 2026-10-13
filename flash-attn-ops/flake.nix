{
  description = "Flake for flash-attn Triton ops";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/nix-pass-gpu-lang";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      pythonCheckInputs =
        pkgs: with pkgs; [
          einops
        ];
    };
}
