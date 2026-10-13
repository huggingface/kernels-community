{
  description = "Flake for Mamba kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.15.0-rc3-fixes";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      # Has many external dependencies, see README.md, this kernel should
      # probably be more lean.
      doGetKernelCheck = false;

      pythonCheckInputs =
        ps: with ps; [
          einops
          huggingface-hub
          transformers
        ];
    };
}
