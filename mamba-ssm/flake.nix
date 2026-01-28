{
  description = "Flake for Mamba kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
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
          causal-conv1d
          einops
          transformers
        ];
    };
}
