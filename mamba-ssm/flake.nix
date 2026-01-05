{
  description = "Flake for Mamba kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/torch-2.10";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
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
