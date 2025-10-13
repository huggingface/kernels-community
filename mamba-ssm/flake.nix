{
  description = "Flake for Mamba kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
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
    };
}
