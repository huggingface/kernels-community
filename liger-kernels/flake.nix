{
  description = "Flake for Unsloth Kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/triton-rocm-fixes";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
