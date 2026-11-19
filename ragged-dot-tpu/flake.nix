{
  description = "Flake for ragged-dot-tpu kernels";

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

      # The check loads the kernel in the build sandbox, which has no TPU:
      # the backend is detected as CPU there, so the TPU-only tokamax kernel
      # dependency cannot be resolved.
      doGetKernelCheck = false;
    };
}
