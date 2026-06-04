{
  description = "Flake for fine-grained FP8 block-wise quantization kernels";

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
