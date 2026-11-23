{
  description = "Flake for vLLM quantization kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/testing-redesign";
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
