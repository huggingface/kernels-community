{
  description = "Flake for vLLM-MoE kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.15.0-final";
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
