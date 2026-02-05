{
  description = "Flake for rwkv kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/remove-backend";
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
