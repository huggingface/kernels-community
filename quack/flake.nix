{
  description = "Flake for QuACK CuTe-DSL kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/kernel-kernel-deps";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      pythonCheckInputs = ps: [ ps.einops ];
    };
}
