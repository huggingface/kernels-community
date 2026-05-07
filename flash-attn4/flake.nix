{
  description = "Flake for Flash Attention 4 kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.12";
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
