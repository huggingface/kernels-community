{
  description = "Flake for Punica SGMV kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.12-fixes";
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
