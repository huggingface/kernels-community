{
  description = "GGUF kernels: compute on the packed blocks of a GGUF checkpoint";
  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.15.0-rc3-fixes";
  };
  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
