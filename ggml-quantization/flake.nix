{
  description = "GGUF kernels: compute on the packed blocks of a GGUF checkpoint";
  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };
  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
