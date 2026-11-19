{
  inputs = {
    kernel-builder.url = "github:huggingface/kernels/build-layout-zip";
  };
  outputs =
    { self, kernel-builder, ... }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
