{
  description = "Flake for metal-int4-sdpa kernel";

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
