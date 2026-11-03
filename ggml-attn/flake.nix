{
  description = "ggml's flash attention as a torch op and a transformers attention implementation";
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
