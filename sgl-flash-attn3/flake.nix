{
  description = "SGLang Flash Attention 3 kernel (forward-only, with sinks)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/nix-pass-gpu-lang";
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
