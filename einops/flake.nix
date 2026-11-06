{
  description = "Flake for einops kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
    upstream = {
      url = "github:arogozhnikov/einops/ad2c8d67d08cbebb45e1aff2be21dc88635fc695";
      flake = false;
    };
  };

  outputs =
    {
      self,
      kernel-builder,
      upstream,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      port = {
        recipe = ./port;
        inherit upstream;
      };
    };
}
