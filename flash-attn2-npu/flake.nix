{
  description = "Flake for flash_attn2_npu kernel";

  inputs = {
    kernel-builder.url = "path:../..";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
      doGetKernelCheck = false;
    };
}
