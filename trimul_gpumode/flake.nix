{
  description = "Flake for trimul_gpumode kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/torch-2.10";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;
    };
}
