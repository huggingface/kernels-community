{
  description = "Flake for triton-kernels kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
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
