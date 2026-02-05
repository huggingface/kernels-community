{
  description = "Flake for Torch kernel extension";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/remove-backend";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;
    };
}
