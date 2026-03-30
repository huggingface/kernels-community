{
  description = "Flake for fine-grained FP8 block-wise quantization kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/ci-tests-arch-variants";
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
