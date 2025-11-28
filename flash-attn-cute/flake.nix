{
  description = "Flake for trimul_gpumode kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder/nvidia-cutlass-dsl-4.3.0";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
      pythonCheckInputs = ps: [ ps.einops ];
    };
}
