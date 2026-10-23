{
  description = "Flake for fine-grained FP8 block-wise quantization kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/torch-2.15.0-rc5";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      # pytest.ini sets `--dist`, which only parses with xdist installed
      pythonCheckInputs = ps: with ps; [ pytest-xdist ];
    };
}
