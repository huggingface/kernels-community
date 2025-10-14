{
  description = "Flake for flash-attn kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;

      pythonCheckInputs =
        pkgs: with pkgs; [
          einops
        ];
    };
}
