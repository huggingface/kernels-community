"""Layer-level entry points, for `kernels`' layer mapping.

`transformers` marks the layers a kernel may replace with `@use_kernel_forward_from_hub("<name>")`, and a
kernel repository supplies a class of that name whose `forward` is grafted onto the real module. The module
keeps its own parameters and constructor -- the class here is stateless, which `kernels` enforces: no
`__init__`, no members besides `forward`, and a signature matching the layer being replaced.

There is a class per norm rather than one generic one because models disagree, silently, on what the
weight means: most compute `x * w`, the zero-centered ones `x * (1 + w)`. Both are an RMS norm and both
type-check; getting it wrong just makes the model slightly wrong. Naming the convention per layer is the
only way that mistake shows up as a missing entry instead of bad output.
"""

import torch
import torch.nn as nn

from ._ops import ops


def _folded_weight(module: nn.Module, fold) -> torch.Tensor:
    """The module's weight in the form the kernel wants, computed once and kept on the module.

    ggml's kernel takes the multiplicand as it will use it, so a convention like `1 + w` has to be
    resolved somewhere. Doing it per call would add back a dispatch, which is the entire point of the
    graft; the weights do not change under inference, so it is done on first use and cached.
    """
    weight = getattr(module, "_ggml_norm_weight", None)
    if weight is None:
        weight = fold(module.weight.detach().float()).contiguous()
        module.register_buffer("_ggml_norm_weight", weight, persistent=False)
    return weight


class RMSNormZeroCentered(nn.Module):
    """The norm whose weight is zero-centered -- the layer computes `x * (1 + w)`.

    Qwen3-Next and Qwen3.5 use it. Named for the convention rather than the model because that is
    what the kernel actually depends on, and because the plain `x * w` name is already taken by the
    118 models that spell it the other way.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `float()` and `type_as` are both no-ops when x is already f32, which is what a GGUF model
        # runs in, so that path is the single dispatch it looks like.
        return ops.rms_norm(x.float(), _folded_weight(self, lambda w: 1.0 + w), self.eps).type_as(x)
