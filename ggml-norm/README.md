---
library_name: kernels
license: mit
tags:
  - kernel
---

## ggml-norm

The fused RMS norm from [llama.cpp](https://github.com/ggml-org/llama.cpp) (`kernel_rms_norm_mul_f32`)
as a single kernel — normalisation and the weight multiply in one dispatch, where eager torch spells the
same thing as five. A decode step runs one of these per norm per layer, so the difference is launch
overhead rather than arithmetic.

`weight` is taken as the kernel will use it: a model whose weight is zero-centered passes `1 + w`, not
`w`. The `RMSNormZeroCentered` layer does that folding itself, once.

## Usage

```python
import torch
from kernels import get_kernel

norm = get_kernel("kernels-community/ggml-norm", version=1)

x = torch.randn(1, 2048, device="mps")
weight = torch.randn(2048, device="mps")

out = norm.rms_norm(x, weight, 1e-6)  # (1, 2048)
```
