---
tags:
- kernels
- moe
- cuda
---

# SonicMoE

Accelerating Mixture-of-Experts with IO and Tile-aware Optimizations.

**SonicMoE** is a blazing-fast MoE implementation optimized for NVIDIA Hopper and Blackwell GPUs.
It leverages CuTe-DSL and Triton to deliver state-of-the-art performance through IO-aware optimizations.

- Paper: [arXiv:2512.14080](https://arxiv.org/abs/2512.14080)
- Source: [Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe)

## Requirements

- NVIDIA Hopper GPUs (H100, H200) or Blackwell GPUs (GB200, B200)
- PyTorch >= 2.7
- CUDA 12.9+
- Python 3.12+

## Usage

```python
import torch
from kernels import get_kernel

sonicmoe = get_kernel("kernels-community/sonic-moe")

from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType

moe = MoE(
    num_experts=128,
    num_experts_per_tok=8,
    hidden_size=4096,
    intermediate_size=1536,
    activation_function=ActivationType.SWIGLU,
    add_bias=False,
    std=0.02,
).to(device="cuda", dtype=torch.bfloat16)

x = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16)
output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
```

## Vendored Dependencies

This kernel vendors [QuACK](https://github.com/Dao-AILab/quack) (quack-kernels) for CuTe-DSL
GEMM infrastructure. The vendored copy is located at `torch-ext/sonicmoe/quack/`.

## License

Apache-2.0 (SonicMoE and QuACK are both Apache-2.0 licensed)
