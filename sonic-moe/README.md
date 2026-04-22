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
- Vendored dependency: [Dao-AILab/quack](https://github.com/Dao-AILab/quack) (v0.3.11)

## Requirements

- NVIDIA Hopper GPUs (H100, H200) or Blackwell GPUs (GB200, B200, B300)
- PyTorch >= 2.7
- CUDA 12.9+ (13.0+ for B300)
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

### Router variants

Since [sonic-moe#39](https://github.com/Dao-AILab/sonic-moe/pull/39) the functional
API exposes both `softmax(topk(logits))` (TC default) and `topk(softmax(logits))`
(Qwen3 style) routing, plus optional top-k probability renormalization:

```python
from sonicmoe import moe_TC_softmax_topk_layer

# Qwen3-style: topk(softmax()) with renormalized probabilities
out, router_logits, expert_freq = moe_TC_softmax_topk_layer(
    x, moe.router.weight,
    moe.c_fc.weight.permute(1, 2, 0), moe.c_fc.bias,
    moe.c_proj.weight.permute(1, 2, 0), moe.c_proj.bias,
    K=8,
    stream_id=torch.cuda.current_stream().cuda_stream,
    is_softmax_over_topk=False,
    norm_topk_probs=True,
)
```

### Weight layout

By default, `w1` (gated up-projection) is expected in **interleaved** layout
`[gate_0, up_0, gate_1, up_1, ...]`. HuggingFace checkpoints (Qwen3, Mixtral,
DeepSeek, ...) store `gate_up_proj` in **concatenated** layout
`[gate_0, ..., gate_{I-1}, up_0, ..., up_{I-1}]`. Pass `concat_layout=True` to
`moe_TC_softmax_topk_layer` / `moe_general_routing_inputs` to consume the
concatenated layout directly without a pre-pass permutation.

## Vendored Dependencies

This kernel vendors [QuACK](https://github.com/Dao-AILab/quack) v0.3.11 for
CuTe-DSL grouped GEMM infrastructure (Hopper + Blackwell). The vendored copy is
under `torch-ext/sonic_moe/quack/`. All `quack::` torch operator names are
rewritten to the `sonicmoe::quack__*` namespace so builds don't collide with a
user-installed `quack-kernels`.

## License

Apache-2.0 (SonicMoE and QuACK are both Apache-2.0 licensed)
