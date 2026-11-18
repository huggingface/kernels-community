---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

This is the repository card of {{ repo_id }} that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels). This card was automatically generated.

## How to use

Use Linux x86_64 with a gfx1201 GPU, ROCm-enabled PyTorch, and a release of
`kernels` that supports kernel dependencies. FlyDSL is provided by the
`kernels-staging/flydsl` kernel dependency, which `kernels` fetches automatically.
FlyDSL compiles kernels at runtime and also requires the ROCm development SDK
and linker: set `ROCM_PATH` to the SDK root and add its LLVM tool directory to
`PATH`. Installing the Python packages alone does not provide these tools.
The wrapper reports a missing
linker or incorrect FlyDSL GPU detection before the first GPU compilation.

```python
import torch
from kernels import get_kernel

# If the org / user isn't a trusted publisher, pass `trust_remote_code=True` to the
# `get_kernel` call. You can find whether this kernel is from a trusted publisher
# by going to the kernel's Hub page and finding the "Trusted publisher" status at
# the top of the page.
kernel_module = get_kernel("{{ repo_id }}", version={{ version }})

# BSHD: batch, sequence, heads, head dimension.
q = torch.randn(1, 4096, 24, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = kernel_module.attention(q, k, v)
out_fp8 = kernel_module.attention(q, k, v, fp8=True)
out_auto = kernel_module.attention(q, k, v, fp8="auto")
```

## Available functions
{% if functions %}
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% else %}

Function list not available.
{% endif %}
{% if layers %}

## Available layers
{% for layer in layers %}
- `{{ layer }}`
{% endfor %}
{% endif %}

## Benchmarks
{% if has_benchmark %}

Run `kernels benchmark {{ repo_id }} --version {{ version }}` for the standard
Hub benchmark. A standalone comparison script is available in the source repository at
`benchmarks/comparison.py`. It compares the published AMD attention kernel with
FlyDSL and SDPA, including FP8 quantization cost and numerical error.
{% else %}

No benchmark available yet.
{% endif %}
{% if upstream %}

## Upstream

The original source code for this kernel comes from {{ upstream }}.
{% endif %}
{% if source %}

## Source

The kernel-builder formatted source for this kernel is available at {{ source }}.
{% endif %}


## Hardware and precision

This package targets gfx1201 on Linux x86_64 and requires ROCm PyTorch and
the `kernels-staging/flydsl` kernel dependency. It provides BF16/FP16 and prequantized E4M3FN FP8 forward attention
in BSHD layout, plus BF16-to-FP8 Q/K/V quantization. It does not provide
backward, LSE, GQA, arbitrary masks, dropout, or packed variable-length inputs.

Q/K/V must have matching batch size, head count, head dimension, and dtype;
K/V shapes must match. Non-causal cross-attention permits different Q and KV
lengths. Causal self-attention is supported; causal cross-attention is not.
Attention head dimensions must be at least 64 and divisible by 32, subject to
the GPU's LDS and buffer-index limits. Prequantized FP8 requires one-element
FP32 device descales for Q/K/V and returns BF16; BF16/FP16 attention returns
the input dtype.

`attention(..., fp8=True)` requires BF16 inputs and includes quantization and
Hadamard Q/K rotation, which requires a power-of-two head dimension (64–1024).
`fp8="auto"` uses xDiT's non-causal shape crossover table; causal attention,
FP16, unknown shapes, and short conditioning K/V retain input precision.
The inherited thresholds are not performance guarantees for every board or
ROCm version. FP8 changes numerical results; assess image/video quality for
the intended model. `flash_attn_func` exposes the raw upstream options, and
`attention` supports full-graph `torch.compile` through registered custom ops.
