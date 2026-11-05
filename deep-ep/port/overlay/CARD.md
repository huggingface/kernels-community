---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

This is the repository card of {{ repo_id }} that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels). This card was automatically generated.

## How to use
{% if functions %}

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("{{ repo_id }}", version={{ version }})
{{ functions[0] }} = kernel_module.{{ functions[0] }}

{{ functions[0] }}(...)
```
{% else %}

Usage example not available.
{% endif %}

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

## Requirements

This is [DeepEP](https://github.com/deepseek-ai/DeepEP) 2.5.0 (`def8651`), with
[DeepJIT](https://github.com/deepseek-ai/DeepJIT) `cba802d`. Only DeepEP's host
extension is prebuilt; its communication kernels are compiled at runtime. On
top of the Hub kernel's own requirements, using it needs:

- NVIDIA Hopper or newer GPUs, connected with NVLink (RDMA for multi-node).
- NCCL >= 2.32.3, loaded by PyTorch. The kernel does not ship NCCL: it uses the
  NCCL library that PyTorch loads, and compiles its kernels against that
  library's headers. PyTorch 2.14 and 2.15 ship NCCL 2.30.7; upgrade it with
  `pip install "nvidia-nccl-cu13>=2.32.3"` (or use a PyTorch build that ships
  2.32.3 or newer).
- A CUDA toolkit >= 13.1 (`nvcc`) and a C++20 host compiler at runtime, found
  through `CUDA_HOME`, `CUDA_PATH`, `nvcc` on `PATH` or `/usr/local/cuda`.

These are checked when the first buffer is created, with an error explaining
what is missing. Compiled kernels are cached in `$HOME/.dj` (`EP_JIT_CACHE_DIR`).

## Benchmarks
{% if has_benchmark %}

Benchmarking script is available for this kernel. Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
{% else %}

No benchmark available yet.
{% endif %}
{% if upstream %}

## Source code

Source code of this kernel originally comes from {{ upstream }} and it was repurposed for compatibility with `kernels`.
{% endif %}
