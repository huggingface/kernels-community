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

## Low-level ops

`flash_attn_interface` is part of the public API and mirrors the upstream
`flash_attn.flash_attn_interface` module. Besides the high-level functions,
these four functions are supported entry points:

- `flash_attn_interface._flash_attn_forward`
- `flash_attn_interface._flash_attn_backward`
- `flash_attn_interface._flash_attn_varlen_forward`
- `flash_attn_interface._flash_attn_varlen_backward`

The upstream `_wrapped_*` aliases of these four functions are available as
well. They are Torch custom ops with upstream's signatures. They have no autograd,
always return the softmax LSE, and are traceable by `torch.compile`. Use them
when you need the LSE or drive the backward pass yourself, for instance in
ring or context-parallel attention. The names keep their upstream leading
underscore so code written against the `flash_attn` package ports unchanged.

Dropout is supported. The CUDA build uses the Torch stable ABI and cannot read
the CUDA generator itself, so the forward is handed a generator and reserves
its Philox seed and offset through the dispatcher, exactly as the non-stable
build did. Masks are therefore bit-identical to earlier kernel versions and to
the pip `flash_attn` package for the same `torch.manual_seed`, including under
activation checkpointing and CUDA graph capture.

Torch 2.10 to 2.12 cannot pass a generator through the stable ABI. There the
kernel draws its seed and offset from the default generator instead, which is
reproducible under `torch.manual_seed` but not bit-identical to other Torch
versions, and the `gen` argument of `fwd`/`varlen_fwd` raises a `ValueError`.

`return_softmax` requires `dropout_p > 0`.
{% if layers %}

## Available layers
{% for layer in layers %}
- `{{ layer }}`
{% endfor %}
{% endif %}

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
