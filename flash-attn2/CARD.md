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

Note that the stable-ABI builds (Torch 2.10 and later) do not support
`dropout_p > 0`, and `return_softmax` requires dropout, so pass `False` for it.
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
