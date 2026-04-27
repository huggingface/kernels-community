---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

This is the repository card of {{ repo_id }} that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels).

## How to use
{% if functions %}

```python
from kernels import get_kernel

kernel_module = get_kernel("{{ repo_id }}")
LiteAttention = kernel_module.LiteAttention

attn = LiteAttention(enable_skipping=True, use_int8=True)
out = attn(q, k, v)
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

## Source code

Source code of this kernel originally comes from https://github.com/moonmath-ai/LiteAttention and was repurposed for compatibility with `kernels`.
