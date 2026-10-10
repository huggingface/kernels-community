---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

# ragged-dot-tpu

MoE experts forward for torch_tpu, built on tokamax's Pallas grouped matmul
(`tokamax.ragged_dot`, `implementation="mosaic_tpu_v2"`) and wrapped with
torch_tpu's `jax_op`. `experts_forward` is a drop-in transformers experts
implementation for gated experts without bias, with a SiLU (e.g. Qwen3-MoE)
or GELU-tanh (e.g. Gemma 4) gate, and supports expert parallelism. The whole
experts forward runs as one op.

```python
from kernels import get_kernel
from transformers import AutoModelForCausalLM
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

kernel = get_kernel("{{ repo_id }}", version={{ version }})
ALL_EXPERTS_FUNCTIONS.register("tokamax_ragged_dot", kernel.experts_forward)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    experts_implementation="tokamax_ragged_dot",
)
```

Requires torch_tpu, jax and tokamax.

On a TPU v6e-8 (expert parallelism over 8 cores), against transformers'
default `grouped_mm`:

- Qwen3-30B-A3B-Instruct-2507 decodes 1.13x faster at batch size 1 and
  21.5x faster at batch size 64 (988 against 52 tokens/s).
- gemma-4-26B-A4B-it decodes 1.07x faster at batch size 1 and 12.8x
  faster at batch size 64 (162 against 13 tokens/s).
