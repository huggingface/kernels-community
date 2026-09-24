---
license: apache-2.0
tags:
- kernels
---

# Metal Flash SDPA

Optimized SDPA kernels inspired by Flash Attention for Metal.

Some components of these kernels are from [mlx](https://github.com/ml-explore/mlx).

## Supported Features

- Variable-length sequences without padding
- Causal masking
- Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
- Softcapping support for attention score regularization
- Attention sinks (`s_aux`, one logit per query head)
- Data types: `float32`, `float16`, `bfloat16`
- Head dimensions: `32`, `64`, `72`, `80`, `96`, `128`, `192`, `256`
- Strided inputs (e.g. views into a packed QKV tensor), as long as `head_dim` is contiguous

## API Reference

### flash_attention_varlen

```python
metal_flash_sdpa.flash_attention_varlen(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    do_causal: bool,
    scale: float,
    softcapping: float,
    s_aux: Optional[torch.Tensor] = None
) -> None
```

- **out**: Output tensor `[total_q_tokens, num_heads, head_dim]`, modified in-place.
- **query/key/value**: Input tensors `[total_tokens, num_heads(_kv), head_dim]`.
- **cu_seqlens_q/cu_seqlens_k**: Cumulative sequence lengths (`torch.int32`), `[batch_size + 1]`.
- **max_seqlen_q/max_seqlen_k**: Maximum sequence lengths.
- **do_causal**: Enable causal masking. When the query and key lengths of a sequence differ, the mask is aligned to the bottom-right corner, as in Flash Attention. Query rows that do not attend to any key produce zeros.
- **scale**: Attention score scaling factor (e.g., `1/sqrt(head_dim)`).
- **softcapping**: Softcapping value for score regularization (use `1.0` for no softcapping).
- **s_aux**: Optional attention sinks `[num_heads]`. Each sink adds a logit to the softmax of its head without attending to a value, as used by gpt-oss.

### flash_attn_varlen_func

Compatibility wrapper matching the original Flash Attention API:

```python
out = metal_flash_sdpa.flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    s_aux: Optional[torch.Tensor] = None
)
```

`softcap` follows Flash Attention: `0.0` disables softcapping.
