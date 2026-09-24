from typing import Optional

import torch

from ._ops import ops


def flash_attention_varlen(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    do_causal: bool = False,
    scale: Optional[float] = None,
    softcapping: float = 1.0,
    s_aux: Optional[torch.Tensor] = None,
) -> None:
    """
    Flash Attention with variable-length sequences.

    Args:
        out: Output tensor of shape [total_q_tokens, num_heads, head_dim]
        query: Query tensor of shape [total_q_tokens, num_heads, head_dim]
        key: Key tensor of shape [total_k_tokens, num_heads_kv, head_dim]
        value: Value tensor of shape [total_k_tokens, num_heads_kv, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1], dtype torch.int32
        cu_seqlens_k: Cumulative sequence lengths for keys, shape [batch_size + 1], dtype torch.int32
        max_seqlen_q: Maximum query sequence length in the batch. Must not be smaller
            than the actual maximum, otherwise part of the output is not written.
        max_seqlen_k: Maximum key sequence length in the batch
        do_causal: Whether to apply causal masking. When the query and key lengths of a
            sequence differ, the mask is aligned to the bottom-right corner of the
            attention matrix, as in flash-attn.
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        softcapping: Softcap value. 1.0 (the default) disables softcapping.
        s_aux: Optional attention sinks, shape [num_heads]. Each sink adds a logit
            to the softmax of its head that does not attend to any value.

    Note:
        - Supported head dimensions: 32, 64, 72, 80, 96, 128, 192, 256
        - Inputs may be strided views, but head_dim must be contiguous.
        - num_heads must be divisible by num_heads_kv (grouped-query attention).
        - Query rows that do not attend to any key produce zeros.
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5

    ops.flash_attention_varlen(
        out,
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        do_causal,
        scale,
        # The op uses flash-attn's convention, where <= 0 disables softcapping.
        0.0 if softcapping == 1.0 else softcapping,
        s_aux,
    )


def flash_attn_varlen_func(
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
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    s_aux: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flash Attention function with API compatible with the original Flash Attention.

    `softcap` follows flash-attn: 0.0 (the default) disables softcapping.
    `s_aux` are optional attention sinks of shape [num_heads], as in vllm-flash-attn3.

    Note: This implementation does not support:
    - dropout
    - window attention
    - alibi slopes
    - returning attention probabilities
    """
    if dropout_p > 0:
        raise NotImplementedError("Dropout is not supported in this implementation")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("Window attention is not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not supported")
    if return_attn_probs:
        raise NotImplementedError("Returning attention probabilities is not supported")
    if softcap < 0:
        raise ValueError("softcap must be non-negative")

    out = torch.empty_like(q)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    ops.flash_attention_varlen(
        out,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        softmax_scale,
        softcap,
        s_aux,
    )

    return out


__all__ = [
    "flash_attention_varlen",
    "flash_attn_varlen_func",
]
