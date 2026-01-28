from typing import List, Optional

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
) -> None:
    """
    Flash Attention with variable-length sequences.
    
    Args:
        out: Output tensor of shape [total_q_tokens, num_heads, head_dim]
        query: Query tensor of shape [total_q_tokens, num_heads, head_dim]
        key: Key tensor of shape [total_k_tokens, num_heads_kv, head_dim]
        value: Value tensor of shape [total_k_tokens, num_heads_kv, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1], dtype must be torch.int32
        cu_seqlens_k: Cumulative sequence lengths for keys, shape [batch_size + 1], dtype must be torch.int32
        max_seqlen_q: Maximum sequence length in the query batch
        max_seqlen_k: Maximum sequence length in the key batch
        do_causal: Whether to apply causal masking
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        softcapping: Softcapping value (default: 1.0, must be 1.0 for this implementation)
    
    Note:
        - cu_seqlens_q and cu_seqlens_k must have dtype torch.int32 for Metal compatibility
        - Supported head dimensions: 32, 64, 72, 80, 96, 128
        - Masks are not supported
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
        softcapping,
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
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    """
    Flash Attention function with API compatible with the original Flash Attention.
    
    Note: This implementation does not support:
    - dropout
    - window attention
    - alibi slopes
    - returning attention probabilities
    """
    if dropout_p > 0:
        raise NotImplementedError("Dropout is not supported in this implementation")
    if window_size != (-1, -1):
        raise NotImplementedError("Window attention is not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not supported")
    if return_attn_probs:
        raise NotImplementedError("Returning attention probabilities is not supported")
    
    # Create output tensor
    out = torch.empty_like(q)
    
    # Call the kernel
    flash_attention_varlen(
        out=out,
        query=q,
        key=k,
        value=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        do_causal=causal,
        scale=softmax_scale,
        softcapping=1.0,
    )
    
    return out


__all__ = [
    "flash_attention_varlen",
    "flash_attn_varlen_func",
]
