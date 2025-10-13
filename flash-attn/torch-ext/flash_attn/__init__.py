from typing import Optional, List
import torch
from ._ops import ops as flash_attn_ops
from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    return_softmax: bool = False,
    gen: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """
    Forward pass for multi-head attention.

    Args:
        q: Query tensor of shape [batch_size, seqlen_q, num_heads, head_size]
        k: Key tensor of shape [batch_size, seqlen_k, num_heads_k, head_size]
        v: Value tensor of shape [batch_size, seqlen_k, num_heads_k, head_size]
        out: Optional output tensor, same shape as q
        alibi_slopes: Optional ALiBi slopes tensor of shape [num_heads] or [batch_size, num_heads]
        p_dropout: Dropout probability
        softmax_scale: Scale factor for softmax
        is_causal: Whether to use causal attention
        window_size_left: Window size for left context (-1 for unlimited)
        window_size_right: Window size for right context (-1 for unlimited)
        softcap: Soft cap for attention weights
        return_softmax: Whether to return softmax weights
        gen: Optional random number generator

    Returns:
        List of tensors: [output, softmax_lse, (softmax if return_softmax)]
    """
    if softmax_scale is None:
        attention_head_dim = q.shape[-1]
        softmax_scale = 1.0 / (attention_head_dim**0.5)

    return flash_attn_ops.fwd(
        q,
        k,
        v,
        out,
        alibi_slopes,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
    )


def varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    zero_tensors: bool = False,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    return_softmax: bool = False,
    gen: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """
    Forward pass for multi-head attention with variable sequence lengths.

    Args:
        q: Query tensor of shape [total_q, num_heads, head_size]
        k: Key tensor of shape [total_k, num_heads_k, head_size] or [num_blocks, page_block_size, num_heads_k, head_size]
        v: Value tensor of shape [total_k, num_heads_k, head_size] or [num_blocks, page_block_size, num_heads_k, head_size]
        cu_seqlens_q: Cumulative sequence lengths for queries of shape [batch_size+1]
        cu_seqlens_k: Cumulative sequence lengths for keys of shape [batch_size+1]
        out: Optional output tensor of shape [total_q, num_heads, head_size]
        seqused_k: Optional tensor specifying how many keys to use per batch element [batch_size]
        leftpad_k: Optional left padding for keys of shape [batch_size]
        block_table: Optional block table of shape [batch_size, max_num_blocks_per_seq]
        alibi_slopes: Optional ALiBi slopes tensor of shape [num_heads] or [batch_size, num_heads]
        max_seqlen_q: Maximum sequence length for queries
        max_seqlen_k: Maximum sequence length for keys
        p_dropout: Dropout probability
        softmax_scale: Scale factor for softmax
        zero_tensors: Whether to zero tensors before computation
        is_causal: Whether to use causal attention
        window_size_left: Window size for left context (-1 for unlimited)
        window_size_right: Window size for right context (-1 for unlimited)
        softcap: Soft cap for attention weights
        return_softmax: Whether to return softmax weights
        gen: Optional random number generator

    Returns:
        List of tensors: [output, softmax_lse, (softmax if return_softmax)]
    """
    if softmax_scale is None:
        attention_head_dim = q.shape[-1]
        softmax_scale = 1.0 / (attention_head_dim**0.5)

    return flash_attn_ops.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen,
    )


def bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    gen: Optional[torch.Generator] = None,
    rng_state: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Backward pass for multi-head attention.

    Args:
        dout: Gradient tensor of shape [batch_size, seqlen_q, num_heads, head_size]
        q: Query tensor of shape [batch_size, seqlen_q, num_heads, head_size]
        k: Key tensor of shape [batch_size, seqlen_k, num_heads_k, head_size]
        v: Value tensor of shape [batch_size, seqlen_k, num_heads_k, head_size]
        out: Output tensor from forward pass of shape [batch_size, seqlen_q, num_heads, head_size]
        softmax_lse: Log-sum-exp values from forward pass of shape [batch_size, num_heads, seqlen_q]
        dq: Optional gradient tensor for queries, same shape as q
        dk: Optional gradient tensor for keys, same shape as k
        dv: Optional gradient tensor for values, same shape as v
        alibi_slopes: Optional ALiBi slopes tensor of shape [num_heads] or [batch_size, num_heads]
        p_dropout: Dropout probability
        softmax_scale: Scale factor for softmax
        is_causal: Whether to use causal attention
        window_size_left: Window size for left context (-1 for unlimited)
        window_size_right: Window size for right context (-1 for unlimited)
        softcap: Soft cap for attention weights
        deterministic: Whether to use deterministic algorithms
        gen: Optional random number generator
        rng_state: Optional RNG state from forward pass

    Returns:
        List of tensors: [dq, dk, dv]
    """
    if softmax_scale is None:
        attention_head_dim = q.shape[-1]
        softmax_scale = 1.0 / (attention_head_dim**0.5)

    return flash_attn_ops.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        gen,
        rng_state,
    )


def varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    zero_tensors: bool = False,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    gen: Optional[torch.Generator] = None,
    rng_state: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Backward pass for multi-head attention with variable sequence lengths.

    Args:
        dout: Gradient tensor of shape [batch_size, seqlen_q, num_heads, head_size]
        q: Query tensor of shape [batch_size, seqlen_q, num_heads, head_size]
        k: Key tensor of shape [batch_size, seqlen_k, num_heads_k, head_size]
        v: Value tensor of shape [batch_size, seqlen_k, num_heads_k, head_size]
        out: Output tensor from forward pass of shape [batch_size, seqlen_q, num_heads, head_size]
        softmax_lse: Log-sum-exp values from forward pass of shape [batch_size, num_heads, seqlen_q]
        cu_seqlens_q: Cumulative sequence lengths for queries of shape [batch_size+1]
        cu_seqlens_k: Cumulative sequence lengths for keys of shape [batch_size+1]
        dq: Optional gradient tensor for queries, same shape as q
        dk: Optional gradient tensor for keys, same shape as k
        dv: Optional gradient tensor for values, same shape as v
        alibi_slopes: Optional ALiBi slopes tensor of shape [num_heads] or [batch_size, num_heads]
        max_seqlen_q: Maximum sequence length for queries
        max_seqlen_k: Maximum sequence length for keys
        p_dropout: Dropout probability
        softmax_scale: Scale factor for softmax
        zero_tensors: Whether to zero tensors before computation
        is_causal: Whether to use causal attention
        window_size_left: Window size for left context (-1 for unlimited)
        window_size_right: Window size for right context (-1 for unlimited)
        softcap: Soft cap for attention weights
        deterministic: Whether to use deterministic algorithms
        gen: Optional random number generator
        rng_state: Optional RNG state from forward pass

    Returns:
        List of tensors: [dq, dk, dv]
    """
    if softmax_scale is None:
        attention_head_dim = q.shape[-1]
        softmax_scale = 1.0 / (attention_head_dim**0.5)

    return flash_attn_ops.varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        gen,
        rng_state,
    )


def fwd_kvcache(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    seqlens_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    is_rotary_interleaved: bool = False,
    num_splits: int = 1,
) -> List[torch.Tensor]:
    """
    Forward pass for multi-head attention with KV cache.

    Args:
        q: Query tensor of shape [batch_size, seqlen_q, num_heads, head_size]
        kcache: Key cache tensor of shape [batch_size_c, seqlen_k, num_heads_k, head_size] or [num_blocks, page_block_size, num_heads_k, head_size]
        vcache: Value cache tensor of shape [batch_size_c, seqlen_k, num_heads_k, head_size] or [num_blocks, page_block_size, num_heads_k, head_size]
        k: Optional new keys tensor of shape [batch_size, seqlen_knew, num_heads_k, head_size]
        v: Optional new values tensor of shape [batch_size, seqlen_knew, num_heads_k, head_size]
        seqlens_k: Optional sequence lengths for keys of shape [batch_size]
        rotary_cos: Optional rotary cosine tensor of shape [seqlen_ro, rotary_dim/2]
        rotary_sin: Optional rotary sine tensor of shape [seqlen_ro, rotary_dim/2]
        cache_batch_idx: Optional indices to index into the KV cache
        leftpad_k: Optional left padding for keys of shape [batch_size]
        block_table: Optional block table of shape [batch_size, max_num_blocks_per_seq]
        alibi_slopes: Optional ALiBi slopes tensor of shape [num_heads] or [batch_size, num_heads]
        out: Optional output tensor, same shape as q
        softmax_scale: Scale factor for softmax
        is_causal: Whether to use causal attention
        window_size_left: Window size for left context (-1 for unlimited)
        window_size_right: Window size for right context (-1 for unlimited)
        softcap: Soft cap for attention weights
        is_rotary_interleaved: Whether rotary embeddings are interleaved
        num_splits: Number of splits for computation

    Returns:
        List of tensors: [output, softmax_lse]
    """
    if softmax_scale is None:
        attention_head_dim = q.shape[-1]
        softmax_scale = 1.0 / (attention_head_dim**0.5)

    return flash_attn_ops.fwd_kvcache(
        q,
        kcache,
        vcache,
        k,
        v,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        is_rotary_interleaved,
        num_splits,
    )
