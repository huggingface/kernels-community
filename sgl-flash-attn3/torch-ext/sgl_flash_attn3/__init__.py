from functools import lru_cache
from typing import Optional, Union

import torch

from ._ops import ops


@lru_cache(maxsize=1)
def is_fa3_supported(device=None) -> bool:
    return (torch.version.cuda >= "12.3") and (
        torch.cuda.get_device_capability(device)[0] == 9
        or torch.cuda.get_device_capability(device)[0] == 8
    )


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk: Optional[int] = None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
):
    """
    FA3 flash_attn_with_kvcache: forward-only attention with paged KV cache,
    optional rotary embedding, sliding window, softcapping, GQA/MQA, and
    inplace KV cache update.

    See sgl-kernel documentation for full argument descriptions.
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)

    q, k_cache, k, v = [maybe_contiguous(x) for x in (q, k_cache, k, v)]
    v_cache = (
        v_cache.contiguous()
        if v_cache.stride(-1) != 1 and v_cache.stride(-3) != 1
        else v_cache
    )
    cu_seqlens_q, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k_new)
    ]
    page_table, cache_batch_idx, cache_leftpad = [
        maybe_contiguous(x) for x in (page_table, cache_batch_idx, cache_leftpad)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    rotary_seqlens = maybe_contiguous(rotary_seqlens)
    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    out, softmax_lse, *rest = ops.fwd(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
        sinks,
    )
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
):
    if not is_fa3_supported():
        raise NotImplementedError(
            "sgl_flash_attn3 is only supported on sm80 and above with CUDA >= 12.3"
        )

    if max_seqlen_q is None or max_seqlen_k is None:
        raise ValueError("max_seqlen_q and max_seqlen_k are required for FA3")

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )
    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    out, softmax_lse, *rest = ops.fwd(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        qv,
        None,  # out
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        None,  # page_table
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,  # rotary cos
        None,  # rotary sin
        None,  # seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        False,  # is_rotary_interleaved
        None,  # scheduler_metadata
        num_splits,
        pack_gqa,
        sm_margin,
        sinks,
    )

    return (out, softmax_lse, *rest) if return_softmax_lse else out
