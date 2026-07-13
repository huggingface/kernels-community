"""FlashAttention forward entry points backed by aiter's Composable-Kernel FMHA.

This is the compiled (HIP / Composable Kernel) counterpart to the Triton
``aiter-flash-attn`` package. Only the *forward* pass is provided (dense and
variable-length), including support for learnable attention sinks via
``window_size[2]`` / ``s_aux`` (e.g. gpt-oss). The op set ships exclusively
the CK ``mha_fwd`` / ``mha_varlen_fwd`` kernels, so the sink path is always
taken (no fall-through to a non-sink ASM kernel).
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from ._ops import ops


def _round_multiple(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _maybe_pad_headdim(t: Tensor, target: int) -> Tensor:
    og = t.size(-1)
    if og == target:
        return t
    return torch.nn.functional.pad(t, [0, target - og])


def _unpack_window(window_size) -> Tuple[int, int, int]:
    left = int(window_size[0])
    right = int(window_size[1])
    sink = int(window_size[2]) if len(window_size) == 3 else 0
    return left, right, sink


def _prep_sink(s_aux: Optional[Tensor], nheads: int, device) -> Optional[Tensor]:
    if s_aux is None:
        return None
    assert s_aux.device == device, "s_aux must be on the same device as q"
    assert s_aux.shape[0] == nheads, "s_aux must have shape [nheads_q]"
    if s_aux.dtype != torch.float32:
        s_aux = s_aux.to(torch.float32)
    return s_aux


def _effective_sink_size(sink_size: int, s_aux: Optional[Tensor]) -> int:
    # The CK kernel only takes the sink path when sink_size > 0. Callers using
    # the transformers attention interface pass the per-head sink logits as
    # ``s_aux`` but leave ``window_size`` a 2-tuple (sink_size == 0), so enable
    # a single learnable sink automatically when s_aux is given.
    if s_aux is not None and sink_size == 0:
        return 1
    return sink_size


def flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, ...] = (-1, -1, 0),
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    s_aux: Optional[Tensor] = None,
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Dense FlashAttention forward.

    ``q, k, v`` have shape ``(batch, seqlen, nheads, headdim)``. ``window_size``
    is ``(left, right, sink_size)``; ``s_aux`` is an optional ``(nheads_q,)``
    fp32 tensor of learnable sink logits. Extra keyword arguments (e.g. those
    passed by the transformers flash-attention interface) are ignored.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    window_size_left, window_size_right, sink_size = _unpack_window(window_size)

    nheads_q = q.size(2)
    s_aux = _prep_sink(s_aux, nheads_q, q.device)
    sink_size = _effective_sink_size(sink_size, s_aux)

    head_q_og = q.size(-1)
    head_v_og = v.size(-1)
    head_q = _round_multiple(head_q_og, 8)
    head_v = _round_multiple(head_v_og, 8)
    q = _maybe_pad_headdim(q, head_q)
    k = _maybe_pad_headdim(k, head_q)
    v = _maybe_pad_headdim(v, head_v)

    return_softmax = return_attn_probs and dropout_p > 0
    out, softmax_lse, S_dmask, _ = ops.mha_fwd(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        sink_size,
        return_lse or return_attn_probs,
        return_softmax,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_kv
        None,  # out
        bias,
        alibi_slopes,
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        s_aux,
        None,  # gen
    )

    out = out[..., :head_v_og]
    if not (return_lse or return_attn_probs):
        return out
    result = (out,)
    if return_lse:
        result = result + (softmax_lse,)
    if return_attn_probs:
        result = result + (S_dmask,)
    return result


def flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, ...] = (-1, -1, 0),
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[Tensor] = None,
    min_seqlen_q: int = 0,
    logits_soft_cap: float = 0.0,
    zero_tensors: bool = False,
    cu_seqlens_q_padded: Optional[Tensor] = None,
    cu_seqlens_k_padded: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Variable-length / packed FlashAttention forward.

    ``q`` has shape ``(total_q, nheads, headdim)``; ``cu_seqlens_*`` are int32
    cumulative offsets of shape ``(batch + 1,)``. Sinks are supported through
    ``window_size[2]`` / ``s_aux`` as in :func:`flash_attn_func`. Extra keyword
    arguments (e.g. from the transformers flash-attention interface) are ignored.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    window_size_left, window_size_right, sink_size = _unpack_window(window_size)

    nheads_q = q.size(1)
    s_aux = _prep_sink(s_aux, nheads_q, q.device)
    sink_size = _effective_sink_size(sink_size, s_aux)

    head_q_og = q.size(-1)
    head_v_og = v.size(-1)
    head_q = _round_multiple(head_q_og, 8)
    head_v = _round_multiple(head_v_og, 8)
    q = _maybe_pad_headdim(q, head_q)
    k = _maybe_pad_headdim(k, head_q)
    v = _maybe_pad_headdim(v, head_v)

    return_softmax = return_attn_probs and dropout_p > 0
    out, softmax_lse, S_dmask, _ = ops.mha_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        sink_size,
        return_lse or return_attn_probs,
        return_softmax,
        None,  # out
        block_table,
        bias,
        alibi_slopes,
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        None,  # gen
        cu_seqlens_q_padded,
        cu_seqlens_k_padded,
        s_aux,
    )

    out = out[..., :head_v_og]
    if not (return_lse or return_attn_probs):
        return out
    result = (out,)
    if return_lse:
        result = result + (softmax_lse,)
    if return_attn_probs:
        result = result + (S_dmask,)
    return result


def _append_to_contiguous_cache(k_cache, v_cache, k, v, cache_seqlens, Sq):
    """In-place append of new k/v into a contiguous cache at per-batch offsets."""
    B = k_cache.shape[0]
    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full((B,), cache_seqlens, device=k_cache.device, dtype=torch.long)
    pos = cache_seqlens.view(B, 1).to(torch.long) + torch.arange(Sq, device=k_cache.device).view(1, Sq)
    bidx = torch.arange(B, device=k_cache.device).view(B, 1).expand(B, Sq)
    k_cache[bidx, pos] = k.to(k_cache.dtype)
    v_cache[bidx, pos] = v.to(v_cache.dtype)


def flash_attn_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k: Optional[Tensor] = None,
    v: Optional[Tensor] = None,
    rotary_cos: Optional[Tensor] = None,
    rotary_sin: Optional[Tensor] = None,
    cache_seqlens=None,
    block_table: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, ...] = (-1, -1, 0),
    s_aux: Optional[Tensor] = None,
    return_lse: bool = False,
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """FlashAttention against a KV cache, built on the CK split-KV kernel.

    ``q`` is ``(batch, seqlen_q, nheads, headdim)``. For a contiguous cache,
    ``k_cache``/``v_cache`` are ``(batch, seqlen_cache, nheads_k, headdim)`` and
    ``cache_seqlens`` (int or ``(batch,)``) is the number of valid cache entries
    *before* appending. If ``k``/``v`` (the new tokens) are given, they are
    written into the cache in place. The query attends to the valid prefix
    (``cache_seqlens`` + ``seqlen_q``) with bottom-right causal alignment.

    Notes / current limitations:
    - Rotary embedding is not applied here; pass ``q``/``k`` already rotated
      (``rotary_cos``/``rotary_sin`` are rejected).
    - Paged caches are addressed via ``block_table`` (forwarded to the split-KV
      varlen kernel); in-cache append for paged layouts must be done by the
      caller (pass ``k=v=None``).
    """
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError(
            "flash_attn_with_kvcache does not apply rotary; rotate q/k beforehand."
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    window_size_left, window_size_right, sink_size = _unpack_window(window_size)
    B, Sq, Hq, D = q.shape
    s_aux = _prep_sink(s_aux, Hq, q.device)
    sink_size = _effective_sink_size(sink_size, s_aux)

    paged = block_table is not None

    # ---- append new tokens into the cache (contiguous layout only) ----
    if k is not None and not paged:
        assert cache_seqlens is not None, "cache_seqlens required to append into the cache"
        _append_to_contiguous_cache(k_cache, v_cache, k, v, cache_seqlens, Sq)
    elif k is not None and paged:
        raise NotImplementedError(
            "in-cache append for paged layouts is not supported; append externally "
            "and call with k=v=None."
        )

    appended = 0 if k is None else Sq

    # ---- key lengths ----
    if cache_seqlens is None:
        klens = torch.full((B,), k_cache.shape[1], device=q.device, dtype=torch.long)
    elif isinstance(cache_seqlens, int):
        klens = torch.full((B,), cache_seqlens + appended, device=q.device, dtype=torch.long)
    else:
        klens = cache_seqlens.to(q.device, torch.long) + appended

    uniform = bool((klens == klens[0]).all().item())

    # ---- fast path: contiguous cache, uniform length -> dense kernel ----
    if not paged and uniform:
        L = int(klens[0].item())
        return flash_attn_func(
            q, k_cache[:, :L], v_cache[:, :L],
            softmax_scale=softmax_scale, causal=causal,
            window_size=(window_size_left, window_size_right, sink_size),
            s_aux=s_aux, return_lse=return_lse,
        )

    # ---- general path: variable cache lengths -> split-KV varlen kernel ----
    cu_q = torch.arange(0, (B + 1) * Sq, Sq, device=q.device, dtype=torch.int32)
    cu_k = torch.zeros(B + 1, device=q.device, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(klens, 0).to(torch.int32)
    q_packed = q.reshape(B * Sq, Hq, D)
    if paged:
        k_packed, v_packed = k_cache, v_cache  # kernel gathers via block_table
    else:
        k_packed = torch.cat([k_cache[b, : int(klens[b])] for b in range(B)], dim=0)
        v_packed = torch.cat([v_cache[b, : int(klens[b])] for b in range(B)], dim=0)

    out = flash_attn_varlen_func(
        q_packed, k_packed, v_packed, cu_q, cu_k, Sq, int(klens.max().item()),
        softmax_scale=softmax_scale, causal=causal,
        window_size=(window_size_left, window_size_right, sink_size),
        block_table=block_table, s_aux=s_aux, return_lse=return_lse,
    )
    if return_lse:
        out, lse = out
        return out.reshape(B, Sq, Hq, D), lse
    return out.reshape(B, Sq, Hq, D)
