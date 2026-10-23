from .flash_attn_interface import (
    _empty_dense_output,
    _empty_like,
    _float_arg,
    bwd,
    fwd,
    fwd_kvcache,
    varlen_bwd,
    varlen_fwd,
)

# Opaque to Dynamo/AOTAutograd; the namespace comes from kernel-builder.
import torch
from typing import Optional, Tuple
from ._ops import torch_add_op_namespace_prefix as add_op_namespace_prefix


# Native calls run inside explicit autograd Functions or no_grad.
def _native_op(name, mutates_args, tags=()):
    def register(fn):
        qualified = add_op_namespace_prefix(name)
        torch.library.define(
            qualified,
            torch.library.infer_schema(fn, mutates_args=mutates_args),
            tags=(torch.Tag.pt2_compliant_tag, *tags),
        )
        torch.library.impl(qualified, "cuda", fn)
        namespace, opname = qualified.split("::")
        return getattr(getattr(torch.ops, namespace), opname).default

    return register


# A fresh CPU token per compiled call keeps seeded RNG ops distinct.
@_native_op("_dense_forward", (), tags=(torch.Tag.nondeterministic_seeded,))
def _dense_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    scale: float,
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    slopes: Optional[torch.Tensor],
    return_probs: bool,
    call_token: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        fwd(
            q,
            k,
            v,
            alibi_slopes=slopes,
            p_dropout=dropout_p,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=left,
            window_size_right=right,
            softcap=softcap,
            return_softmax=return_probs,
        )
    )


@torch.library.register_fake(add_op_namespace_prefix("_dense_forward"))
def _dense_forward_fake(
    q,
    k,
    v,
    dropout_p,
    scale,
    causal,
    left,
    right,
    softcap,
    slopes,
    return_probs,
    call_token,
):
    b, sq, h, _ = q.shape
    p_shape = (
        (b, h, (sq + 127) // 128 * 128, (k.shape[1] + 127) // 128 * 128)
        if return_probs
        else (0,)
    )
    swap_groups = (
        sq == 1
        and h > k.shape[2]
        and dropout_p == 0
        and slopes is None
        and (left < 0 or left >= k.shape[1])
        and (right < 0 or right >= k.shape[1])
    )
    return (
        _empty_dense_output(q, k.shape[2], swap_groups),
        torch.empty((b, h, sq), device=q.device, dtype=torch.float32),
        torch.empty(p_shape, device=q.device, dtype=q.dtype),
        torch.empty(2, device=q.device, dtype=torch.int64),
    )


@_native_op("_dense_backward", ())
def _dense_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    state: torch.Tensor,
    dropout_p: float,
    scale: float,
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    slopes: Optional[torch.Tensor],
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            alibi_slopes=slopes,
            p_dropout=dropout_p,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=left,
            window_size_right=right,
            softcap=softcap,
            deterministic=deterministic,
            rng_state=state,
        )[:3]
    )


@torch.library.register_fake(add_op_namespace_prefix("_dense_backward"))
def _dense_backward_fake(
    dout,
    q,
    k,
    v,
    out,
    lse,
    state,
    dropout_p,
    scale,
    causal,
    left,
    right,
    softcap,
    slopes,
    deterministic,
):
    return _empty_like(q), _empty_like(k), _empty_like(v)


class _DenseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        packing,
    ):
        if packing == 3:
            if q.ndim != 5 or q.shape[2] != 3:
                raise ValueError("qkv must have shape [B, S, 3, H, D]")
            q, k, v = q.unbind(2)
        elif packing == 2:
            if k.ndim != 5 or k.shape[2] != 2:
                raise ValueError("kv must have shape [B, Sk, 2, Hk, D]")
            k, v = k.unbind(2)
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, v must have shape [B, S, H, D]")
        dim = q.shape[-1]
        if not 0 < dim <= 256:
            raise ValueError("head size must be in 1..256")
        scale = (
            dim**-0.5
            if softmax_scale is None
            else _float_arg(softmax_scale, "softmax_scale")
        )
        if dim % 8:
            q, k, v = (torch.nn.functional.pad(x, (0, 8 - dim % 8)) for x in (q, k, v))
        else:
            q, k, v = (x.contiguous() if x.stride(-1) != 1 else x for x in (q, k, v))
        left, right = window_size
        out, lse, probabilities, state = _dense_forward(
            q,
            k,
            v,
            dropout_p,
            scale,
            causal,
            left,
            right,
            softcap,
            alibi_slopes,
            return_attn_probs and dropout_p > 0,
            torch.empty(0, device="cpu") if torch.compiler.is_compiling() else None,
        )
        ctx.save_for_backward(q, k, v, out, lse, state, alibi_slopes)
        ctx.options = (dropout_p, scale, causal, left, right, softcap, deterministic)
        ctx.dim, ctx.packing = dim, packing
        out = out[..., :dim]
        return (out, lse, probabilities) if return_attn_probs else out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout, *diagnostic_grads):
        q, k, v, out, lse, state, slopes = ctx.saved_tensors
        p, scale, causal, left, right, cap, deterministic = ctx.options
        if ctx.dim % 8:
            dout = torch.nn.functional.pad(dout, (0, 8 - ctx.dim % 8))
        elif dout.stride(-1) != 1:
            dout = dout.contiguous()
        dq, dk, dv = _dense_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            state,
            p,
            scale,
            causal,
            left,
            right,
            cap,
            slopes,
            deterministic,
        )
        dq, dk, dv = (x[..., : ctx.dim] for x in (dq, dk, dv))
        if ctx.packing == 3:
            gradients = (torch.stack((dq, dk, dv), dim=2), None, None)
        elif ctx.packing == 2:
            gradients = (dq, torch.stack((dk, dv), dim=2), None)
        else:
            gradients = (dq, dk, dv)
        # v2 differentiates Q/K/V only; LSE/probabilities are diagnostics.
        return (*gradients, *(None,) * 9)


@_native_op("_varlen_forward", (), tags=(torch.Tag.nondeterministic_seeded,))
def _varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cuq: torch.Tensor,
    cuk: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    scale: float,
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    slopes: Optional[torch.Tensor],
    return_probs: bool,
    block_table: Optional[torch.Tensor],
    call_token: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        varlen_fwd(
            q,
            k,
            v,
            cuq,
            cuk,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            p_dropout=dropout_p,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=left,
            window_size_right=right,
            softcap=softcap,
            alibi_slopes=slopes,
            return_softmax=return_probs,
            block_table=block_table,
        )
    )


@torch.library.register_fake(add_op_namespace_prefix("_varlen_forward"))
def _varlen_forward_fake(
    q,
    k,
    v,
    cuq,
    cuk,
    max_q,
    max_k,
    dropout_p,
    scale,
    causal,
    left,
    right,
    softcap,
    slopes,
    return_probs,
    block_table,
    call_token,
):
    total, h, _ = q.shape
    p_shape = (
        (cuq.numel() - 1, h, (max_q + 127) // 128 * 128, (max_k + 127) // 128 * 128)
        if return_probs
        else (0,)
    )
    kv_heads = k.shape[2] if block_table is not None else k.shape[1]
    swap_groups = (
        max_q == 1
        and total == cuq.numel() - 1
        and h > kv_heads
        and dropout_p == 0
        and slopes is None
        and left < 0
        and right < 0
    )
    output = (
        torch.empty(q.shape, device=q.device, dtype=q.dtype)
        if swap_groups
        else _empty_like(q)
    )
    return (
        output,
        torch.empty((h, total), device=q.device, dtype=torch.float32),
        torch.empty(p_shape, device=q.device, dtype=q.dtype),
        torch.empty(2, device=q.device, dtype=torch.int64),
    )


@_native_op("_varlen_backward", ())
def _varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cuq: torch.Tensor,
    cuk: torch.Tensor,
    state: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    scale: float,
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    slopes: Optional[torch.Tensor],
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cuq,
            cuk,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            p_dropout=dropout_p,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=left,
            window_size_right=right,
            softcap=softcap,
            alibi_slopes=slopes,
            deterministic=deterministic,
            rng_state=state,
        )[:3]
    )


@torch.library.register_fake(add_op_namespace_prefix("_varlen_backward"))
def _varlen_backward_fake(
    dout,
    q,
    k,
    v,
    out,
    lse,
    cuq,
    cuk,
    state,
    max_q,
    max_k,
    dropout_p,
    scale,
    causal,
    left,
    right,
    softcap,
    slopes,
    deterministic,
):
    return _empty_like(q), _empty_like(k), _empty_like(v)


class _VarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cuq,
        cuk,
        max_q,
        max_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        slopes,
        deterministic,
        return_probs,
        block_table,
        packing,
    ):
        ctx.paged = block_table is not None
        # Dynamo rejects duplicate tensor arguments to Function.apply.
        if cuk is None:
            cuk = cuq
        if packing == 3:
            if q.ndim != 4 or q.shape[1] != 3:
                raise ValueError("qkv must have shape [total, 3, H, D]")
            q, k, v = q.unbind(1)
        elif packing == 2:
            if k.ndim != 4 or k.shape[1] != 2:
                raise ValueError("kv must have shape [total_k, 2, Hk, D]")
            k, v = k.unbind(1)
        if (
            q.ndim != 3
            or k.ndim != (4 if ctx.paged else 3)
            or v.ndim != (4 if ctx.paged else 3)
        ):
            raise ValueError(
                "q must be rank 3; k/v must be rank 4 with block_table, otherwise rank 3"
            )
        dim = q.shape[-1]
        if not 0 < dim <= 256:
            raise ValueError("head size must be in 1..256")
        scale = (
            dim**-0.5
            if softmax_scale is None
            else _float_arg(softmax_scale, "softmax_scale")
        )
        if dim % 8:
            q, k, v = (torch.nn.functional.pad(x, (0, 8 - dim % 8)) for x in (q, k, v))
        else:
            q, k, v = (x.contiguous() if x.stride(-1) != 1 else x for x in (q, k, v))
        left, right = window_size
        out, lse, probabilities, state = _varlen_forward(
            q,
            k,
            v,
            cuq,
            cuk,
            max_q,
            max_k,
            dropout_p,
            scale,
            causal,
            left,
            right,
            softcap,
            slopes,
            return_probs and dropout_p > 0,
            block_table,
            torch.empty(0, device="cpu") if torch.compiler.is_compiling() else None,
        )
        ctx.save_for_backward(q, k, v, out, lse, cuq, cuk, state, slopes)
        ctx.options = (
            max_q,
            max_k,
            dropout_p,
            scale,
            causal,
            left,
            right,
            softcap,
            deterministic,
        )
        ctx.dim, ctx.packing = dim, packing
        out = out[..., :dim]
        return (out, lse, probabilities) if return_probs else out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout, *diagnostic_grads):
        if ctx.paged:
            raise NotImplementedError(
                "paged K/V backward is not supported by the v2 API"
            )
        q, k, v, out, lse, cuq, cuk, state, slopes = ctx.saved_tensors
        max_q, max_k, p, scale, causal, left, right, cap, deterministic = ctx.options
        if ctx.dim % 8:
            dout = torch.nn.functional.pad(dout, (0, 8 - ctx.dim % 8))
        elif dout.stride(-1) != 1:
            dout = dout.contiguous()
        dq, dk, dv = _varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cuq,
            cuk,
            state,
            max_q,
            max_k,
            p,
            scale,
            causal,
            left,
            right,
            cap,
            slopes,
            deterministic,
        )
        dq, dk, dv = (x[..., : ctx.dim] for x in (dq, dk, dv))
        if ctx.packing == 3:
            gradients = (torch.stack((dq, dk, dv), dim=1), None, None)
        elif ctx.packing == 2:
            gradients = (dq, torch.stack((dk, dv), dim=1), None)
        else:
            gradients = (dq, dk, dv)
        return (*gradients, *(None,) * 14)


@_native_op("_cached_forward", ("kcache", "vcache"))
def _cached_forward(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    lengths: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    leftpad: Optional[torch.Tensor],
    table: Optional[torch.Tensor],
    slopes: Optional[torch.Tensor],
    scale: Optional[float],
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    interleaved: bool,
    splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        fwd_kvcache(
            q,
            kcache,
            vcache,
            k,
            v,
            lengths,
            cos,
            sin,
            indices,
            leftpad,
            table,
            slopes,
            None,
            scale,
            causal,
            left,
            right,
            softcap,
            interleaved,
            splits,
        )
    )


@_native_op("_cached_forward_readonly", ())
def _cached_forward_readonly(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    lengths: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    leftpad: Optional[torch.Tensor],
    table: Optional[torch.Tensor],
    slopes: Optional[torch.Tensor],
    scale: Optional[float],
    causal: bool,
    left: int,
    right: int,
    softcap: float,
    interleaved: bool,
    splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        fwd_kvcache(
            q,
            kcache,
            vcache,
            k,
            v,
            lengths,
            cos,
            sin,
            indices,
            leftpad,
            table,
            slopes,
            None,
            scale,
            causal,
            left,
            right,
            softcap,
            interleaved,
            splits,
        )
    )


@torch.library.register_fake(add_op_namespace_prefix("_cached_forward_readonly"))
@torch.library.register_fake(add_op_namespace_prefix("_cached_forward"))
def _cached_forward_fake(
    q,
    kcache,
    vcache,
    k,
    v,
    lengths,
    cos,
    sin,
    indices,
    leftpad,
    table,
    slopes,
    scale,
    causal,
    left,
    right,
    softcap,
    interleaved,
    splits,
):
    b, sq, h, d = q.shape
    if d % 8 == 0:
        swap_groups = (
            sq == 1
            and h > kcache.shape[2]
            and left < 0
            and right < 0
            and slopes is None
        )
        output = _empty_dense_output(q, kcache.shape[2], swap_groups)
    else:
        padded = torch.empty(
            (b, sq, h, (d + 7) // 8 * 8), device=q.device, dtype=q.dtype
        )
        output = padded[..., :d]
    return output, torch.empty((b, h, sq), device=q.device, dtype=torch.float32)


@torch.no_grad()
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    """Run cached inference and optionally append K/V; no backward is provided."""
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = (
        x.contiguous() if x is not None and x.stride(-1) != 1 else x for x in (q, k, v)
    )
    cache_batch_idx, block_table = (
        x.contiguous() if x is not None and x.stride(-1) != 1 else x
        for x in (cache_batch_idx, block_table)
    )
    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, device=q.device, dtype=torch.int32
        )
    # Accurate mutation metadata avoids functionalization copies.
    forward = (
        _cached_forward
        if k is not None and (k.ndim != 4 or k.shape[1] > 0)
        else _cached_forward_readonly
    )
    output, lse = forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
    )
    return (output, lse) if return_softmax_lse else output
