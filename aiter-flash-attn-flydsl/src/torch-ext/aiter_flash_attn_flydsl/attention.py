"""Compile-friendly attention adapted from xDiT's AITER FlyDSL backend.

xDiT revision 4ac65db2fa4add8ff5426cfbb021d96d0be6eab0, Apache-2.0.
The Hub builder supplies a distinct operator namespace for each loaded build.
"""

import torch

from ._ops import add_op_namespace_prefix
from .fmha import flydsl_flash_attn_func
from .fp8_quant import flydsl_fp8_quant
from .selection import fp8_eligible


@torch.library.custom_op(add_op_namespace_prefix("attention"), mutates_args=())
def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    return flydsl_flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)


@_attention.register_fake
def _attention_fake(q, k, v, causal, softmax_scale):
    return q.new_empty(q.shape)


@torch.library.custom_op(add_op_namespace_prefix("attention_fp8"), mutates_args=())
def _attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=True)
    return flydsl_flash_attn_func(
        q8,
        k8,
        v8,
        causal=causal,
        softmax_scale=softmax_scale,
        q_descale=sq,
        k_descale=sk,
        v_descale=sv,
    )


@_attention_fp8.register_fake
def _attention_fp8_fake(q, k, v, causal, softmax_scale):
    return q.new_empty(q.shape, dtype=torch.bfloat16)


@torch.library.custom_op(
    add_op_namespace_prefix("attention_fp8_prequant"), mutates_args=()
)
def _attention_fp8_prequant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sq: torch.Tensor,
    sk: torch.Tensor,
    sv: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    return flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        softmax_scale=softmax_scale,
        q_descale=sq,
        k_descale=sk,
        v_descale=sv,
    )


@_attention_fp8_prequant.register_fake
def _attention_fp8_prequant_fake(q, k, v, sq, sk, sv, causal, softmax_scale):
    return q.new_empty(q.shape, dtype=torch.bfloat16)


def attention(
    q,
    k,
    v,
    causal=False,
    *,
    fp8=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
):
    """BSHD forward attention with full-graph torch.compile support.

    fp8=True forces BF16-to-FP8 quantization with Q/K rotation; fp8="auto"
    uses xDiT's measured shape crossovers. Unknown shapes retain input precision.
    FP8 inputs use supplied descales and are already quantized/rotated by caller.
    The raw flash_attn_func API provides stream, scale, and output-buffer options.
    """
    if fp8 not in (False, True, "auto"):
        raise ValueError('fp8 must be False, True, or "auto"')
    if q.dtype == torch.float8_e4m3fn:
        if fp8 is not False:
            raise ValueError("FP8 inputs are already quantized; use fp8=False")
        if q_descale is None or k_descale is None or v_descale is None:
            raise ValueError("FP8 inputs require q/k/v descales")
        return _attention_fp8_prequant(
            q, k, v, q_descale, k_descale, v_descale, causal, softmax_scale
        )
    if any(scale is not None for scale in (q_descale, k_descale, v_descale)):
        raise ValueError("descales are only accepted for FP8 inputs")
    use_fp8 = fp8 is True or (
        fp8 == "auto"
        and not causal
        and fp8_eligible(q.dtype, q.shape[1], k.shape[1], q.shape[2], q.shape[3])
    )
    if use_fp8:
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("FP8 quantization requires matching BF16 q/k/v")
        return _attention_fp8(q, k, v, causal, softmax_scale)
    return _attention(q, k, v, causal, softmax_scale)
