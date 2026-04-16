import torch
from ._ops import ops


def sdpa_int4(
    queries: torch.Tensor,
    k_quant: torch.Tensor,
    k_scales: torch.Tensor,
    k_biases: torch.Tensor,
    v_quant: torch.Tensor,
    v_scales: torch.Tensor,
    v_biases: torch.Tensor,
    gqa_factor: int,
    N: int,
    scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Fused int4 SDPA for Apple Silicon.

    Computes softmax(Q @ dequant(K_int4)^T * scale) @ dequant(V_int4)
    in a single Metal kernel dispatch with online softmax.

    Args:
        queries: (num_heads, D) float32 — single-token decode query
        k_quant: (num_kv_heads, N, D//8) uint32 — packed int4 keys
        k_scales: (num_kv_heads, N, D//64) float32 — per-group scale
        k_biases: (num_kv_heads, N, D//64) float32 — per-group bias
        v_quant: (num_kv_heads, N, D//8) uint32 — packed int4 values
        v_scales: (num_kv_heads, N, D//64) float32
        v_biases: (num_kv_heads, N, D//64) float32
        gqa_factor: num_heads // num_kv_heads
        N: sequence length
        scale: attention scale (typically 1/sqrt(D))
        sliding_window: if > 0, attend only to last `sliding_window` tokens

    Returns:
        (num_heads, D) float32 — attention output
    """
    return ops.sdpa_int4(
        queries, k_quant, k_scales, k_biases,
        v_quant, v_scales, v_biases,
        gqa_factor, N, scale, sliding_window,
    )
