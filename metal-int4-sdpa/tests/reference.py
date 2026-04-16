"""
Pure PyTorch reference implementation of fused int4 SDPA.

This implements scaled dot-product attention where K and V are stored
in int4 format (4-bit quantized with per-group scale+bias). The key
insight is that we dequantize K/V on-the-fly during attention computation,
never materializing the full fp32/fp16 K/V matrices.

int4 format:
  - uint32 packs 8 x 4-bit values
  - group_size=64: every 64 elements share one (scale, bias) pair
  - dequantize: value = scale * nibble + bias

Attention:
  score_i = (Q * scale) @ dequantize(K_i)
  output = softmax(scores) @ dequantize(V)

Uses online softmax (Milakov & Gimelshein 2018) to avoid materializing
the full score matrix — O(1) memory in sequence length.
"""

import torch
import torch.nn.functional as F


def quantize_int4(
    x: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize fp32 tensor to int4 with per-group scale+bias.

    Args:
        x: (..., D) float tensor where D is divisible by group_size
        group_size: number of elements per quantization group

    Returns:
        packed: (..., D//8) uint32 tensor, 8 nibbles per uint32
        scales: (..., D//group_size) float tensor
        biases: (..., D//group_size) float tensor
    """
    shape = x.shape
    D = shape[-1]
    assert D % group_size == 0
    assert D % 8 == 0

    # Reshape into groups
    x_grouped = x.reshape(*shape[:-1], D // group_size, group_size)

    # Per-group min/max
    x_min = x_grouped.min(dim=-1).values
    x_max = x_grouped.max(dim=-1).values

    # Scale and bias: value = scale * nibble + bias
    # nibble in [0, 15], so: scale = (max - min) / 15, bias = min
    scale = (x_max - x_min) / 15.0
    scale = scale.clamp(min=1e-8)
    bias = x_min

    # Quantize to [0, 15]
    x_scaled = (x_grouped - bias.unsqueeze(-1)) / scale.unsqueeze(-1)
    x_clamped = x_scaled.round().clamp(0, 15).to(torch.uint8)

    # Pack 8 nibbles into int32 (PyTorch doesn't support uint32 bitops)
    x_flat = x_clamped.reshape(*shape[:-1], D // 8, 8)
    packed = torch.zeros(*shape[:-1], D // 8, dtype=torch.int32, device=x.device)
    for i in range(8):
        packed |= x_flat[..., i].to(torch.int32) << (4 * i)

    return packed, scale, bias


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Dequantize int4 packed tensor back to float.

    Args:
        packed: (..., D//8) uint32 tensor
        scales: (..., D//group_size) float tensor
        biases: (..., D//group_size) float tensor
        group_size: quantization group size

    Returns:
        (..., D) float tensor
    """
    shape = packed.shape
    D = shape[-1] * 8

    # Unpack 8 nibbles from each uint32
    nibbles = torch.zeros(*shape, 8, dtype=torch.float32, device=packed.device)
    for i in range(8):
        nibbles[..., i] = ((packed >> (4 * i)) & 0xF).float()

    nibbles = nibbles.reshape(*shape[:-1], D)

    # Apply scale + bias per group
    nibbles_grouped = nibbles.reshape(*shape[:-1], D // group_size, group_size)
    result = scales.unsqueeze(-1) * nibbles_grouped + biases.unsqueeze(-1)
    return result.reshape(*shape[:-1], D)


def sdpa_int4_reference(
    queries: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    k_biases: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    v_biases: torch.Tensor,
    scale: float,
    gqa_factor: int = 1,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Reference: dequantize everything, then standard SDPA.

    This is the naive baseline that materializes full K and V.

    Args:
        queries: (num_heads, D) float — single-token decode query
        k_packed: (num_kv_heads, N, D//8) uint32
        k_scales: (num_kv_heads, N, D//group_size) float
        k_biases: (num_kv_heads, N, D//group_size) float
        v_packed: (num_kv_heads, N, D//8) uint32
        v_scales: (num_kv_heads, N, D//group_size) float
        v_biases: (num_kv_heads, N, D//group_size) float
        scale: attention scale factor (typically 1/sqrt(D))
        gqa_factor: num_heads // num_kv_heads for grouped-query attention
        sliding_window: if > 0, only attend to last `sliding_window` tokens

    Returns:
        (num_heads, D) float — attention output
    """
    num_heads, D = queries.shape
    num_kv_heads = k_packed.shape[0]
    N = k_packed.shape[1]

    # Full dequantization
    K = dequantize_int4(k_packed, k_scales, k_biases)  # (num_kv_heads, N, D)
    V = dequantize_int4(v_packed, v_scales, v_biases)  # (num_kv_heads, N, D)

    # GQA: expand KV heads to match query heads
    if gqa_factor > 1:
        K = K.repeat_interleave(gqa_factor, dim=0)  # (num_heads, N, D)
        V = V.repeat_interleave(gqa_factor, dim=0)

    # Scores: (num_heads, N)
    scores = scale * torch.einsum("hd,hnd->hn", queries, K)

    # Sliding window mask
    if sliding_window > 0:
        mask = torch.arange(N, device=queries.device)
        mask = (N - 1 - mask) >= sliding_window
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    # Softmax + weighted sum
    attn = F.softmax(scores, dim=-1)  # (num_heads, N)
    output = torch.einsum("hn,hnd->hd", attn, V)  # (num_heads, D)

    return output


def sdpa_int4_fused(
    queries: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    k_biases: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    v_biases: torch.Tensor,
    scale: float,
    gqa_factor: int = 1,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Fused: dequantize K/V on-the-fly with online softmax.

    This is the PyTorch equivalent of what the Metal kernel does:
    iterate over tokens one at a time, dequantize K/V in registers,
    and maintain running softmax statistics.

    Same interface as sdpa_int4_reference.
    """
    num_heads, D = queries.shape
    num_kv_heads = k_packed.shape[0]
    N = k_packed.shape[1]
    group_size = 64
    k_packed_dim = D // 8
    k_scale_dim = D // group_size

    output = torch.zeros(num_heads, D, device=queries.device, dtype=queries.dtype)

    for h in range(num_heads):
        kv_h = h // gqa_factor
        q = queries[h] * scale  # (D,)

        max_score = torch.tensor(float("-inf"), device=queries.device)
        sum_exp = torch.tensor(0.0, device=queries.device)
        acc = torch.zeros(D, device=queries.device)

        for ki in range(N):
            # Sliding window
            if sliding_window > 0 and (N - 1 - ki) >= sliding_window:
                continue

            # Dequantize K[ki] on the fly
            k_packed_row = k_packed[kv_h, ki]  # (D//8,)
            k_s = k_scales[kv_h, ki]  # (D//group_size,)
            k_b = k_biases[kv_h, ki]

            k_vec = torch.zeros(D, device=queries.device)
            for g in range(k_scale_dim):
                for j in range(group_size // 8):
                    w_idx = g * (group_size // 8) + j
                    w = k_packed_row[w_idx].item()
                    for n in range(8):
                        nibble = (w >> (4 * n)) & 0xF
                        elem_idx = g * group_size + j * 8 + n
                        k_vec[elem_idx] = k_s[g] * nibble + k_b[g]

            # Score
            score = torch.dot(q, k_vec)

            # Online softmax update
            new_max = torch.maximum(max_score, score)
            factor = torch.exp(max_score - new_max)
            exp_score = torch.exp(score - new_max)
            max_score = new_max
            sum_exp = sum_exp * factor + exp_score

            # Dequantize V[ki] on the fly
            v_packed_row = v_packed[kv_h, ki]
            v_s = v_scales[kv_h, ki]
            v_b = v_biases[kv_h, ki]

            v_vec = torch.zeros(D, device=queries.device)
            for g in range(k_scale_dim):
                for j in range(group_size // 8):
                    w_idx = g * (group_size // 8) + j
                    w = v_packed_row[w_idx].item()
                    for n in range(8):
                        nibble = (w >> (4 * n)) & 0xF
                        elem_idx = g * group_size + j * 8 + n
                        v_vec[elem_idx] = v_s[g] * nibble + v_b[g]

            # Accumulate
            acc = acc * factor + exp_score * v_vec

        # Normalize
        output[h] = acc / (sum_exp + 1e-8)

    return output


def sdpa_int4_fused_vectorized(
    queries: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    k_biases: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    v_biases: torch.Tensor,
    scale: float,
    gqa_factor: int = 1,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Vectorized fused SDPA — processes all tokens at once per head.

    This is the practical PyTorch implementation: still fused (no full K/V
    materialization across heads), but vectorized over the sequence dimension
    for each head. Matches Metal kernel semantics but uses PyTorch ops.
    """
    num_heads, D = queries.shape
    num_kv_heads = k_packed.shape[0]
    N = k_packed.shape[1]
    group_size = 64
    k_scale_dim = D // group_size

    output = torch.zeros(num_heads, D, device=queries.device, dtype=queries.dtype)

    for kv_h in range(num_kv_heads):
        # Batch-dequantize K for this KV head: (N, D)
        K = dequantize_int4(
            k_packed[kv_h], k_scales[kv_h], k_biases[kv_h]
        )
        V = dequantize_int4(
            v_packed[kv_h], v_scales[kv_h], v_biases[kv_h]
        )

        # Process all query heads that map to this KV head
        h_start = kv_h * gqa_factor
        h_end = h_start + gqa_factor

        for h in range(h_start, h_end):
            q = queries[h] * scale  # (D,)
            scores = K @ q  # (N,)

            if sliding_window > 0:
                mask = torch.arange(N, device=queries.device)
                scores[(N - 1 - mask) >= sliding_window] = float("-inf")

            attn = F.softmax(scores, dim=0)  # (N,)
            output[h] = attn @ V  # (D,)

    return output


if __name__ == "__main__":
    torch.manual_seed(42)

    # Test configuration matching Gemma 4 sliding attention layer
    num_heads = 8
    num_kv_heads = 4
    gqa_factor = num_heads // num_kv_heads
    D = 256
    N = 128
    group_size = 64

    device = "cpu"

    # Generate random Q, K, V
    Q = torch.randn(num_heads, D, device=device)
    K_full = torch.randn(num_kv_heads, N, D, device=device)
    V_full = torch.randn(num_kv_heads, N, D, device=device)
    attn_scale = 1.0 / (D**0.5)

    # Quantize K and V to int4
    k_packed, k_scales, k_biases = quantize_int4(K_full, group_size)
    v_packed, v_scales, v_biases = quantize_int4(V_full, group_size)

    # Run all three implementations
    out_ref = sdpa_int4_reference(
        Q, k_packed, k_scales, k_biases,
        v_packed, v_scales, v_biases,
        attn_scale, gqa_factor,
    )
    out_fused = sdpa_int4_fused(
        Q, k_packed, k_scales, k_biases,
        v_packed, v_scales, v_biases,
        attn_scale, gqa_factor,
    )
    out_vec = sdpa_int4_fused_vectorized(
        Q, k_packed, k_scales, k_biases,
        v_packed, v_scales, v_biases,
        attn_scale, gqa_factor,
    )

    # Compare
    diff_fused = (out_ref - out_fused).abs().max().item()
    diff_vec = (out_ref - out_vec).abs().max().item()

    print(f"Reference vs Fused (loop):       max diff = {diff_fused:.6e}")
    print(f"Reference vs Fused (vectorized): max diff = {diff_vec:.6e}")
    print(f"Shape: ({num_heads}, {D}), KV heads: {num_kv_heads}, N: {N}")
    print(f"Quantization: int4, group_size={group_size}")

    assert diff_fused < 1e-4, f"Fused loop too far off: {diff_fused}"
    assert diff_vec < 1e-4, f"Vectorized too far off: {diff_vec}"
    print("\nAll implementations match!")
