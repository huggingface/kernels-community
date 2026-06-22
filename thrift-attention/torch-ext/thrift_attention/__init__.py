import math

import torch

from ._ops import ops


BLOCK_SIZE = 64
DEFAULT_FRACTION = 0.05


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    check_flash_options(
        q,
        dropout_p,
        softmax_scale,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
    )
    check_qkv(q, k, v)

    q_hnd = q.transpose(1, 2).contiguous()
    k_hnd = k.transpose(1, 2).contiguous()
    v_hnd = v.transpose(1, 2).contiguous()

    if q_hnd.shape[2] == 1:
        out = single_query_attention(q_hnd, k_hnd, v_hnd)
    else:
        out = tiled_attention(q_hnd, k_hnd, v_hnd, causal)
    return out.transpose(1, 2).contiguous()


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    if block_table is not None:
        raise NotImplementedError("ThriftAttention does not support paged varlen block tables")
    check_flash_options(
        q,
        dropout_p,
        softmax_scale,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
    )
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("varlen q, k, and v must be shaped [total_tokens, heads, head_dim]")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1 tensors")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")

    outputs = []
    for idx in range(cu_seqlens_q.numel() - 1):
        q_start = int(cu_seqlens_q[idx].item())
        q_end = int(cu_seqlens_q[idx + 1].item())
        k_start = int(cu_seqlens_k[idx].item())
        k_end = int(cu_seqlens_k[idx + 1].item())
        out = flash_attn_func(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
        outputs.append(out.squeeze(0))

    if outputs:
        return torch.cat(outputs, dim=0)
    return q.new_empty(q.shape)


def check_flash_options(q, dropout_p, softmax_scale, window_size, softcap, alibi_slopes, return_attn_probs):
    if dropout_p != 0.0:
        raise NotImplementedError("ThriftAttention only supports dropout_p=0.0")
    if softmax_scale is not None:
        expected = q.shape[-1] ** -0.5
        if not math.isclose(float(softmax_scale), expected, rel_tol=1e-5, abs_tol=1e-8):
            raise NotImplementedError("ThriftAttention only supports the default softmax scale")
    if window_size != (-1, -1):
        raise NotImplementedError("ThriftAttention does not support sliding-window attention")
    if softcap != 0.0:
        raise NotImplementedError("ThriftAttention does not support softcap")
    if alibi_slopes is not None:
        raise NotImplementedError("ThriftAttention does not support ALiBi slopes")
    if return_attn_probs:
        raise NotImplementedError("ThriftAttention does not return attention probabilities")


def check_qkv(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be shaped [batch, seq, heads, head_dim]")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q, k, and v must be float16 or bfloat16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same sequence length")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same KV head count")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("q heads must be divisible by KV heads")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("q, k, and v must have the same head_dim")
    if q.shape[3] not in (64, 128):
        raise ValueError("ThriftAttention supports head_dim 64 or 128")


def single_query_attention(q, k, v):
    if k.shape[2] % BLOCK_SIZE != 0:
        raise ValueError("ThriftAttention single-query attention requires KV length divisible by 64")

    is_bf16 = q.dtype == torch.bfloat16
    batch, q_heads, _, head_dim = q.shape
    kv_heads = k.shape[1]
    groups = q_heads // kv_heads
    if groups > 16:
        raise NotImplementedError("ThriftAttention single-query kernels support at most 16 Q heads per KV head")

    q_grouped = q.reshape(batch, kv_heads, groups, head_dim).contiguous()
    num_kv_blocks = k.shape[2] // BLOCK_SIZE
    selected = select_key_blocks(q_grouped, k, num_kv_blocks, is_bf16)

    q_packed, q_scale = ops.nvfp4_quantize(q_grouped, is_bf16)
    k_packed, k_scale = ops.nvfp4_quantize(k.contiguous(), is_bf16)
    v_packed_t, v_scale_t = ops.nvfp4_quantize_transposed(v.contiguous(), is_bf16)
    out = ops.thrift_attention_single_query_nvfp4_packed(
        q_grouped,
        k.contiguous(),
        v.contiguous(),
        selected,
        q_packed,
        k_packed,
        v_packed_t,
        q_scale,
        k_scale,
        v_scale_t,
        is_bf16,
    )
    return out.reshape(batch, q_heads, 1, head_dim).contiguous()


def tiled_attention(q, k, v, causal):
    original_q_len = q.shape[2]
    q, k, v = pad_tiled_inputs(q, k, v, causal)

    is_bf16 = q.dtype == torch.bfloat16
    selected = select_block_pairs(q, k, causal, is_bf16)

    q_packed, q_scale = ops.nvfp4_quantize(q.contiguous(), is_bf16)
    k_packed, k_scale = ops.nvfp4_quantize_permuted(k.contiguous(), is_bf16)
    v_packed_t, v_scale_t = ops.nvfp4_quantize_transposed(v.contiguous(), is_bf16)
    if causal:
        attention = ops.thrift_attention_causal_nvfp4_packed
    else:
        attention = ops.thrift_attention_noncausal_nvfp4_packed
    out = attention(
        q,
        k,
        v,
        selected,
        q_packed,
        k_packed,
        v_packed_t,
        q_scale,
        k_scale,
        v_scale_t,
        is_bf16,
    )
    return out[:, :, :original_q_len, :].contiguous()


def pad_tiled_inputs(q, k, v, causal):
    q_len = q.shape[2]
    kv_len = k.shape[2]
    if q_len == kv_len and q_len % BLOCK_SIZE == 0:
        return q.contiguous(), k.contiguous(), v.contiguous()
    if q_len == kv_len and causal:
        padded_len = ((q_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        if padded_len == q_len:
            return q.contiguous(), k.contiguous(), v.contiguous()
        pad = padded_len - q_len
        q_pad = q.new_zeros(q.shape[0], q.shape[1], pad, q.shape[3])
        k_pad = k.new_zeros(k.shape[0], k.shape[1], pad, k.shape[3])
        v_pad = v.new_zeros(v.shape[0], v.shape[1], pad, v.shape[3])
        return (
            torch.cat((q, q_pad), dim=2).contiguous(),
            torch.cat((k, k_pad), dim=2).contiguous(),
            torch.cat((v, v_pad), dim=2).contiguous(),
        )
    if q_len % BLOCK_SIZE == 0 and kv_len % BLOCK_SIZE == 0:
        return q.contiguous(), k.contiguous(), v.contiguous()
    raise ValueError("ThriftAttention tiled attention requires block-aligned sequence lengths")


def select_block_pairs(q, k, causal, is_bf16):
    num_kv_blocks = k.shape[2] // BLOCK_SIZE
    selected_count = resolve_top_k(num_kv_blocks, causal)
    q_mean = block_means(q, is_bf16)
    k_mean = block_means(k, is_bf16)
    if q.shape[1] != k.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k_mean = k_mean.repeat_interleave(groups, dim=1).contiguous()
    if num_kv_blocks <= 2048:
        return ops.block_mean_topk(q_mean, k_mean, selected_count, causal, is_bf16)

    scores = (
        q_mean.reshape(q.shape[0] * q.shape[1], q_mean.shape[2], q.shape[3]).float()
        @ k_mean.reshape(q.shape[0] * q.shape[1], num_kv_blocks, q.shape[3])
        .float()
        .transpose(-1, -2)
    )
    if causal:
        mask = torch.triu(
            torch.ones(q_mean.shape[2], num_kv_blocks, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    indices = scores.topk(selected_count, dim=-1).indices.to(torch.int32)
    if causal:
        valid_counts = torch.arange(1, q_mean.shape[2] + 1, device=q.device).clamp(max=num_kv_blocks)
        ranks = torch.arange(selected_count, device=q.device)
        indices.masked_fill_(ranks.view(1, 1, -1) >= valid_counts.view(1, -1, 1), -1)
    return indices.contiguous()


def select_key_blocks(q_grouped, k, num_kv_blocks, is_bf16):
    selected_count = resolve_top_k(num_kv_blocks, False)
    k_mean = block_means(k, is_bf16)
    if num_kv_blocks <= 2048:
        return ops.single_query_key_mean_topk(q_grouped, k_mean, selected_count, num_kv_blocks, is_bf16)
    scores = (q_grouped.float().unsqueeze(3) * k_mean.float().unsqueeze(2)).sum(dim=-1)
    scores = scores.amax(dim=2).reshape(q_grouped.shape[0] * q_grouped.shape[1], num_kv_blocks)
    return scores.topk(selected_count, dim=-1).indices.to(torch.int32).contiguous()


def block_means(x, is_bf16):
    batch, heads, seq_len, head_dim = x.shape
    out_dtype = torch.bfloat16 if is_bf16 else torch.float16
    return (
        x.reshape(batch, heads, seq_len // BLOCK_SIZE, BLOCK_SIZE, head_dim)
        .float()
        .mean(dim=3)
        .to(out_dtype)
        .contiguous()
    )


def resolve_top_k(num_blocks, causal):
    if num_blocks <= 0:
        return 0
    if not causal:
        return max(1, min(num_blocks, round(DEFAULT_FRACTION * num_blocks)))

    b = -(2 * num_blocks + 1)
    c = DEFAULT_FRACTION * num_blocks * (num_blocks + 1)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return num_blocks
    top_k_float = (-b - math.sqrt(discriminant)) / 2.0
    return max(1, min(num_blocks, round(top_k_float)))
