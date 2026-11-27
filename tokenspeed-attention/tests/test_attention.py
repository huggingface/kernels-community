import math

import pytest
import torch


def _make_cu_seqlens(seqlens, device):
    cu_cpu = [0]
    for seqlen in seqlens:
        cu_cpu.append(cu_cpu[-1] + seqlen)
    return cu_cpu, torch.tensor(cu_cpu, device=device, dtype=torch.int32)


def _varlen_attention_reference(
    q,
    k,
    v,
    cu_q,
    cu_kv,
    *,
    causal,
    scale,
):
    out = torch.empty((q.shape[0], q.shape[1], v.shape[-1]), device=q.device)
    kv_group = q.shape[1] // k.shape[1]
    for batch in range(len(cu_q) - 1):
        q_start, q_end = cu_q[batch], cu_q[batch + 1]
        kv_start, kv_end = cu_kv[batch], cu_kv[batch + 1]
        sq = q_end - q_start
        sk = kv_end - kv_start
        for head in range(q.shape[1]):
            kv_head = head // kv_group
            scores = (
                q[q_start:q_end, head].float()
                @ k[kv_start:kv_end, kv_head].float().T
            ) * scale
            if causal:
                q_idx = torch.arange(sq, device=q.device).view(-1, 1)
                k_idx = torch.arange(sk, device=q.device).view(1, -1)
                scores = scores.masked_fill(k_idx > q_idx + (sk - sq), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[q_start:q_end, head] = probs @ v[kv_start:kv_end, kv_head].float()
    return out.to(q.dtype)


def _paged_mla_reference(
    q,
    kv_cache,
    page_table,
    cache_seqlens,
    *,
    kv_lora_rank,
    qk_rope_head_dim,
    softmax_scale,
):
    out = torch.empty(q.shape[:-1] + (kv_lora_rank,), device=q.device)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    page_size = kv_cache.shape[1]
    for batch in range(q.shape[0]):
        tokens = []
        for pos in range(int(cache_seqlens[batch].item())):
            page = int(page_table[batch, pos // page_size].item())
            tokens.append(kv_cache[page, pos % page_size, 0])
        kv = torch.stack(tokens).float()
        k_latent = kv[:, :kv_lora_rank]
        k_rope = kv[:, kv_lora_rank : kv_lora_rank + qk_rope_head_dim]
        for head in range(q.shape[2]):
            q_head = q[batch, 0, head].float()
            scores = (
                k_latent @ q_head[:kv_lora_rank]
                + k_rope @ q_head[kv_lora_rank : kv_lora_rank + qk_rope_head_dim]
            ) * softmax_scale
            probs = torch.softmax(scores, dim=0)
            out[batch, 0, head] = probs @ k_latent
            lse[batch, 0, head] = torch.logsumexp(scores, dim=0)
    return out.to(q.dtype), lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_mha_prefill_matches_reference():
    from tokenspeed_attention import mha_prefill

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    seqlens = [8, 5]
    cu_cpu, cu = _make_cu_seqlens(seqlens, device)
    total = cu_cpu[-1]
    q = torch.randn(total, 4, 32, device=device, dtype=dtype)
    k = torch.randn(total, 2, 32, device=device, dtype=dtype)
    v = torch.randn(total, 2, 32, device=device, dtype=dtype)

    out = mha_prefill(q, k, v, cu, cu_cpu, max(seqlens))
    ref = _varlen_attention_reference(
        q,
        k,
        v,
        cu_cpu,
        cu_cpu,
        causal=True,
        scale=1.0 / math.sqrt(q.shape[-1]),
    )

    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_mla_decode_with_kvcache_matches_reference():
    from tokenspeed_attention import mla_decode_with_kvcache

    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    batch, q_len, heads = 2, 1, 4
    rank, rope_dim, page_size = 16, 8, 8
    cache_lens = torch.tensor([7, 11], device=device, dtype=torch.int32)
    pages_per_batch = (cache_lens + page_size - 1) // page_size
    total_pages = int(pages_per_batch.sum().item())
    max_pages = int(pages_per_batch.max().item())

    q = torch.randn(batch, q_len, heads, rank + rope_dim, device=device, dtype=dtype)
    kv_cache = torch.randn(
        total_pages,
        page_size,
        1,
        rank + rope_dim,
        device=device,
        dtype=dtype,
    )
    page_table = torch.zeros(batch, max_pages, device=device, dtype=torch.int32)
    next_page = 0
    for batch_idx, num_pages in enumerate(pages_per_batch.tolist()):
        page_table[batch_idx, :num_pages] = torch.arange(
            next_page,
            next_page + num_pages,
            device=device,
            dtype=torch.int32,
        )
        next_page += num_pages

    scale = 1.0 / math.sqrt(rank + rope_dim)
    out, lse = mla_decode_with_kvcache(
        q,
        kv_cache,
        page_table,
        cache_lens,
        int(cache_lens.max().item()),
        rank,
        rank,
        rope_dim,
        scale,
        return_lse=True,
    )
    ref, ref_lse = _paged_mla_reference(
        q,
        kv_cache,
        page_table,
        cache_lens,
        kv_lora_rank=rank,
        qk_rope_head_dim=rope_dim,
        softmax_scale=scale,
    )

    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(lse, ref_lse, atol=8e-2, rtol=8e-2)
