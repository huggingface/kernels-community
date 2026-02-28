import math
import torch

from kernels.benchmark import Benchmark


def _cdiv(a, b):
    return (a + b - 1) // b


def _extract_output(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def _reference_mla_decode(q, blocked_k, block_table, cache_seqlens, head_dim_v, causal=False):
    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)

    out = torch.empty(b, s_q, h_q, head_dim_v, dtype=torch.float32, device=q.device)

    for i in range(b):
        cur_len = int(cache_seqlens[i].item())
        num_blocks = _cdiv(cur_len, block_size)
        cur_blocks = block_table[i][:num_blocks]
        kv = blocked_k[cur_blocks].reshape(-1, h_kv, d)[:cur_len]

        query = q[i].transpose(0, 1).float()  # [h_q, s_q, d]
        key_val = kv.transpose(0, 1).float()  # [h_kv, s_k, d]

        if h_kv != h_q:
            key_val = key_val.repeat_interleave(h_q // h_kv, dim=0)

        attn = query @ key_val.transpose(-2, -1) / math.sqrt(d)

        s_k = key_val.size(1)
        if causal and s_q > 1:
            mask = torch.ones(s_q, s_k, dtype=torch.bool, device=q.device).tril(
                diagonal=s_k - s_q
            )
            attn.masked_fill_(~mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        output = attn @ key_val[..., :head_dim_v]
        out[i] = output.transpose(0, 1)

    return out.to(q.dtype)


def _varlen_reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False):
    batch_size = cu_seqlens_q.shape[0] - 1
    total_tokens_q = q.shape[0]
    num_heads = q.shape[1]
    head_dim_v = v.shape[2]
    scale = q.shape[-1] ** (-0.5)

    out = torch.zeros(
        (total_tokens_q, num_heads, head_dim_v), device=q.device, dtype=q.dtype
    )

    for b in range(batch_size):
        start_q, end_q = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        start_k, end_k = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        q_b = q[start_q:end_q].transpose(0, 1).float()  # [H, seq_q, D_qk]
        k_b = k[start_k:end_k].transpose(0, 1).float()  # [H, seq_k, D_qk]
        v_b = v[start_k:end_k].transpose(0, 1).float()  # [H, seq_k, D_v]

        attn = q_b @ k_b.transpose(-2, -1) * scale

        if causal:
            seq_q, seq_k = q_b.size(1), k_b.size(1)
            mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device).tril(
                diagonal=seq_k - seq_q
            )
            attn.masked_fill_(~mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        result = attn @ v_b  # [H, seq_q, D_v]
        out[start_q:end_q] = result.transpose(0, 1).to(q.dtype)

    return out


# MLA decode constants (DeepSeek V3 architecture)
_HEAD_DIM = 576  # Q/K head dimension
_HEAD_DIM_V = 512  # V head dimension
_NUM_HEADS_K = 1  # MLA uses single KV head
_PAGE_BLOCK_SIZE = 64  # Page block size


def _setup_mla_decode(bench, batch_size, seq_k, num_heads_q):
    max_num_blocks = _cdiv(seq_k, _PAGE_BLOCK_SIZE)
    total_blocks = batch_size * max_num_blocks

    bench.q = (
        torch.randn(
            batch_size, 1, num_heads_q, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        / 10
    )
    bench.blocked_k = (
        torch.randn(
            total_blocks,
            _PAGE_BLOCK_SIZE,
            _NUM_HEADS_K,
            _HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 10
    )
    bench.block_table = torch.arange(
        total_blocks, device="cuda", dtype=torch.int32
    ).view(batch_size, max_num_blocks)
    bench.cache_seqlens = torch.full(
        (batch_size,), seq_k, device="cuda", dtype=torch.int32
    )
    bench.tile_scheduler_metadata, _ = bench.kernel.get_mla_metadata()
    bench.out = torch.empty(
        batch_size, 1, num_heads_q, _HEAD_DIM_V, device="cuda", dtype=torch.bfloat16
    )


def _run_mla_decode(bench, causal=False):
    out, lse = bench.kernel.flash_mla_with_kvcache(
        q=bench.q,
        k_cache=bench.blocked_k,
        block_table=bench.block_table,
        cache_seqlens=bench.cache_seqlens,
        head_dim_v=_HEAD_DIM_V,
        tile_scheduler_metadata=bench.tile_scheduler_metadata,
        causal=causal,
    )
    bench.out = out


def _verify_mla_decode(bench, causal=False):
    return _reference_mla_decode(
        bench.q,
        bench.blocked_k,
        bench.block_table,
        bench.cache_seqlens,
        _HEAD_DIM_V,
        causal=causal,
    )


class FlashMLABenchmark(Benchmark):
    seed: int = 42

    # Workload: small (B=2, S_k=256, H_q=64)
    def setup_small(self):
        _setup_mla_decode(self, batch_size=2, seq_k=256, num_heads_q=64)

    def benchmark_small(self):
        _run_mla_decode(self, causal=False)

    def verify_small(self) -> torch.Tensor:
        return _verify_mla_decode(self, causal=False)

    # Workload: medium (B=4, S_k=1024, H_q=64)
    def setup_medium(self):
        _setup_mla_decode(self, batch_size=4, seq_k=1024, num_heads_q=64)

    def benchmark_medium(self):
        _run_mla_decode(self, causal=False)

    def verify_medium(self) -> torch.Tensor:
        return _verify_mla_decode(self, causal=False)

    # Workload: large (B=8, S_k=4096, H_q=128)
    def setup_large(self):
        _setup_mla_decode(self, batch_size=8, seq_k=4096, num_heads_q=128)

    def benchmark_large(self):
        _run_mla_decode(self, causal=False)

    def verify_large(self) -> torch.Tensor:
        return _verify_mla_decode(self, causal=False)


class FlashMLACausalBenchmark(Benchmark):
    seed: int = 42

    # Workload: small (B=2, S_k=256, H_q=64)
    def setup_small(self):
        _setup_mla_decode(self, batch_size=2, seq_k=256, num_heads_q=64)

    def benchmark_small(self):
        _run_mla_decode(self, causal=True)

    def verify_small(self) -> torch.Tensor:
        return _verify_mla_decode(self, causal=True)

    # Workload: medium (B=4, S_k=1024, H_q=64)
    def setup_medium(self):
        _setup_mla_decode(self, batch_size=4, seq_k=1024, num_heads_q=64)

    def benchmark_medium(self):
        _run_mla_decode(self, causal=True)

    def verify_medium(self) -> torch.Tensor:
        return _verify_mla_decode(self, causal=True)

    # Workload: large (B=8, S_k=4096, H_q=128)
    def setup_large(self):
        _setup_mla_decode(self, batch_size=8, seq_k=4096, num_heads_q=128)

    def benchmark_large(self):
        _run_mla_decode(self, causal=True)

    def verify_large(self) -> torch.Tensor:
        return _verify_mla_decode(self, causal=True)


# class FlashMLAVarlenBenchmark(Benchmark):
#     seed: int = 42

#     # Workload: small (3 sequences, max_seqlen=64)
#     def setup_small(self):
#         H, D = 8, 64
#         seqlens = [32, 48, 64]
#         total = sum(seqlens)
#         self.q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.k = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.v = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.cu_seqlens = torch.tensor(
#             [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
#             device="cuda",
#             dtype=torch.int32,
#         )
#         self.max_seqlen = max(seqlens)
#         self.out = torch.empty(total, H, D, device="cuda", dtype=torch.bfloat16)

#     def benchmark_small(self):
#         self.out = _extract_output(
#             self.kernel.flash_attn_varlen_func(
#                 self.q,
#                 self.k,
#                 self.v,
#                 self.cu_seqlens,
#                 self.cu_seqlens,
#                 self.max_seqlen,
#                 self.max_seqlen,
#             )
#         )

#     def verify_small(self) -> torch.Tensor:
#         return _varlen_reference_attention(
#             self.q, self.k, self.v, self.cu_seqlens, self.cu_seqlens, causal=False
#         )

#     # Workload: medium (5 sequences, max_seqlen=256)
#     def setup_medium(self):
#         H, D = 16, 64
#         seqlens = [128, 192, 256, 200, 150]
#         total = sum(seqlens)
#         self.q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.k = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.v = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.cu_seqlens = torch.tensor(
#             [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
#             device="cuda",
#             dtype=torch.int32,
#         )
#         self.max_seqlen = max(seqlens)
#         self.out = torch.empty(total, H, D, device="cuda", dtype=torch.bfloat16)

#     def benchmark_medium(self):
#         self.out = _extract_output(
#             self.kernel.flash_attn_varlen_func(
#                 self.q,
#                 self.k,
#                 self.v,
#                 self.cu_seqlens,
#                 self.cu_seqlens,
#                 self.max_seqlen,
#                 self.max_seqlen,
#             )
#         )

#     def verify_medium(self) -> torch.Tensor:
#         return _varlen_reference_attention(
#             self.q, self.k, self.v, self.cu_seqlens, self.cu_seqlens, causal=False
#         )

#     # Workload: large (8 sequences, max_seqlen=512)
#     def setup_large(self):
#         H, D = 32, 128
#         seqlens = [256, 384, 512, 448, 320, 480, 400, 512]
#         total = sum(seqlens)
#         self.q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.k = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.v = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
#         self.cu_seqlens = torch.tensor(
#             [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
#             device="cuda",
#             dtype=torch.int32,
#         )
#         self.max_seqlen = max(seqlens)
#         self.out = torch.empty(total, H, D, device="cuda", dtype=torch.bfloat16)

#     def benchmark_large(self):
#         self.out = _extract_output(
#             self.kernel.flash_attn_varlen_func(
#                 self.q,
#                 self.k,
#                 self.v,
#                 self.cu_seqlens,
#                 self.cu_seqlens,
#                 self.max_seqlen,
#                 self.max_seqlen,
#             )
#         )

#     def verify_large(self) -> torch.Tensor:
#         return _varlen_reference_attention(
#             self.q, self.k, self.v, self.cu_seqlens, self.cu_seqlens, causal=False
#         )
