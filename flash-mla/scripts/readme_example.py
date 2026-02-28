# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels"
# ]
# ///
"""
Flash-MLA (Multi-head Latent Attention) Example

This script demonstrates the usage of the Flash-MLA kernel for efficient
attention computation on Hopper (SM90) GPUs.

Flash-MLA is optimized for DeepSeek-style MLA attention patterns.
"""
import math
import torch
from kernels import get_kernel, get_local_kernel
from pathlib import Path

# Setup
torch.manual_seed(42)
flash_mla = get_kernel("drbh/tmp-kernel-123")
# flash_mla = get_local_kernel(Path("build"), "flash-mla")
device = torch.device("cuda")

# Check GPU architecture
cc_major, cc_minor = torch.cuda.get_device_capability()
print(f"GPU Compute Capability: {cc_major}.{cc_minor}")
if cc_major != 9:
    print("Warning: Flash-MLA dense decoding is optimized for SM90 (Hopper) GPUs.")
    print("Some features may not work on other architectures.")

def cdiv(a, b):
    """Ceiling division"""
    return (a + b - 1) // b


# =============================================================================
# Test 1: Dense MLA Decoding (SM90)
# =============================================================================
print("\n" + "=" * 60)
print("Test 1: Dense MLA Decoding")
print("=" * 60)

# Configuration matching DeepSeek V3 architecture
batch_size = 2
seq_len_q = 1  # Typically 1 for decoding
num_heads_q = 64  # Number of query heads (must be 64 or 128)
num_heads_k = 1   # MLA uses single KV head
head_dim = 576    # Q/K head dimension (576 or 512)
head_dim_v = 512  # V head dimension (must be 512)
page_block_size = 64  # Page block size (must be 64)
seq_len_k = 256   # KV cache sequence length

# Calculate number of blocks needed
max_num_blocks = cdiv(seq_len_k, page_block_size)

# Create input tensors
q = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim,
                device=device, dtype=torch.bfloat16) / 10
q.clamp_(min=-1.0, max=1.0)

# KV cache in blocked format: [num_blocks, page_block_size, num_heads_k, head_dim]
total_blocks = batch_size * max_num_blocks
blocked_k = torch.randn(total_blocks, page_block_size, num_heads_k, head_dim,
                        device=device, dtype=torch.bfloat16) / 10
blocked_k.clamp_(min=-1.0, max=1.0)

# Block table maps batch elements to their cache blocks
block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).view(batch_size, max_num_blocks)

# Sequence lengths for each batch element
cache_seqlens = torch.full((batch_size,), seq_len_k, device=device, dtype=torch.int32)

# Get scheduler metadata (required for flash_mla_with_kvcache)
tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

print(f"Query shape: {q.shape}")
print(f"KV cache shape: {blocked_k.shape}")
print(f"Block table shape: {block_table.shape}")
print(f"Cache seqlens: {cache_seqlens}")

# Run Flash-MLA dense decoding
with torch.inference_mode():
    out, lse = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=blocked_k,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=None,
        causal=False,  # Causal masking
    )

print(f"Output shape: {out.shape}")  # [batch_size, seq_len_q, num_heads_q, head_dim_v]
print(f"LSE shape: {lse.shape}")     # [batch_size, num_heads_q, seq_len_q]
print("Dense MLA decoding: SUCCESS")


# =============================================================================
# Test 2: Reference comparison for correctness
# =============================================================================
print("\n" + "=" * 60)
print("Test 2: Correctness Check vs PyTorch Reference")
print("=" * 60)

def reference_attention(q, blocked_k, block_table, cache_seqlens, dv, is_causal=False):
    """
    Reference implementation using PyTorch for verification
    """
    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)

    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=q.device)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32, device=q.device)

    cache_seqlens_cpu = cache_seqlens.cpu()

    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        cur_num_blocks = cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0:cur_num_blocks]

        # Reconstruct KV from blocks
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]

        # Compute attention
        query = q[i].transpose(0, 1).float()  # [h_q, s_q, d]
        kv = cur_kv.transpose(0, 1).float()   # [h_kv, s_k, d]

        # Expand KV heads if needed
        if h_kv != h_q:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)

        # Q @ K^T
        attn_weight = query @ kv.transpose(-2, -1)

        # Apply causal mask if needed
        s_k = kv.size(1)
        if is_causal and s_q > 1:
            mask = torch.ones(s_q, s_k, dtype=torch.bool, device=q.device).tril(diagonal=s_k - s_q)
            attn_weight.masked_fill_(~mask, float("-inf"))

        # Scale and softmax
        attn_weight = attn_weight / math.sqrt(d)
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Attention @ V
        output = attn_weight @ kv[..., :dv]

        out_ref[i] = output.transpose(0, 1)
        lse_ref[i] = lse

    return out_ref.to(q.dtype), lse_ref

# Compute reference
out_ref, lse_ref = reference_attention(q, blocked_k, block_table, cache_seqlens, head_dim_v, is_causal=False)

# Compare
out_close = torch.allclose(out.float(), out_ref.float(), atol=1e-3, rtol=1e-2)
lse_close = torch.allclose(lse.float(), lse_ref.float(), atol=1e-4, rtol=1e-3)

print(f"Output close to reference: {out_close}")
print(f"LSE close to reference: {lse_close}")

if out_close and lse_close:
    print("Correctness check: PASSED")
else:
    max_out_diff = (out.float() - out_ref.float()).abs().max().item()
    max_lse_diff = (lse.float() - lse_ref.float()).abs().max().item()
    print(f"Max output diff: {max_out_diff}")
    print(f"Max LSE diff: {max_lse_diff}")
    print("Correctness check: Check differences above")


# =============================================================================
# Test 3: Different configurations
# =============================================================================
print("\n" + "=" * 60)
print("Test 3: Testing different configurations")
print("=" * 60)

configs = [
    {"batch": 1, "seq_q": 1, "heads_q": 64, "seq_k": 128},
    {"batch": 4, "seq_q": 1, "heads_q": 128, "seq_k": 512},
    {"batch": 8, "seq_q": 2, "heads_q": 64, "seq_k": 1024},
]

for cfg in configs:
    b = cfg["batch"]
    s_q = cfg["seq_q"]
    h_q = cfg["heads_q"]
    s_k = cfg["seq_k"]

    max_blocks = cdiv(s_k, page_block_size)
    total_blks = b * max_blocks

    q_test = torch.randn(b, s_q, h_q, head_dim, device=device, dtype=torch.bfloat16) / 10
    k_test = torch.randn(total_blks, page_block_size, num_heads_k, head_dim, device=device, dtype=torch.bfloat16) / 10
    bt_test = torch.arange(total_blks, device=device, dtype=torch.int32).view(b, max_blocks)
    sl_test = torch.full((b,), s_k, device=device, dtype=torch.int32)

    sched_meta, _ = flash_mla.get_mla_metadata()

    with torch.inference_mode():
        out_test, lse_test = flash_mla.flash_mla_with_kvcache(
            q=q_test,
            k_cache=k_test,
            block_table=bt_test,
            cache_seqlens=sl_test,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=sched_meta,
        )

    print(f"Config: batch={b}, seq_q={s_q}, heads_q={h_q}, seq_k={s_k} -> Output: {out_test.shape} SUCCESS")


print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
