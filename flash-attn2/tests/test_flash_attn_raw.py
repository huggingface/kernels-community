import torch
import flash_attn

# make reproducible
torch.manual_seed(0)


def _attention_torch(query, key, value, *, backend):
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(backend):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    out = out.transpose(1, 2).contiguous()
    return out


def test_flash_attn():
    """Test standard flash attention with mha_fwd"""
    print("===== Testing mha_fwd =====")

    batch_size = 1
    seq_len = 4224
    num_attention_heads = 24
    attention_head_dim = 128

    shape = (batch_size, seq_len, num_attention_heads, attention_head_dim)

    print(f"Testing shape: {shape}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Num heads: {num_attention_heads}, Head dim: {attention_head_dim}")

    query = torch.randn(shape, device="cuda", dtype=torch.float16)
    key = torch.randn(shape, device="cuda", dtype=torch.float16)
    value = torch.randn(shape, device="cuda", dtype=torch.float16)

    # Get reference implementation using PyTorch SDPA
    golden_truth = _attention_torch(
        query, key, value, backend=torch.nn.attention.SDPBackend.MATH
    )

    print(f"Golden truth shape: {golden_truth.shape}")
    print(f"Query sum: {query.sum().item()}")

    # Test non-causal flash attention
    out, softmax_lse, p, rng_state = flash_attn.fwd(
        q=query,
        k=key,
        v=value,
        is_causal=False,
    )

    print(f"Flash attention output shape: {out.shape}")
    print(f"Query sum after attention: {query.sum().item()}")

    # Compare outputs
    diff = (out - golden_truth).abs().max()
    print(f"Max absolute difference (non-causal): {diff.item()}")

    assert out.shape == shape
    assert diff < 1e-2, f"Difference too large: {diff.item()}"

    # Test causal attention
    print("\n--- Testing with causal=True ---")
    out_causal, _, _, _ = flash_attn.fwd(
        q=query,
        k=key,
        v=value,
        is_causal=True,
    )

    print(f"Causal attention output shape: {out_causal.shape}")
    assert out_causal.shape == shape

    # Compare causal vs non-causal (should be different)
    diff_causal = (out - out_causal).abs().max()
    print(f"Difference between causal and non-causal: {diff_causal.item()}")
    assert diff_causal > 1e-3, "Causal and non-causal should produce different results"

    print("✓ mha_fwd test passed!")


def test_mha_varlen_fwd():
    """Test variable-length sequences with mha_varlen_fwd"""
    print("\n===== Testing mha_varlen_fwd =====")

    # Create variable length sequences
    # Batch with 3 sequences of lengths: 512, 1024, 256
    seq_lens = [512, 1024, 256]
    total_seq_len = sum(seq_lens)
    num_attention_heads = 16
    attention_head_dim = 64

    # Create cumulative sequence lengths (required for varlen)
    cu_seqlens = torch.tensor(
        [0] + [sum(seq_lens[: i + 1]) for i in range(len(seq_lens))],
        device="cuda",
        dtype=torch.int32,
    )

    print(f"Sequence lengths: {seq_lens}")
    print(f"Cumulative sequence lengths: {cu_seqlens}")
    print(f"Total sequence length: {total_seq_len}")

    # Create packed tensors (all sequences concatenated)
    query = torch.randn(
        total_seq_len,
        num_attention_heads,
        attention_head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    key = torch.randn(
        total_seq_len,
        num_attention_heads,
        attention_head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn(
        total_seq_len,
        num_attention_heads,
        attention_head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")

    # Create reference truth by running attention on individual sequences
    # and concatenating the results
    golden_truth_parts = []
    for i, seq_len in enumerate(seq_lens):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]

        # Extract individual sequence
        q_seq = query[start_idx:end_idx].unsqueeze(0)  # Add batch dimension
        k_seq = key[start_idx:end_idx].unsqueeze(0)
        v_seq = value[start_idx:end_idx].unsqueeze(0)

        # Run reference attention on this sequence
        golden_seq = _attention_torch(
            q_seq, k_seq, v_seq, backend=torch.nn.attention.SDPBackend.MATH
        )
        golden_truth_parts.append(golden_seq.squeeze(0))  # Remove batch dimension

    # Concatenate all sequences back together
    golden_truth = torch.cat(golden_truth_parts, dim=0)
    print(f"Golden truth shape: {golden_truth.shape}")

    # Run flash attention varlen
    out, softmax_lse, p, rng_state = flash_attn.varlen_fwd(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seq_lens),
        max_seqlen_k=max(seq_lens),
        is_causal=False,
    )

    print(f"Flash attention varlen output shape: {out.shape}")
    print(f"Output should match input: {out.shape == query.shape}")

    # Compare with reference truth
    diff = (out - golden_truth).abs().max()
    print(f"Max absolute difference (non-causal): {diff.item()}")

    # Verify output shape
    assert out.shape == (total_seq_len, num_attention_heads, attention_head_dim)
    assert diff < 1e-2, f"Difference too large: {diff.item()}"

    # Test with causal attention
    print("\n--- Testing with causal=True ---")
    out_causal, _, _, _ = flash_attn.varlen_fwd(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seq_lens),
        max_seqlen_k=max(seq_lens),
        is_causal=True,
    )

    print(f"Causal attention output shape: {out_causal.shape}")
    assert out_causal.shape == (total_seq_len, num_attention_heads, attention_head_dim)

    # The causal and non-causal outputs should be different
    diff_causal = (out - out_causal).abs().max()
    print(f"Difference between causal and non-causal: {diff_causal.item()}")
    assert diff_causal > 1e-3, "Causal and non-causal should produce different results"

    print("✓ mha_varlen_fwd test passed!")


if __name__ == "__main__":
    test_flash_attn()
    test_mha_varlen_fwd()
