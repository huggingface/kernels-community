import torch
import pytest
import metal_flash_sdpa


def create_cu_seqlens(seq_lengths):
    """Create cumulative sequence lengths tensor."""
    cu_seqlens = [0]
    for length in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return torch.tensor(cu_seqlens, dtype=torch.int32, device="mps")


def compute_attention_reference(query, key, value, scale, causal=False, softcapping=1.0, gqa_ratio=1):
    """Compute reference attention output for validation."""
    num_heads = query.shape[1]
    expected = torch.zeros_like(query)
    
    for h in range(num_heads):
        kv_h = h // gqa_ratio if gqa_ratio > 1 else h
        q_h = query[:, h, :]
        k_h = key[:, kv_h, :]
        v_h = value[:, kv_h, :]
        
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
        
        # Apply softcapping if not 1.0
        if softcapping != 1.0:
            scores = scores / softcapping
            scores = torch.tanh(scores) * softcapping
        
        # Apply causal mask if needed
        if causal:
            seq_len = query.shape[0]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device="mps"), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float("-inf"))
        
        attn_weights = torch.softmax(scores, dim=-1)
        expected[:, h, :] = torch.matmul(attn_weights, v_h)
    
    return expected


def get_tolerance(dtype, head_dim):
    """Get appropriate tolerance based on dtype and head dimension."""
    if dtype == torch.bfloat16:
        return (2e-2, 2e-2) if head_dim >= 96 else (1.6e-2, 1.6e-2)
    elif dtype == torch.float16:
        return (2e-3, 2e-3)
    else:
        return (1e-3, 1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64, 72, 80, 96, 128, 256])
@pytest.mark.parametrize("seq_config", [
    # (seq_lengths_q, seq_lengths_k, description)
    ([32], [32], "single_sequence"),
    ([8, 16, 12], [10, 20, 15], "variable_lengths"),
    ([16, 24], [16, 24], "multiple_sequences"),
    ([2], [2], "small_sequence_2"),
    ([4], [4], "small_sequence_4"),
    ([8], [8], "small_sequence_8"),
    ([16], [32], "cross_attention_q_lt_k"),
    ([32], [16], "cross_attention_q_gt_k"),
    ([8], [128], "cross_attention_large_diff"),
    ([1], [64], "single_query_token"),
])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_comprehensive(dtype, head_dim, seq_config, causal):
    """Comprehensive test for Flash Attention with various configurations."""
    torch.manual_seed(42)
    
    seq_lengths_q, seq_lengths_k, _ = seq_config
    
    # Skip causal tests for cross-attention cases
    if causal and seq_lengths_q != seq_lengths_k:
        pytest.skip("Causal attention only valid when q_seq == k_seq")
    
    # Test parameters
    num_heads = 4
    
    # Create cumulative sequence lengths
    cu_seqlens_q = create_cu_seqlens(seq_lengths_q)
    cu_seqlens_k = create_cu_seqlens(seq_lengths_k)
    
    total_q = sum(seq_lengths_q)
    total_k = sum(seq_lengths_k)
    max_seqlen_q = max(seq_lengths_q)
    max_seqlen_k = max(seq_lengths_k)
    
    # Create input tensors
    query = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device="mps")
    key = torch.randn(total_k, num_heads, head_dim, dtype=dtype, device="mps")
    value = torch.randn(total_k, num_heads, head_dim, dtype=dtype, device="mps")
    
    # Scale factor
    scale = 1.0 / (head_dim ** 0.5)
    
    # Call Flash Attention
    out = torch.empty_like(query)
    metal_flash_sdpa.flash_attention_varlen(
        out=out,
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        do_causal=causal,
        scale=scale,
        softcapping=1.0,
    )
    
    # Compute ground truth for each sequence
    expected = torch.zeros_like(out)
    batch_size = len(seq_lengths_q)
    
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i+1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i+1].item()
        
        if q_end > q_start and k_end > k_start:  # Skip empty sequences
            q_i = query[q_start:q_end]
            k_i = key[k_start:k_end]
            v_i = value[k_start:k_end]
            
            expected_i = compute_attention_reference(q_i, k_i, v_i, scale, causal=causal)
            expected[q_start:q_end] = expected_i
    
    # Check results
    rtol, atol = get_tolerance(dtype, head_dim)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64, 72, 80, 96, 128, 256])
@pytest.mark.parametrize("gqa_config", [
    # (num_heads, num_kv_heads, seq_len)
    (8, 2, 32),    # 4:1 ratio
    (16, 4, 32),   # 4:1 ratio
    (16, 8, 32),   # 2:1 ratio
    (16, 2, 32),   # 8:1 ratio
    (16, 4, 128),  # 4:1 ratio with larger sequence
])
def test_flash_attention_gqa(dtype, head_dim, gqa_config):
    """Test Flash Attention with Grouped Query Attention configurations."""
    torch.manual_seed(42)
    
    num_heads, num_kv_heads, seq_len = gqa_config
    gqa_ratio = num_heads // num_kv_heads
    
    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens([seq_len])
    
    # Create input tensors
    query = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="mps")
    key = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="mps")
    value = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="mps")
    
    # Scale factor
    scale = 1.0 / (head_dim ** 0.5)
    
    # Call Flash Attention
    out = torch.empty_like(query)
    metal_flash_sdpa.flash_attention_varlen(
        out=out,
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        do_causal=False,
        scale=scale,
        softcapping=1.0,
    )
    
    # Compute ground truth with GQA
    expected = compute_attention_reference(query, key, value, scale, gqa_ratio=gqa_ratio)
    
    # Check results
    rtol, atol = get_tolerance(dtype, head_dim)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcapping_config", [
    # (softcapping_value, seq_lengths, head_dim)
    (1.0, [32], 64),        # No softcapping
    (50.0, [32, 24], 64),   # Regular softcapping
    (10.0, [16], 128),      # Strong softcapping
    (1000.0, [16], 64),     # Very weak softcapping
    (30.0, [48], 96),       # Medium softcapping
])
def test_flash_attention_softcapping(dtype, softcapping_config):
    """Test Flash Attention with various softcapping values."""
    torch.manual_seed(42)
    
    softcapping, seq_lengths, head_dim = softcapping_config
    num_heads = 4
    
    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens(seq_lengths)
    total_tokens = sum(seq_lengths)
    max_seqlen = max(seq_lengths)
    
    # Create input tensors
    query = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="mps")
    key = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="mps")
    value = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="mps")
    
    # Scale factor
    scale = 1.0 / (head_dim ** 0.5)
    
    # Call Flash Attention with softcapping
    out = torch.empty_like(query)
    metal_flash_sdpa.flash_attention_varlen(
        out=out,
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        do_causal=False,
        scale=scale,
        softcapping=softcapping,
    )
    
    # Compute ground truth with softcapping
    expected = torch.zeros_like(query)
    
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        if end > start:
            q_seq = query[start:end]
            k_seq = key[start:end]
            v_seq = value[start:end]
            
            expected_seq = compute_attention_reference(
                q_seq, k_seq, v_seq, scale, softcapping=softcapping
            )
            expected[start:end] = expected_seq
    
    # Check results (higher tolerance for softcapping)
    # Note: Softcapping with strong values (< 50) has higher error due to 
    # the interaction between tanh transformation and exp2-based softmax
    if dtype == torch.bfloat16:
        if softcapping < 50:
            rtol, atol = 1.5e-1, 1.5e-1  # Higher tolerance for strong softcapping
        else:
            rtol, atol = 3e-2, 3e-2
    elif dtype == torch.float16:
        if softcapping < 50:
            rtol, atol = 1e-1, 1e-1
        else:
            rtol, atol = 2e-2, 2e-2
    else:
        if softcapping < 50:
            rtol, atol = 1.5e-1, 1.5e-1  # Higher tolerance for strong softcapping with float32
        else:
            rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("large_seq_config", [
    # (q_seq, k_seq, head_dim, dtype)
    (32, 2048, 64, torch.float32),
    (16, 1024, 96, torch.float16),
    (64, 1536, 64, torch.bfloat16),
])
def test_flash_attention_large_sequences(large_seq_config):
    """Test Flash Attention with large k sequences (>= 1024)."""
    torch.manual_seed(42)
    
    q_seq, k_seq, head_dim, dtype = large_seq_config
    num_heads = 4
    
    # Create cumulative sequence lengths
    cu_seqlens_q = create_cu_seqlens([q_seq])
    cu_seqlens_k = create_cu_seqlens([k_seq])
    
    # Create input tensors
    query = torch.randn(q_seq, num_heads, head_dim, dtype=dtype, device="mps")
    key = torch.randn(k_seq, num_heads, head_dim, dtype=dtype, device="mps")
    value = torch.randn(k_seq, num_heads, head_dim, dtype=dtype, device="mps")
    
    # Scale factor
    scale = 1.0 / (head_dim ** 0.5)
    
    # Call Flash Attention
    out = torch.empty_like(query)
    metal_flash_sdpa.flash_attention_varlen(
        out=out,
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_seq,
        max_seqlen_k=k_seq,
        do_causal=False,
        scale=scale,
        softcapping=1.0,
    )
    
    # Compute ground truth
    expected = compute_attention_reference(query, key, value, scale)
    
    # Check results (higher tolerance for large sequences)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 3e-2
    elif dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 2e-3, 2e-3
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_flash_attention_edge_cases():
    """Test Flash Attention edge cases."""
    torch.manual_seed(42)
    
    # Test 1: Single token sequence
    query = torch.randn(1, 1, 64, device="mps")
    key = torch.randn(1, 1, 64, device="mps")
    value = torch.randn(1, 1, 64, device="mps")
    cu_seqlens = create_cu_seqlens([1])
    out = torch.empty_like(query)
    
    metal_flash_sdpa.flash_attention_varlen(
        out=out,
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=1,
        max_seqlen_k=1,
        do_causal=False,
        scale=0.125,
        softcapping=1.0,
    )
    
    # With single token, output should equal value
    torch.testing.assert_close(out, value, rtol=1e-5, atol=1e-5)
    
    # Test 2: Empty sequence in batch
    seq_lengths = [8, 0, 12]  # Middle sequence is empty
    cu_seqlens = create_cu_seqlens(seq_lengths)
    total_tokens = sum(seq_lengths)
    
    query = torch.randn(total_tokens, 4, 64, device="mps")
    key = torch.randn(total_tokens, 4, 64, device="mps")
    value = torch.randn(total_tokens, 4, 64, device="mps")
    out = torch.empty_like(query)
    
    # This should handle empty sequences gracefully
    metal_flash_sdpa.flash_attention_varlen(
        out=out,
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seq_lengths) if seq_lengths else 0,
        max_seqlen_k=max(seq_lengths) if seq_lengths else 0,
        do_causal=False,
        scale=0.125,
        softcapping=1.0,
    )


def test_flash_attention_unsupported_cases():
    """Test that unsupported cases raise appropriate errors."""
    
    # Test 1: Unsupported head dimension
    query = torch.randn(16, 4, 48, device="mps")  # head_dim = 48 (not supported)
    key = torch.randn(16, 4, 48, device="mps")
    value = torch.randn(16, 4, 48, device="mps")
    cu_seqlens = create_cu_seqlens([16])
    out = torch.empty_like(query)
    
    with pytest.raises(RuntimeError, match="Head dimension .* is not supported"):
        metal_flash_sdpa.flash_attention_varlen(
            out=out,
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=16,
            max_seqlen_k=16,
            do_causal=False,
            scale=0.144,
            softcapping=1.0,
        )
    
    # Test 2: Wrong dtype for cu_seqlens (should be int32)
    cu_seqlens_wrong = torch.tensor([0, 16], dtype=torch.int64, device="mps")
    query = torch.randn(16, 4, 64, device="mps")
    key = torch.randn(16, 4, 64, device="mps")
    value = torch.randn(16, 4, 64, device="mps")
    
    # This will silently fail (output will be unchanged)
    out = torch.full_like(query, -999.0)
    metal_flash_sdpa.flash_attention_varlen(
        out=out,
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_wrong,
        cu_seqlens_k=cu_seqlens_wrong,
        max_seqlen_q=16,
        max_seqlen_k=16,
        do_causal=False,
        scale=0.125,
        softcapping=1.0,
    )
    
    # Check that output wasn't modified (kernel didn't run)
    assert (out == -999.0).all(), "cu_seqlens with wrong dtype should cause kernel to not run"


def test_flash_attn_varlen_func():
    """Test the flash_attn_varlen_func compatibility function."""
    torch.manual_seed(42)
    
    # Test dimensions
    seq_lengths = [8, 12]
    num_heads = 4
    head_dim = 64
    
    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens(seq_lengths)
    total_tokens = sum(seq_lengths)
    max_seqlen = max(seq_lengths)
    
    # Create input tensors
    q = torch.randn(total_tokens, num_heads, head_dim, device="mps")
    k = torch.randn(total_tokens, num_heads, head_dim, device="mps")
    v = torch.randn(total_tokens, num_heads, head_dim, device="mps")
    
    # Call the compatibility function
    out = metal_flash_sdpa.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,  # Will use 1/sqrt(head_dim)
        causal=False,
    )
    
    # Check that output has correct shape and is not zeros
    assert out.shape == q.shape
    assert out.abs().max().item() > 0
    
    # Test with causal
    out_causal = metal_flash_sdpa.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        softmax_scale=0.125,
        causal=True,
    )
    
    assert out_causal.shape == q.shape
    assert out_causal.abs().max().item() > 0
