# /// script
# dependencies = [
#   "numpy", 
#   "torch", 
#   "kernels"
# ]
# ///
import torch
from kernels import get_kernel

# Setup
torch.manual_seed(42)
flash_attn = get_kernel("kernels-community/flash-attn")
device = torch.device("cuda")

# Create test tensors
B, S, H, D = 2, 5, 4, 8  # batch, seq_len, heads, head_dim
q = k = v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)

# Reference implementation using PyTorch SDPA
def reference_attention(query, key, value, causal=False):
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=causal)
    return out.transpose(1, 2).contiguous()

# 1. Standard attention
print("\n1. Standard attention:")
out_ref = reference_attention(q, k, v)
out_flash = flash_attn.fwd(
    q=q, 
    k=k, 
    v=v, 
    is_causal=False,
)[0]
print(f"Reference output: {out_ref.shape}")
print(f"Flash output: {out_flash.shape}")
print(f"Outputs close: {torch.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-3)}")

# 2. Causal attention (for autoregressive models)
print("\n2. Causal attention:")

out_ref_causal = reference_attention(q, k, v, causal=True)
out_causal = flash_attn.fwd(
    q=q, 
    k=k, 
    v=v, 
    is_causal=True,
)[0]
print(f"Reference causal output: {out_ref_causal.shape}")
print(f"Flash causal output: {out_causal.shape}")
print(f"Outputs close: {torch.allclose(out_causal, out_ref_causal, atol=1e-2, rtol=1e-3)}")

def var_reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False):
    batch_size = cu_seqlens_q.shape[0] - 1
    # Return output in packed format (same as flash attention)
    total_tokens_q = q.shape[0]
    out = torch.zeros((total_tokens_q, q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)
    
    for b in range(batch_size):
        start_q, end_q = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        start_k, end_k = cu_seqlens_k[b], cu_seqlens_k[b + 1]
        
        # Extract slices for this batch
        q_slice = q[start_q:end_q]  # Shape: (seq_len_q, H, D)
        k_slice = k[start_k:end_k]  # Shape: (seq_len_k, H, D)
        v_slice = v[start_k:end_k]  # Shape: (seq_len_k, H, D)
        
        # Add batch dimension for reference_attention
        q_slice = q_slice.unsqueeze(0)  # Shape: (1, seq_len_q, H, D)
        k_slice = k_slice.unsqueeze(0)  # Shape: (1, seq_len_k, H, D)
        v_slice = v_slice.unsqueeze(0)  # Shape: (1, seq_len_k, H, D)
        
        # Compute attention and remove batch dimension
        attn_out = reference_attention(q_slice, k_slice, v_slice, causal=causal)
        attn_out = attn_out.squeeze(0)  # Shape: (seq_len_q, H, D)
        
        # Place result in output tensor (packed format)
        out[start_q:end_q] = attn_out
    
    return out

# 3. Variable length sequences (packed format)
print("\n3. Variable length sequences:")
# Pack sequences of lengths [3,4,3] for q and [4,5,3] for k into single tensors
q_var = torch.randn(10, H, D, device=device, dtype=torch.float16)  # total_q=10
k_var = v_var = torch.randn(12, H, D, device=device, dtype=torch.float16)  # total_k=12
cu_q = torch.tensor([0, 3, 7, 10], device=device, dtype=torch.int32)  # cumulative sequence lengths
cu_k = torch.tensor([0, 4, 9, 12], device=device, dtype=torch.int32)

out_var_ref = var_reference_attention(q_var, k_var, v_var, cu_q, cu_k, max_seqlen_q=4, max_seqlen_k=5, causal=False)
# Custom function to handle variable
out_var = flash_attn.varlen_fwd(
    q=q_var,
    k=k_var,
    v=v_var,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    max_seqlen_q=4,
    max_seqlen_k=5,
)[0]
print(f"Variable length output: {out_var.shape}")
print(f"Reference variable length output: {out_var_ref.shape}")
print(f"Outputs close: {torch.allclose(out_var, out_var_ref, atol=1e-2, rtol=1e-3)}")
