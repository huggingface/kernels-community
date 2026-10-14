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
flash_attn3 = get_kernel("kernels-community/flash-attn3")
device = torch.device("cuda")

# Create test tensors for Flash Attention 3
batch_size, seqlen, num_heads, head_dim = 2, 512, 8, 64
query = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16
)
key = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16
)
value = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16
)


# Reference implementation using PyTorch SDPA
def reference_attention(q, k, v, causal=False):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal
        )
    return out.transpose(1, 2)


out_ref = reference_attention(query, key, value, causal=False)

# Flash Attention 3
out_flash, softmax_lse = flash_attn3.flash_attn_func(
    query, key, value, softmax_scale=None, causal=False
)

print(f"Reference output: {out_ref.shape}")
print(f"Flash Attention 3 output: {out_flash.shape}")
print(f"Outputs close: {torch.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)}")
# Reference output: torch.Size([2, 512, 8, 64])
# Flash Attention 3 output: torch.Size([2, 512, 8, 64])
# Outputs close: True
