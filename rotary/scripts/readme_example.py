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
rotary = get_kernel("kernels-community/rotary")
device = torch.device("cuda")

# Create test tensors for rotary position embeddings
# Used in transformer models like LLaMA, GPT-NeoX
batch_size, seqlen, num_heads, head_dim = 2, 128, 8, 64
rotary_dim = 32  # typically head_dim // 2

# Query and Key tensors
query = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.float32
)
key = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.float32
)

# Rotary position embeddings (cos and sin)
cos = torch.randn(seqlen, 1, rotary_dim, device=device, dtype=torch.float32)
sin = torch.randn(seqlen, 1, rotary_dim, device=device, dtype=torch.float32)


# Reference implementation
def apply_rotary_ref(x1, x2, cos, sin):
    return x1 * cos - x2 * sin, x1 * sin + x2 * cos


# Split query into rotary parts
q1 = query[..., :rotary_dim].clone()
q2 = query[..., rotary_dim : 2 * rotary_dim].clone()

# Reference
q1_ref, q2_ref = apply_rotary_ref(q1.clone(), q2.clone(), cos, sin)

# Kernel (in-place operation)
rotary.apply_rotary(q1, q2, cos, sin, q1, q2, conj=False)

print(f"Query rotary output shape: {q1.shape}")
print(f"Outputs close: {torch.allclose(q1, q1_ref, atol=1e-5, rtol=1e-5)}")
# Query rotary output shape: torch.Size([2, 128, 8, 32])
# Outputs close: True
