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
deformable_detr = get_kernel("kernels-community/deformable-detr")
device = torch.device("cuda")

# Multi-Scale Deformable Attention parameters
batch_size = 2
num_heads = 8
num_levels = 4
num_points = 4
embed_dim = 256
head_dim = embed_dim // num_heads

# Spatial shapes for each feature level (height, width)
# Typically from FPN outputs at different scales
spatial_shapes = torch.tensor(
    [[64, 64], [32, 32], [16, 16], [8, 8]], device=device, dtype=torch.long
)
level_start_index = torch.cat(
    [torch.zeros(1, device=device, dtype=torch.long),
     (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]]
)
total_keys = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item()
num_queries = 100

# Input tensors
# value: flattened multi-scale feature maps [B, sum(H*W), num_heads, head_dim]
value = torch.randn(
    batch_size, total_keys, num_heads, head_dim, device=device, dtype=torch.float32
)
# sampling_loc: normalized sampling locations [B, num_queries, num_heads, num_levels, num_points, 2]
sampling_loc = torch.rand(
    batch_size, num_queries, num_heads, num_levels, num_points, 2,
    device=device, dtype=torch.float32
)
# attn_weight: attention weights [B, num_queries, num_heads, num_levels, num_points]
attn_weight = torch.rand(
    batch_size, num_queries, num_heads, num_levels, num_points,
    device=device, dtype=torch.float32
)
attn_weight = attn_weight / attn_weight.sum(-1, keepdim=True)  # normalize

# Run multi-scale deformable attention forward pass
im2col_step = 64
output = deformable_detr.ms_deform_attn_forward(
    value, spatial_shapes, level_start_index,
    sampling_loc, attn_weight, im2col_step
)

print(f"Value shape: {value.shape}")
print(f"Sampling locations shape: {sampling_loc.shape}")
print(f"Attention weights shape: {attn_weight.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
# Value shape: torch.Size([2, 5440, 8, 32])
# Sampling locations shape: torch.Size([2, 100, 8, 4, 4, 2])
# Attention weights shape: torch.Size([2, 100, 8, 4, 4])
# Output shape: torch.Size([2, 100, 256])
# Output dtype: torch.float32
