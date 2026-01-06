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
scattermoe = get_kernel("kernels-community/scattermoe")
device = torch.device("cuda")

# ScatterMoE: Efficient Mixture of Experts implementation
# Uses scatter operations for parallel expert computation
num_tokens = 64
hidden_dim = 256
num_experts = 8
top_k = 2  # number of experts per token
expert_dim = 512

# Input tokens [num_tokens, hidden_dim]
inputs = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float16)

# Router selects top-k experts per token
# expert_idxs: selected expert indices [num_tokens, top_k]
expert_idxs = torch.randint(0, num_experts, (num_tokens, top_k), device=device)

# Flatten, sort, and count expert assignments
sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = scattermoe.flatten_sort_count(
    expert_idxs, num_experts=num_experts
)

print(f"Input shape: {inputs.shape}")
print(f"Expert indices shape: {expert_idxs.shape}")
print(f"Sorted expert indices shape: {sorted_expert_idxs.shape}")
print(f"Sorted scattered indices shape: {sorted_scattered_idxs.shape}")
print(f"Expert offsets shape: {expert_offsets.shape}")

# Create expert weights [num_experts, expert_dim, hidden_dim]
expert_weights = torch.randn(
    num_experts, expert_dim, hidden_dim, device=device, dtype=torch.float16
) * 0.02

# Run parallel linear across experts
output = scattermoe.parallel_linear(
    inputs, expert_weights, top_k,
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets
)

print(f"Expert weights shape: {expert_weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
# Input shape: torch.Size([64, 256])
# Expert indices shape: torch.Size([64, 2])
# Sorted expert indices shape: torch.Size([128])
# Sorted scattered indices shape: torch.Size([128])
# Expert offsets shape: torch.Size([8])
# Expert weights shape: torch.Size([8, 512, 256])
# Output shape: torch.Size([128, 512])
# Output dtype: torch.float16
