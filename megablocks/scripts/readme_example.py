# /// script
# requires-python = "==3.10"
# dependencies = [
#     "numpy",
#     "kernels",
#     "torch"
# ]
# ///

import torch
from collections import namedtuple

from kernels import get_kernel

# Make reproducible
torch.manual_seed(42)
# torch.cuda.manual_seed(42)
device = "cpu"

# Download optimized kernels from the Hugging Face hub
# megablocks = get_kernel("kernels-community/megablocks")
import megablocks
print("MegaBlocks kernel downloaded successfully.")

model = megablocks.cpu_fused_moe.MegaBlocksMoeMLP()
model.experts = namedtuple("Experts", ["gate_up_proj", "gate_down_proj", "down_proj", "hidden_size"])
print("MegaBlocksMoeMLP instance created successfully.")

# Config
ne, hs, isz = 128, 1152, 3072

# Router with proper initialization
model.router = torch.nn.Linear(hs, ne, device=device)
torch.nn.init.kaiming_uniform_(model.router.weight)

# Expert layers with realistic weights
e = model.experts
e.gate_up_proj = torch.nn.Parameter(torch.randn(ne, hs, isz, device=device) * 0.02)
e.gate_up_proj_bias = torch.nn.Parameter(torch.zeros(ne, isz, device=device))
e.down_proj = torch.nn.Parameter(torch.randn(ne, 1536, hs, device=device) * 0.02)
e.down_proj_bias = torch.nn.Parameter(torch.zeros(ne, hs, device=device))
e.hidden_size = hs
print("Expert layers initialized successfully.")

# Test with normalized input
x = torch.randn(1, 1, hs, device=device) * 0.1
output, expert_weights = model(x)
print("Model forward pass completed successfully.")

print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
print(f"Output: {output.flatten()[:10]}")
print(f"Expert weights sum: {expert_weights.sum():.3f}")