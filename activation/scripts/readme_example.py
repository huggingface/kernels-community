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
activation = get_kernel("kernels-community/activation")
device = torch.device("cuda")

# Create test tensor for SwiGLU (SiLU and Mul)
# Input has shape (num_tokens, 2 * hidden_dim)
num_tokens, hidden_dim = 128, 512
input_tensor = torch.randn(
    num_tokens, 2 * hidden_dim, device=device, dtype=torch.float16
)


# Reference implementation: SwiGLU computes silu(x[:d]) * x[d:]
def reference_silu_and_mul(x):
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


out_ref = reference_silu_and_mul(input_tensor)

# Custom kernel SwiGLU
out_shape = input_tensor.shape[:-1] + (hidden_dim,)
out_kernel = torch.empty(out_shape, dtype=input_tensor.dtype, device=device)
out_kernel = activation.silu_and_mul(out_kernel, input_tensor)

print(f"Reference output: {out_ref.shape}")
print(f"Kernel output: {out_kernel.shape}")
print(f"Outputs close: {torch.allclose(out_kernel, out_ref, atol=1e-3, rtol=1e-3)}")
# Reference output: torch.Size([128, 512])
# Kernel output: torch.Size([128, 512])
# Outputs close: True
