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
layer_norm = get_kernel("kernels-community/layer-norm")
device = torch.device("cuda")

# Create test tensor
batch_size, seq_len, hidden_dim = 2, 5, 768
input_tensor = torch.randn(
    batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16
)
weight = torch.ones(hidden_dim, device=device, dtype=torch.float16)
epsilon = 1e-5

# Reference implementation using PyTorch LayerNorm
ref_ln = torch.nn.LayerNorm(
    hidden_dim,
    eps=epsilon,
    elementwise_affine=False,
    device=device,
    dtype=torch.float16,
)
out_ref = ref_ln(input_tensor)

# Custom kernel LayerNorm
out_kernel = layer_norm.dropout_add_ln_fwd(
    input=input_tensor.view(-1, hidden_dim),
    gamma=weight,
    beta=None,
    rowscale=None,
    colscale=None,
    x0_subset=None,
    z_subset=None,
    dropout_p=0.0,
    epsilon=epsilon,
    rowscale_const=1.0,
    z_numrows=seq_len,
    gen=None,
    residual_in_fp32=False,
    is_rms_norm=False,
)[0].view(batch_size, seq_len, hidden_dim)

print(f"Reference output: {out_ref.shape}")
print(f"Kernel output: {out_kernel.shape}")
print(f"Outputs close: {torch.allclose(out_kernel, out_ref, atol=1e-2, rtol=1e-3)}")
