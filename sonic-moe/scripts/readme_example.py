# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels",
#   "nvidia-cutlass-dsl",
#   "quack-kernels",
# ]
# ///
import torch
from kernels import get_kernel, get_local_kernel
from pathlib import Path

# Setup
torch.manual_seed(42)
# sonic_moe = get_kernel("kernels-community/sonic-moe")
sonic_moe = get_local_kernel(Path("build"), "sonic_moe")
device = torch.device("cuda")

# ---------------------------------------------------------------------------
# 1. count_cumsum: count expert assignments and compute cumulative sum
# ---------------------------------------------------------------------------
num_tokens = 1024
num_experts = 8  # must be divisible by 4
expert_indices = torch.randint(
    0, num_experts, (num_tokens,), device=device, dtype=torch.int32
)

count_output, cumsum_output = sonic_moe.count_cumsum(
    x=expert_indices, E=num_experts, do_cumsum=True
)

try:
    torch.cuda.synchronize()
except Exception as e:
    print(f"CUDA error (kernel may not be supported on this GPU): {e}")
    raise SystemExit(1)

ref_count = expert_indices.bincount(minlength=num_experts).to(torch.int32)
ref_cumsum = ref_count.cumsum(-1)

print("=== count_cumsum ===")
print(f"Counts match:     {torch.equal(count_output, ref_count)}")
print(f"Cumsum match:     {torch.equal(cumsum_output, ref_cumsum)}")

count_only = sonic_moe.count_cumsum(
    x=expert_indices, E=num_experts, do_cumsum=False
)
torch.cuda.synchronize()
print(f"Count-only match: {torch.equal(count_only, ref_count)}")

# ---------------------------------------------------------------------------
# 2. MoE forward: full mixture-of-experts layer
#    (requires quack-kernels and a Hopper+ GPU for sonicmoe backend)
# ---------------------------------------------------------------------------
try:
    MoE = sonic_moe.MoE
    KernelBackendMoE = sonic_moe.KernelBackendMoE
    ActivationType = sonic_moe.enums.ActivationType
except AttributeError:
    print("\nSkipping MoE test (requires quack-kernels and Hopper+ GPU)")
    raise SystemExit(0)

T, H, I, E, K = 8192, 768, 256, 128, 8
dtype = torch.bfloat16

with torch.device(device):
    moe = MoE(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        intermediate_size=I,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
    ).to(dtype=dtype)

x = 0.02 * torch.randn(T, H, device=device, dtype=dtype)

# Forward pass with torch backend (reference, works on any GPU)
with torch.no_grad():
    y_torch, _ = moe(x, kernel_backend_moe=KernelBackendMoE.torch)

print(f"\n=== MoE (torch backend) ===")
print(f"Input:  {x.shape}")
print(f"Output: {y_torch.shape}")

# Forward pass with sonicmoe backend (requires Hopper+ GPU)
try:
    with torch.no_grad():
        y_sonic, _, expert_freq = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    torch.cuda.synchronize()

    close = torch.allclose(y_sonic.float(), y_torch.float(), atol=1.4e-2, rtol=2e-2)
    print(f"\n=== MoE (sonicmoe backend) ===")
    print(f"Output: {y_sonic.shape}")
    print(f"Matches torch: {close}")
except Exception as e:
    print(f"\nSkipping sonicmoe backend (requires Hopper+ GPU): {e}")
