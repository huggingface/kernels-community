# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "apache-tvm-ffi==0.1.14.post1",
#     "kernels==0.17.2",
#     "torch==2.13.0",
# ]
# ///
# Use a local build with LOCAL_KERNELS=kernels-community/flash-attn2-rs=result.
import torch
from kernels import get_kernel

mod = get_kernel("kernels-community/flash-attn2-rs", version=1)
q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
out, lse, probabilities, rng_state = mod.fwd(q, k, v)
torch.cuda.synchronize()
print(out.shape, lse.shape)
