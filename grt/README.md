---
license: apache-2.0
tags:
  - kernel
---

## GRT (Geometric Rotation Transformer) Kernels

Fused CUDA and ROCm kernels for the **GRT (Geometric Rotation Transformer)** architecture ([`MatterWaveAI/grt-30m`](https://huggingface.co/MatterWaveAI/grt-30m)):

- **`apply_fused_ffn`** (`givens_forward` / `givens_backward`): Multi-stage butterfly Givens rotation cascade for GRT feed-forward expand/contract projections.
- **`apply_fused_coupling`** (`givens_coupling_forward` / `givens_coupling_backward`): Multi-stage butterfly Givens rotation cascade with 4-component `Cl(2,0)` Clifford bivector coupling for GRT attention `Q, K, V, O` projections.
- **`apply_fused_mor`** (`givens_mor_forward` / `givens_mor_backward`): Mixture-of-Rotations soft expert blending cascade.
- **`clifford_coupling`**: 4-component `Cl(2,0)` Clifford feature computation (`r^2, x0, x1, bivector`).
- **`fused_embedding`**: Polar token embedding lookup (`phase_angles` + `log_magnitude`).
- **`fused_pearl_norm`**: Pair-radius RMSNorm (`PearlNorm`).
- **`fused_radial_gelu`**: Pair-magnitude `RadialGELU` activation.
- **`fused_mixer_stage`**: Single-stage group-coupled rotational mixer.
- **`fused_zloss`**: Masked `logsumexp^2` z-loss.

## Usage

```python
# /// script
# dependencies = [
#   "torch",
#   "kernels"
# ]
# ///
import torch
from kernels import get_kernel

grt = get_kernel("kernels-community/grt", version=1)
device = torch.device("cuda")

x = torch.randn(32, 256, device=device, dtype=torch.float32)
angles = torch.randn(9, 128, device=device, dtype=torch.float32) * 0.1
raw_scales = torch.randn(9, 256, device=device, dtype=torch.float32) * 0.1

out = grt.apply_fused_ffn(x, angles, raw_scales)
print(f"Output: {out.shape}")
```
