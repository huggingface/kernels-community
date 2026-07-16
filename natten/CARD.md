---
license: mit
tags:
- kernel
---

![Status](https://hubwebhook.dholtz.com/shield?repo=kernels-community/natten)

# NATTEN — Neighborhood Attention kernels

CUDA kernels for [NATTEN](https://natten.org) (Neighborhood Attention
Extension, [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN)), packaged
for the [kernels](https://github.com/huggingface/kernels) library. Vendored
from NATTEN v0.21.6 (MIT license).

Neighborhood attention restricts each token's attention to a sliding local
window over 1-D, 2-D, or 3-D token layouts, with optional stride, dilation,
and per-dimension causal masking. This kernel ships NATTEN's fused
neighborhood attention (FNA) and FMHA kernels along with the thin functional
frontend (`na1d`, `na2d`, `na3d`, `attention`, `merge_attentions`) and its
backend/config selection logic.

## Usage

```python
import torch
from kernels import get_kernel

natten = get_kernel("kernels-community/natten", version=1)

# 2-D neighborhood attention over a 32x32 token layout:
# [batch, *token_layout, heads, head_dim]
q = torch.randn(1, 32, 32, 8, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = natten.functional.na2d(q, k, v, kernel_size=(7, 7))

# 3-D, with stride/dilation/causal masking per dim:
q3 = torch.randn(1, 8, 16, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
k3, v3 = torch.randn_like(q3), torch.randn_like(q3)
out3 = natten.functional.na3d(
    q3, k3, v3, kernel_size=(3, 5, 5), dilation=(1, 2, 2), is_causal=(True, False, False)
)
```

All functions are differentiable and torch.compile-compatible (the underlying
ops are registered via `TORCH_LIBRARY` with fake/meta implementations).

## Backend / architecture support

| Backend | Compute capabilities | Notes |
|---|---|---|
| Reference CUDA kernels | 7.0 – 12.0 | correctness fallback, all dims |
| CUTLASS 2.X FNA / FMHA (`cutlass-fna`) | 7.0 – 12.0 (SM80 tensor-core paths) | default on pre-Hopper |
| Hopper FNA / FMHA (`hopper-fna`) | 9.0a | H100/H200 |
| Blackwell FNA / FMHA (`blackwell-fna`) | 10.0a | B200/GB200; requires CUDA ≥ 12.8, so cu126 builds fall back to `cutlass-fna` / reference on these GPUs |
| Flex Attention backend (`flex-fna`) | any | pure PyTorch, via `torch.nn.attention.flex_attention` |

Known gaps vs. upstream wheels:

- No SM103 (B300 / Blackwell Ultra) binaries — kernel-builder does not
  currently target `10.3a`.
- SM120 (RTX 50 series) gets the CUTLASS 2.X and reference paths only, same
  as upstream (NATTEN has no SM120-specific fused kernels).
- No ROCm or CPU support (upstream dropped these in v0.20).

The backend is chosen automatically per GPU; pass `backend="..."` to
`na1d`/`na2d`/`na3d` to override.

## Developing

The ~144 kernel instantiation TUs under `csrc/autogen/` are committed
codegen output. To regenerate (e.g. when bumping the vendored NATTEN
version), run:

```bash
scripts/regen.sh                 # re-runs autogen + rewrites build.toml src lists
```

Build and test with kernel-builder:

```bash
nix run .#build-and-copy -L
nix run .#ci-test -L
```

## Credits

All kernels and the Python frontend are by Ali Hassani and the NATTEN
contributors ([SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN), MIT).
This repository only repackages them for the Kernel Hub: pybind11 bindings
are replaced with `TORCH_LIBRARY` registrations (Python limited API), and
build-time codegen is committed statically.

If you use NATTEN, please cite:

```bibtex
@inproceedings{hassani2023neighborhood,
  title   = {Neighborhood Attention Transformer},
  author  = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
  year    = 2023,
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
@misc{hassani2024faster,
  title   = {Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level},
  author  = {Ali Hassani and Wen-Mei Hwu and Humphrey Shi},
  year    = 2024,
  eprint  = {2403.04690},
  archivePrefix = {arXiv},
}
```
