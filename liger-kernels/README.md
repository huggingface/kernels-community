---
license: apache-2.0
tags:
  - kernels
---

![Status](https://hubwebhook.dholtz.com/shield?repo=kernels-community/liger_kernels)


# Liger Kernels

This repository contains the kernels for the Liger project [Liger Kernels](https://github.com/linkedin/Liger-Kernel)



## Tiled MLP tests

The tiled SwiGLU and GEGLU adapters recompute each token tile with non-reentrant
checkpointing. The tiling helper keeps parameter dependencies in the autograd
graph, including when inputs are frozen or `compute_params` is omitted. In
distributed runs it coordinates tile counts across ranks, including uneven
local sequence lengths.

Run these commands inside the kernel's test environment, with `LOCAL_KERNELS`
pointing to the local build as in the other kernel tests:

```bash
python -m pytest tests/test_tiled_mlp.py

torchrun --standalone --nproc-per-node=2 -m pytest tests/test_tiled_mlp_distributed.py

torchrun --standalone --nproc-per-node=4 -m pytest tests/test_tiled_mlp_distributed.py
```

Two ranks cover DDP, FSDP2, tensor parallelism, and the token-local MLP portion of
context parallelism. Four ranks additionally cover FSDP2 × TP, FSDP2 × CP,
TP × CP, and DP × CP. Each layout checks outputs, input and parameter gradients,
and optimizer updates over two iterations, with frozen or differentiable inputs
and with or without an outer checkpoint. CP cases use uneven local sequence
shards and reduce weight gradients across the CP group; they do not test attention.
The same pytest module can run across nodes with a distributed launcher that sets
`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, and `LOCAL_RANK`.
