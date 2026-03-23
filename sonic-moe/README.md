# SonicMoE

[kernel-builder](https://github.com/huggingface/kernels) port of [Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe) @ [`e9190f9`](https://github.com/Dao-AILab/sonic-moe/commit/e9190f9a807ed445816702307ec813a5c9962a93).

High-performance Mixture-of-Experts for Hopper and Blackwell GPUs. `count_cumsum` is AOT-compiled and works on any CUDA GPU. The full MoE layer (CuTeDSL/Triton, JIT at runtime) requires Hopper+ and `pip install nvidia-cutlass-dsl quack-kernels`.

```python
from kernels import get_kernel
sonic_moe = get_kernel("kernels-community/sonic-moe")

# Any CUDA GPU
count, cumsum = sonic_moe.count_cumsum(expert_indices, E=num_experts, do_cumsum=True)

# Hopper+ GPU
output, aux_loss = sonic_moe.MoE(...).cuda()(hidden_states)
```
