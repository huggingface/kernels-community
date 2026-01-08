---
tags:
  - kernels
  - gptoss
---

# gpt-oss-metal-kernels

Metal kernels that back the OpenAI GPT-OSS reference implementation, repackaged for local experiments on Apple Silicon GPUs. The GPT-OSS project distributes optimized inference primitives for the `gpt-oss-20b` and `gpt-oss-120b` open-weight models, including MXFP4-packed linear layers and fused attention paths that target Metal Performance Shaders on macOS [gpt-oss](https://github.com/openai/gpt-oss).

## Installation

```bash
pip install kernels  # we just need to install the kernels package
```

The package exposes Python bindings through `gpt_oss_metal_kernels.ops`; these symbols are re-exported in `gpt_oss_metal_kernels.__init__` for convenience. All kernels expect Metal (`mps`) tensors and operate in place on user-provided outputs to minimize additional allocations.

## Available Ops

- `f32_bf16w_matmul`, `f32_bf16w_matmul_add`
- `f32_bf16w_dense_matmul_qkv`, `f32_bf16w_dense_matmul_attn_output`, `f32_bf16w_dense_matmul_mlp_gate`
- `f32_bf16w_rmsnorm`
- `bf16_f32_embeddings`
- `f32_rope`
- `f32_bf16w_matmul_qkv`
- `f32_sdpa`
- `f32_topk`, `expert_routing_metadata`, `f32_scatter`

For implementation details, inspect the `.metal` shader files.

## Usage & Consistency Checks

Each example below compares a Metal kernel against the canonical PyTorch equivalent using shared random inputs. The snippets assume an Apple Silicon machine with an `mps` device and that `kernels` installed in the active environment.

### 1. BF16-weight matmul vs PyTorch `matmul`

```python
import torch
from kernels import get_kernel

gptoss_kernels = get_kernel("kernels-community/gpt-oss-metal-kernels")

torch.manual_seed(0)
device = "mps"
batch, rows, cols = 2, 128, 1024

activations = torch.randn(batch, rows, device=device, dtype=torch.float32)
weights = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
bias = torch.zeros(cols, device=device, dtype=torch.bfloat16)
out_ref = activations @ weights.float() + bias.float()

out_kernel = torch.empty(batch, cols, device=device, dtype=torch.float32)
gptoss_kernels.f32_bf16w_matmul(
    activations,
    weights,
    bias,
    out_kernel,
    num_tokens=batch,
    num_cols=rows,
    num_rows=cols,
    threadgroup_size=32,
)

print(out_kernel)
print(out_ref)

torch.testing.assert_close(out_kernel, out_ref, atol=1e-3, rtol=1e-3)
```

### 2. RMSNorm vs PyTorch layer norm equivalent

```python
from kernels import get_kernel
import torch

gptoss_kernels = get_kernel("kernels-community/gpt-oss-metal-kernels")
device = "mps"

hidden = 4096
eps = 1e-5
x = torch.randn(4, hidden, device=device, dtype=torch.float32)
weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)

variance = x.pow(2).mean(dim=-1, keepdim=True)
out_ref = (x * torch.rsqrt(variance + eps)) * weight.float()

out_kernel = torch.empty_like(x)
gptoss_kernels.f32_bf16w_rmsnorm(x, weight, out_kernel, epsilon=eps)

print(out_kernel)
print(out_ref)

torch.testing.assert_close(out_kernel, out_ref, atol=1e-3, rtol=1e-3)
```

### 3. Embedding lookup with BF16 tables

```python
from kernels import get_kernel
import torch

device = "mps"
gptoss_kernels = get_kernel("kernels-community/gpt-oss-metal-kernels")

vocab, dim = 1024, 256
token_ids = torch.randint(0, vocab, (16,), device=device, dtype=torch.int32)
emb_table = torch.randn(vocab, dim, device=device, dtype=torch.bfloat16)

out_ref = emb_table.float().index_select(0, token_ids.long())
out_kernel = torch.empty_like(out_ref)
gptoss_kernels.bf16_f32_embeddings(token_ids, emb_table, out_kernel, threadgroup_size=32)

print(out_kernel)
print(out_ref)

torch.testing.assert_close(out_kernel, out_ref, atol=1e-4, rtol=1e-3)
```

### 4. Scaled dot-product attention (SDPA)

```python
from kernels import get_kernel
import torch
import torch.nn as nn

device = "mps"
gptoss_kernels = get_kernel("kernels-community/gpt-oss-metal-kernels")


head_dim = 64
kv_heads = 8
qmul = 8
num_q_heads = kv_heads * qmul
history_tokens = 3
num_q_tokens = 2
total_tokens = history_tokens + num_q_tokens
max_tokens = total_tokens
num_kv_tokens = history_tokens

# Generate Q/K/V tensors
Q_chunk = torch.randn(num_q_tokens, num_q_heads, head_dim, device=device, dtype=torch.float32)
K_all = torch.randn(kv_heads, total_tokens, head_dim, device=device, dtype=torch.float32)
V_all = torch.randn(kv_heads, total_tokens, head_dim, device=device, dtype=torch.float32)

qkv_dim = head_dim * (num_q_heads + 2 * kv_heads)
q_buffer = torch.zeros(num_q_tokens, qkv_dim, device=device, dtype=torch.float32)
for t in range(num_q_tokens):
    q_buffer[t, : num_q_heads * head_dim] = Q_chunk[t].reshape(-1)
    token_idx = history_tokens + t
    q_buffer[
        t,
        num_q_heads * head_dim : num_q_heads * head_dim + kv_heads * head_dim,
    ] = K_all[:, token_idx, :].reshape(-1)
    q_buffer[
        t,
        num_q_heads * head_dim + kv_heads * head_dim :,
    ] = V_all[:, token_idx, :].reshape(-1)

token_stride = 2 * head_dim
kv_stride = token_stride * max_tokens
kv_cache = torch.zeros(kv_heads, kv_stride, device=device, dtype=torch.float32)
for h in range(kv_heads):
    for t in range(total_tokens):
        base = t * token_stride
        kv_cache[h, base : base + head_dim] = K_all[h, t]
        kv_cache[h, base + head_dim : base + 2 * head_dim] = V_all[h, t]

sink = torch.full((num_q_heads,), -1e4, device=device, dtype=torch.bfloat16)
output = torch.empty(num_q_tokens, num_q_heads, head_dim, device=device, dtype=torch.float32)

gptoss_kernels.f32_sdpa(
    q_buffer,
    0,
    kv_cache,
    0,
    sink,
    0,
    output,
    0,
    window=total_tokens,
    kv_stride=kv_stride,
    num_q_tokens=num_q_tokens,
    num_kv_tokens=num_kv_tokens,
    num_q_heads=num_q_heads,
    num_kv_heads=kv_heads,
    head_dim=head_dim,
)
```
For this kernel, the outputs match 97% of the time, It should be related to how the reference implementation is implemented below:

```python
def sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, S: torch.Tensor, sm_scale: float, sliding_window: int = 0) -> torch.Tensor:
    Q = Q.reshape(Q.shape[0], Q.shape[1], -1, Q.shape[-1])
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)

scale = head_dim ** -0.5
q_buffer_cpu = q_buffer.detach().cpu()
kv_cache_cpu = kv_cache.detach().cpu()
sinks_cpu = sink.detach().to(torch.float32).cpu()

Q_total_cpu = torch.zeros(total_tokens, kv_heads, qmul, head_dim, dtype=torch.float32)
for idx, abs_idx in enumerate(range(num_kv_tokens, total_tokens)):
    q_flat = q_buffer_cpu[idx, : num_q_heads * head_dim]
    Q_total_cpu[abs_idx] = q_flat.view(kv_heads, qmul, head_dim)

K_total_cpu = torch.empty(total_tokens, kv_heads, head_dim, dtype=torch.float32)
V_total_cpu = torch.empty(total_tokens, kv_heads, head_dim, dtype=torch.float32)
for t in range(total_tokens):
    base = t * token_stride
    K_total_cpu[t] = kv_cache_cpu[:, base : base + head_dim]
    V_total_cpu[t] = kv_cache_cpu[:, base + head_dim : base + 2 * head_dim]

output_ref = sdpa(
    Q_total_cpu,
    K_total_cpu,
    V_total_cpu,
    sinks_cpu,
    scale,
    sliding_window=0,
)

```


These kernels form the core of the GPT-OSS inference stack, enabling BF16 activations with MXFP4 weights while keeping latency low on Metal GPUs [gpt-oss](https://github.com/openai/gpt-oss). Use the snippets as templates when validating your own model integrations or when extending the kernel set.
