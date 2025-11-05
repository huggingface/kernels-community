---
license: apache-2.0
tags:
  - kernel
---

## Punica sgmv

[Punica](https://github.com/punica-ai/punica) sgmv kernels with modifications
from [Lorax](https://github.com/predibase/lorax).

## Usage

```python
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
punica_sgmv = get_kernel("kernels-community/punica-sgmv")
device = torch.device("cuda")

# SGMV for LoRA operations
batch_size, hidden_dim, lora_rank = 3, 1024, 16
num_layers, layer_idx = 2, 0

input_tensor = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)
output = torch.zeros(batch_size, hidden_dim, device=device, dtype=torch.float16)

# LoRA weights
lora_a = torch.randn(
    num_layers, hidden_dim, lora_rank, device=device, dtype=torch.float16
)
lora_b = torch.randn(
    num_layers, lora_rank, hidden_dim, device=device, dtype=torch.float16
)

# Segment indices
s_start = torch.tensor([0, 2], dtype=torch.int32, device=device)
s_end = torch.tensor([2, 3], dtype=torch.int32, device=device)
wa_ptr = torch.tensor(
    [lora_a.data_ptr(), lora_a.data_ptr()], dtype=torch.int64, device=device
)
wb_ptr = torch.tensor(
    [lora_b.data_ptr(), lora_b.data_ptr()], dtype=torch.int64, device=device
)

# Apply LoRA
tmp_shrink, tmp_expand = punica_sgmv.get_tmp_tensors(wa_ptr.size(0), lora_rank, device)
intermediate = punica_sgmv.lora_a_sgmv_cutlass(
    input_tensor, tmp_shrink, wa_ptr, s_start, s_end, layer_idx, lora_rank
)
punica_sgmv.lora_b_sgmv_cutlass(
    output, intermediate, tmp_expand, wb_ptr, s_start, s_end, layer_idx
)

print(f"Output shape: {output.shape}")
# Output shape: torch.Size([3, 1024])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.
