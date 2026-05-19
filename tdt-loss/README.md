---
tags:
- kernel
---
This CUDA extension implements TDT (Token-and-Duration Transducer) loss, originally from [eustlb/tdt-loss](https://huggingface.co/eustlb/tdt-loss).

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

tdt_loss_module = get_kernel("kernels-community/tdt-loss")

# Example usage
batch_size = 2
max_input_len = 100       # T: number of encoder frames
max_target_len = 20       # U: includes blank column (= num_labels + 1)
vocab_size = 256
durations = [0, 1, 2, 3, 4]  # possible duration values
num_durations = len(durations)
blank_id = 0
device = torch.device("cuda")

token_logits = torch.randn(batch_size, max_input_len, max_target_len, vocab_size, device=device, requires_grad=True)
duration_logits = torch.randn(batch_size, max_input_len, max_target_len, num_durations, device=device, requires_grad=True)
targets = torch.randint(1, vocab_size, (batch_size, max_target_len - 1), device=device)
source_lengths = torch.full((batch_size,), max_input_len, device=device, dtype=torch.int32)
target_lengths = torch.full((batch_size,), max_target_len - 1, device=device, dtype=torch.int32)

loss = tdt_loss_module.tdt_loss(
    token_logits, duration_logits, targets,
    source_lengths, target_lengths, durations,
    blank_id, sigma=0.05, reduction="mean",
)
loss.backward()
```

## Benchmarks

See the [benchmark script](https://gist.github.com/eustlb/42bb54363e82ad6c0f6fc8e24f604c18).
