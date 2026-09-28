---
tags:
- kernel
---
CUDA kernels for the TDT (Token-and-Duration Transducer, [arXiv:2304.06795](https://arxiv.org/abs/2304.06795)) loss,
used to train [Parakeet TDT](https://huggingface.co/docs/transformers/model_doc/parakeet) models in 🤗 Transformers.

The forward and backward passes are computed with dedicated kernels:

- a log-softmax over the vocabulary for every lattice node (one thread block per node, online softmax),
- the alpha / beta recursions over the anti-diagonals of the lattice (one thread block per sample),
- a fused gradient w.r.t. the token and duration logits.

Logits can be float32, float16 or bfloat16 and only need a contiguous last dimension, so slices of a joint
`(..., vocab_size + num_durations)` output are used without a copy. The lattice recursions run in float64.

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

tdt = get_kernel("kernels-community/tdt-loss", version=1)

batch_size, max_frames, max_labels, vocab_size = 2, 100, 20, 1024
durations = [0, 1, 2, 3, 4]
blank_id = vocab_size  # blank is the last token
device = "cuda"

# Joint network output: vocabulary (with blank) followed by the durations.
logits = torch.randn(
    batch_size, max_frames, max_labels + 1, vocab_size + 1 + len(durations), device=device, requires_grad=True
)
targets = torch.randint(0, vocab_size, (batch_size, max_labels), device=device)
logit_lengths = torch.tensor([max_frames, max_frames - 10], device=device)
target_lengths = torch.tensor([max_labels, max_labels - 5], device=device)

loss = tdt.tdt_loss(
    logits[..., : vocab_size + 1],
    logits[..., vocab_size + 1 :],
    targets,
    logit_lengths,
    target_lengths,
    blank_id,
    durations,
    sigma=0.0,
    reduction="mean",  # or "mean_volume", "mean_batch", "sum", "none"
)
loss.backward()
```

Samples without a valid alignment get an infinite loss and a zero gradient.

The `layers.TDTLoss` layer has the same signature as `transformers.loss.loss_tdt.tdt_loss`, so 🤗 Transformers
swaps it in for Parakeet TDT models loaded with `use_kernels=True`.

## Benchmarks

Forward + backward of the loss, with the logits given as slices of a joint output, compared with the vectorized
PyTorch implementation from `transformers.loss.loss_tdt`. Run `benchmarks/benchmark.py` with `kernels benchmark`.

| batch, frames, labels, vocab | dtype | GPU | PyTorch | kernel | speedup |
|---|---|---|---:|---:|---:|
| 8, 100, 20, 1025 | float32 | L4 | 536 ms | 4.3 ms | 125x |
| 4, 200, 40, 8193 | float32 | L4 | 1252 ms | 53 ms | 24x |
| 4, 200, 40, 8193 | bfloat16 | L4 | 1242 ms | 39 ms | 32x |
| 8, 100, 20, 1025 | float32 | A100 | 624 ms | 1.5 ms | 410x |
| 4, 200, 40, 8193 | float32 | A100 | 1259 ms | 9.5 ms | 133x |
| 4, 200, 40, 8193 | bfloat16 | A100 | 1267 ms | 8.6 ms | 148x |
| 8, 100, 20, 1025 | float32 | H200 | 488 ms | 1.0 ms | 480x |
| 4, 200, 40, 8193 | float32 | H200 | 993 ms | 5.0 ms | 199x |
| 4, 200, 40, 8193 | bfloat16 | H200 | 896 ms | 4.9 ms | 183x |

Fine-tuning [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) on LibriSpeech
(batch size 8, bfloat16 autocast, 300 steps, same data order and seed) on an H200 gives the same loss curve as the
PyTorch loss (identical first-step loss, median per-step difference of 0.0015), with a full training step going from
1305 ms to 144 ms (9x) and peak memory from 33.2 GiB to 21.5 GiB.
