"""Minimal runnable example for `ShortConvolution` from the fla kernel.

    python scripts/short_conv_example.py

Requires a CUDA device plus `kernels` (`pip install -U kernels`).
"""

import torch
import torch.nn.functional as F
from kernels import get_kernel

fla = get_kernel("kernels-community/fla", version=1)
ShortConvolution = fla.modules.ShortConvolution


def reference(conv, x):
    """Causal depthwise conv1d in plain PyTorch, for comparison."""
    y = F.conv1d(
        x.transpose(1, 2),
        conv.weight,
        conv.bias,
        groups=conv.hidden_size,
        padding=conv.kernel_size[0] - 1,
    )[..., : x.shape[1]].transpose(1, 2)
    return F.silu(y) if conv.activation == "silu" else y


def main():
    device, dtype = "cuda", torch.bfloat16
    batch, seq_len, hidden_size, kernel_size = 2, 128, 512, 4

    conv = ShortConvolution(hidden_size, kernel_size, activation="silu", device=device, dtype=dtype)
    print(conv)

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)

    # Prefill: [B, T, D] -> [B, T, D], keeping the conv state for decoding.
    y, cache = conv(x, output_final_state=True)
    print(f"prefill out: {tuple(y.shape)}, cache: {tuple(cache.shape)}")
    print(f"max diff vs torch reference: {(y - reference(conv, x)).abs().max().item():.4f}")

    # Decode: one token at a time, `cache` is updated in place.
    x_new = torch.randn(batch, 1, hidden_size, device=device, dtype=dtype)
    y_new, cache = conv(x_new, cache=cache, output_final_state=True)
    print(f"decode out: {tuple(y_new.shape)}, cache: {tuple(cache.shape)}")

    # Same step through the full sequence: decoding matches prefilling.
    y_full, _ = conv(torch.cat([x, x_new], dim=1))
    print(f"max diff decode vs full prefill: {(y_new - y_full[:, -1:]).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
