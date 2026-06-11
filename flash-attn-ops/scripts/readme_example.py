"""Minimal runnable example for the flash-attn-ops kernel.

    python scripts/readme_example.py

Requires a CUDA device plus `kernels` (`pip install -U kernels`).
"""

import torch
from kernels import get_kernel

flash_attn_ops = get_kernel("kernels-community/flash-attn-ops")


def main():
    device = "cuda"
    batch, vocab = 8, 32000

    # Cross-entropy (the motivating import: flash_attn.losses.cross_entropy).
    logits = torch.randn(batch, vocab, device=device, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, vocab, (batch,), device=device)
    loss = flash_attn_ops.CrossEntropyLoss(reduction="mean")(logits, labels)
    loss.backward()
    print(f"cross_entropy loss: {loss.item():.4f}")

    # RMSNorm.
    x = torch.randn(batch, 1024, device=device, dtype=torch.bfloat16)
    w = torch.randn(1024, device=device, dtype=torch.bfloat16)
    y = flash_attn_ops.rms_norm_fn(x, w, None, eps=1e-6)
    print(f"rms_norm out: {tuple(y.shape)}")

    # Rotary embedding.
    q = torch.randn(batch, 128, 8, 64, device=device, dtype=torch.bfloat16)
    angle = torch.randn(128, 32, device=device)
    cos, sin = angle.cos().to(q.dtype), angle.sin().to(q.dtype)
    q_rot = flash_attn_ops.apply_rotary(q, cos, sin)
    print(f"apply_rotary out: {tuple(q_rot.shape)}")


if __name__ == "__main__":
    main()
