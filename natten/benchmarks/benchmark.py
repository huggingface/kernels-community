# Micro-benchmark for the fused neighborhood attention path.
#
# Compares NATTEN's auto-selected fused backend against PyTorch SDPA over the
# full sequence (which computes strictly more attention, but is the baseline
# people care about) across a few window sizes.
#
# Run inside `kernel-builder devshell` / `testshell`, or any env where the
# built `natten` package is importable.

import argparse
import time

import torch
from torch.nn.functional import scaled_dot_product_attention

from natten.functional import na2d


def benchmark(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--size", type=int, default=64, help="2-D token layout side")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="bfloat16"
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    shape = (args.batch, args.size, args.size, args.heads, args.head_dim)

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # [B, H, seq, D] view for SDPA
    q_sdpa = q.flatten(1, 2).permute(0, 2, 1, 3).contiguous()
    k_sdpa = k.flatten(1, 2).permute(0, 2, 1, 3).contiguous()
    v_sdpa = v.flatten(1, 2).permute(0, 2, 1, 3).contiguous()

    sdpa_ms = benchmark(lambda: scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa))
    print(
        f"device={torch.cuda.get_device_name()} dtype={args.dtype} "
        f"layout={args.size}x{args.size} heads={args.heads} head_dim={args.head_dim}"
    )
    print(f"{'kernel':>24} {'ms':>10} {'vs SDPA':>10}")
    print(f"{'sdpa (full self-attn)':>24} {sdpa_ms:>10.3f} {'1.00x':>10}")

    for window in (7, 13, 21, 33):
        if window > args.size:
            continue
        na_ms = benchmark(lambda: na2d(q, k, v, kernel_size=(window, window)))
        print(
            f"{f'na2d k={window}x{window}':>24} {na_ms:>10.3f} "
            f"{f'{sdpa_ms / na_ms:.2f}x':>10}"
        )


if __name__ == "__main__":
    main()
