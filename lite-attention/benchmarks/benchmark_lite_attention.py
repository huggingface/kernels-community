import argparse
import time
import warnings

import torch

from lite_attention import LiteAttention


def make_qkv(batch, seqlen, heads, head_dim, dtype=torch.bfloat16):
    q = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def bench(name, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iters
    print(f"{name}: {elapsed_ms:.3f} ms")
    return elapsed_ms


def skip_fraction(attn):
    if attn.read_list is None:
        return 0.0
    return 1.0 - float(attn.calc_percentage_per_head(attn.read_list).mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError("LiteAttention Hopper path requires SM90+")

    torch.manual_seed(0)
    q, k, v = make_qkv(args.batch, args.seqlen, args.heads, args.head_dim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        baseline = LiteAttention(enable_skipping=False)
    baseline_ms = bench(
        "baseline_no_skip",
        lambda: baseline(q, k, v),
        args.warmup,
        args.iters,
    )

    for threshold in [-10.0, -3.0]:
        attn = LiteAttention(threshold=threshold)
        ms = bench(
            f"bf16_skip_threshold_{threshold:g}",
            lambda: attn(q, k, v),
            args.warmup,
            args.iters,
        )
        print(
            f"  speedup={baseline_ms / ms:.3f}x "
            f"skip_fraction={skip_fraction(attn):.3f}"
        )

    int8_attn = LiteAttention(threshold=-10.0, use_int8=True)
    int8_ms = bench(
        "int8_skip_threshold_-10",
        lambda: int8_attn(q, k, v),
        args.warmup,
        args.iters,
    )
    print(
        f"  speedup={baseline_ms / int8_ms:.3f}x "
        f"skip_fraction={skip_fraction(int8_attn):.3f}"
    )


if __name__ == "__main__":
    main()
