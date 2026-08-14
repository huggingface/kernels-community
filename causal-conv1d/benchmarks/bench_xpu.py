"""Standalone XPU sweep for causal-conv1d.

Reports per-shape latency and achieved bandwidth so that kernel changes can be
compared against a recorded baseline. causal-conv1d is memory bound, so the
GB/s column is the number to watch.

Usage:
    PYTHONPATH=$(readlink -f result/torch213-cxx11-xpu20260-x86_64-linux) \
        python benchmarks/bench_xpu.py
    ... --json baseline.json          # record a baseline
    ... --compare baseline.json       # diff against a recorded baseline
"""

import argparse
import itertools
import json

import torch
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

DEVICE = "xpu"

# (batch, dim, seqlen). Covers decode (seqlen 1-4), the prefill/chunk sizes real
# models use, and deliberately ragged shapes so tail handling is exercised.
# dim values follow d_inner = 2 * d_model for Mamba-style blocks.
SHAPES = [
    # decode / short
    (1, 4096, 1),
    (32, 4096, 1),
    (128, 4096, 4),
    (32, 1024, 64),
    # prefill
    (16, 4096, 128),
    (8, 2048, 512),
    (1, 2048, 1024),
    (4, 2048, 2048),
    (1, 4096, 4096),
    (1, 2048, 16384),
    # ragged: neither dim nor seqlen is a multiple of the tile sizes
    (3, 1536, 333),
    (5, 2560, 1023),
]
WIDTHS = [2, 3, 4]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def torch_fwd(x, weight, bias):
    """The stock-PyTorch equivalent: a depthwise causal conv plus silu.

    This is what a user gets without this kernel, so it is the reference the
    absolute numbers should be judged against.
    """
    seqlen = x.shape[-1]
    width = weight.shape[-1]
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=x.shape[1])
    return F.silu(out[..., :seqlen])


def timeit(fn, warmup=25, iters=100, repeats=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    # The fastest block is reported: interference from other work on the device
    # only ever makes a block slower, so the minimum is the stable estimator.
    best = float("inf")
    for _ in range(repeats):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.xpu.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / iters)
    return best  # microseconds


def bench_fwd(batch, dim, seqlen, width, dtype, channel_last, ref=False):
    x = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=dtype)
    if channel_last:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype)
    us = timeit(lambda: causal_conv1d_fn(x, weight, bias, activation="silu"))
    # One read of x plus one write of out dominates the traffic.
    gbs = 2 * x.numel() * x.element_size() / (us * 1e-6) / 1e9
    ref_us = timeit(lambda: torch_fwd(x, weight, bias)) if ref else None
    return us, gbs, ref_us


def bench_bwd(batch, dim, seqlen, width, dtype, channel_last, ref=False):
    x = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=dtype)
    if channel_last:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    x = x.requires_grad_(True)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype, requires_grad=True)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype, requires_grad=True)
    out = causal_conv1d_fn(x, weight, bias, activation="silu")
    g = torch.randn_like(out)

    def step():
        x.grad = weight.grad = bias.grad = None
        out.backward(g, retain_graph=True)

    us = timeit(step)
    gbs = 4 * x.numel() * x.element_size() / (us * 1e-6) / 1e9

    ref_us = None
    if ref:
        ref_out = torch_fwd(x, weight, bias)

        def ref_step():
            x.grad = weight.grad = bias.grad = None
            ref_out.backward(g, retain_graph=True)

        ref_us = timeit(ref_step)
    return us, gbs, ref_us


def bench_update(batch, dim, width, dtype):
    x = torch.randn(batch, dim, 1, device=DEVICE, dtype=dtype)
    conv_state = torch.randn(batch, dim, width, device=DEVICE, dtype=dtype)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype)
    return timeit(
        lambda: causal_conv1d_update(x, conv_state, weight, bias, activation="silu")
    )


def key(*parts):
    return "/".join(str(p) for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--compare", help="compare against a previously recorded file")
    ap.add_argument("--bwd", action="store_true", help="also benchmark backward")
    ap.add_argument(
        "--ref",
        action="store_true",
        help="also time the stock-PyTorch depthwise conv and report the speedup",
    )
    args = ap.parse_args()

    assert torch.xpu.is_available(), "no XPU device"
    print(f"device: {torch.xpu.get_device_name(0)}  torch: {torch.__version__}\n")

    baseline = json.load(open(args.compare)) if args.compare else {}
    results = {}

    header = f"{'shape':>22} {'w':>2} {'dtype':>9} {'layout':>7} {'us':>9} {'GB/s':>8}"
    if args.ref:
        header += f" {'torch us':>9} {'vs torch':>9}"
    if baseline:
        header += f" {'vs base':>9}"
    print("== forward ==")
    print(header)
    for (batch, dim, seqlen), width, dtype in itertools.product(
        SHAPES, WIDTHS, DTYPES
    ):
        for channel_last in (False, True):
            us, gbs, ref_us = bench_fwd(
                batch, dim, seqlen, width, dtype, channel_last, args.ref
            )
            k = key(
                "fwd", batch, dim, seqlen, width, dtype, "cl" if channel_last else "cf"
            )
            results[k] = us
            line = (
                f"{f'{batch}x{dim}x{seqlen}':>22} {width:>2} "
                f"{str(dtype).replace('torch.', ''):>9} "
                f"{'chan-last' if channel_last else 'contig':>7} {us:>9.1f} {gbs:>8.1f}"
            )
            if ref_us is not None:
                line += f" {ref_us:>9.1f} {ref_us / us:>8.2f}x"
            if k in baseline:
                line += f" {baseline[k] / us:>8.2f}x"
            print(line)

    if args.bwd:
        print("\n== backward ==")
        print(header)
        for (batch, dim, seqlen), width, dtype in itertools.product(
            SHAPES, WIDTHS, DTYPES
        ):
            for channel_last in (False, True):
                us, gbs, ref_us = bench_bwd(
                    batch, dim, seqlen, width, dtype, channel_last, args.ref
                )
                k = key(
                    "bwd",
                    batch,
                    dim,
                    seqlen,
                    width,
                    dtype,
                    "cl" if channel_last else "cf",
                )
                results[k] = us
                line = (
                    f"{f'{batch}x{dim}x{seqlen}':>22} {width:>2} "
                    f"{str(dtype).replace('torch.', ''):>9} "
                    f"{'chan-last' if channel_last else 'contig':>7} "
                    f"{us:>9.1f} {gbs:>8.1f}"
                )
                if ref_us is not None:
                    line += f" {ref_us:>9.1f} {ref_us / us:>8.2f}x"
                if k in baseline:
                    line += f" {baseline[k] / us:>8.2f}x"
                print(line)

    print("\n== update (decode) ==")
    print(f"{'batch x dim':>22} {'w':>2} {'dtype':>9} {'us':>9}")
    for batch, dim in [(1, 2048), (32, 2048), (128, 4096)]:
        for width in WIDTHS:
            dtype = torch.float16
            us = bench_update(batch, dim, width, dtype)
            k = key("upd", batch, dim, width, dtype)
            results[k] = us
            line = f"{f'{batch}x{dim}':>22} {width:>2} {'float16':>9} {us:>9.1f}"
            if k in baseline:
                line += f" {baseline[k] / us:>8.2f}x"
            print(line)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
