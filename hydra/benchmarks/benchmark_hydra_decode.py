from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch
from kernels import get_local_kernel


def load_local_kernel(repo: Path):
    for variant in (repo / "build", repo):
        if (variant / "metadata.json").exists():
            return get_local_kernel(variant)

    sys.path.insert(0, str(repo / "torch-ext"))
    return importlib.import_module("hydra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydra decode microbenchmark")
    parser.add_argument("--repo", default=".", help="Path to the hydra kernel directory")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--window", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    kernel = load_local_kernel(Path(args.repo))
    q = torch.randn(
        args.batch,
        args.heads,
        1,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        args.batch,
        args.kv_heads,
        args.tokens,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    window = None if args.window <= 0 else args.window
    for _ in range(args.warmup):
        kernel.hydra(q, k, v, sliding_window=window)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        kernel.hydra(q, k, v, sliding_window=window)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = elapsed * 1000.0 / args.iters
    print(
        "hydra_decode "
        f"B={args.batch} H={args.heads} Hkv={args.kv_heads} "
        f"Tkv={args.tokens} D={args.head_dim} window={window or 0} "
        f"iters={args.iters} ms_per_iter={ms:.4f}"
    )


if __name__ == "__main__":
    main()
