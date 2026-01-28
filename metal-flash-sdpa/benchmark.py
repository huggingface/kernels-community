#!/usr/bin/env python3
"""Benchmark causal mask performance scaling with sequence length"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import kernels

metal_flash_sdpa = kernels.get_kernel("kernels-community/metal-flash-sdpa")


def create_cu_seqlens(seq_lengths: List[int]) -> torch.Tensor:
    """Create cumulative sequence lengths tensor."""
    cu_seqlens = [0]
    for length in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return torch.tensor(cu_seqlens, dtype=torch.int32, device="mps")


def benchmark_flash_sdpa_causal(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    num_iterations: int = 20,
) -> float:
    """Benchmark Flash SDPA with causal mask"""

    seq_lengths = [seq_len] * batch_size
    cu_seqlens = create_cu_seqlens(seq_lengths)
    total_tokens = sum(seq_lengths)

    # Create input tensors
    query = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="mps")
    key = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="mps")
    value = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="mps")
    out = torch.empty_like(query)

    scale = 1.0 / (head_dim**0.5)

    # Warmup
    for _ in range(5):
        metal_flash_sdpa.flash_attention_varlen(
            out=out,
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            do_causal=True,
            scale=scale,
            softcapping=1.0,
        )
    torch.mps.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        metal_flash_sdpa.flash_attention_varlen(
            out=out,
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            do_causal=True,
            scale=scale,
            softcapping=1.0,
        )
    torch.mps.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_iterations


def benchmark_naive_sdpa_causal(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    num_iterations: int = 20,
) -> float:
    """Benchmark naive SDPA with causal mask"""

    # Create input tensors
    query = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="mps"
    )
    key = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="mps"
    )
    value = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="mps"
    )

    scale = 1.0 / (head_dim**0.5)

    # Precompute causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device="mps"), diagonal=1).bool()

    # Warmup
    for _ in range(5):
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, value)
    torch.mps.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, value)
    torch.mps.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_iterations


def run_scaling_benchmark():
    """Run causal mask scaling benchmark"""

    print("=" * 80)
    print("Causal Mask Performance Scaling Benchmark")
    print("Batch Size: 4, Head Dimension: 64")
    print("=" * 80)

    # Configuration
    batch_size = 4
    num_heads = 16
    head_dim = 64
    dtype = torch.float16

    # Sequence lengths from 512 to 4096
    seq_lengths = [512, 768, 1024, 1536, 2048, 3072, 4096]

    flash_times = []
    naive_times = []
    speedups = []

    print(f"{'Seq Len':<8} {'Flash (ms)':<12} {'Naive (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    for seq_len in seq_lengths:
        # Benchmark Flash SDPA
        flash_time = benchmark_flash_sdpa_causal(
            batch_size, num_heads, seq_len, head_dim, dtype
        )
        flash_times.append(flash_time)

        # Benchmark Naive SDPA
        naive_time = benchmark_naive_sdpa_causal(
            batch_size, num_heads, seq_len, head_dim, dtype
        )
        naive_times.append(naive_time)

        speedup = naive_time / flash_time
        speedups.append(speedup)

        print(f"{seq_len:<8} {flash_time:<12.2f} {naive_time:<12.2f} {speedup:<10.2f}x")

    return seq_lengths, flash_times, naive_times, speedups


def create_line_plot(seq_lengths, flash_times, naive_times, speedups):
    """Create line graph visualization"""

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        "Causal Mask Performance Scaling\n(Batch Size: 4, Head Dimension: 64)",
        fontsize=16,
    )

    # Plot execution times
    ax.plot(
        seq_lengths,
        flash_times,
        marker="o",
        linewidth=3,
        markersize=10,
        label="Flash SDPA",
        color="blue",
    )
    ax.plot(
        seq_lengths,
        naive_times,
        marker="s",
        linewidth=3,
        markersize=10,
        label="Naive SDPA",
        color="red",
    )

    ax.set_xlabel("Sequence Length", fontsize=14)
    ax.set_ylabel("Time (ms)", fontsize=14)
    ax.set_title("Execution Time vs Sequence Length", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Add value annotations for all points
    for i, (seq_len, flash_time, naive_time) in enumerate(
        zip(seq_lengths, flash_times, naive_times)
    ):
        ax.annotate(
            f"{flash_time:.1f}ms",
            xy=(seq_len, flash_time),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            color="blue",
        )
        ax.annotate(
            f"{naive_time:.1f}ms",
            xy=(seq_len, naive_time),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            color="red",
        )

    # Set axis limits to better show the data
    ax.set_xlim(seq_lengths[0] - 100, seq_lengths[-1] + 100)
    ax.set_ylim(0, max(naive_times) * 1.1)

    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_analysis(seq_lengths, flash_times, naive_times, speedups):
    """Print detailed analysis of the results"""

    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Performance scaling analysis
    print("\n1. Performance Scaling:")
    print(
        f"   • Flash SDPA: {flash_times[0]:.2f}ms → {flash_times[-1]:.2f}ms ({flash_times[-1] / flash_times[0]:.1f}x increase)"
    )
    print(
        f"   • Naive SDPA: {naive_times[0]:.2f}ms → {naive_times[-1]:.2f}ms ({naive_times[-1] / naive_times[0]:.1f}x increase)"
    )

    # Speedup analysis
    print("\n2. Speedup Analysis:")
    print(f"   • Average Speedup: {np.mean(speedups):.2f}x")
    print(
        f"   • Max Speedup: {np.max(speedups):.2f}x (at seq_len={seq_lengths[np.argmax(speedups)]})"
    )
    print(
        f"   • Min Speedup: {np.min(speedups):.2f}x (at seq_len={seq_lengths[np.argmin(speedups)]})"
    )

    # Efficiency analysis
    print("\n3. Efficiency Analysis:")
    speedup_improvement = speedups[-1] / speedups[0]
    print(f"   • Speedup improvement from 512→4096: {speedup_improvement:.2f}x")

    if speedup_improvement > 1.1:
        print("   • Flash SDPA becomes MORE efficient at longer sequences")
    elif speedup_improvement < 0.9:
        print("   • Flash SDPA becomes LESS efficient at longer sequences")
    else:
        print("   • Flash SDPA maintains consistent efficiency across sequence lengths")

    # Memory complexity analysis
    print("\n4. Theoretical Complexity:")
    print(f"   • Sequence length increased by: {seq_lengths[-1] / seq_lengths[0]:.1f}x")
    print(
        f"   • Theoretical O(n²) complexity increase: {(seq_lengths[-1] / seq_lengths[0]) ** 2:.1f}x"
    )
    print(f"   • Actual Flash SDPA increase: {flash_times[-1] / flash_times[0]:.1f}x")
    efficiency_ratio = (flash_times[-1] / flash_times[0]) / (
        (seq_lengths[-1] / seq_lengths[0]) ** 2
    )
    print(f"   • Flash SDPA efficiency ratio: {efficiency_ratio:.3f} (lower is better)")


def main():
    # Run the scaling benchmark
    seq_lengths, flash_times, naive_times, speedups = run_scaling_benchmark()

    # Create line plot visualization
    create_line_plot(seq_lengths, flash_times, naive_times, speedups)

    # Print detailed analysis
    print_analysis(seq_lengths, flash_times, naive_times, speedups)


if __name__ == "__main__":
    main()
