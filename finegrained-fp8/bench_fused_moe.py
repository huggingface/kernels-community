"""Benchmark MoE dispatch methods: correctness matrix + performance sweep + plot.

Usage:
    python bench_fused_moe.py --grouped    # grouped variants
    python bench_fused_moe.py --batched    # batched variants
    python bench_fused_moe.py --all        # all variants
"""

import argparse
import sys

sys.path.append("torch-ext")

import torch
import triton
import matplotlib
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt

from finegrained_fp8 import (
    moe_grouped,
    moe_batched,
    moe_grouped_fused,
    moe_batched_fused,
    moe_grouped_atomic,
    moe_batched_atomic,
)

matplotlib.use("Agg")

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

console = Console()

GROUPED_METHODS = {
    "grouped": moe_grouped,
    "grouped_fused": moe_grouped_fused,
    "grouped_atomic": moe_grouped_atomic,
}

BATCHED_METHODS = {
    "batched": moe_batched,
    "batched_fused": moe_batched_fused,
    "batched_atomic": moe_batched_atomic,
}

ALL_METHODS = {**GROUPED_METHODS, **BATCHED_METHODS}


def quantize_weights_block(W, block_n=128, block_k=128):
    E, N, K = W.shape
    Wq = torch.empty_like(W, dtype=FP8_DTYPE)
    Bs = torch.empty(
        E, N // block_n, K // block_k, dtype=torch.float32, device=W.device
    )
    for e in range(E):
        for ni in range(N // block_n):
            for ki in range(K // block_k):
                block = W[
                    e,
                    ni * block_n : (ni + 1) * block_n,
                    ki * block_k : (ki + 1) * block_k,
                ]
                amax = block.abs().amax().clamp(min=1e-12)
                scale = FP8_MAX / amax
                Wq[
                    e,
                    ni * block_n : (ni + 1) * block_n,
                    ki * block_k : (ki + 1) * block_k,
                ] = (block * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
                Bs[e, ni, ki] = 1.0 / scale
    return Wq, Bs


def diff_emoji(d):
    if d == 0:
        return "✅", "exact", "green"
    elif d < 1:
        return "🟢", f"{d:.4f}", "green"
    elif d < 100:
        return "🟡", f"{d:.1f}", "yellow"
    elif d < 1000:
        return "🟠", f"{d:.1f}", "dark_orange"
    else:
        return "🔴", f"{d:.1f}", "red"


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE dispatch methods")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--grouped", action="store_true", help="Benchmark grouped variants")
    group.add_argument("--batched", action="store_true", help="Benchmark batched variants")
    group.add_argument("--all", action="store_true", help="Benchmark all variants")
    cli_args = parser.parse_args()

    if cli_args.all:
        METHODS = ALL_METHODS
        variant = "all"
    elif cli_args.batched:
        METHODS = BATCHED_METHODS
        variant = "batched"
    else:
        METHODS = GROUPED_METHODS
        variant = "grouped"

    device = "cuda"
    block_size = [128, 128]

    # Qwen3-30B-A3B-Instruct dimensions
    model_name = "Qwen3-30B-A3B"
    E, N_inter, K, top_k = 128, 768, 2048, 8

    torch.manual_seed(42)
    W_gu = torch.randn(E, 2 * N_inter, K, dtype=torch.float32, device=device)
    gate_up_proj, gate_up_proj_scale_inv = quantize_weights_block(W_gu)
    W_down = torch.randn(E, K, N_inter, dtype=torch.float32, device=device)
    down_proj, down_proj_scale_inv = quantize_weights_block(W_down)

    names = list(METHODS.keys())

    # ═══════════════════════════════════════════════════════════════════════════
    # Correctness
    # ═══════════════════════════════════════════════════════════════════════════
    console.rule("[bold]Correctness[/bold]")

    num_tokens = 32
    hidden = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16)
    top_k_idx = torch.randint(0, E, (num_tokens, top_k), device=device)
    top_k_wts = torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16)

    args = (
        hidden,
        top_k_idx,
        top_k_wts,
        gate_up_proj,
        down_proj,
        gate_up_proj_scale_inv,
        down_proj_scale_inv,
        block_size,
    )

    # For correctness, fused methods use simulate_unfused=True to match unfused precision
    def call_method(name, fn):
        if name not in ("grouped", "batched"):
            return fn(*args, simulate_unfused=True)
        return fn(*args)

    outputs = {}
    with torch.no_grad():
        for name, fn in METHODS.items():
            outputs[name] = call_method(name, fn)

    outputs2 = {}
    with torch.no_grad():
        for name, fn in METHODS.items():
            outputs2[name] = call_method(name, fn)

    # Self-consistency table
    det_table = Table(title="Self-consistency (determinism)")
    det_table.add_column("Method", style="cyan")
    det_table.add_column("Max diff", justify="right")
    det_table.add_column("Status", justify="center")

    for name in names:
        d = (outputs[name].float() - outputs2[name].float()).abs().max().item()
        if d == 0:
            det_table.add_row(name, "0.00", "[green]✅ deterministic[/green]")
        else:
            det_table.add_row(
                name, f"{d:.2f}", f"[yellow]⚠️  non-deterministic ({d:.1f})[/yellow]"
            )

    console.print(det_table)

    # Parity matrix
    parity_table = Table(title="Parity matrix (max abs diff)")
    parity_table.add_column("", style="cyan")
    for name in names:
        parity_table.add_column(name[:14], justify="center")

    for n1 in names:
        cells = []
        for n2 in names:
            d = (outputs[n1].float() - outputs[n2].float()).abs().max().item()
            emoji, text, color = diff_emoji(d)
            cells.append(f"[{color}]{emoji} {text}[/{color}]")
        parity_table.add_row(n1, *cells)

    console.print(parity_table)

    # ═══════════════════════════════════════════════════════════════════════════
    # Benchmark
    # ═══════════════════════════════════════════════════════════════════════════
    console.rule(
        f"[bold]Benchmark — {model_name} (E={E}, N_inter={N_inter}, K={K}, top_k={top_k})[/bold]"
    )

    def run_sweep(bench_fn_factory, title, mode="eager"):
        """Run adaptive token sweep, stopping when TFLOPS plateau (<5% gain)."""
        table = Table(title=title)
        table.add_column("Tokens", justify="right", style="bold")
        for name in names:
            table.add_column(name[:14], justify="right")
        table.add_column("Winner", justify="center", style="bold green")

        all_res = []
        prev_tflops = {}  # per-method previous TFLOPS
        active = set(names)  # methods still progressing
        num_tokens = 1

        while num_tokens <= 131072 and active:  # stop when all methods plateau
            hidden = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16)
            top_k_idx = torch.randint(0, E, (num_tokens, top_k), device=device)
            top_k_wts = torch.randn(
                num_tokens, top_k, device=device, dtype=torch.bfloat16
            )

            args = (
                hidden,
                top_k_idx,
                top_k_wts,
                gate_up_proj,
                down_proj,
                gate_up_proj_scale_inv,
                down_proj_scale_inv,
                block_size,
            )
            results = {}
            for name, fn in METHODS.items():
                msg = f"  [dim]benching {name} @ {num_tokens} tokens...[/dim]"
                console.print(msg + " " * 40, end="\r")

                @torch.no_grad()
                def _bench(fn=fn):
                    return fn(*args)

                try:
                    results[name] = bench_fn_factory(_bench)
                except Exception as e:
                    console.print(f"  [red]{name} @ {num_tokens} tokens: {e}[/red]")
                    results[name] = float("inf")

            all_res.append((num_tokens, results))

            # Compute per-method TFLOPS and check progress
            S = num_tokens * top_k
            flops = 2 * S * K * 2 * N_inter + 2 * S * N_inter * K
            valid = {n: ms for n, ms in results.items() if ms != float("inf")}
            best_name = min(valid, key=valid.get) if valid else "n/a"

            cells = []
            for name in names:
                ms = results[name]
                if ms == float("inf"):
                    cells.append("[red]n/a[/red]")
                    continue
                tflops = flops / (ms * 1e-3) / 1e12

                # Check if this method is still progressing
                if name in prev_tflops and num_tokens >= 1024:
                    increase = (tflops - prev_tflops[name]) / max(
                        prev_tflops[name], 1e-12
                    )
                    if increase < 0.05:
                        active.discard(name)

                prev_tflops[name] = tflops

                if name not in active:
                    cells.append(f"[dim]{ms:.3f}[/dim]")
                elif name == best_name:
                    cells.append(f"[bold green]{ms:.3f}[/bold green]")
                else:
                    cells.append(f"{ms:.3f}")

            table.add_row(
                str(num_tokens),
                *cells,
                best_name if results[best_name] != float("inf") else "n/a",
            )

            num_tokens *= 2

        console.print(table)
        return all_res

    all_results = run_sweep(
        lambda fn: triton.testing.do_bench(fn, warmup=10, rep=50),
        "Latency — Eager (ms)",
    )

    all_results_cg = run_sweep(
        lambda fn: triton.testing.do_bench_cudagraph(fn),
        "Latency — CUDA Graphs (ms)",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Save CSV
    # ═══════════════════════════════════════════════════════════════════════════
    import csv

    csv_path = f"moe_results_{variant}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tokens", "method", "mode", "time_ms", "tflops"])
        for mode, results_list in [
            ("eager", all_results),
            ("cudagraph", all_results_cg),
        ]:
            for num_tokens, results in results_list:
                S = num_tokens * top_k
                flops = 2 * S * K * 2 * N_inter + 2 * S * N_inter * K
                for name, ms in results.items():
                    if ms == float("inf"):
                        continue
                    tflops = flops / (ms * 1e-3) / 1e12
                    writer.writerow(
                        [num_tokens, name, mode, f"{ms:.3f}", f"{tflops:.2f}"]
                    )
    console.print(f"Results saved to [bold]{csv_path}[/bold]")

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot
    # ═══════════════════════════════════════════════════════════════════════════
    colors = {
        "grouped": "#e74c3c",
        "grouped_fused": "#2ecc71",
        "grouped_atomic": "#3498db",
        "batched": "#c0392b",
        "batched_fused": "#27ae60",
        "batched_atomic": "#2980b9",
    }
    linestyles = {
        "grouped": "--",
        "grouped_fused": "-",
        "grouped_atomic": ":",
        "batched": "--",
        "batched_fused": "-",
        "batched_atomic": ":",
    }

    def compute_tflops(results_list):
        data = {}
        for name in names:
            xs, ys = [], []
            for num_tokens, results in results_list:
                S = num_tokens * top_k
                flops = 2 * S * K * 2 * N_inter + 2 * S * N_inter * K
                ms = results.get(name, float("inf"))
                if ms == float("inf"):
                    continue
                tflops = flops / (ms * 1e-3) / 1e12
                xs.append(num_tokens)
                ys.append(tflops)
            data[name] = (xs, ys)
        return data

    tflops_eager = compute_tflops(all_results)
    tflops_cg = compute_tflops(all_results_cg)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for col, (tflops_data, mode) in enumerate(
        [(tflops_eager, "Eager"), (tflops_cg, "CUDA Graphs")]
    ):
        ax = axes[col]
        for name in names:
            xs, ys = tflops_data[name]
            if xs:
                ax.plot(
                    xs,
                    ys,
                    linestyles.get(name, "-"),
                    marker="o",
                    markersize=4,
                    color=colors.get(name, "gray"),
                    label=name,
                    linewidth=2,
                )
        ax.set_xlabel("Tokens", fontsize=12)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_title(f"{mode} — TFLOPS", fontsize=13, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"FP8 MoE Expert Dispatch — {model_name} (E={E}, N_inter={N_inter}, K={K}, top_k={top_k})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plot_path = f"moe_tflops_{variant}.png"
    fig.savefig(plot_path, dpi=150)
    console.print(f"\nPlot saved to [bold]{plot_path}[/bold]")
    plt.close(fig)


if __name__ == "__main__":
    main()
