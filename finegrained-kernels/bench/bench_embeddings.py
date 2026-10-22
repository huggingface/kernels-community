# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Selected-row embedding lookup: fused kernels vs native and compiled PyTorch.

Every arm gathers the same quantized rows and returns BF16; no arm dequantizes the
whole table. Native references use only PyTorch operations, including FP4/UE8M0
unpacking. Compile uses fullgraph=True without automatic CUDA graphs, then all
three arms run through the same eager, graph-replay, and cold-weight timings.
Cold timings flush L2 before each lookup, outside the timed CUDA events. These
individual-event measurements have coarser resolution than hot replay timings;
use hot replay to compare sub-microsecond differences.

Run: python bench/bench_embeddings.py
     python bench/bench_embeddings.py --formats nvfp4 mxfp4 --widths 160
     python bench/bench_embeddings.py --smoke
     python bench/bench_embeddings.py --replot
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import torch
from triton.testing import do_bench, do_bench_cudagraph


matplotlib.use("Agg")
import matplotlib.pyplot as plt


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "torch-ext"))
from finegrained_kernels.embedding import embedding_lookup  # noqa: E402


FORMATS = ("fp8", "fp8_row", "fp8_block", "nvfp4", "mxfp8", "mxfp4", "int8", "int4")
IMPLS = ("finegrained-kernels", "torch", "torch.compile")
COLORS = ("#1f77b4", "#e377c2", "#ff7f0e")


def make_table(fmt, vocab, width):
    """Build checkpoint-shaped operands once; setup is excluded from timings."""
    if fmt.startswith("fp8") or fmt == "mxfp8":
        weight = (torch.randn(vocab, width, device="cuda") * 3).to(torch.float8_e4m3fn)
    elif fmt in ("nvfp4", "mxfp4", "int4"):
        shape = (vocab // 2, width) if fmt == "int4" else (vocab, width // 2)
        weight = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
    else:
        weight = torch.randint(-128, 128, (vocab, width), device="cuda", dtype=torch.int8)
    scale_shape = {
        "fp8": (1,),
        "fp8_row": (vocab,),
        "fp8_block": ((vocab + 127) // 128, (width + 127) // 128),
        "nvfp4": (vocab, width // 16),
        "mxfp8": (vocab, width // 32),
        "mxfp4": (vocab, width // 32),
        "int8": (vocab,),
        "int4": (vocab, width // 32),
    }[fmt]
    if fmt.startswith("mx"):
        scale = torch.randint(122, 129, scale_shape, device="cuda", dtype=torch.uint8)
    else:
        scale = torch.rand(scale_shape, device="cuda") * 0.2 + 0.02
        if fmt == "nvfp4":
            scale = scale.to(torch.float8_e4m3fn)
    global_scale = torch.tensor([0.0371], device="cuda") if fmt == "nvfp4" else None
    return weight, scale, global_scale


def torch_embedding(weight, ids, scale, global_scale, fmt, width, fp4_lut):
    """Gather first: FP4 packs adjacent columns, INT4 adjacent vocabulary rows."""
    if fmt == "int4":
        packed = weight[ids // 2].to(torch.int16)
        code = (packed >> ((ids % 2)[:, None] * 4)) & 15
        value = ((code ^ 8) - 8).float()
    elif fmt in ("nvfp4", "mxfp4"):
        cols = torch.arange(width, device=weight.device)
        packed = weight[ids[:, None], cols[None, :] // 2]
        code = (packed >> ((cols[None, :] % 2) * 4)) & 15
        value = fp4_lut[code.long()]
    else:
        value = weight[ids].float()
    if fmt == "fp8":
        selected_scale = scale
    elif fmt in ("fp8_row", "int8"):
        selected_scale = scale[ids, None]
    elif fmt == "fp8_block":
        cols = torch.arange(width, device=weight.device)
        selected_scale = scale[ids[:, None] // 128, cols[None, :] // 128]
    elif fmt in ("nvfp4", "mxfp4"):
        selected_scale = scale[ids[:, None], cols[None, :] // (16 if fmt == "nvfp4" else 32)].float()
        if fmt == "mxfp4":
            selected_scale = torch.exp2(selected_scale - 127)
    else:
        selected_scale = scale[ids].float()
        if fmt.startswith("mx"):
            selected_scale = torch.exp2(selected_scale - 127)
        selected_scale = selected_scale.repeat_interleave(16 if fmt == "nvfp4" else 32, -1)
    value = value * selected_scale
    if global_scale is not None:
        value = value * global_scale
    return value.to(torch.bfloat16)


def cold_latency(run, flush, repeats):
    """Measure lookup events after an L2 eviction, without charging for eviction."""
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flush.zero_()
        start.record()
        _output = run()
        end.record()
    samples = []
    for _ in range(repeats):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return sorted(samples)[len(samples) // 2]


def benchmark(args):
    torch.manual_seed(19)
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    fp4_lut = torch.tensor(magnitudes + tuple(-v for v in magnitudes), device="cuda")
    torch._dynamo.mark_static_address(fp4_lut)
    # B200 L2 is 126 MiB; twice the cache size also covers other GPUs.
    cache_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(max(256 * 1024**2, 2 * cache_bytes), device="cuda", dtype=torch.uint8)
    results = []
    for width in args.widths:
        for fmt in args.formats:
            weight, scale, global_scale = make_table(fmt, args.vocab, width)
            for tensor in (weight, scale, global_scale):
                if tensor is not None:
                    torch._dynamo.mark_static_address(tensor)
            for regime, rows in (("decode", 1), ("prefill", args.prefill)):
                torch._dynamo.reset()
                ids = torch.randint(args.vocab, (rows,), device="cuda")
                weight_format = "fp8" if fmt.startswith("fp8") else fmt

                def fused():
                    return embedding_lookup(
                        weight,
                        ids,
                        scale,
                        global_scale,
                        torch.bfloat16,
                        weight_format=weight_format,
                    )

                def native():
                    return torch_embedding(weight, ids, scale, global_scale, fmt, width, fp4_lut)

                compiled = torch.compile(native, mode="max-autotune-no-cudagraphs", fullgraph=True)
                reference = native()
                for impl, run in zip(IMPLS, (fused, native, compiled)):
                    output = run()
                    torch.testing.assert_close(output, reference, rtol=8e-3, atol=1e-5)
                    parity = (output.float() - reference.float()).abs().max().item()
                    times = {
                        "eager": do_bench(run, rep=args.rep, return_mode="median") * 1000,
                        "cudagraph": do_bench_cudagraph(run, rep=args.rep, return_mode="median") * 1000,
                        "cold": cold_latency(run, flush, 5 if args.smoke else 25),
                    }
                    print(
                        f"{fmt:9s} D={width:4d} {regime:7s} {impl:19s} "
                        f"eager={times['eager']:.2f} graph={times['cudagraph']:.2f} "
                        f"cold={times['cold']:.2f} us; max_abs={parity:.3g}",
                        flush=True,
                    )
                    for mode, latency in times.items():
                        results.append(
                            {
                                "format": fmt,
                                "width": width,
                                "vocab": args.vocab,
                                "regime": regime,
                                "rows": rows,
                                "impl": impl,
                                "mode": mode,
                                "latency_us": latency,
                                "max_abs": parity,
                            }
                        )
    return results


def plot(results, path, device):
    widths = list(dict.fromkeys(int(row["width"]) for row in results))
    fig, axes = plt.subplots(2 * len(widths), 2, figsize=(18, 7 * len(widths)), squeeze=False)
    for width_index, width in enumerate(widths):
        for col, regime in enumerate(("decode", "prefill")):
            cells = [r for r in results if int(r["width"]) == width and r["regime"] == regime]
            formats = list(dict.fromkeys(r["format"] for r in cells))
            times = {(r["format"], r["impl"]): float(r["latency_us"]) for r in cells if r["mode"] == "cudagraph"}
            for focused in (False, True):
                ax = axes[2 * width_index + int(focused), col]
                arms = ((IMPLS[0], COLORS[0]), (IMPLS[2], COLORS[2])) if focused else tuple(zip(IMPLS, COLORS))
                bar_width = 0.8 / len(arms)
                for arm, (impl, color) in enumerate(arms):
                    positions = [i + (arm - (len(arms) - 1) / 2) * bar_width for i in range(len(formats))]
                    bars = ax.bar(positions, [times[fmt, impl] for fmt in formats], bar_width, color=color, label=impl)
                    if focused:
                        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
                ax.set_xticks(range(len(formats)), formats)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
                ax.set_ylabel("latency (µs)")
                comparison = "custom vs compiled" if focused else "all implementations"
                ax.set_title(f"D={width} — {regime}, {cells[0]['rows']} rows — {comparison}")
                ax.legend(fontsize=8)
    fig.suptitle(
        f"Quantized embedding lookup — {device} — CUDA graph replay\n"
        "Linear axes; identical selected-row work. Cold-weight and host timings are retained in the CSV.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=FORMATS)
    parser.add_argument("--widths", nargs="+", type=int, default=(160, 4096))
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--prefill", type=int, default=2048)
    parser.add_argument("--rep", type=int, default=100, help="timing window in milliseconds")
    parser.add_argument("--smoke", action="store_true", help="256-row prefill and shorter timings")
    parser.add_argument("--replot", action="store_true", help="read the CSV without running kernels")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if (
        args.vocab <= 0
        or args.vocab % 2
        or any(width <= 0 or width % 32 for width in args.widths)
        or args.prefill <= 0
        or args.rep <= 0
    ):
        parser.error("vocab must be positive and even; widths multiples of 32; prefill and rep positive")
    if args.smoke:
        args.prefill, args.rep = 256, 20
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"bench_embeddings{args.suffix}.csv"
    png_path = csv_path.with_suffix(".png")
    if args.replot:
        with csv_path.open() as file:
            results = list(csv.DictReader(file))
    else:
        results = benchmark(args)
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)
    plot(results, png_path, torch.cuda.get_device_name(0))
    print(f"wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
