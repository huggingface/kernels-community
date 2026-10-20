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
"""NVFP4 CSF scales: decode + GEMM latency and scale memory against dense scales.

Two 512-expert MoE projections (gate: N=640, K=2560; down: N=2560, K=640) are quantized per
expert from random bf16 weights with ``nvfp4_quantize_two_level``, and their E4M3 scales are
CSF-encoded with the tests' fixture encoder. Both arms run the same NVFP4 GEMM:

  dense   dense scales resident
  csf     ``decompress_nvfp4_csf_scales`` into one shared workspace, then the GEMM

Decode routes 1 token per expert through ``matmul_batched``; prefill routes ``--prefill``
tokens per expert through ``matmul_grouped``. Each arm is captured in a CUDA graph and timed
warm (``do_bench_cudagraph``) and L2-flushed (``do_bench`` on the replay, which clears L2
outside the timed interval). Tuning finishes before timing. The csf arm's scales and outputs
are checked bit-identical to the dense arm's.

Scales come from synthetic weights. How well CSF compresses depends on how far the scale
bytes spread within a row, so real checkpoints can compress better or worse.

Writes ``bench/bench_moe_csf.json``, ``bench/bench_moe_csf_latency.png`` and
``bench/bench_moe_csf_memory.png`` beside this file.

Run: python bench/bench_moe_csf.py                 (measure, then plot)
     python bench/bench_moe_csf.py --replot        (rebuild the figures from the JSON)
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "torch-ext"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import torch  # noqa: E402

import finegrained_kernels  # noqa: E402
from finegrained_kernels.quant import nvfp4_quantize_two_level  # noqa: E402
from utils import encode_csf_scales  # noqa: E402  tests/utils.py fixture encoder

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(OUT_DIR, "bench_moe_csf.json")
PROJECTIONS = (("gate", 640, 2560), ("down", 2560, 640))
BLUE, ORANGE, GRAY = "#377eb8", "#ff9d36", "#969696"


def latency(fn):
    """Warm and L2-flushed CUDA-graph replay of ``fn``, in microseconds."""
    from triton.testing import do_bench, do_bench_cudagraph

    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return dict(
        warm_us=1000 * do_bench_cudagraph(fn, rep=200),
        cold_us=1000 * do_bench(graph.replay, warmup=100, rep=200, return_mode="median"),
    )


def quantize_stack(experts, N, K):
    """Per-expert two-level NVFP4: packed weights, E4M3 scales and fp32 globals."""
    weights, scales, globals_ = [], [], []
    for _ in range(experts):
        packed, scale, global_scale = nvfp4_quantize_two_level(
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        )
        weights.append(packed)
        scales.append(scale)
        globals_.append(global_scale.reshape(()))
    return torch.stack(weights), torch.stack(scales), torch.stack(globals_).float()


def measure(args):
    torch.manual_seed(1234)
    records = []
    for projection, N, K in PROJECTIONS:
        weights, dense, global_scale = quantize_stack(args.experts, N, K)
        fixed, exceptions = encode_csf_scales(dense)
        streams = finegrained_kernels.prepare_nvfp4_csf_scales(
            fixed.cuda(), [stream.cuda() for stream in exceptions]
        )
        workspace = torch.empty_like(dense)
        decoded = finegrained_kernels.decompress_nvfp4_csf_scales(*streams)
        assert torch.equal(decoded.view(torch.uint8), dense.view(torch.uint8))
        memory = dict(
            csf_bytes=sum(t.numel() * t.element_size() for t in streams),
            dense_bytes=dense.numel(),
        )
        for regime, tokens in (("decode", 1), ("prefill", args.prefill)):
            A = torch.randn(
                tokens * args.experts, K, device="cuda", dtype=torch.bfloat16
            )
            kwargs = dict(activation_format="nvfp4", b_global_scale=global_scale)
            if regime == "decode":
                op = finegrained_kernels.matmul_batched
                kwargs["expert_ids"] = torch.arange(
                    args.experts, device="cuda", dtype=torch.int32
                )
            else:
                op = finegrained_kernels.matmul_grouped
                kwargs["expert_start"] = tokens * torch.arange(
                    args.experts + 1, device="cuda", dtype=torch.int32
                )

            def run_dense():
                return op(A, weights, Bs=dense, **kwargs)

            def run_csf():
                finegrained_kernels.decompress_nvfp4_csf_scales(*streams, out=workspace)
                return op(A, weights, Bs=workspace, **kwargs)

            assert torch.equal(run_csf(), run_dense())
            record = dict(
                device=torch.cuda.get_device_name(),
                projection=projection,
                N=N,
                K=K,
                experts=args.experts,
                regime=regime,
                tokens_per_expert=tokens,
                **memory,
                dense=latency(run_dense),
                csf=latency(run_csf),
            )
            print(json.dumps(record), flush=True)
            records.append(record)
    with open(RESULTS, "w") as f:
        json.dump(records, f, indent=2)
    return records


def plot(records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    device, experts = records[0]["device"], records[0]["experts"]
    sizes = {r["projection"]: r for r in records}
    dense_mb = [sizes[p]["dense_bytes"] / 1e6 for p, _, _ in PROJECTIONS]
    csf_mb = [sizes[p]["csf_bytes"] / 1e6 for p, _, _ in PROJECTIONS]
    workspace_mb = max(dense_mb)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax = axes[0]
    x = np.arange(len(PROJECTIONS))
    ax.bar_label(ax.bar(x - 0.19, dense_mb, 0.38, label="Dense scales", color=BLUE), fmt="%.2f", padding=3)
    ax.bar_label(ax.bar(x + 0.19, csf_mb, 0.38, label="CSF streams", color=ORANGE), fmt="%.2f", padding=3)
    ax.set(
        xticks=x,
        xticklabels=[f"{p.title()} projection" for p, _, _ in PROJECTIONS],
        ylabel="Scale storage (MB, decimal)",
        ylim=(0, max(dense_mb) * 1.25),
        title="Stored scales per expert stack",
    )
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    layers = np.array([1, 2, 4, 8, 16])
    x = np.arange(len(layers))
    dense = layers * sum(dense_mb)
    csf = layers * sum(csf_mb)
    ax.bar_label(ax.bar(x - 0.19, dense, 0.38, label="Dense scales", color=BLUE), fmt="%.0f", padding=3, fontsize=9)
    ax.bar(x + 0.19, csf, 0.38, label="CSF streams", color=ORANGE)
    bars = ax.bar(
        x + 0.19, np.full(len(x), workspace_mb), 0.38, bottom=csf,
        label="One shared dense workspace", color=GRAY, hatch="///", edgecolor="white",
    )
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in csf + workspace_mb], padding=3, fontsize=9)
    ax.set(
        xticks=x,
        xticklabels=layers,
        xlabel="Resident layers (one gate + down stack per layer)",
        ylabel="Scales + shared workspace (MB)",
        ylim=(0, max(dense) * 1.18),
        title="Reusing one workspace across layers",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"NVFP4 CSF scale memory — {experts} experts, scale tensors only", fontsize=15, y=1.01)
    fig.text(
        0.5, -0.06,
        "Synthetic scales from random weights; the right panel scales the measured per-stack sizes.\n"
        "FP4 weights, activations and other model memory are excluded. Compression is lossless.",
        ha="center", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bench_moe_csf_memory.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for ax, key, title in zip(
        axes, ("warm_us", "cold_us"), ("Warm CUDA-graph replay", "L2-flushed CUDA-graph replay")
    ):
        x = np.arange(len(records))
        dense = [r["dense"][key] for r in records]
        csf = [r["csf"][key] for r in records]
        ax.bar_label(ax.bar(x - 0.19, dense, 0.38, color=BLUE, label="Dense scales + NVFP4 GEMM"), fmt="%.1f", padding=3, fontsize=10)
        ax.bar_label(ax.bar(x + 0.19, csf, 0.38, color=ORANGE, label="CSF decode + same NVFP4 GEMM"), fmt="%.1f", padding=3, fontsize=10)
        labels = [
            f"{r['projection'].title()} {r['regime']}\n{r['tokens_per_expert']} token{'s' if r['tokens_per_expert'] != 1 else ''}/expert"
            for r in records
        ]
        ax.set(xticks=x, xticklabels=labels, ylabel="Latency (µs)", title=title, ylim=(0, max(dense + csf) * 1.28))
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"{device}: CSF decode cost on top of the NVFP4 GEMM", fontsize=15, y=1.01)
    fig.text(
        0.5, -0.06,
        f"{experts} experts; decode + GEMM measured together. Lower is faster.\n"
        "Synthetic NVFP4 weights and scales. Kernel timings, not model throughput.",
        ha="center", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bench_moe_csf_latency.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--prefill", type=int, default=16, help="tokens per expert in the prefill regime")
    parser.add_argument("--replot", action="store_true", help="rebuild the figures from the JSON")
    args = parser.parse_args()
    if args.replot:
        with open(RESULTS) as f:
            records = json.load(f)
    else:
        records = measure(args)
    plot(records)


if __name__ == "__main__":
    main()
