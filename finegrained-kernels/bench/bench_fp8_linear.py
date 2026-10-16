# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dense rowwise FP8 / coarse W8A16 benchmarks through the public kernel loader.

Example (LOCAL_KERNELS selects the build being reviewed):
  FINEGRAINED_AUTOTUNE_TRIALS=100 python bench/bench_fp8_linear.py \
      --family rowwise --shapes '1,16384,5120;4096,16384,5120' --output results.jsonl

Warm timings replay one operation repeatedly. Cold timings capture one operation,
then flush L2 between replays, excluding Python/descriptor setup. These are kernel
microbenchmarks, not whole-model throughput. Conversion costs and the persistent
BF16-weight baseline are reported separately.
"""

import argparse
import contextlib
import importlib
import json
import os
import statistics
from pathlib import Path

import kernels
import torch
import triton
import triton.testing


def timing(fn):
    for _ in range(3):
        fn()
    warm = [triton.testing.do_bench_cudagraph(fn, rep=60) * 1000 for _ in range(5)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    cold = [
        triton.testing.do_bench(
            graph.replay, warmup=50, rep=150, quantiles=[0.5, 0.2, 0.8]
        )[0]
        * 1000
        for _ in range(3)
    ]
    # Retain captured outputs while collecting both sets of measurements.
    del result
    return {
        "warm_median_us": statistics.median(warm),
        "warm_samples_us": warm,
        "cold_median_us": statistics.median(cold),
        "cold_samples_us": cold,
    }


@contextlib.contextmanager
def disable_scaled_mm(disabled):
    previous = os.environ.get("FINEGRAINED_DISABLE_SCALED_MM")
    if disabled:
        os.environ["FINEGRAINED_DISABLE_SCALED_MM"] = "1"
    else:
        os.environ.pop("FINEGRAINED_DISABLE_SCALED_MM", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FINEGRAINED_DISABLE_SCALED_MM", None)
        else:
            os.environ["FINEGRAINED_DISABLE_SCALED_MM"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["rowwise", "weight_only"], required=True)
    parser.add_argument(
        "--shapes", default="1,16384,5120;16,16384,5120;256,16384,5120;4096,16384,5120"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = int(os.environ.get("FINEGRAINED_AUTOTUNE_TRIALS", "100"))
    if trials < 100:
        parser.error("performance runs require FINEGRAINED_AUTOTUNE_TRIALS>=100")
    kernel = kernels.get_kernel("kernels-community/finegrained-kernels", version=0)
    quant = importlib.import_module(kernel.__name__ + ".quant")
    shapes = [tuple(map(int, shape.split(","))) for shape in args.shapes.split(";")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "trials": trials,
            }
        ),
        flush=True,
    )
    with torch.no_grad():
        for m, n, k in shapes:
            torch.manual_seed(42)
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(n, k, device="cuda") * 0.05
            row = {"family": args.family, "M": m, "N": n, "K": k, "trials": trials}
            print("START " + json.dumps(row), flush=True)
            if args.family == "rowwise":
                scales = weight.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
                b = (weight / scales).to(torch.float8_e4m3fn)
                aq, asc = quant.fp8_act_quant_tensor_wide(a, k)
                reference = (aq.float() * asc.reshape(-1, 1)) @ (b.float() * scales).T
                dequantized = (b.float() * scales).to(a.dtype)

                def scaled_mm(q, scale):
                    if hasattr(torch.nn.functional, "scaled_mm"):
                        recipe = torch.nn.functional.ScalingType.RowWise
                        return torch.nn.functional.scaled_mm(
                            q,
                            b.T,
                            scale.reshape(-1, 1),
                            recipe,
                            scales.T,
                            recipe,
                            output_dtype=a.dtype,
                        )
                    return torch._scaled_mm(
                        q, b.T, scale.reshape(-1, 1), scales.T, out_dtype=a.dtype
                    )

                def scaled_e2e():
                    q, scale = quant.fp8_act_quant_tensor_wide(a, k)
                    return scaled_mm(q, scale)

                def run():
                    return kernel.matmul_2d(a, b, None, scales, output_dtype=a.dtype)

                baselines = {
                    "scaled_mm_gemm_only": lambda: scaled_mm(aq, asc),
                    "scaled_mm_including_quant": scaled_e2e,
                    "activation_quant_only": lambda: quant.fp8_act_quant_tensor_wide(
                        a, k
                    ),
                    "bf16_predequantized_weights": lambda: a @ dequantized.T,
                }
            else:
                if n % 128 or k % 128:
                    parser.error(
                        "weight_only benchmark quantization requires N and K divisible by128"
                    )
                scales = (
                    weight.reshape(n // 128, 128, k // 128, 128)
                    .abs()
                    .amax((1, 3))
                    .clamp_min(1e-12)
                    / 448
                )
                expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
                b = (weight / expanded).to(torch.float8_e4m3fn)
                dequantized = (b.float() * expanded).to(a.dtype)
                reference = a.float() @ dequantized.float().T

                def run():
                    return kernel.matmul_2d(
                        a,
                        b,
                        None,
                        scales,
                        activation_format="bf16",
                        output_dtype=a.dtype,
                    )

                def torch_dequant_mm():
                    full_scale = scales.repeat_interleave(128, 0).repeat_interleave(
                        128, 1
                    )
                    return a @ (b.float() * full_scale).to(a.dtype).T

                baselines = {
                    # This baseline holds a persistent BF16 weight matrix (2x the value storage).
                    "bf16_predequantized_weights": lambda: a @ dequantized.T,
                    "torch_dequant_including_mm": torch_dequant_mm,
                }
            if args.family == "weight_only":
                try:
                    workspace = importlib.import_module(
                        kernel.__name__ + ".weight_only"
                    )
                except ModuleNotFoundError:
                    pass  # The baseline revision may predate the fused dequantizer.
                else:

                    def fused_dequant_mm():
                        decoded = workspace.dequantize_weight(
                            b, scales, a.dtype, 128, 128
                        )
                        return a @ decoded.T

                    baselines["fused_dequant_including_mm"] = fused_dequant_mm
                    baselines["fused_dequant_only"] = (
                        lambda: workspace.dequantize_weight(
                            b, scales, a.dtype, 128, 128
                        )
                    )
            del weight
            with disable_scaled_mm(False):
                actual = run()
                torch.testing.assert_close(
                    actual.float(), reference, rtol=0.02, atol=0.02
                )
                row["finegrained_default"] = timing(run)
            if args.family == "rowwise":
                with disable_scaled_mm(True):
                    torch.testing.assert_close(
                        run().float(), reference, rtol=0.02, atol=0.02
                    )
                    row["finegrained_triton"] = timing(run)
            for name, fn in baselines.items():
                try:
                    row[name] = timing(fn)
                except (RuntimeError, NotImplementedError) as exc:
                    row[name] = {"unsupported": str(exc)}
            row["relative_l2"] = (
                torch.linalg.vector_norm(actual.float() - reference)
                / torch.linalg.vector_norm(reference)
            ).item()
            with args.output.open("a") as output:
                output.write(json.dumps(row) + "\n")
            print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
