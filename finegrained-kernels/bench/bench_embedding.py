# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Compare fused lookup with selected-row PyTorch paths. Run through LOCAL_KERNELS.

Tables have one million rows, not the full PLE vocabulary. Hot trials reuse at most 16 rows;
cache-cold trials flush 256 MiB outside each timed CUDA-graph replay. The latter includes
single-replay/event overhead and is not directly comparable with batched hot timings.
"""

import argparse
import json

import kernels
import torch
import triton


def reference(weight, ids, scale, global_scale):
    rows = torch.nn.functional.embedding(ids, weight)
    if global_scale is None:
        return rows.to(scale.dtype) * scale
    packed = rows.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
    )
    scales = torch.nn.functional.embedding(ids, scale).float().repeat_interleave(16, -1)
    return (values[codes] * scales * global_scale).to(torch.bfloat16)


def capture(fn):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def cold_latency(graph, flush):
    timings = []
    for _ in range(40):
        flush.zero_()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    return sorted(timings)[len(timings) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=1_000_000)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 32, 2048])
    args = parser.parse_args()
    torch.manual_seed(17)
    kernel = kernels.get_kernel("kernels-community/finegrained-kernels", version=0)
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for packed in (False, True):
        weight = (
            torch.randint(
                0,
                256,
                (args.vocab_size, args.width // 2),
                device="cuda",
                dtype=torch.uint8,
            ).view(torch.int8)
            if packed
            else torch.full(
                (args.vocab_size, args.width),
                2.0,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
        )
        scale = (
            torch.ones(
                args.vocab_size,
                args.width // 16,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            if packed
            else torch.tensor([0.375], device="cuda")
        )
        global_scale = torch.tensor([0.03125], device="cuda") if packed else None
        for mode in ("eager", "compiled"):
            # The ordinary NV4 implementation constructs a CPU LUT: only compilation lifts
            # that constant out of the forward and makes this baseline graph-capturable.
            if packed and mode == "eager":
                continue
            ordinary = (
                torch.compile(reference, fullgraph=True)
                if mode == "compiled"
                else reference
            )
            for rows in args.rows:
                for cold in (False, True):
                    ids = torch.randint(
                        0, args.vocab_size if cold else 16, (rows,), device="cuda"
                    )

                    def baseline():
                        return ordinary(weight, ids, scale, global_scale)

                    def fused():
                        return kernel.embedding_lookup(
                            weight,
                            ids,
                            scale,
                            global_scale,
                            torch.bfloat16 if packed else scale.dtype,
                        )

                    torch.testing.assert_close(fused(), baseline(), rtol=0, atol=0)
                    if cold:
                        baseline_graph, baseline_out = capture(baseline)
                        fused_graph, fused_out = capture(fused)
                        a, b = (
                            cold_latency(baseline_graph, flush),
                            cold_latency(fused_graph, flush),
                        )
                        del baseline_graph, baseline_out, fused_graph, fused_out
                    else:
                        a = triton.testing.do_bench_cudagraph(baseline, rep=50) * 1000
                        b = triton.testing.do_bench_cudagraph(fused, rep=50) * 1000
                    print(
                        json.dumps(
                            dict(
                                format="nvfp4" if packed else "fp8",
                                baseline=mode,
                                rows=rows,
                                cache="cold" if cold else "hot",
                                pytorch_us=a,
                                fused_us=b,
                                speedup=a / b,
                            )
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
