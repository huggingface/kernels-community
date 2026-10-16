# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Routed rowwise FP8 / W8A16 microbenchmarks with full tuning and warm/cold replay.

Eight active experts, N1280/K5120; --rows controls rows per expert. Includes optional
gate fusion and a gather permutation. The BF16 baseline holds dequantized weights;
W8A16 also reports per-call fused dequantization plus BF16 bmm.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import kernels
import torch

from bench_fp8_linear import timing

p = argparse.ArgumentParser()
p.add_argument("--family", choices=["rowwise", "weight_only"], required=True)
p.add_argument("--kind", choices=["batched", "grouped"], required=True)
p.add_argument("--output", required=True)
p.add_argument("--rows", default="1,16,256")
p.add_argument("--gate", action="store_true")
p.add_argument("--gather", action="store_true")
a = p.parse_args()
if int(os.environ.get("FINEGRAINED_AUTOTUNE_TRIALS", "100")) < 100:
    p.error("performance runs require at least 100 tuning trials")
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
print(
    json.dumps(
        {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "trials": int(os.environ.get("FINEGRAINED_AUTOTUNE_TRIALS", "100")),
        }
    ),
    flush=True,
)
kernel = kernels.get_kernel("kernels-community/finegrained-kernels", version=0)
q = importlib.import_module(kernel.__name__ + ".quant")

for r in map(int, a.rows.split(",")):
    e, n, k = 8, 1280, 5120
    wn = 2 * n if a.gate else n
    m = e * r
    torch.manual_seed(42)
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(e, wn, k, device="cuda") * 0.05
    ids = torch.arange(e, device="cuda", dtype=torch.int32).repeat_interleave(r)
    starts = torch.arange(e + 1, device="cuda", dtype=torch.int32) * r
    if a.family == "rowwise":
        S = W.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        B = (W / S).to(torch.float8_e4m3fn)
        deq = B.float() * S
        Aq, As = q.fp8_act_quant_tensor_wide(A, k)
        Ar = Aq.float() * As.reshape(-1, 1)
        actfmt = None
    else:
        if a.gate:
            S = (
                W.reshape(e, n // 128, 128, 2, k // 128, 128)
                .abs()
                .amax((2, 5))
                .clamp_min(1e-12)
                / 448
            )
            expanded = (
                S.repeat_interleave(128, 1)
                .reshape(e, wn, k // 128)
                .repeat_interleave(128, 2)
            )
            S = S.reshape(e, wn // 128, k // 128)
        else:
            S = (
                W.reshape(e, wn // 128, 128, k // 128, 128)
                .abs()
                .amax((2, 4))
                .clamp_min(1e-12)
                / 448
            )
            expanded = S.repeat_interleave(128, 1).repeat_interleave(128, 2)
        B = (W / expanded).to(torch.float8_e4m3fn)
        deq = (B.float() * expanded).to(A.dtype)
        Ar = A
        actfmt = "bf16"
    del W
    gather = torch.randperm(m, device="cuda").to(torch.int32) if a.gather else None
    kwargs = {"expert_ids": ids} if a.kind == "batched" else {"expert_start": starts}
    if gather is not None:
        kwargs["gather_idx"] = gather
    op = getattr(kernel, "matmul_" + a.kind)

    def run():
        return op(
            A,
            B,
            None,
            S,
            activation_format=actfmt,
            gate=a.gate,
            output_dtype=A.dtype,
            **kwargs,
        )

    def epilogue(x):
        return torch.nn.functional.silu(x[..., 0::2]) * x[..., 1::2] if a.gate else x

    RefA = Ar if gather is None else Ar[gather.long()]
    want = epilogue(
        torch.bmm(RefA.reshape(e, r, k).float(), deq.float().transpose(1, 2))
    ).reshape(m, n)
    row = {
        "family": a.family,
        "kind": a.kind,
        "E": e,
        "rows_per_expert": r,
        "N": n,
        "K": k,
        "gate": a.gate,
        "gather": a.gather,
    }
    print("START", json.dumps(row), flush=True)
    got = run()
    torch.testing.assert_close(got.float(), want.float(), rtol=0.03, atol=0.03)
    row["finegrained"] = timing(run)
    bf16a = RefA.reshape(e, r, k).to(A.dtype)
    bf16b = deq.transpose(1, 2).to(A.dtype)
    row["loadtime_dequantized_bf16_bmm"] = timing(
        lambda: epilogue(torch.bmm(bf16a, bf16b))
    )
    if a.family == "weight_only":
        wo = importlib.import_module(kernel.__name__ + ".weight_only")

        def dequant_bmm():
            weights = wo.dequantize_weight(B, S, A.dtype, 128, 128, a.gate)
            return epilogue(torch.bmm(bf16a, weights.transpose(1, 2)))

        row["fused_dequant_including_bmm"] = timing(dequant_bmm)
    with Path(a.output).open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("RESULT", json.dumps(row), flush=True)
