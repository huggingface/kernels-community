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

"""Compare fused CSF and dense scales using an actual checkpoint's scale streams.

Example (fixed shape: [rows/16, 16*(1+cols/2)]):
  python bench/bench_csf.py --fixed gate.fixed --exceptions gate.exceptions \
      --rows 640 --cols 160 --tokens 1 128 --experts 8

LOCAL_KERNELS can select an isolated local build. Both paths use the same packed
weights and quantization globals. Autotuning completes before CUDA-graph timing.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from kernels import get_kernel
from triton.testing import do_bench_cudagraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--exceptions", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128])
    parser.add_argument(
        "--layout", choices=["auto", "batched", "grouped", "all"], default="auto"
    )
    parser.add_argument(
        "--activation-formats",
        nargs="+",
        choices=["nvfp4", "bf16"],
        default=["nvfp4", "bf16"],
    )
    args = parser.parse_args()
    kernel = get_kernel("kernels-community/finegrained-kernels", version=0)
    fixed = torch.frombuffer(bytearray(args.fixed.read_bytes()), dtype=torch.uint8)
    fixed = fixed.reshape(args.rows // 16, 16 * (1 + args.cols // 2))
    exceptions = torch.frombuffer(
        bytearray(args.exceptions.read_bytes()), dtype=torch.uint32
    )
    scales = kernel.prepare_nvfp4_csf_scales(
        fixed.unsqueeze(0).repeat(args.experts, 1, 1), [exceptions] * args.experts
    ).to("cuda")
    dense = kernel.decompress_nvfp4_csf_scales(scales)
    K = args.cols * 16
    weights = torch.randint(
        -128, 127, (args.experts, args.rows, K // 2), device="cuda", dtype=torch.int8
    )
    global_scale = torch.ones(args.experts, device="cuda", dtype=torch.float32)
    for tokens in args.tokens:
        A = torch.randn((tokens * args.experts, K), device="cuda", dtype=torch.bfloat16)
        layouts = (
            [False, True]
            if args.layout == "all"
            else [tokens > 1 if args.layout == "auto" else args.layout == "grouped"]
        )
        for activation in args.activation_formats:
            for grouped in layouts:
                routing = torch.arange(
                    args.experts, device="cuda", dtype=torch.int32
                ).repeat_interleave(tokens)
                kwargs = dict(activation_format=activation, b_global_scale=global_scale)
                if grouped:
                    kwargs["expert_start"] = (
                        torch.arange(args.experts + 1, device="cuda", dtype=torch.int32)
                        * tokens
                    )
                    csf_op, dense_op = kernel.matmul_csf_grouped, kernel.matmul_grouped
                else:
                    kwargs["expert_ids"] = routing
                    csf_op, dense_op = kernel.matmul_csf_batched, kernel.matmul_batched

                def csf():
                    return csf_op(A, weights, scales, **kwargs)

                def ordinary():
                    return dense_op(A, weights, Bs=dense, **kwargs)

                torch.testing.assert_close(csf(), ordinary(), rtol=0.03, atol=0.03)
                csf_us = do_bench_cudagraph(csf, rep=500, return_mode="median") * 1000
                dense_us = (
                    do_bench_cudagraph(ordinary, rep=500, return_mode="median") * 1000
                )
                print(
                    json.dumps(
                        dict(
                            device=torch.cuda.get_device_name(),
                            rows=args.rows,
                            cols=args.cols,
                            experts=args.experts,
                            tokens_per_expert=tokens,
                            grouped=grouped,
                            activation_format=activation,
                            trials=os.getenv("FINEGRAINED_AUTOTUNE_TRIALS", "default"),
                            csf_bytes=scales.storage_bytes,
                            dense_bytes=dense.numel(),
                            csf_us=csf_us,
                            dense_us=dense_us,
                            ratio=csf_us / dense_us,
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
