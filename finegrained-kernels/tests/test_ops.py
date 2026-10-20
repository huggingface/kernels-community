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
"""Op-level scenario tests: the (weight format x epilogue x input/output format)
support matrix of ``matmul_grouped`` / ``matmul_batched`` / ``matmul_2d``, each cell checked
against an independent dequantize-and-matmul torch reference (``tests/utils.py``'s ``WEIGHTS``
registry — shared with ``test_moe``'s fused-vs-unfused problems).

The scenario list is GENERATED from the same support matrix the ``check_activation_format``
docstring documents — every weight format crosses its valid input and output formats
once, activation-function variants ride one format (the GLU math is format-independent),
and routing variants (sentinel, noncontiguous ids, empty expert, decode / native-M / launch-scale
shapes, torch.compile) ride one format each. One ``Problem`` list feeds all three ops via the
``op`` axis (``test_op_scenarios``): the routed ops (``batched`` / ``grouped``) run
every Problem; ``matmul`` — the single-GEMM sibling — runs each Problem it can represent (no
expert routing, a quantized format it routes, requant on MX weights only), one weight matrix and
no gather/scatter. Two orthogonal knobs ride the same list: ``static`` (calibrated activation
quant — a shared scale on all three ops, one per expert on the routed pair) and ``swizzled`` (MX
weight scales pre-swizzled into the 5D SWIZZLE_32_4_4 tcgen05 layout — a pure layout variant
checked against the affine reference).
Nothing in this file uses a kernel under test as the oracle."""

from dataclasses import dataclass, field, replace

import pytest
import torch
import triton
import triton.language as tl

from utils import (  # type: ignore
    DTYPE_TO_TOL,
    INT_BITS,
    INT_GROUP,
    REQUANT_FN,
    REQUANT_GROUP,
    SUPPORTS_SWIZZLED_SCALES,
    TEST_DEVICE,
    WEIGHTS,
    dequant_gate_up,
    dq_block_fp8,
    dq_grouped,
    dq_int,
    dq_scale,
    make_gate_up,
    make_static_activation_scale,
    make_gptq_act_order,
    maybe_compile,
    quant_dequant_a,
    prepare_weight_scales,
)

import finegrained_kernels  # type: ignore
from finegrained_kernels.embedding import embedding_lookup
from finegrained_kernels import swizzle_mx_scales, unswizzle_mx_scales, repack_modelopt_int4, repack_modelopt_int8, repack_ggml_int, dequantize_int_weight, repack_awq_int4, repack_gptq_int4  # type: ignore
from finegrained_kernels.batched import (  # type: ignore
    int_weight_only_matmul_batched_kernel,
    mx_dynamic_matmul_batched_kernel,
    w8a8_tensor_dynamic_fp8_matmul_batched_kernel,
)
from finegrained_kernels.compat import NVFP4_SCALE_GROUP_K  # type: ignore
from finegrained_kernels.formats import is_mxfp4, is_mxfp8, mx_scale_family, ue8m0_as_uint8
from finegrained_kernels.grouped import (  # type: ignore
    block_fp8_weight_only_matmul_grouped_kernel,
    full_precision_matmul_grouped_kernel,
    int_weight_only_matmul_grouped_kernel,
    mx_dynamic_matmul_grouped_kernel,
    w8a8_tensor_dynamic_fp8_matmul_grouped_kernel,
)
from finegrained_kernels.loading.scales import decode_group_scale
from finegrained_kernels.matmul import (  # type: ignore
    int_weight_only_matmul_2d_kernel,
    mx_dynamic_matmul_kernel,
    w8a8_tensor_dynamic_fp8_matmul_kernel,
)
from finegrained_kernels.quant import nvfp4_act_quant  # type: ignore
from finegrained_kernels.epilogue import apply_glu  # type: ignore


# ── the scenario spec ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Problem:
    """One op-level scenario: a weight format (a ``WEIGHTS`` registry row) plus the
    transform and routing knobs the public ops expose. Validity of the format fields
    against the registry row is enforced at generation, not runtime."""

    weights: str
    S: int = 64
    E: int = 4
    N: int = 128
    K: int = 256
    gate: bool = False
    act_fn: str = "silu"
    swiglu_alpha: float | None = None
    swiglu_limit: float | None = None
    activation_format: str | None = None
    activation_scale_ub: float | None = None
    quantize_output: bool = False
    per_expert_globals: bool = False  # activation (and requant output) globals calibrated per expert
    prequant: bool = False  # pass As explicitly (must be bit-identical to raw A)
    static: bool = False  # calibrated (static) activation scale instead of a runtime one
    csf_encoding: bool = False  # lossless CSF compression of NVFP4 scales
    swizzled: bool = False  # pass MX weight scales pre-swizzled (5D SWIZZLE_32_4_4 fast path)
    sentinel_fraction: float = 0.0
    noncontiguous: bool = False
    empty_expert: bool = False
    compile: bool = False
    bias: bool = False  # a per-expert output bias, added on the accumulator before the GLU
    zero_points: bool = False  # asymmetric integer weights: a zero point per weight scale group
    int_group: int | None = INT_GROUP  # None spans K (per-channel)
    pre_quant_scale: bool = False  # AWQ input transform, shared for dense or per expert when routed
    svdquant_rank: int = 0  # dense NVFP4 residual plus BF16 low-rank correction
    dtype: torch.dtype = torch.bfloat16
    # in the CI subset (kernels_ci); the rest run in test_op_scenarios_extended
    ci: bool = field(default=True, compare=False)

    @property
    def id(self):
        tag = self.weights
        if self.activation_scale_ub is not None:
            tag += "_capped"
        if self.weights in INT_BITS and self.int_group != INT_GROUP:
            tag += f"_group{self.int_group or 'channel'}"
        if self.gate:
            tag += f"_gate_{self.act_fn}"
            if self.swiglu_alpha is not None:
                tag += "_alpha"
            if self.swiglu_limit is not None:
                tag += "_limit"
        if self.quantize_output:
            tag += f"_out{self.activation_format}"
        elif self.activation_format is not None:
            tag += f"_in{self.activation_format}"
        if self.per_expert_globals:
            tag += "_expertglobals"
        if self.prequant:
            tag += "_prequant"
        if self.static:
            tag += "_static"
        if self.csf_encoding:
            tag += "_csf"
        if self.swizzled:
            tag += "_swizzled"
        if self.sentinel_fraction:
            tag += "_sentinel"
        if self.noncontiguous:
            tag += "_noncontig"
        if self.empty_expert:
            tag += "_emptyexpert"
        if self.compile:
            tag += "_compile"
        if self.bias:
            tag += "_bias"
        if self.zero_points:
            tag += "_zp"
        if self.pre_quant_scale:
            tag += "_prequantscale"
        if self.svdquant_rank:
            tag += f"_svdquant_r{self.svdquant_rank}"
        if self.dtype != torch.bfloat16:
            tag += f"_{str(self.dtype).rsplit('.', 1)[-1]}"  # float16 / float32
        return f"{tag}_S{self.S}_E{self.E}_N{self.N}_K{self.K}"


def scenarios() -> list[Problem]:
    """The curated matrix. Coverage rules:
    - every weight format: one plain GEMM cell;
    - every requant format of a weight family: one gate cell with ``quantize_output`` in that
      activation format (requant lives under gate);
    - every non-default activation format: one cell (the W4A4 chains);
    - one prequant mirror per quantized family (raw-vs-As bit-equality);
    - GLU variants (gelu, swiglu alpha+limit) on one format — the math is format-blind;
    - routing/launch variants (sentinel, noncontiguous, empty expert, decode shape,
      launch-scale shape, torch.compile) one format each."""
    out: list[Problem] = []
    for w, row in WEIGHTS.items():
        if w == "fp16":
            # emitted explicitly below with fp16 activations (matching dtypes required)
            continue
        out.append(Problem(weights=w))
        for fmt in row["requant_formats"]:
            out.append(Problem(weights=w, gate=True, activation_format=fmt, quantize_output=True))
                # swizzled-in -> swizzled-out: swizzled MX weights + requant emit a swizzled (5D
                # SWIZZLE_32_4_4) Cs — the down's fast-path input. Format-general (nvfp4 group-16
                # differs only in column count). Reference un-swizzles the 5D Cs to the affine cell.
            if w in ("mxfp8", "mxfp8_u8", "mxfp4", "nvfp4"):
                out.append(
                    Problem(weights=w, gate=True, activation_format=fmt, quantize_output=True, swizzled=True)
                )
        default_in = {"fp8_128x128": "fp8", "fp8_tensor": "fp8", "fp8_rowwise": "fp8"}.get(w)
        for fmt in row["activation_formats"]:
            if fmt != default_in:
                out.append(Problem(weights=w, activation_format=fmt))
    out += [
        # Coarse and per-tensor FP8 with raw BF16/FP16 activations.
        Problem(ci=False, weights="fp8_128x128", activation_format="bf16", S=1, N=192, K=384, bias=True, dtype=torch.float16),
        Problem(ci=False, weights="fp8_128x128_ue8m0", activation_format="bf16", S=8, empty_expert=True, sentinel_fraction=0.25),
        Problem(weights="fp8_tensor", activation_format="bf16", S=1, gate=True, bias=True, compile=True),
        Problem(weights="fp8_128x128", activation_format="bf16", S=257, N=256, K=512, gate=True, bias=True, compile=True),
        Problem(ci=False, weights="fp8_128x128", activation_format="bf16", S=1025, E=1, N=192, K=384, compile=True, bias=True),
        Problem(weights="fp16", dtype=torch.float16),
        # gate WITHOUT requant (raw GLU intermediate) — one quantized + one full format
        Problem(weights="mxfp8", gate=True),
        Problem(weights="bf16", gate=True),
        Problem(weights="mxfp8", gate=True, act_fn="gelu", activation_format="mxfp8", quantize_output=True),
        Problem(
            weights="mxfp8",
            gate=True,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            activation_format="mxfp8", quantize_output=True,
        ),
        Problem(weights="fp8_128x128", prequant=True),
        Problem(weights="mxfp8", prequant=True),
        Problem(weights="nvfp4", prequant=True),
        Problem(weights="mxfp4", activation_format="fp8", static=True, gate=True, pre_quant_scale=True, compile=True),
        Problem(weights="nvfp4", activation_format="fp8", static=True, gate=True, pre_quant_scale=True, compile=True),
        Problem(weights="nvfp4", S=8, gate=True, pre_quant_scale=True, compile=True),
        Problem(weights="nvfp4", S=1, svdquant_rank=32, pre_quant_scale=True, bias=True),
        Problem(ci=False, weights="nvfp4", S=64, N=129, K=320, svdquant_rank=128, pre_quant_scale=True, compile=True),
        Problem(ci=False, weights="nvfp4", S=128, svdquant_rank=32, pre_quant_scale=True, gate=True,
                activation_format="nvfp4", quantize_output=True, swizzled=True),
        Problem(weights="mxfp8", sentinel_fraction=0.25),
        # the prefill expansion (>= 4096 routed rows below 1024 per expert) copies only the local rows:
        # values + 2D scales, and raw bf16 rows
        Problem(weights="fp8_128x128", S=4096, E=8, sentinel_fraction=0.5),
        Problem(weights="bf16", S=4096, E=8, sentinel_fraction=0.5),
        Problem(weights="mxfp8", noncontiguous=True),
        Problem(weights="mxfp8", empty_expert=True),
        # decode shape (small M — inline act-quant on MX, the software/scalar arms elsewhere)
        Problem(weights="mxfp8", S=8),
        Problem(weights="nvfp4", S=8),
        Problem(weights="bf16", S=4, gate=True),  # the full-precision swapped decode tiles
        # the TOP of the 2D swap decode band (mx_2d_swap_scope_pruner scopes SWAP_AB to M <= 16):
        # the arm's tile choices differ across the band, and an A-descriptor trap that is clean at
        # S=4 fires by S=16, so the edge is the cell that sees it
        Problem(weights="nvfp4", S=16),
        Problem(weights="fp8_128x128", S=8),
        Problem(weights="fp8_128x128_ue8m0", S=8),
        Problem(weights="mxfp4", S=8),
        # native-M tile (BM>=128): the format matrix above rides sub-native S, so these carry the
        # native-only compute arms — block-dynamic UE8M0 dot_scaled fold and the native mxfp/nvfp4
        # MMA (mxfp8's is covered by the S=2048 launch-scale case below).
        Problem(weights="fp8_128x128_ue8m0", S=128),
        Problem(weights="mxfp4", S=128),
        Problem(weights="mxfp4", activation_format="mxfp4", S=128),  # W4A4 native (packed acts)
        Problem(weights="mxfp4", activation_format="bf16", S=128),  # W4A16 prefill (TMA arm reachable)
        # W4A16 at prefill scale — enough tokens/expert for BM>=64, so the grouped WS+descriptor arm
        # (tma-gather A, no other=0.0) is reachable/tuned (the matmul_ogs-style fast path).
        Problem(weights="mxfp4", activation_format="bf16", S=2048, E=16, N=512, K=1024),
        Problem(weights="mxfp8", activation_format="bf16", S=128),  # W8A16 prefill
        Problem(weights="mxfp4", activation_format="bf16", S=8),  # W4A16 decode
        Problem(weights="nvfp4", S=128),
        # swizzled MX weight scales (5D SWIZZLE_32_4_4 — the tcgen05 fast path): the same values in
        # the swizzled layout, so results match the affine cells. One per MX family + a gate case
        # (the (E, 2N) gate|up swizzle). Runs on all three ops; the reference stays on the affine Bs.
        Problem(weights="mxfp8", swizzled=True),
        Problem(weights="mxfp4", swizzled=True),
        Problem(weights="nvfp4", swizzled=True),
        Problem(weights="mxfp8", gate=True, swizzled=True),
        # N>128 gate: 2N/expert spans >1 128-block/projection, so the gate|up interleave
        # [g0,u0,g1,u1,...] diverges from a flat [g0,g1,..,u0,u1,..] slab (they coincide only at
        # N=128). Guards every op's swizzled gate reader against the two-slab layout.
        Problem(weights="mxfp8", gate=True, swizzled=True, N=256, K=512),
        # the interleave-divergence guard per fp4 family: packed-E2M1 byte-halving and nvfp4's
        # group-16 column count are exactly the axes the mxfp8 cell above can't cover
        Problem(weights="mxfp4", gate=True, activation_format="mxfp4", quantize_output=True, swizzled=True, N=256, K=512),
        Problem(weights="nvfp4", gate=True, activation_format="nvfp4", quantize_output=True, swizzled=True, N=256, K=512),
        # NVFP4 second-level globals as modelopt calibrates them: one weight global per (expert,
        # half) of a gate|up stack, and per-expert activation / requant-output globals. Affine and
        # per-expert activation and requant-output globals (a checkpoint's calibrated
        # per-projection input_scale), which the act quant applies by the row's expert
        Problem(weights="nvfp4", per_expert_globals=True),
        Problem(weights="nvfp4", gate=True, activation_format="nvfp4", quantize_output=True, per_expert_globals=True),
        Problem(weights="nvfp4", csf_encoding=True, S=8),
        Problem(weights="nvfp4", csf_encoding=True, empty_expert=True, sentinel_fraction=0.25, compile=True),
        Problem(weights="nvfp4", csf_encoding=True, activation_format="bf16"),
        Problem(weights="nvfp4", csf_encoding=True, gate=True, activation_format="nvfp4", quantize_output=True, compile=True),
        # swizzled decode (S=8) — the bench's pre-swizzled batched decode arm per fp4 family
        Problem(weights="mxfp4", swizzled=True, S=8),
        Problem(weights="nvfp4", swizzled=True, S=8),
        # NVFP4 tiny-M: the 2D kernel's SWAP_AB decode arm (BM=1 swapped dot_scaled — the
        # only native E4M3 M=1 path) — affine and swizzled, checked against the same oracle
        Problem(weights="nvfp4", S=2),
        Problem(weights="nvfp4", swizzled=True, S=2),
        Problem(weights="mxfp4", gate=True, activation_format="mxfp4", quantize_output=True, swizzled=True, S=8),
        Problem(weights="nvfp4", gate=True, activation_format="nvfp4", quantize_output=True, swizzled=True, S=8),
        # launch-scale smoke (the matrix rides small shapes; this catches scale-dependent
        # scheduling/tiling regressions)
        Problem(weights="mxfp8", S=2048, E=16, N=512, K=1024),
        Problem(weights="fp8_128x128", compile=True),
        Problem(weights="mxfp4", compile=True),
        Problem(weights="bf16", compile=True),  # the fp kernel's pre_hook under compile
        Problem(ci=False, weights="fp8_rowwise", S=256, activation_scale_ub=0.5, compile=True),
        Problem(weights="fp8_tensor", static=True, compile=True),  # the static arm's operand marshalling
        # calibrated (static) activation quant, reached when As is a calibrated scale rather than
        # per-block. One value per quantized module, so the OP fixes the shape: a dense linear
        # calibrates once, a MoE calibrates each expert separately. Runs on all three ops.
        Problem(weights="fp8_128x128", static=True),
        Problem(weights="fp8_128x128", gate=True, static=True),
        # per-TENSOR weights: the shipped static form (Ministral-3 dense, Mistral-4 MoE — both
        # write qscheme_act="TENSOR" with weight_block_size=None)
        Problem(weights="fp8_tensor", static=True),
        # row-wise weights (Llama 4 FP8): the stacked gate|up rows carry their own scales, the swapped decode
        # tile folds them on its other axis, a caller's per-token As rides as-is, and the launch-scale shape
        # reaches the grouped kernel's descriptor / warp-specialized arms
        Problem(weights="fp8_rowwise", gate=True),
        Problem(ci=False, weights="fp8_rowwise", S=8),
        Problem(ci=False, weights="fp8_rowwise", gate=True, S=8),
        Problem(ci=False, weights="fp8_rowwise", prequant=True, compile=True),
        Problem(ci=False, weights="fp8_rowwise", S=4096, E=8, prequant=True, noncontiguous=True),
        Problem(weights="fp8_rowwise", static=True, S=17),
        Problem(ci=False, weights="fp8_rowwise", S=257, bias=True, empty_expert=True, compile=True),
        Problem(ci=False, weights="fp8_rowwise", S=2048, E=16, N=512, K=1024),
        # the prefill expansion lays the routed rows out, and their per-token scales with them
        Problem(ci=False, weights="fp8_rowwise", S=4096, E=8, sentinel_fraction=0.5),
        Problem(ci=False, weights="fp8_tensor", S=4096, E=8),
        # non-aligned N (64-grid, off the 128-grid — gpt-oss H=I=2880 shape). matmul_2d masks the
        # N-tail; routed MX runs the affine arm (per-row scales, any BN|N); routed FP8 rejects it
        # (its scales are 128-blocked along N — raises pointing to matmul_2d).
        Problem(weights="fp8_128x128", N=320, K=1024),
        Problem(weights="mxfp8", N=320, K=1024),
        # weight-only GATE at non-dividing N: the affine gate|up scale leaf must clamp its rows
        # (the up half reads N + offs_bn — unclamped it runs past 2N on the partial last tile)
        Problem(weights="mxfp8", gate=True, activation_format="bf16", N=320, K=1024),
        Problem(weights="mxfp8", N=320, K=1024, swizzled=True),  # non-128 N on the swizzled arm (bf16 out, all 3 ops)
        Problem(weights="mxfp4", activation_format="mxfp4", N=320, K=320),  # W4A4 non-128 N and K
        Problem(weights="mxfp4", gate=True, activation_format="mxfp4", quantize_output=True, N=320, K=320),  # gated W4A4 non-128 (gpt-oss gate_up)
        # output/input dtype coverage (fp16 + fp32) across the FP8 and MX kernels — the format matrix
        # above rides bf16.
        Problem(weights="fp8_128x128", dtype=torch.float16),
        Problem(weights="fp8_128x128", dtype=torch.float32),
        Problem(weights="mxfp4", dtype=torch.float16),
        Problem(weights="mxfp4", dtype=torch.float32),
        Problem(weights="mxfp8", dtype=torch.float16),
        Problem(weights="mxfp8", dtype=torch.float32),
        # output bias (gpt-oss ships one on both projections), added in the shared epilogue: one cell per kernel
        # (full precision, block-FP8 dynamic and static, per-tensor FP8, MX dynamic, MX weight-only), then the
        # epilogue paths — the two-level global applied before it, the gated columns, the swapped decode tile,
        # block-FP8's unstacked-gate decode band, the requantized output — and the gpt-oss gate_up itself
        Problem(weights="bf16", bias=True),
        Problem(weights="fp8_128x128", bias=True),
        Problem(weights="fp8_128x128", static=True, bias=True),
        Problem(weights="fp8_tensor", bias=True),
        Problem(weights="mxfp8", bias=True),
        Problem(weights="mxfp8", activation_format="bf16", bias=True),
        Problem(weights="nvfp4", bias=True),
        Problem(weights="mxfp8", gate=True, bias=True),
        Problem(weights="nvfp4", S=8, bias=True),
        Problem(weights="fp8_128x128", gate=True, S=8, bias=True),
        Problem(weights="mxfp8", gate=True, activation_format="mxfp8", quantize_output=True, bias=True),
        Problem(weights="mxfp4", activation_format="bf16", gate=True, swiglu_alpha=1.702, swiglu_limit=7.0, bias=True),
        Problem(weights="mxfp4", activation_format="bf16", gate=True, swiglu_alpha=1.702, swiglu_limit=7.0, bias=True, S=8),
        # int4 weight-only (the format matrix above gives each row its plain cell): the gated GLU with
        # a bias, decode (the batched swap reduce at S=1), a K off the BK grid (the batched tail tile
        # slides back) and fp16 activations, each with and without zero points
        Problem(ci=False, weights="int4", gate=True, swiglu_alpha=1.702, swiglu_limit=7.0, bias=True),
        Problem(weights="int4", zero_points=True, gate=True, bias=True),
        Problem(ci=False, weights="int4", S=1),
        Problem(ci=False, weights="int4", zero_points=True, S=8),
        Problem(ci=False, weights="int4", zero_points=True, K=640),
        Problem(ci=False, weights="int4", dtype=torch.float16),
        Problem(ci=False, weights="int4", zero_points=True, dtype=torch.float16),
        # the 8- and 2-bit widths ride the same kernels (the width follows from the weight's word count)
        Problem(ci=False, weights="int8", zero_points=True, S=1),
        Problem(ci=False, weights="int8", zero_points=True, gate=True, bias=True),
        Problem(ci=False, weights="int8", dtype=torch.float16),
        Problem(ci=False, weights="int2", zero_points=True, S=1),
        Problem(weights="int2", zero_points=True, gate=True, bias=True),
        # Wide groups and per-channel scales retain rounded weight dequantization.
        Problem(ci=False, weights="int2", int_group=512, K=1024),
        Problem(ci=False, weights="int4", int_group=512, K=1024, zero_points=True, gate=True, bias=True),
        Problem(ci=False, weights="int8", int_group=None, K=1024, zero_points=True),
        # per-channel with K off the 128 grid (gpt-oss K=2880): only a 64-wide K tile divides K
        Problem(weights="int8", int_group=None, K=320),
        Problem(ci=False, weights="int4", int_group=None, K=1024, compile=True),
        Problem(ci=False, weights="int4", zero_points=True, compile=True),
        Problem(weights="int4", S=8, gate=True, bias=True, pre_quant_scale=True, compile=True),
        Problem(weights="int4", S=32, gate=True, bias=True, activation_format="fp8", static=True,
                pre_quant_scale=True, compile=True),
        Problem(weights="int8", int_group=None, gate=True, bias=True, activation_format="int8", static=True,
                pre_quant_scale=True, compile=True),
    ]
    return out


PROBLEMS = scenarios()


# ── inputs, reference, and checks ────────────────────────────────────────────────


def _routed(problem: Problem):
    torch.manual_seed(0)
    A = torch.randn(problem.S, problem.K, device=TEST_DEVICE, dtype=problem.dtype)
    high = problem.E - 1 if problem.empty_expert else problem.E
    expert_ids = torch.randint(
        0, high, (problem.S,), device=TEST_DEVICE, dtype=torch.int32
    )
    if problem.sentinel_fraction:
        idx = torch.randperm(problem.S, device=TEST_DEVICE)[
            : int(problem.S * problem.sentinel_fraction)
        ]
        expert_ids[idx] = problem.E
    if problem.noncontiguous:
        expert_ids = _make_noncontig(expert_ids)
    return A, expert_ids


def _bias(problem: Problem, op):
    """The scenario's output bias, else None: ``(E, N_out)`` for the routed ops, ``(N_out,)`` for
    ``matmul``, ``N_out`` the GEMM's width (the interleaved gate|up columns under gate). Seeded, so the
    reference and the op read the identical values; large enough that dropping it fails the check."""
    if not problem.bias:
        return None
    g = torch.Generator(device=TEST_DEVICE).manual_seed(1)
    rows = 2 * problem.N if problem.gate else problem.N
    bias = torch.randn(problem.E, rows, device=TEST_DEVICE, generator=g).to(problem.dtype)
    return bias[0] if op == "matmul" else bias


def _zero_points(problem: Problem, E):
    """The scenario's integer zero points, else None: one per weight scale group, ``(E, rows, K // group)``
    over the stack the weights were made at. Seeded, so the reference and the op read the identical values."""
    if not problem.zero_points:
        return None
    g = torch.Generator(device=TEST_DEVICE).manual_seed(2)
    rows = 2 * problem.N if problem.gate else problem.N
    shape = (E, rows, problem.K // (problem.int_group or problem.K))
    return torch.randint(0, 1 << INT_BITS[problem.weights], shape, device=TEST_DEVICE, dtype=torch.uint8, generator=g)


def _make_noncontig(x):
    base = torch.empty((x.numel(), 2), dtype=x.dtype, device=x.device)
    base[:, 0] = x
    base[:, 1] = x
    return base[:, 0]


def _static_scale(problem: Problem, op, A):
    """The calibrated activation scale, else None. A checkpoint calibrates one per quantized
    module, so the op fixes the shape: one value for ``matmul``, the single linear, and one per
    expert for the routed ops, whose experts are each their own module — fanned around the
    per-tensor value so an entry read against the wrong expert shows up. Deterministic in ``A``,
    so the reference and the op derive identical numbers."""
    if not problem.static:
        return None
    scale = make_static_activation_scale(A)
    if problem.activation_format == "int8":
        scale = scale * (448.0 / 127.0)
    if op == "matmul":
        return scale
    fan = torch.linspace(0.5, 2.0, problem.E, device=A.device, dtype=torch.float32)
    return (scale * fan).contiguous()


def _nvfp4_global(x):
    """The canonical NVFP4 second-level global of ``x``: ``amax / (6·448)`` — the smallest
    global that keeps every block scale in e4m3 range. In deployment this is CALIBRATED
    offline (the checkpoint's ``input_scale``); the tests \"calibrate\" on the tensor itself,
    deterministic so the reference and the op share the identical scalar."""
    return (x.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float().reshape(1)


def _act_global(problem: Problem, A, expert_ids=None):
    """NVFP4 is ALWAYS two-level — every nvfp4 quantized activation carries its calibrated global
    ``g_a`` (no single-level nvfp4 exists). ``per_expert_globals`` spreads it into the ``(E,)``
    vector modelopt writes (one ``input_scale`` per expert), fanned around the per-tensor value so
    an ignored entry shows up. Weight-only (bf16 activations) and non-nvfp4 formats have no second
    level (None)."""
    if problem.weights != "nvfp4" or problem.activation_format in ("bf16", "fp8"):
        return None
    g = _nvfp4_global(A)
    if not problem.per_expert_globals:
        return g
    fan = torch.linspace(0.5, 2.0, problem.E, device=A.device, dtype=torch.float32)
    return (g * fan).contiguous()


def _rowwise(problem: Problem, g, expert_ids):
    """A per-expert global as a per-ROW column vector for the torch reference; a per-tensor one
    passes through unchanged."""
    if g is None or g.numel() == 1:
        return g
    return g[expert_ids.to(torch.long).clamp(max=problem.E - 1)].reshape(-1, 1)


def _dequant_a(problem: Problem, op, A, expert_ids=None):
    """``A`` dequantized to fp32 on the format's grid (the exact host quant the op calls, or the
    calibrated static scale of each row's expert), plus the pre-quantized ``(Aq, As)`` form for the prequant round-trip
    check (``None`` where ``A`` stays raw)."""
    row = WEIGHTS[problem.weights]
    if problem.activation_scale_ub is not None:
        return quant_dequant_a(A, problem.K, activation_scale_ub=problem.activation_scale_ub), None
    static_scale = _static_scale(problem, op, A)
    act_global = _act_global(problem, A, expert_ids)
    if static_scale is not None:  # static (calibrated) activation quant
        return quant_dequant_a(A, problem.K, scale=_rowwise(problem, static_scale, expert_ids)), None
    if act_global is not None:
        # nvfp4 acts are always two-level: quantize A/g_a per block (the exact host fn the op
        # calls), dequantize × g_a; the pre-quantized form is the bare block scale (the g_a
        # global rides separately as As_global, computed from raw A in the caller).
        # per-expert globals quantize each row against its own: the op hands the quant the
        # routed rows' experts, sentinels and all, exactly as this reference does
        Aq, As_block = nvfp4_act_quant(A, global_scale=act_global, expert_index=expert_ids)
        A_dq = dq_grouped(Aq.view(torch.int8), As_block, NVFP4_SCALE_GROUP_K) * _rowwise(
            problem, act_global, expert_ids
        )
        return A_dq, (Aq, As_block)
    if problem.activation_format == "bf16":  # weight-only: raw bf16/fp16 activation, never quantized
        return A.float(), None
    quant = row["act_quant"][problem.activation_format]
    if quant is None:
        return A.float(), None
    Aq, As = quant(A)
    return row["dq_act"](Aq, As), (Aq, As)


def _prequant_args(problem: Problem, op, A, expert_ids=None):
    """The pre-quantized ``(Aq, As)`` form of ``A`` (``As`` = bare block scale) — the ``As`` half of
    ``_dequant_a``, handed to the op (and the reference) exactly as the op would compute it. The nvfp4
    activation global rides separately (``_act_global`` on the raw ``A``)."""
    return _dequant_a(problem, op, A, expert_ids)[1]


def _act_dequant(problem: Problem, op, A, As=None, As_global=None, expert_ids=None):
    """The fp32 activation the op multiplies by — from raw ``A`` (``As`` None: quantize+dequant on
    the format grid, the exact host quant the op applies) or from a pre-quantized ``(Aq, As)`` (dequant
    it, folding the nvfp4 ``As_global`` back). Both land on the same values, so the reference reads
    whatever the op was handed."""
    if As is None:
        return _dequant_a(problem, op, A, expert_ids)[0]
    if As_global is not None:  # nvfp4 two-level: block scale As, global As_global
        return dq_grouped(A.view(torch.int8), As, NVFP4_SCALE_GROUP_K) * _rowwise(
            problem, As_global, expert_ids
        )
    return WEIGHTS[problem.weights]["dq_act"](A, As)


def _pre_quant_scale(problem: Problem, op):
    E = 1 if op == "matmul" else problem.E
    scale = torch.linspace(0.5, 1.5, E * problem.K, device=TEST_DEVICE, dtype=problem.dtype)
    return scale if op == "matmul" else scale.view(E, problem.K)


def _svdquant_factors(problem: Problem):
    g = torch.Generator(device=TEST_DEVICE).manual_seed(87)
    rows = problem.N * (2 if problem.gate else 1)
    a = torch.randn(problem.svdquant_rank, problem.K, generator=g, device=TEST_DEVICE, dtype=problem.dtype) / problem.K**0.5
    b = torch.randn(rows, problem.svdquant_rank, generator=g, device=TEST_DEVICE, dtype=problem.dtype) * 20
    return a, b


def _fp32_intermediate(problem: Problem, op, A, expert_ids, B, Bs, Bs_global, As=None, As_global=None):
    """The fp32 GLU output on the format-quantized operands — the pre-requant oracle, shared by the
    global calibration (``_out_global``) and the reference (``_reference``), reading the SAME ``A``/
    ``As`` (and their globals) the op takes. ``matmul`` has no routing (single ``W[0]``); routed ops
    gather ``W[expert]`` and zero sentinel rows; GLU in fp32 (the production epilogue applies it to
    the fp32 accumulator directly)."""
    row = WEIGHTS[problem.weights]
    A_raw = A
    if problem.pre_quant_scale:
        scale = _pre_quant_scale(problem, op)
        if op != "matmul":
            scale = scale[expert_ids.long().clamp(max=problem.E - 1)]
        A_raw = A * scale
    if problem.static and problem.activation_format in ("fp8", "int8"):
        act_scale = _rowwise(problem, _static_scale(problem, op, A), expert_ids)
        normalized = A_raw.float() / act_scale
        A_quant = normalized.round().clamp(-128, 127) if problem.activation_format == "int8" else normalized.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        A_dq = A_quant * act_scale
    elif problem.pre_quant_scale and problem.weights == "nvfp4" and As_global is not None:
        aq, scales = nvfp4_act_quant(A_raw, global_scale=As_global, expert_index=expert_ids)
        A_dq = dq_grouped(aq.view(torch.int8), scales, NVFP4_SCALE_GROUP_K) * _rowwise(problem, As_global, expert_ids)
    else:
        A_dq = _act_dequant(problem, op, A_raw, As, As_global, expert_ids)
    # (E, rows, K) fp32
    if problem.weights in INT_BITS:  # dequantized to the activation dtype, as the GEMM does
        W = dq_int(B, Bs, _zero_points(problem, B.shape[0]),
                   None if problem.activation_format in ("fp8", "int8") else problem.dtype, problem.int_group or problem.K)
        if problem.activation_format == "fp8":
            global_scale = Bs_global.view(-1, 1, 1)
            W = (W / global_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * global_scale
    elif problem.weights == "nvfp4" and problem.activation_format == "fp8":
        W = dq_grouped(B, Bs, 32) * Bs_global.view(-1, 1, 1)
    elif problem.gate:
        W = dequant_gate_up(row, B, Bs, Bs_global)
    else:
        W = row["dequant"](B, Bs, Bs_global)
    if problem.activation_format == "bf16":
        W = W.to(A.dtype).float()
    bias = _bias(problem, op)
    if op == "matmul":
        ref = A_dq @ W[0].T  # single linear, no routing
        if bias is not None:
            ref = ref + bias.float()
    else:
        local = expert_ids.long().clamp(max=problem.E - 1)
        ref = torch.einsum("sk,snk->sn", A_dq, W[local])
        if bias is not None:
            ref = ref + bias.float()[local]
        ref[expert_ids.long() >= problem.E] = 0
    if problem.svdquant_rank:
        lora_a, lora_b = _svdquant_factors(problem)
        down = (A_raw.float() @ lora_a.float().T).to(problem.dtype)
        correction = (down.float() @ lora_b.float().T).to(problem.dtype)
        ref = ref.to(problem.dtype) + correction
    if problem.gate and problem.act_fn is not None:  # act_fn=None: the raw gate|up columns
        gate_v, up_v = ref[..., 0::2], ref[..., 1::2]
        ref = apply_glu(
            gate_v, up_v, problem.act_fn, problem.swiglu_alpha, problem.swiglu_limit
        ).float()
    return ref


def _out_global(problem: Problem, op, A, expert_ids, B, Bs, Bs_global, As=None, As_global=None):
    """The PROVIDED NVFP4 output global — the next proj's calibrated ``input_scale``. The tests
    'calibrate' it deterministically off the fp32 intermediate (``amax/(6·448)``) so the reference
    and the op share the identical scalar; None for MX / dense output (no second level)."""
    if not (problem.quantize_output and problem.activation_format == "nvfp4"):
        return None
    g = _nvfp4_global(
        _fp32_intermediate(problem, op, A, expert_ids, B, Bs, Bs_global, As, As_global)
    )
    if not problem.per_expert_globals:
        return g
    # the NEXT projection's input_scale is calibrated per expert too (what modelopt writes);
    # the requant epilogue normalizes each row by ITS expert's value
    fan = torch.linspace(0.5, 2.0, problem.E, device=g.device, dtype=torch.float32)
    return (g * fan).contiguous()


def _reference(problem: Problem, op, A, expert_ids, B, Bs, Bs_global, As=None, As_global=None, out_global=None):
    """The op written with torch only: the SAME inputs as ``_op`` (``A``/``As``/their globals/
    ``out_global``), returning the op's OWN output format so the shared ``_dequant`` reads reference
    and op identically. Dequant/gather the operands (``_fp32_intermediate`` reads whatever the op was
    handed — raw ``A`` or a pre-quantized ``(Aq, As)``) → matmul → GLU, then mirror the fused requant:
    divide by the provided ``out_global`` and snap to the format grid, returning the ``[C, Cs]`` the
    op returns. No ``quantize_output`` → the dense fp32 intermediate. ``"fp8"`` output has no torch
    block-quant (``REQUANT_FN["fp8"]`` is None), so it stays the unsnapped oracle — the dequant-
    closeness tolerance in ``_check`` absorbs the block-fp8 rounding."""
    inter = _fp32_intermediate(problem, op, A, expert_ids, B, Bs, Bs_global, As, As_global)
    if not problem.quantize_output or problem.activation_format == "fp8":
        return inter
    g_rows = _rowwise(problem, out_global, expert_ids)
    scaled = inter / g_rows if g_rows is not None else inter
    return list(REQUANT_FN[problem.activation_format](scaled.to(problem.dtype)))


def _op(problem: Problem, op, A, expert_ids, B, Bs, Bs_global, As=None, As_global=None, out_global=None,
        output_dtype=None):
    """The op half of the symmetric ``(ref, op)`` pair — the same inputs as ``_reference`` (plus
    ``As`` for the pre-quantized-input test). Runs the kernel on the format's operands (swizzling
    weight scales; the two-level per-tensor globals ride as the decoupled ``a_global_scale`` /
    ``b_global_scale`` kwargs) and returns its RAW output (dense tensor, or ``[C, Cs]`` under
    ``quantize_output``) — the shared ``_dequant`` brings it to fp32, the same call the reference goes
    through. The NVFP4 output global ``out_global`` is PROVIDED (the calibrated ``input_scale`` of
    the next proj): it rides in as ``output_global_scale`` and the op normalizes the intermediate by
    it before requant — the same scalar the reference uses, so the two spaces match."""
    # a requant cell names its format explicitly (the reference snaps to that grid)
    assert not problem.quantize_output or problem.activation_format not in (None, "bf16")
    kw = dict(activation_format=problem.activation_format, activation_scale_ub=problem.activation_scale_ub, quantize_output=problem.quantize_output)
    if problem.svdquant_rank:
        lora_a, lora_b = _svdquant_factors(problem)
        kw.update(svdquant_lora_a=lora_a, svdquant_lora_b=lora_b, simulate_unfused=True)
    if problem.pre_quant_scale:
        kw["a_pre_quant_scale"] = _pre_quant_scale(problem, op)
    if problem.gate:
        kw.update(
            gate=True,
            act_fn=problem.act_fn,
            swiglu_alpha=problem.swiglu_alpha,
            swiglu_limit=problem.swiglu_limit,
        )
    if not problem.quantize_output:
        kw["output_dtype"] = output_dtype or (None if problem.prequant else problem.dtype)
    if problem.bias:
        kw["bias"] = _bias(problem, op)
    if problem.zero_points:
        zero_points = _zero_points(problem, B.shape[0])
        kw["b_zero"] = zero_points[0] if op == "matmul" else zero_points
    if out_global is not None:  # provided NVFP4 output global (next proj's input_scale)
        kw["output_global_scale"] = out_global
    if problem.static:  # fused static activation quant — As is the calibrated scale
        As = _static_scale(problem, op, A)
    # matmul is the single-GEMM sibling: slice to the one weight matrix (expert 0) and drop the
    # routing maps; the call is otherwise identical to the routed ops.
    if op == "matmul":
        B = B[0]
        Bs = Bs[0] if Bs is not None else None
        Bs_global = Bs_global[:1] if Bs_global is not None else None
    # weight-scale swizzle, shared by all three ops — a pure layout change (values unchanged),
    # so the op's result still matches the affine-Bs reference.
    if problem.swizzled and not SUPPORTS_SWIZZLED_SCALES:
        pytest.skip("the SWIZZLE_32_4_4 scale layout is the tcgen05 fast path (CUDA-only)")
    bs = swizzle_mx_scales(Bs) if problem.swizzled and Bs is not None else Bs
    globals_kw = dict(a_global_scale=As_global, b_global_scale=Bs_global)
    if op == "matmul":
        fn = prepare_weight_scales(bs, problem.csf_encoding, finegrained_kernels.matmul_2d, problem.compile)
        return fn(A, B, As, bs, **globals_kw, **kw)
    if op == "batched":
        fn = prepare_weight_scales(bs, problem.csf_encoding, finegrained_kernels.matmul_batched, problem.compile)
        return fn(A, B, As, bs, expert_ids=expert_ids, **globals_kw, **kw)
    expert_start, gather_idx, scatter_idx = finegrained_kernels.scheduling.compute_grouped_scheduling(
        expert_ids, problem.E, 1
    )
    fn = prepare_weight_scales(bs, problem.csf_encoding, finegrained_kernels.matmul_grouped, problem.compile)
    raw = fn(
        A,
        B,
        As,
        bs,
        expert_start=expert_start,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        **globals_kw,
        **kw,
    )
    return raw


def _dequant(problem: Problem, out, out_global=None, expert_ids=None):
    """Bring a raw op-format output — reference OR kernel — back to fp32; the one function both sides
    go through. Dense (no ``quantize_output``, or the fp8-output oracle) → ``.float()``. ``"fp8"`` op
    output → dequant the per-(row, N-block) scale. MX/NVFP4 → un-swizzle a 5D SWIZZLE_32_4_4 Cs (the
    kernel's tcgen05 fast-path layout; the torch reference is always row-major) and dequant the group
    scales, folding the provided ``out_global`` back (``dq(q, Cs)·g_out ≈ intermediate``; None = MX,
    single-level)."""
    if not isinstance(out, (list, tuple)):  # dense intermediate (or the fp8-output oracle)
        return out.float()
    C, Cs = out
    if problem.activation_format == "fp8":
        # dq_scale decodes by dtype — fp32 passes through, UE8M0 (uint8) is 2^(exp-127).
        return C.float() * torch.repeat_interleave(
            dq_scale(Cs), C.shape[1] // Cs.shape[1], dim=-1
        )
    group = REQUANT_GROUP[problem.activation_format]
    if Cs.ndim == 5:
        # packed-E2M1 output (mxfp4/nvfp4) stores N/2 bytes, but the scale spans the logical N —
        # unswizzle over N columns, not the packed byte count (nvfp4 group-16 crosses a 4-block
        # boundary here; mxfp4 group-32 happens not to, which is why only nvfp4 exposed it).
        packed_out = problem.activation_format in ("mxfp4", "nvfp4")
        n_logical = C.shape[1] * (2 if packed_out else 1)
        Cs = unswizzle_mx_scales(Cs, C.shape[0], n_logical // group)
    dq = dq_grouped(C, Cs, group)
    g_rows = _rowwise(problem, out_global, expert_ids)
    return dq * g_rows if g_rows is not None else dq  # fold the provided global back


def _assert_op_layout(problem: Problem, op, out):
    """Op-only layout check (the torch reference never swizzles, so this can't live in ``_dequant``):
    a swizzled MX/NVFP4 block MUST emit a 5D SWIZZLE_32_4_4 Cs, else a silent row-major fallback would
    still pass the value check — but only where the layout survives to the output (matmul_2d always;
    grouped only with scatter_idx=None, so a scattered grouped output is legitimately row-major)."""
    if not problem.quantize_output or problem.activation_format == "fp8":
        return
    _, Cs = out
    expect = problem.swizzled and op == "matmul"
    assert (Cs.ndim == 5) == expect, (
        f"swizzled={problem.swizzled} op={op} but Cs.ndim={Cs.ndim} "
        f"({'expected 5D SWIZZLE_32_4_4' if expect else 'expected row-major'})"
    )


def _check(problem: Problem, dq_out, ref_cmp, expert_ids, op):
    """Compare two fp32 tensors already in the same space (both via the shared ``_dequant``): an
    exact-ish ``assert_close`` when the op returns dense values, a mean-relative bound for the requant
    paths (agreement is grid-boundary flips only — bit-equality across independently tuned launches is
    not the contract; fp32 accumulation order differs per config). Sentinel rows drop."""
    keep = (
        torch.ones(ref_cmp.shape[0], dtype=torch.bool, device=ref_cmp.device)
        if op == "matmul"  # no routing — every row is valid
        else expert_ids.long() < problem.E
    )
    if not problem.quantize_output:
        # Use output-dtype tolerances: Torch and tuned kernels may sum FP32
        # products in different orders, and GLU can amplify near-zero differences.
        atol, rtol = DTYPE_TO_TOL[problem.dtype]
        torch.testing.assert_close(
            dq_out[keep], ref_cmp[keep].to(problem.dtype).float(), atol=atol, rtol=rtol
        )
        return
    rel = (
        (dq_out[keep] - ref_cmp[keep]).abs().mean()
        / ref_cmp[keep].abs().mean().clamp(min=1e-6)
    ).item()
    assert rel < 0.06, f"requant dequant mean-rel {rel:.4f} vs offline reference"
    # The mean bound alone is blind to a corrupted minority (one garbage row in 64 still
    # passes it — the historical broken-winner signature). Bound the per-row error too:
    # every kept row must individually agree with the reference within a loose factor.
    # The bound sits above 0.5 because a SINGLE adjacent E2M1 code flip at the row amax is a
    # legal grid-boundary difference (ref 4s vs op 6s -> exactly 0.5) — garbage rows land ~1.0.
    row_err = (dq_out[keep] - ref_cmp[keep]).abs().amax(dim=-1)
    row_ref = ref_cmp[keep].abs().amax(dim=-1).clamp(min=1e-6)
    worst = (row_err / row_ref).max().item()
    assert worst < 0.75, (
        f"requant dequant worst-row rel {worst:.4f} (mean {rel:.4f}) — a localized "
        f"corruption the mean bound cannot see"
    )


def _make_weights(problem: Problem, row, E):
    """The scenario's weights: ``(B, Bs, Bs_global)``; under gate the interleaved gate|up stack."""
    kwargs = {"group": problem.int_group or problem.K} if problem.weights in INT_BITS else {}
    if problem.gate and problem.weights not in INT_BITS:
        B, Bs, Bs_global = make_gate_up(row, problem.N, problem.K, E)
    else:  # INT groups run along K per row: the stack quantizes as is
        B, Bs, Bs_global = row["make"](2 * problem.N if problem.gate else problem.N, problem.K, E, **kwargs)
    if problem.weights == "nvfp4" and problem.activation_format == "fp8":
        Bs = Bs[..., ::2].contiguous()  # valid group-32 E4M3 payload; both FP4 halves use one scale
    if problem.weights in INT_BITS and problem.activation_format == "fp8":
        Bs = Bs.float()
        Bs_global = dq_int(B, Bs, group=problem.int_group or problem.K).abs().amax(dim=(-2, -1)).div(448).clamp(min=1e-12)
    elif problem.weights in INT_BITS and problem.activation_format == "int8":
        Bs = Bs.float()
    if problem.csf_encoding:
        # E4M3 codes 24 apart (more than the 15 a CSF offset holds): an exception in every expert
        Bs[..., 0, :2] = Bs.new_tensor([0.125, 1.0])
    return B, Bs, Bs_global


def _skip_unsupported(problem: Problem, op: str) -> None:
    """The scenarios an op can't represent. matmul_2d is the single-GEMM
    sibling: it skips expert routing (sentinel / noncontiguous / empty-expert / the MoE prequant-As
    check) and non-MX input/output format knobs (its FP8 paths infer the quant from the scale shape and
    return the intermediate dense). Everything else — including full-precision (BF16/FP16) weights and
    a shared static activation scale — runs on all three ops."""
    if problem.svdquant_rank and op != "matmul":
        pytest.skip("SVDQuant checkpoint support is dense")
    if op != "matmul":
        return
    if problem.sentinel_fraction or problem.noncontiguous or problem.empty_expert:
        pytest.skip("expert-routing scenario (MoE only)")
    if problem.prequant and problem.weights != "fp8_rowwise":
        pytest.skip("prequantized block-scale inputs need routing")
    if problem.per_expert_globals:
        pytest.skip("per-expert globals need routing (MoE only)")

    mx = problem.weights in ("mxfp8", "mxfp8_u8", "mxfp4", "nvfp4")
    int_quantized = problem.weights in INT_BITS and problem.activation_format in ("fp8", "int8")
    if not mx and not int_quantized and (problem.activation_format not in (None, "bf16") or problem.quantize_output):
        pytest.skip("an explicit activation format / requant is MX-only for matmul_2d")


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize("op", ["batched", "grouped", "matmul"])
@pytest.mark.parametrize("problem", [p for p in PROBLEMS if p.ci], ids=lambda p: p.id)
def test_op_scenarios(problem: Problem, op):
    """Reference (the op written in torch) vs op (the kernel): same inputs, each returning the op's
    own output format, compared once through the shared ``_dequant``."""
    _run_scenario(problem, op)


@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize("op", ["batched", "grouped", "matmul"])
@pytest.mark.parametrize("problem", [p for p in PROBLEMS if not p.ci], ids=lambda p: p.id)
def test_op_scenarios_extended(problem: Problem, op):
    """The scenarios outside the CI subset: variants of a format the CI cells already cover."""
    _run_scenario(problem, op)


def _run_scenario(problem: Problem, op):
    _skip_unsupported(problem, op)
    A, expert_ids = _routed(problem)
    row = WEIGHTS[problem.weights]
    E = 1 if op == "matmul" else problem.E  # matmul is a single weight matrix
    B, Bs, Bs_global = _make_weights(problem, row, E)
    if (op != "matmul" and problem.N % 128 != 0 and problem.weights.startswith("fp8")
            and problem.activation_format != "bf16"):
        # fp8 weight scales are 128-blocked along N, so routed fp8 rejects non-128 N. MX (per-row
        # scales, BN | N) handles it on the affine arm — falls through to the normal ref-vs-op run.
        with pytest.raises(ValueError, match="matmul_2d"):
            _op(problem, op, A, expert_ids, B, Bs, Bs_global)
        return
    # ref is the op written in torch; both take the same inputs — including the PROVIDED nvfp4 output
    # global — and return the op's own format, so _dequant reads them identically and _check compares
    # once. _assert_op_layout is the one op-only check (the reference never swizzles its output).
    # prequant scenarios hand the op its activations already quantized (As set); otherwise
    # raw A + As=None and the op quantizes it. Reference
    # and op take the identical (A, As): pre-quantizing just hands the op the values it would
    # otherwise compute, and _fp32_intermediate reads whichever form it's given. The nvfp4 activation
    # global comes off the RAW A (before prequant replaces it) and rides separately as As_global.
    _run_ref_vs_op(problem, op, A, expert_ids, B, Bs, Bs_global)


def _run_ref_vs_op(problem: Problem, op, A, expert_ids, B, Bs, Bs_global, shared=None):
    """The shared ref-vs-op body: build the activation-side operands, run both sides, compare
    through the one ``_dequant``. Factored out so the forced-config sweep runs the identical
    check per config; ``shared`` (a dict) caches the config-independent half (operands +
    reference) across repeated calls at identical inputs, so the sweep re-runs only the op."""
    if not shared:
        As = None
        As_global = _act_global(problem, A, expert_ids)
        if problem.prequant:
            A, As = _prequant_args(problem, op, A, expert_ids)
            if problem.noncontiguous and As is not None:
                As = _make_noncontig(As.reshape(-1)).view(As.shape)
        g_out = _out_global(problem, op, A, expert_ids, B, Bs, Bs_global, As, As_global)
        ref = _reference(problem, op, A, expert_ids, B, Bs, Bs_global, As=As, As_global=As_global, out_global=g_out)
        if shared is not None:
            shared.update(A=A, As=As, As_global=As_global, g_out=g_out, ref=ref)
    else:
        A, As, As_global, g_out, ref = (
            shared["A"], shared["As"], shared["As_global"], shared["g_out"], shared["ref"]
        )
    out = _op(problem, op, A, expert_ids, B, Bs, Bs_global, As=As, As_global=As_global, out_global=g_out)
    _assert_op_layout(problem, op, out)
    if problem.gate and not problem.quantize_output:
        # The GLU scales the gate|up accumulation error by |act(gate)|: on a cancelling up column
        # the summation-order noise of any fp32 GEMM (cuBLAS included) exceeds the output
        # tolerance. So check the GEMM (raw gate|up, act_fn=None) against the reference, then the
        # GLU against the reference GLU of the kernel's own gate|up (fp32, as the epilogue reads its
        # accumulators) — each at its own error scale.
        raw_problem = replace(problem, act_fn=None)
        raw_ref = shared.get("raw_ref") if shared else None
        if raw_ref is None:
            raw_ref = _fp32_intermediate(raw_problem, op, A, expert_ids, B, Bs, Bs_global, As, As_global)
            if shared is not None:
                shared["raw_ref"] = raw_ref
        raw = _dequant(raw_problem, _op(
            raw_problem, op, A, expert_ids, B, Bs, Bs_global, As=As, As_global=As_global, out_global=g_out,
            output_dtype=torch.float32,
        ), g_out, expert_ids)
        _check(raw_problem, raw, raw_ref, expert_ids, op)
        ref = apply_glu(
            raw[..., 0::2], raw[..., 1::2], problem.act_fn, problem.swiglu_alpha, problem.swiglu_limit
        ).float()
    _check(
        problem,
        _dequant(problem, out, g_out, expert_ids),
        _dequant(problem, ref, g_out, expert_ids),
        expert_ids,
        op,
    )


# Cells for the forced-config sweep: one mx grouped gate|up + requant launch (the arm-dense
# kernel: dot_scaled/dot x memory modes x WS) and one batched decode launch (the swap/scalar
# arms — the family where a BN=256 winner at N=128 historically corrupted 40/64 rows), plus
# the SWIZZLED_SCALES/INTERLEAVED_SCALES tune-key axes (their own pruner scopes), the 2D mx
# kernel (the dense gate fusion and its gate_pointer_only/WS pruners), and one nvfp4 cell
# (E4M3 scales: the nvfp4_native_ok fence + software decode arms).
_SWEEP_CELLS = [
    (Problem(weights="mxfp8", gate=True, activation_format="mxfp8", quantize_output=True), "grouped", mx_dynamic_matmul_grouped_kernel),
    # K off the BK=128 grid admits the BK=64 rows, the family a K=256 cell never reaches
    (Problem(weights="mxfp4", K=320), "grouped", mx_dynamic_matmul_grouped_kernel),
    (Problem(weights="mxfp8", K=320), "grouped", mx_dynamic_matmul_grouped_kernel),
    (Problem(weights="mxfp4", S=8), "batched", mx_dynamic_matmul_batched_kernel),
    (Problem(weights="mxfp8", gate=True, activation_format="mxfp8", quantize_output=True, swizzled=True), "grouped", mx_dynamic_matmul_grouped_kernel),
    (Problem(weights="mxfp8", gate=True, activation_format="mxfp8", quantize_output=True, swizzled=True), "matmul", mx_dynamic_matmul_kernel),
    # non-128 N on the 2D gate|up tile (gpt-oss-style N=320): the BN=128 tile overhangs N
    (Problem(weights="mxfp4", gate=True, activation_format="mxfp4", quantize_output=True, N=320, K=320), "matmul", mx_dynamic_matmul_kernel),
    (Problem(weights="nvfp4", gate=True, activation_format="nvfp4", quantize_output=True, swizzled=True, S=8), "batched", mx_dynamic_matmul_batched_kernel),
    (Problem(weights="fp8_128x128", activation_format="bf16"), "grouped", block_fp8_weight_only_matmul_grouped_kernel),
    # the calibrated (static) arm on per-tensor weights: it hands the kernel a RAW A to quantize
    # per tile, the one activation form that cannot ride the TMA gather, so its admitted set is
    # the one a memory-mode fence gets wrong (silently — the tuner forgives what will not lower)
    (Problem(weights="fp8_tensor", static=True), "grouped", w8a8_tensor_dynamic_fp8_matmul_grouped_kernel),
    # row-wise weights: the per-row scale vector each arm folds on its own weight-row axis — the
    # stacked gate|up span, the swapped decode tile, and the 2D kernel's tile
    (Problem(weights="fp8_rowwise", gate=True, S=257, bias=True, empty_expert=True), "grouped", w8a8_tensor_dynamic_fp8_matmul_grouped_kernel),
    (Problem(weights="fp8_rowwise", gate=True, S=8), "batched", w8a8_tensor_dynamic_fp8_matmul_batched_kernel),
    (Problem(weights="fp8_rowwise", gate=True, S=256), "matmul", w8a8_tensor_dynamic_fp8_matmul_kernel),

    # the int4 weight-only kernels: the 2D kernel's tile x WS x memory-mode space and the batched
    # dot / swapped-reduce arms, asymmetric so the zero-point reads are in every config
    (Problem(weights="int4", zero_points=True, gate=True), "matmul", int_weight_only_matmul_2d_kernel),
    (Problem(weights="int4", zero_points=True, S=8), "batched", int_weight_only_matmul_batched_kernel),
    # the 2D kernel's single-token scalar reduce: its narrow tiles and the gate|up rows
    (Problem(weights="int4", zero_points=True, gate=True, S=1), "matmul", int_weight_only_matmul_2d_kernel),
    (Problem(weights="int4", zero_points=True, gate=True), "grouped", int_weight_only_matmul_grouped_kernel),
    (Problem(weights="int8", zero_points=True, S=4), "matmul", int_weight_only_matmul_2d_kernel),
    (Problem(weights="int2", zero_points=True, S=4), "matmul", int_weight_only_matmul_2d_kernel),
    # Sweeps include BK < group, so a tuned winner cannot hide wide-group addressing.
    (Problem(weights="int4", int_group=512, K=1024, S=4, zero_points=True, gate=True), "matmul", int_weight_only_matmul_2d_kernel),
    (Problem(weights="int4", int_group=512, K=1024, zero_points=True, gate=True), "batched", int_weight_only_matmul_batched_kernel),
    (Problem(weights="int4", int_group=None, K=1024, zero_points=True, gate=True), "grouped", int_weight_only_matmul_grouped_kernel),
]


@pytest.mark.slow
@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize(
    "problem, op, tuner",
    _SWEEP_CELLS,
    ids=[f"{p.id}_{op}" for p, op, _ in _SWEEP_CELLS],
)
def test_every_admitted_config_is_correct(problem: Problem, op, tuner):
    """Force-run EVERY config the pruners admit for one launch and hard-check each against the
    torch oracle. The tuner benches configs by SPEED, so a wrong-but-fast config that slips the
    pruners gets crowned silently — the class no winner-only test can see. A forced config may
    fail to compile/launch (the tuner's forgiven-inf path — reported, not asserted); a config
    that RUNS must be correct."""

    A, expert_ids = _routed(problem)
    row = WEIGHTS[problem.weights]
    B, Bs, Bs_global = _make_weights(problem, row, problem.E)

    admitted: dict = {}
    orig_prune, orig_configs = tuner.early_config_prune, tuner.configs

    def spy(configs, named_args, **kwargs):
        kept = orig_prune(configs, named_args, **kwargs) if orig_prune else configs
        admitted["configs"] = list(kept)
        return kept

    tuner.early_config_prune = spy
    shared: dict = {}
    try:
        tuner.cache.clear()
        _run_ref_vs_op(problem, op, A, expert_ids, B, Bs, Bs_global, shared=shared)
        assert admitted.get("configs"), "the spy never saw a prune pass"
        forgiven = []
        for cfg in admitted["configs"]:
            tuner.configs = [cfg]
            tuner.early_config_prune = None
            tuner.cache.clear()
            try:
                _run_ref_vs_op(problem, op, A, expert_ids, B, Bs, Bs_global, shared=shared)
            except AssertionError as e:
                raise AssertionError(f"admitted config computes WRONG results: {cfg}") from e
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                # a sticky device fault poisons the context: every LATER config would raise
                # too and be forgiven, silently gutting the rest of the sweep. Probe the
                # context directly instead of matching error strings — Triton formats
                # faults many ways ("Triton Error [CUDA]", "CUDA driver error", ...).
                try:
                    torch.ones(1, device=TEST_DEVICE).item()
                except Exception as probe:
                    raise AssertionError(
                        f"config left a STICKY device fault (the context is poisoned): "
                        f"{cfg}\n{err}\nprobe: {type(probe).__name__}: {str(probe)[:120]}"
                    ) from e
                forgiven.append((cfg, err[:120]))
    finally:
        tuner.early_config_prune, tuner.configs = orig_prune, orig_configs
        tuner.cache.clear()
    if forgiven:
        print(f"\n[sweep] {len(forgiven)} admitted config(s) failed to compile/run (forgiven):")
        for cfg, err in forgiven:
            print(f"  {cfg}: {err}")


@triton.jit
def _decode_ue8m0_scales(Scales, Out):
    offsets = tl.arange(0, 8)
    tl.store(Out + offsets, decode_group_scale(tl.load(Scales + offsets)))


@pytest.mark.kernels_ci
def test_ue8m0_scale_bytes():
    values = [0, 1, 2, 126, 127, 128, 254, 255]
    scales = torch.tensor(values, dtype=torch.uint8, device=TEST_DEVICE)
    actual = torch.empty(8, dtype=torch.float32, device=TEST_DEVICE)
    _decode_ue8m0_scales[(1,)](scales, actual)
    expected = torch.tensor(
        [2.0 ** (byte - 127) for byte in values[:-1]] + [float("nan")],
        dtype=torch.float32, device=TEST_DEVICE,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.kernels_ci
def test_uint8_mx_scales_without_native_ue8m0(monkeypatch):
    """MX format checks accept raw exponent bytes without a native UE8M0 dtype."""
    scale = torch.full((16, 4), 127, dtype=torch.uint8)
    if (native_dtype := finegrained_kernels.compat.UE8M0_DTYPE) is not None:
        native_scale = scale.view(native_dtype)
        assert torch.equal(ue8m0_as_uint8(native_scale), scale)
        assert mx_scale_family(native_scale, 128) == 32

    # What torch < 2.7 resolves at import time.
    monkeypatch.setattr(finegrained_kernels.formats, "UE8M0_DTYPE", None)
    monkeypatch.setattr(finegrained_kernels.formats, "UE8M0_SCALE_DTYPES", (torch.uint8,))
    assert ue8m0_as_uint8(scale) is scale
    assert is_mxfp8(
        torch.empty((16, 128), dtype=torch.float8_e4m3fn), scale
    )
    assert is_mxfp4(torch.empty((16, 64), dtype=torch.int8), scale)
    assert mx_scale_family(scale, 128) == 32
    fp32 = torch.ones((16, 4), dtype=torch.float32)
    assert ue8m0_as_uint8(fp32) is fp32


# Embedding lookup has no GEMM operands or epilogue: check the selected-row operation directly.


_EMBEDDING_CASES = [
    pytest.param("fp8", "tensor", dtype, id=f"fp8_tensor_{dtype}")
    for dtype in (torch.float32, torch.bfloat16, torch.float16)
] + [
    pytest.param("fp8", "row", torch.bfloat16, id="fp8_row"),
    pytest.param("fp8", "block", torch.float32, id="fp8_block"),
    pytest.param("nvfp4", "group", torch.bfloat16, id="nvfp4"),
    pytest.param("mxfp8", "group", torch.bfloat16, id="mxfp8"),
    pytest.param("mxfp4", "group", torch.float16, id="mxfp4"),
    pytest.param("int8", "row", torch.float32, id="int8"),
    pytest.param("int4", "group", torch.bfloat16, id="int4"),
]


def _embedding_table(weight_format, layout, dtype, sliced=False):
    rows, width = 38, 256 if sliced else 160
    if weight_format in ("nvfp4", "mxfp4", "int4"):
        shape = (rows // 2, width) if weight_format == "int4" else (rows, width // 2)
        weight = torch.randint(0, 256, shape, device=TEST_DEVICE, dtype=torch.uint8)
        # Every nibble, including signed INT4's -8/-1 and E2M1's signed zero.
        weight[0] = (
            torch.arange(16, device=TEST_DEVICE, dtype=torch.uint8)
            .mul(17)
            .repeat(shape[1] // 16)
        )
    elif weight_format == "int8":
        weight = torch.randint(
            -128, 128, (rows, width), device=TEST_DEVICE, dtype=torch.int8
        )
    else:
        weight = (torch.randn(rows, width, device=TEST_DEVICE) * 3).to(
            torch.float8_e4m3fn
        )
    group = 16 if weight_format == "nvfp4" else 32
    scale_shape = {
        "tensor": (1,),
        "row": (rows,),
        "block": ((rows + 31) // 32, (width + 31) // 32),
        "group": (rows, width // group),
    }[layout]
    if weight_format in ("mxfp8", "mxfp4"):
        scale = torch.randint(
            124, 131, scale_shape, device=TEST_DEVICE, dtype=torch.uint8
        )
    else:
        scale_dtype = torch.float8_e4m3fn if weight_format == "nvfp4" else dtype
        scale = (torch.rand(scale_shape, device=TEST_DEVICE) * 3 + 0.125).to(
            scale_dtype
        )
    global_scale = (
        torch.tensor([0.0371], device=TEST_DEVICE) if weight_format == "nvfp4" else None
    )
    if sliced:
        if layout == "row":
            scale = scale[:, None]
        # Retain shape while exercising non-unit column strides and offset storage.
        views = []
        for tensor in (weight, scale):
            backing = torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] * 2 + 1),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            view = backing[..., 1::2]
            view.copy_(tensor)
            views.append(view)
        weight, scale = views
        if weight_format == "nvfp4":
            weight = weight.view(torch.int8)
            # Zero column stride exercises scalar scale loads in packed decoding.
            scale = scale[:, :1].expand_as(scale)
        if weight_format in ("mxfp8", "mxfp4"):
            scale = scale.view(torch.float8_e8m0fnu)
    return weight, scale, global_scale


def _embedding_reference(
    weight, ids, scale, global_scale, dtype, weight_format, layout
):
    if weight_format == "int4":
        u = weight.to(torch.uint8).long()
        values = torch.stack((u & 15, u >> 4), dim=1).reshape(weight.shape[0] * 2, -1)
        values = torch.where(values >= 8, values - 16, values).float()
        values = values * scale.repeat_interleave(32, -1)
    elif layout == "group":
        values = dq_grouped(weight, scale, 16 if weight_format == "nvfp4" else 32)
    elif layout == "block":
        values = dq_block_fp8(weight, scale, 32, 32)
    else:
        values = weight.float() * (scale.reshape(-1, 1) if layout == "row" else scale)
    if global_scale is not None:
        values = values * global_scale
    return torch.nn.functional.embedding(ids, values.to(dtype))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("weight_format,layout,dtype", _EMBEDDING_CASES)
@pytest.mark.parametrize("sliced", [False, True])
def test_embedding_lookup_matches_selected_rows(weight_format, layout, dtype, sliced):
    weight, scale, global_scale = _embedding_table(weight_format, layout, dtype, sliced)
    ids = torch.tensor([[3, 0, 3], [37, 4, 1]], device=weight.device).t()
    for indices in (ids, ids.to(torch.int32), ids.new_empty((2, 0)), ids.new_tensor(3)):
        actual = embedding_lookup(
            weight,
            indices,
            scale,
            global_scale,
            dtype,
            weight_format=weight_format,
            block_size=32,
        )
        torch.testing.assert_close(
            actual,
            _embedding_reference(
                weight, indices, scale, global_scale, dtype, weight_format, layout
            ),
            rtol=0,
            atol=0,
        )
        assert actual.dtype == dtype
    # Keep the original positional FP8/NVFP4 entry points covered.
    if layout == "tensor" or weight_format == "nvfp4":
        default = embedding_lookup(weight, ids, scale, global_scale)
        default_dtype = torch.bfloat16 if weight_format == "nvfp4" else scale.dtype
        assert default.dtype == default_dtype
        torch.testing.assert_close(
            default,
            _embedding_reference(
                weight, ids, scale, global_scale, default_dtype, weight_format, layout
            ),
            rtol=0,
            atol=0,
        )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("weight_format,layout,dtype", _EMBEDDING_CASES)
def test_embedding_lookup_fullgraph_compile(weight_format, layout, dtype):
    weight, scale, global_scale = _embedding_table(
        weight_format, layout, dtype, sliced=True
    )
    ids = torch.tensor([[3, 0, 3], [37, 4, 1]], device=weight.device).t()
    lookup = maybe_compile(embedding_lookup, True)
    actual = lookup(
        weight,
        ids,
        scale,
        global_scale,
        dtype,
        weight_format=weight_format,
        block_size=32,
    )
    torch.testing.assert_close(
        actual,
        _embedding_reference(
            weight, ids, scale, global_scale, dtype, weight_format, layout
        ),
        rtol=0,
        atol=0,
    )


def _embedding_invalid_index(_rank, index, packed):
    weight = torch.ones(37, 16 if packed else 32, device=TEST_DEVICE).to(
        torch.uint8 if packed else torch.float8_e4m3fn
    )
    scale = (
        torch.ones(37, 2, device=TEST_DEVICE).to(torch.float8_e4m3fn)
        if packed
        else torch.ones(1, device=TEST_DEVICE)
    )
    global_scale = torch.ones(1, device=TEST_DEVICE) if packed else None
    embedding_lookup(
        weight, torch.tensor([index], device=TEST_DEVICE), scale, global_scale
    )
    getattr(torch, torch.device(TEST_DEVICE).type).synchronize()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("index", [-1, 37])
def test_embedding_invalid_index_fails_in_separate_process(index, packed):
    # Isolate the poisoned device context, as for native embedding's bounds assertions.
    with pytest.raises(
        torch.multiprocessing.ProcessRaisedException,
        match="(?i)device-side assert|assertion failure",
    ):
        torch.multiprocessing.spawn(
            _embedding_invalid_index, args=(index, packed), nprocs=1
        )


@pytest.mark.skipif(not SUPPORTS_SWIZZLED_SCALES, reason="native scaled MMA required")
@pytest.mark.parametrize("strategy", ["split", "persistent"])
def test_svdquant_scheduling(monkeypatch, strategy):
    """Force both schedules: a short correctness tune can miss split-K or multiple tile iterations."""
    tuner = mx_dynamic_matmul_kernel
    persistent = strategy == "persistent"
    config = next(c for c in tuner.configs if c.kwargs.get("SPLIT_K") == (1 if persistent else 4)
                  and c.kwargs["PERSISTENT"] == persistent
                  and c.kwargs["BLOCK_SIZE_N"] == c.kwargs["BLOCK_SIZE_K"] == 128
                  and c.kwargs["WARP_SPEC"] and c.num_warps == 4 and c.num_stages == 3)
    baseline = next(c for c in tuner.configs if c.kwargs == dict(config.kwargs, PERSISTENT=False)
                    and c.num_warps == config.num_warps and c.num_stages == config.num_stages)
    monkeypatch.setattr(tuner, "configs", [config])
    monkeypatch.setattr(tuner, "early_config_prune", None)
    tuner.cache.clear()
    if persistent:
        monkeypatch.setattr("finegrained_kernels.matmul.persistent_program_count", lambda _: 2)
    problem = Problem(weights="nvfp4", S=384 if persistent else 1, N=256,
                      K=256 if persistent else 1024, svdquant_rank=32,
                      pre_quant_scale=True, bias=True, swizzled=True, compile=True)
    A, ids = _routed(problem)
    B, Bs, global_scale = _make_weights(problem, WEIGHTS[problem.weights], 1)
    try:
        if persistent:
            # Scheduling must preserve every bit, including BF16 midpoint choices from FP32 sum order.
            act_global = _act_global(problem, A, ids)
            actual = _op(problem, "matmul", A, ids, B, Bs, global_scale, As_global=act_global)
            tuner.configs = [baseline]
            tuner.cache.clear()
            expected = _op(problem, "matmul", A, ids, B, Bs, global_scale, As_global=act_global)
            assert torch.equal(actual, expected)
        else:
            _run_ref_vs_op(problem, "matmul", A, ids, B, Bs, global_scale)
    finally:
        tuner.cache.clear()


@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="two concurrent CUDA streams required")
def test_int4_split_k_concurrent_streams():
    """Two split-K int4 matmuls launched concurrently on two streams, many times over, must equal their serial results
    bit for bit: each launch reduces through its own arrival counters and partials (shared counters would let one
    launch's arrivals finish another's tile early and sum unwritten partials). Both streams first sleep, so the
    launches queue up behind the sleep and then run overlapped, with small grids that co-reside on the GPU."""
    tuner = int_weight_only_matmul_2d_kernel
    split = [c for c in tuner.configs if c.kwargs.get("SPLIT_K") == 8 and c.kwargs["BLOCK_SIZE_N"] == 16
             and c.kwargs["BLOCK_SIZE_K"] == 128 and c.kwargs["A_MEMORY_MODE"] == "pointer"
             and c.num_warps == 2 and c.num_stages == 3]
    assert split, "no SPLIT_K=8 decode config in the grid"
    g = torch.Generator(device=TEST_DEVICE).manual_seed(3)
    N, K, M = 512, 4096, 4
    problems = []
    for _ in range(2):
        q = torch.randint(0, 16, (N, K), device=TEST_DEVICE, dtype=torch.int32, generator=g)
        B = (q.reshape(N, K // 8, 8) << (4 * torch.arange(8, device=TEST_DEVICE, dtype=torch.int32))).sum(-1, dtype=torch.int32).view(torch.uint32)
        Bs = (torch.rand(N, K // 128, device=TEST_DEVICE, generator=g) * 0.02 + 0.005).to(torch.bfloat16)
        A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16, generator=g)
        problems.append((A, B, Bs))
    orig_prune, orig_configs = tuner.early_config_prune, tuner.configs
    try:
        tuner.configs, tuner.early_config_prune = split[:1], None
        tuner.cache.clear()
        serial = [finegrained_kernels.matmul_2d(A, B, None, Bs) for A, B, Bs in problems]
        torch.cuda.synchronize()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        outs = [[], []]
        for stream in streams:
            with torch.cuda.stream(stream):
                torch.cuda._sleep(200_000_000)
        for _ in range(100):
            for i, (A, B, Bs) in enumerate(problems):
                with torch.cuda.stream(streams[i]):
                    outs[i].append(finegrained_kernels.matmul_2d(A, B, None, Bs))
        torch.cuda.synchronize()
    finally:
        tuner.early_config_prune, tuner.configs = orig_prune, orig_configs
        tuner.cache.clear()
    for i in range(2):
        for out in outs[i]:
            assert torch.equal(out, serial[i]), "a concurrent split-K launch differs from its serial result"


@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize("op", ["matmul", "batched", "grouped"])
def test_int_natural_layout_refused(op):
    """int32 words, a checkpoint's natural packing, are refused by every integer op: only uint32 words, the
    interleaved layout ``interleave_int_words`` and the repacks emit, are read."""
    B, Bs, _ = WEIGHTS["int4"]["make"](256, 512, 1)
    A = torch.randn(4, 512, device=TEST_DEVICE, dtype=torch.bfloat16)
    expert_ids = torch.zeros(4, device=TEST_DEVICE, dtype=torch.int32)
    expert_start = torch.tensor([0, 4], device=TEST_DEVICE, dtype=torch.int32)
    run = {
        "matmul": lambda W: finegrained_kernels.matmul_2d(A, W[0], None, Bs[0]),
        "batched": lambda W: finegrained_kernels.matmul_batched(A, W, None, Bs, expert_ids=expert_ids),
        "grouped": lambda W: finegrained_kernels.matmul_grouped(A, W, None, Bs, expert_start=expert_start),
    }[op]
    run(B)
    with pytest.raises(AssertionError, match="interleave_int_words"):
        run(B.view(torch.int32))


@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize("checkpoint", ["gptq", "gptq_v2", "gptq_symmetric", "awq", "modelopt_int4", "modelopt_int8", "ggml_q4_0", "ggml_q8_0"])
def test_int_repacks(checkpoint):
    """GPTQ / AWQ int4 linears packed by their own conventions (GPTQ: values along K, zero points along N minus 1,
    0 under ``gptq_v2``; AWQ: both along N in its ``[0, 2, 4, 6, 1, 3, 5, 7]`` column order) repack to weights whose
    dequant is ``(q - zero) * scale`` of the source values. ModelOpt stores signed values, INT4 across
    output-channel pairs or INT8 unpacked, and both must preserve the original signed-weight reference."""

    g = torch.Generator(device=TEST_DEVICE).manual_seed(4)
    K, N, group = 512, 256, 128
    if checkpoint.startswith(("modelopt", "ggml")):
        ggml = checkpoint.startswith("ggml")
        bits = 4 if checkpoint in ("modelopt_int4", "ggml_q4_0") else 8
        group = 32 if ggml else (128 if bits == 4 else K)
        # Cover dense 2D and stacked experts without a separate codec/metadata test matrix.
        leading = () if bits == 4 else (2,)
        signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (*leading, N, K),
                               device=TEST_DEVICE, dtype=torch.int8, generator=g)
        scales = (torch.rand(*leading, N, K // group, device=TEST_DEVICE, generator=g) * 0.02 + 0.005).to(
            torch.float16 if ggml else torch.float32
        )
        if ggml:
            d = scales.half()
            d[..., 1::2] *= -1  # canonical Q4_0 scales can be negative
            d[..., 0] = 0
            blocks = torch.empty(*leading, N, K // group, 2 + 4 * bits, device=TEST_DEVICE, dtype=torch.uint8)
            blocks[..., :2] = d.contiguous().view(torch.uint8).view(*leading, N, K // group, 2)
            values = signed.reshape(*leading, N, K // group, group)
            if bits == 4:
                values = values.int() + 8
                blocks[..., 2:] = (values[..., :16] | (values[..., 16:] << 4)).to(torch.uint8)
            else:
                blocks[..., 2:] = values.view(torch.uint8)
            W, Ws, Wz = repack_ggml_int(blocks, "Q4_0" if bits == 4 else "Q8_0")
            scales = d.float()  # GGML reconstructs q*d in FP32 before casting the weight.
        elif bits == 4:
            qweight = ((signed[0::2].int() & 15) | ((signed[1::2].int() & 15) << 4)).to(torch.uint8)
            W, Ws, Wz = repack_modelopt_int4(qweight, scales)
        else:
            W, Ws, Wz = repack_modelopt_int8(signed, scales.squeeze(-1))
        ref = signed.to(scales.dtype) * scales.repeat_interleave(group, dim=-1)
        assert W.dtype == torch.uint32 and Wz is None
        assert torch.equal(dequantize_int_weight(W, Ws, K), ref)
        A = torch.randn(8, K, device=TEST_DEVICE, dtype=torch.float16, generator=g)
        if bits == 4:
            out = finegrained_kernels.matmul_2d(A, W, None, Ws)
            expected = A.float() @ ref.half().float().T
        else:
            expert_ids = torch.arange(8, device=TEST_DEVICE, dtype=torch.int32) % 2
            run = torch.compile(finegrained_kernels.matmul_batched, fullgraph=True)
            out = run(A, W, None, Ws, expert_ids=expert_ids)
            expected = torch.bmm(ref[expert_ids.long()].half().float(), A.float().unsqueeze(-1)).squeeze(-1)
        atol, rtol = DTYPE_TO_TOL[torch.float16]
        torch.testing.assert_close(out.float(), expected.half().float(), atol=atol, rtol=rtol)
        return
    q = torch.randint(0, 16, (K, N), device=TEST_DEVICE, dtype=torch.int32, generator=g)
    z = torch.randint(0, 16, (K // group, N), device=TEST_DEVICE, dtype=torch.int32, generator=g)
    if checkpoint == "gptq_symmetric":
        z.fill_(8)
    scales = (torch.rand(K // group, N, device=TEST_DEVICE, generator=g) * 0.02 + 0.005).to(torch.float16)
    nibble = 4 * torch.arange(8, device=TEST_DEVICE, dtype=torch.int32)

    def pack_n(x, order=range(8)):  # (rows, N) -> (rows, N // 8), word nibble i = column order[i]
        cols = x.reshape(x.shape[0], N // 8, 8)[..., list(order)]
        return (cols << nibble).sum(-1, dtype=torch.int32)

    if checkpoint == "awq":
        awq_order = (0, 2, 4, 6, 1, 3, 5, 7)
        W, Ws, Wz = repack_awq_int4(pack_n(q, awq_order), pack_n(z, awq_order), scales)
    else:
        qweight = (q.reshape(K // 8, 8, N) << nibble[None, :, None]).sum(1, dtype=torch.int32)
        offset = 0 if checkpoint == "gptq_v2" else 1
        W, Ws, Wz = repack_gptq_int4(
            qweight, pack_n((z - offset) & 0xF), scales, zeros_offset=offset,
            symmetric=checkpoint == "gptq_symmetric",
        )
    assert W.dtype == torch.uint32
    ref = (q.T - z.T.repeat_interleave(group, dim=1)).to(torch.float16) * scales.T.repeat_interleave(group, dim=1)
    assert torch.equal(dequantize_int_weight(W, Ws, K, Wz), ref)


@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize(
    "weights,tuner,activation_format",
    [
        ("bf16", full_precision_matmul_grouped_kernel, None),
        ("mxfp8", mx_dynamic_matmul_grouped_kernel, None),
        ("int4", int_weight_only_matmul_grouped_kernel, None),
        ("fp8_tensor", block_fp8_weight_only_matmul_grouped_kernel, "bf16"),
    ],
)
def test_grouped_packed_schedule_built_by_its_configs(weights, tuner, activation_format):
    """A ``PACKED_SCHEDULE`` config reads the tile schedule its own pre_hook builds (``fill_packed_schedule``), the
    other configs launch without it: forced to one packed config, a grouped op equals its inline-layout twin (same
    tile, same K order) bit for bit."""
    torch.manual_seed(0)
    E, S, N, K = 4, 512, 256, 512
    B, Bs, _ = WEIGHTS[weights]["make"](N, K, E)
    A = torch.randn(S, K, device=TEST_DEVICE, dtype=torch.bfloat16)
    expert_ids = torch.randint(0, E, (S,), device=TEST_DEVICE, dtype=torch.int32)
    order = torch.argsort(expert_ids, stable=True).to(torch.int32)
    expert_start = torch.zeros(E + 1, device=TEST_DEVICE, dtype=torch.int32)
    expert_start[1:] = torch.histc(expert_ids.float(), bins=E, min=0, max=E - 1).to(torch.int32).cumsum(0)
    admitted: dict = {}
    orig_prune, orig_configs = tuner.early_config_prune, tuner.configs

    def spy(configs, named_args, **kwargs):
        admitted["configs"] = orig_prune(configs, named_args, **kwargs)
        return admitted["configs"]

    def run():
        tuner.cache.clear()
        return finegrained_kernels.matmul_grouped(A, B, None, Bs, expert_start=expert_start, gather_idx=order, activation_format=activation_format)

    try:
        tuner.early_config_prune = spy
        run()
        packed = next(c for c in admitted["configs"] if c.kwargs.get("PACKED_SCHEDULE"))
        twin = next(
            c for c in orig_configs
            if not c.kwargs.get("PACKED_SCHEDULE") and c.num_warps == packed.num_warps
            and c.num_stages == packed.num_stages
            and {k: v for k, v in c.kwargs.items() if k != "PACKED_SCHEDULE"}
            == {k: v for k, v in packed.kwargs.items() if k != "PACKED_SCHEDULE"}
        )
        tuner.early_config_prune = None
        outs = []
        for c in (packed, twin):
            tuner.configs = [c]
            outs.append(run())
    finally:
        tuner.early_config_prune, tuner.configs = orig_prune, orig_configs
        tuner.cache.clear()
    assert torch.equal(outs[0], outs[1]), "the packed-schedule config differs from its inline twin"


@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize("M", [1, 16, 256])
@pytest.mark.parametrize("op", ["matmul", "grouped", "batched"])
def test_int4_act_order(op, M):
    """Act-order GPTQ (``desc_act=True``, groups scattered across K): repacked to contiguous groups plus a K order the
    ops gather A's columns by (``b_perm``), every op matches ``A @ W_ref.T`` with the weight dequantized in A's own
    order; an identity ``g_idx`` repacks exactly as no ``g_idx`` (three tensors, same bits)."""

    torch.manual_seed(0)
    E, N, K = (1 if op == "matmul" else 4), 256, 512
    W, Ws, Wz, perm, W_ref = make_gptq_act_order(N, K, E)
    A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16)
    if op == "matmul":
        out = finegrained_kernels.matmul_2d(A, W[0], None, Ws[0], b_zero=Wz[0], b_perm=perm[0])
        ref = A.float() @ W_ref[0].T
    else:
        expert_ids = torch.randint(0, E, (M,), device=TEST_DEVICE, dtype=torch.int32)
        ref = torch.einsum("sk,snk->sn", A.float(), W_ref[expert_ids.long()])
        if op == "batched":
            out = finegrained_kernels.matmul_batched(A, W, None, Ws, expert_ids=expert_ids, b_zero=Wz, b_perm=perm)
        else:
            order = torch.argsort(expert_ids, stable=True)
            expert_start = torch.zeros(E + 1, device=TEST_DEVICE, dtype=torch.int32)
            expert_start[1:] = torch.histc(expert_ids.float(), bins=E, min=0, max=E - 1).to(torch.int32).cumsum(0)
            out = finegrained_kernels.matmul_grouped(
                A, W, None, Ws, expert_start=expert_start, gather_idx=order.to(torch.int32), b_zero=Wz, b_perm=perm,
            )
            ref = ref[order]
    err = ((out.float() - ref).norm() / ref.norm()).item()
    assert err < 4e-3, f"{op} M={M}: act-order vs dequantized reference {err:.2e}"

    q = torch.randint(0, 2**31 - 1, (K // 8, N), device=TEST_DEVICE, dtype=torch.int32)
    z = torch.randint(0, 2**31 - 1, (K // 128, N // 8), device=TEST_DEVICE, dtype=torch.int32)
    s = torch.rand(K // 128, N, device=TEST_DEVICE).to(torch.bfloat16)
    identity = repack_gptq_int4(q, z, s, g_idx=torch.arange(K, device=TEST_DEVICE, dtype=torch.int32) // 128)
    plain = repack_gptq_int4(q, z, s)
    assert len(identity) == 3 and all(torch.equal(x, y) for x, y in zip(identity, plain))
