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

from dataclasses import dataclass

import pytest
import torch

from utils import (  # type: ignore
    DTYPE_TO_TOL,
    REQUANT_FN,
    REQUANT_GROUP,
    TEST_DEVICE,
    WEIGHTS,
    dq_grouped,
    dq_scale,
    make_static_activation_scale,
    maybe_compile,
    quant_dequant_a,
)

import finegrained_kernels  # type: ignore
from finegrained_kernels import swizzle_mx_scales, unswizzle_mx_scales  # type: ignore
from finegrained_kernels.compat import NVFP4_SCALE_GROUP_K  # type: ignore
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
    quantize_output: bool = False
    per_expert_globals: bool = False  # activation (and requant output) globals calibrated per expert
    prequant: bool = False  # pass As explicitly (must be bit-identical to raw A)
    static: bool = False  # calibrated (static) activation scale instead of a runtime one
    swizzled: bool = False  # pass MX weight scales pre-swizzled (5D SWIZZLE_32_4_4 fast path)
    sentinel_fraction: float = 0.0
    noncontiguous: bool = False
    empty_expert: bool = False
    compile: bool = False
    dtype: torch.dtype = torch.bfloat16

    @property
    def id(self):
        tag = self.weights
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
        default_in = {"fp8_128x128": "fp8", "fp8_tensor": "fp8"}.get(w)
        for fmt in row["activation_formats"]:
            if fmt != default_in:
                out.append(Problem(weights=w, activation_format=fmt))
    out += [
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
        Problem(weights="mxfp8", sentinel_fraction=0.25),
        Problem(weights="mxfp8", noncontiguous=True),
        Problem(weights="mxfp8", empty_expert=True),
        # decode shape (small M — inline act-quant on MX, the software/scalar arms elsewhere)
        Problem(weights="mxfp8", S=8),
        Problem(weights="nvfp4", S=8),
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
        # calibrated (static) activation quant, reached when As is a calibrated scale rather than
        # per-block. One value per quantized module, so the OP fixes the shape: a dense linear
        # calibrates once, a MoE calibrates each expert separately. Runs on all three ops.
        Problem(weights="fp8_128x128", static=True),
        Problem(weights="fp8_128x128", gate=True, static=True),
        # per-TENSOR weights: the shipped static form (Ministral-3 dense, Mistral-4 MoE — both
        # write qscheme_act="TENSOR" with weight_block_size=None)
        Problem(weights="fp8_tensor", static=True),
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
    if problem.weights != "nvfp4" or problem.activation_format == "bf16":
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


def _fp32_intermediate(problem: Problem, op, A, expert_ids, B, Bs, Bs_global, As=None, As_global=None):
    """The fp32 GLU output on the format-quantized operands — the pre-requant oracle, shared by the
    global calibration (``_out_global``) and the reference (``_reference``), reading the SAME ``A``/
    ``As`` (and their globals) the op takes. ``matmul`` has no routing (single ``W[0]``); routed ops
    gather ``W[expert]`` and zero sentinel rows; GLU in fp32 (the production epilogue applies it to
    the fp32 accumulator directly)."""
    row = WEIGHTS[problem.weights]
    A_dq = _act_dequant(problem, op, A, As, As_global, expert_ids)
    W = row["dequant"](B, Bs, Bs_global)  # (E, rows, K) fp32
    if op == "matmul":
        ref = A_dq @ W[0].T  # single linear, no routing
    else:
        local = expert_ids.long().clamp(max=problem.E - 1)
        ref = torch.einsum("sk,snk->sn", A_dq, W[local])
        ref[expert_ids.long() >= problem.E] = 0
    if problem.gate:
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


def _op(problem: Problem, op, A, expert_ids, B, Bs, Bs_global, As=None, As_global=None, out_global=None):
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
    kw = dict(activation_format=problem.activation_format, quantize_output=problem.quantize_output)
    if problem.gate:
        kw.update(
            gate=True,
            act_fn=problem.act_fn,
            swiglu_alpha=problem.swiglu_alpha,
            swiglu_limit=problem.swiglu_limit,
        )
    if not problem.quantize_output:
        kw["output_dtype"] = problem.dtype
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
    bs = swizzle_mx_scales(Bs) if problem.swizzled and Bs is not None else Bs
    globals_kw = dict(a_global_scale=As_global, b_global_scale=Bs_global)
    if op == "matmul":
        fn = maybe_compile(finegrained_kernels.matmul_2d, problem.compile)
        return fn(A, B, As, bs, **globals_kw, **kw)
    if op == "batched":
        fn = maybe_compile(finegrained_kernels.matmul_batched, problem.compile)
        return fn(A, B, As, bs, expert_ids=expert_ids, **globals_kw, **kw)
    expert_start, gather_idx, scatter_idx = finegrained_kernels.scheduling.compute_grouped_scheduling(
        expert_ids, problem.E, 1
    )
    fn = maybe_compile(finegrained_kernels.matmul_grouped, problem.compile)
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
    """The scenario's weights: ``(B, Bs, Bs_global)`` at the stack's row count (doubled under gate)."""
    return row["make"](2 * problem.N if problem.gate else problem.N, problem.K, E)


def _skip_moe_only(problem: Problem, op: str) -> None:
    """matmul_2d is the single-GEMM sibling: skip only the scenarios it can't represent — expert
    routing (sentinel / noncontiguous / empty-expert / the MoE prequant-As check) and non-MX
    input/output format knobs (its FP8 paths infer the quant from the scale shape and return the
    intermediate dense). Everything else — including full-precision (BF16/FP16) weights and a
    shared static activation scale — runs on all three ops."""
    if op != "matmul":
        return
    if problem.sentinel_fraction or problem.noncontiguous or problem.empty_expert or problem.prequant:
        pytest.skip("expert-routing scenario (MoE only)")
    if problem.per_expert_globals:
        pytest.skip("per-expert globals need routing (MoE only)")

    mx = problem.weights in ("mxfp8", "mxfp8_u8", "mxfp4", "nvfp4")
    if not mx and (problem.activation_format is not None or problem.quantize_output):
        pytest.skip("an explicit activation format / requant is MX-only for matmul_2d")


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize("op", ["batched", "grouped", "matmul"])
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_op_scenarios(problem: Problem, op):
    """Reference (the op written in torch) vs op (the kernel): same inputs, each returning the op's
    own output format, compared once through the shared ``_dequant``."""
    _skip_moe_only(problem, op)
    if problem.per_expert_globals and op == "grouped":
        # the grouped op quantizes one row per SOURCE token and gathers it per routed slot, so a
        # per-expert activation global needs expert-sorted rows — the fused down, covered end to
        # end by the MoE chain tests
        pytest.skip("grouped takes per-expert activation globals on expert-sorted rows only")
    A, expert_ids = _routed(problem)
    row = WEIGHTS[problem.weights]
    E = 1 if op == "matmul" else problem.E  # matmul is a single weight matrix
    B, Bs, Bs_global = _make_weights(problem, row, E)
    if op != "matmul" and problem.N % 128 != 0 and problem.weights.startswith("fp8"):
        # fp8 weight scales are 128-blocked along N, so routed fp8 rejects non-128 N. MX (per-row
        # scales, BN | N) handles it on the affine arm — falls through to the normal ref-vs-op run.
        with pytest.raises(ValueError, match="matmul_2d"):
            _op(problem, op, A, expert_ids, B, Bs, Bs_global)
        return
    # ref is the op written in torch; both take the same inputs — including the PROVIDED nvfp4 output
    # global — and return the op's own format, so _dequant reads them identically and _check compares
    # once. _assert_op_layout is the one op-only check (the reference never swizzles its output).
    # prequant scenarios hand the op its activations already quantized (As set — routed ops only,
    # _skip_moe_only excludes matmul); otherwise raw A + As=None and the op quantizes it. Reference
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
    (Problem(weights="mxfp8", gate=True, activation_format="mxfp8", quantize_output=True), "grouped", "mx_dynamic_matmul_grouped_kernel"),
    (Problem(weights="mxfp4", S=8), "batched", "mx_dynamic_matmul_batched_kernel"),
    (Problem(weights="mxfp8", gate=True, activation_format="mxfp8", quantize_output=True, swizzled=True), "grouped", "mx_dynamic_matmul_grouped_kernel"),
    (Problem(weights="mxfp8", gate=True, activation_format="mxfp8", quantize_output=True, swizzled=True), "matmul", "mx_dynamic_matmul_kernel"),
    (Problem(weights="nvfp4", gate=True, activation_format="nvfp4", quantize_output=True, swizzled=True, S=8), "batched", "mx_dynamic_matmul_batched_kernel"),
    # the calibrated (static) arm on per-tensor weights: it hands the kernel a RAW A to quantize
    # per tile, the one activation form that cannot ride the TMA gather, so its admitted set is
    # the one a memory-mode fence gets wrong (silently — the tuner forgives what will not lower)
    (Problem(weights="fp8_tensor", static=True), "grouped", "w8a8_tensor_dynamic_fp8_matmul_grouped_kernel"),
]


@pytest.mark.slow
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize(
    "problem, op, kernel_name",
    _SWEEP_CELLS,
    ids=[f"{p.id}_{op}" for p, op, _ in _SWEEP_CELLS],
)
def test_every_admitted_config_is_correct(problem: Problem, op, kernel_name):
    """Force-run EVERY config the pruners admit for one launch and hard-check each against the
    torch oracle. The tuner benches configs by SPEED, so a wrong-but-fast config that slips the
    pruners gets crowned silently — the class no winner-only test can see. A forced config may
    fail to compile/launch (the tuner's forgiven-inf path — reported, not asserted); a config
    that RUNS must be correct."""
    import finegrained_kernels.batched as batched_mod
    import finegrained_kernels.grouped as grouped_mod
    import finegrained_kernels.matmul as matmul_mod
    import finegrained_kernels.quant as quant_mod

    # the op's own kernel, or one of the shared passes it launches (the activation quant)
    op_mod = {"batched": batched_mod, "grouped": grouped_mod, "matmul": matmul_mod}[op]
    tuner = getattr(op_mod, kernel_name, None) or getattr(quant_mod, kernel_name)
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
        assert admitted.get("configs"), "the spy never saw a prune pass — wrong kernel_name?"
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
