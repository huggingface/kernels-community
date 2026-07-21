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
"""Op-level scenario tests: the (weight recipe x epilogue x input/output recipe)
support matrix of ``matmul_grouped`` / ``matmul_batched`` / ``matmul_2d``, each cell checked
against an independent dequantize-and-matmul torch reference (``tests/utils.py``'s ``WEIGHTS``
registry — shared with ``test_moe``'s fused-vs-unfused problems).

The scenario list is GENERATED from the same support matrix the ``Quantization``
docstring documents — every weight recipe crosses its valid input and output recipes
once, activation-function variants ride one recipe (the GLU math is recipe-independent),
and routing variants (sentinel, noncontiguous ids, empty expert, decode / native-M / launch-scale
shapes, torch.compile) ride one recipe each. One ``Problem`` list feeds all three ops via the
``op`` axis (``test_op_scenarios``): the routed ops (``batched`` / ``grouped``) run
every Problem; ``matmul`` — the single-GEMM sibling — runs each Problem it can represent (no
expert routing, a quantized recipe it routes, requant on MX weights only), one weight matrix and
no gather/scatter. Two orthogonal knobs ride the same list: ``static`` (per-tensor calibrated
activation quant, all three ops) and ``swizzled`` (MX weight scales pre-swizzled into the 5D
SWIZZLE_32_4_4 tcgen05 layout — a pure layout variant checked against the affine reference).
Nothing in this file uses a kernel under test as the oracle."""

from dataclasses import dataclass

import pytest
import torch
import triton

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
    unswizzle_mx_scales,
)

import finegrained_fp8  # type: ignore
from finegrained_fp8 import Epilogue, Quantization, swizzle_mx_scales  # type: ignore
from finegrained_fp8.utils import (  # type: ignore
    NVFP4_SCALE_GROUP_K,
    apply_glu,
    nvfp4_act_quant,
    swizzle_gateup_weight_scales,
    ue8m0_as_uint8,
)


# ── the scenario spec ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Problem:
    """One op-level scenario: a weight recipe (a ``WEIGHTS`` registry row) plus the
    transform and routing knobs the public ops expose. Validity of the recipe fields
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
    input_recipe: str | None = None
    output_recipe: str | None = None
    prequant: bool = False  # pass As explicitly (must be bit-identical to raw A)
    static: bool = False  # per-tensor calibrated activation scale (block-scale FP8 path)
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
        if self.input_recipe:
            tag += f"_in{self.input_recipe}"
        if self.output_recipe:
            tag += f"_out{self.output_recipe}"
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
    - every weight recipe: one plain GEMM cell;
    - every valid (weights, output_recipe) pair: one gate cell (requant lives under gate);
    - every non-default input recipe: one cell (the W4A4 chains);
    - one prequant mirror per quantized family (raw-vs-As bit-equality);
    - GLU variants (gelu, swiglu alpha+limit) on one recipe — the math is recipe-blind;
    - routing/launch variants (sentinel, noncontiguous, empty expert, decode shape,
      launch-scale shape, torch.compile) one recipe each."""
    out: list[Problem] = []
    for w, row in WEIGHTS.items():
        if w == "fp16":
            # emitted explicitly below with fp16 activations (matching dtypes required)
            continue
        out.append(Problem(weights=w))
        for orecipe in row["output_recipes"]:
            if orecipe is not None:
                out.append(Problem(weights=w, gate=True, output_recipe=orecipe))
                # swizzled-in -> swizzled-out: swizzled MX weights + requant emit a swizzled (5D
                # SWIZZLE_32_4_4) Cs — the down's fast-path input. Recipe-general (nvfp4 group-16
                # differs only in column count). Reference un-swizzles the 5D Cs to the affine cell.
                if w in ("mxfp8", "mxfp8_u8", "mxfp4", "nvfp4"):
                    out.append(
                        Problem(weights=w, gate=True, output_recipe=orecipe, swizzled=True)
                    )
        default_in = {"fp8_128x128": "fp8", "fp8_tensor": "fp8"}.get(w)
        for irecipe in row["input_recipes"]:
            if irecipe is not None and irecipe != default_in:
                out.append(Problem(weights=w, input_recipe=irecipe))
    out += [
        Problem(weights="fp16", dtype=torch.float16),
        # gate WITHOUT requant (raw GLU intermediate) — one quantized + one full recipe
        Problem(weights="mxfp8", gate=True),
        Problem(weights="bf16", gate=True),
        Problem(weights="mxfp8", gate=True, act_fn="gelu", output_recipe="mxfp8"),
        Problem(
            weights="mxfp8",
            gate=True,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            output_recipe="mxfp8",
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
        Problem(weights="fp8_128x128", S=8),
        Problem(weights="fp8_128x128_ue8m0", S=8),
        Problem(weights="mxfp4", S=8),
        # native-M tile (BM>=128): the recipe matrix above rides sub-native S, so these carry the
        # native-only compute arms — block-dynamic UE8M0 dot_scaled fold and the native mxfp/nvfp4
        # MMA (mxfp8's is covered by the S=2048 launch-scale case below).
        Problem(weights="fp8_128x128_ue8m0", S=128),
        Problem(weights="mxfp4", S=128),
        Problem(weights="mxfp4", input_recipe="mxfp4", S=128),  # W4A4 native (packed acts)
        Problem(weights="nvfp4", S=128),
        # swizzled MX weight scales (5D SWIZZLE_32_4_4 — the tcgen05 fast path): the same values in
        # the swizzled layout, so results match the affine cells. One per MX family + a gate case
        # (the (E, 2N) gate|up swizzle). Runs on all three ops; the reference stays on the affine Bs.
        Problem(weights="mxfp8", swizzled=True),
        Problem(weights="mxfp4", swizzled=True),
        Problem(weights="nvfp4", swizzled=True),
        Problem(weights="mxfp8", gate=True, swizzled=True),
        # launch-scale smoke (the matrix rides small shapes; this catches scale-dependent
        # scheduling/tiling regressions)
        Problem(weights="mxfp8", S=2048, E=16, N=512, K=1024),
        Problem(weights="fp8_128x128", compile=True),
        Problem(weights="mxfp4", compile=True),
        Problem(weights="bf16", compile=True),  # the fp kernel's pre_hook under compile
        # static (per-tensor calibrated) activation quant — the block_static path, reached when
        # As is a per-tensor scalar; runs on all three ops (2D / grouped / batched).
        Problem(weights="fp8_128x128", static=True),
        Problem(weights="fp8_128x128", gate=True, static=True),
        # non-aligned N (a partial last block): matmul_2d ONLY — the routed MoE ops require
        # N % 128 == 0 (experts are 128-padded). Exercises matmul_2d's block-size inference (FP8)
        # and the MX N-tail. The MLA kv_a_proj-style dense shape.
        Problem(weights="fp8_128x128", N=320, K=1024),
        Problem(weights="mxfp8", N=320, K=1024),
        # output/input dtype coverage (fp16 + fp32) across the FP8 and MX kernels — the recipe matrix
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


def _static_scale(problem: Problem, A):
    """Per-tensor calibrated activation scale for the static (block-scale FP8) path, else None —
    deterministic in ``A``, so the reference and the op derive the identical scalar."""
    return make_static_activation_scale(A) if problem.static else None


def _nvfp4_global(x):
    """The canonical NVFP4 second-level global of ``x``: ``amax / (6·448)`` — the smallest
    global that keeps every block scale in e4m3 range. In deployment this is CALIBRATED
    offline (the checkpoint's ``input_scale``); the tests \"calibrate\" on the tensor itself,
    deterministic so the reference and the op share the identical scalar."""
    return (x.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float().reshape(1)


def _act_global(problem: Problem, A):
    """NVFP4 is ALWAYS two-level — every nvfp4 activation carries its calibrated global
    ``g_a`` (no single-level nvfp4 exists). Non-nvfp4 recipes have no second level (None)."""
    return _nvfp4_global(A) if problem.weights == "nvfp4" else None


def _swizzle_bs(op, gate, Bs, N, K):
    """Reorder affine MX weight scales into the 5D SWIZZLE_32_4_4 layout the ops read on the
    tcgen05 fast path — a pure layout change (values unchanged), so the op's result matches the
    affine-Bs reference. ``Bs`` is 2D ``(rows, K//g)`` for matmul (one matrix), else the routed
    ops' ``(E, rows, K//g)``. Uses the deployment's own swizzle helpers."""
    bs_u8 = ue8m0_as_uint8(Bs)
    g = K // Bs.shape[-1]
    cb = triton.cdiv(K // g, 4)
    if op == "matmul":  # single matrix (rows = N or 2N under gate)
        return swizzle_mx_scales(bs_u8).reshape(1, triton.cdiv(Bs.shape[0], 128), cb, 2, 256)
    E, rows, _ = Bs.shape
    if gate:  # (E, 2N, K//g) -> the interleaved gate|up weight-scale swizzle
        return swizzle_gateup_weight_scales(bs_u8, E, N)
    return swizzle_mx_scales(bs_u8.reshape(E * rows, K // g)).reshape(
        1, E * (rows // 128), cb, 2, 256
    )


def _dequant_a(problem: Problem, A):
    """``A`` dequantized to fp32 on the recipe's grid (the exact host quant the op calls, or the
    static per-tensor scale), plus the pre-quantized ``(Aq, As)`` form for the prequant round-trip
    check (``None`` where ``A`` stays raw)."""
    row = WEIGHTS[problem.weights]
    static_scale = _static_scale(problem, A)
    act_global = _act_global(problem, A)
    if static_scale is not None:  # static per-tensor activation quant
        return quant_dequant_a(A, problem.K, scale=static_scale), None
    if act_global is not None:
        # nvfp4 acts are always two-level: quantize A/g_a per block (the exact host fn the op
        # calls), dequantize × g_a; the pre-quantized form is the [block, g_a] As pair.
        Aq, As_block = nvfp4_act_quant(A, global_scale=act_global)
        A_dq = dq_grouped(Aq.view(torch.int8), As_block, NVFP4_SCALE_GROUP_K) * act_global
        return A_dq, (Aq, [As_block, act_global])
    quant = row["act_quant"][problem.input_recipe]
    if quant is None:
        return A.float(), None
    Aq, As = quant(A)
    return row["dq_act"](Aq, As), (Aq, As)


def _prequant_args(problem: Problem, A):
    """The pre-quantized ``(Aq, As)`` form of ``A`` (nvfp4: ``As = [block, g_a]``) — the ``As`` half
    of ``_dequant_a``, handed to the op (and the reference) exactly as the op would compute it."""
    return _dequant_a(problem, A)[1]


def _act_dequant(problem: Problem, A, As=None):
    """The fp32 activation the op multiplies by — from raw ``A`` (``As`` None: quantize+dequant on
    the recipe grid, the exact host quant the op applies) or from a pre-quantized ``(Aq, As)`` (dequant
    it). Both land on the same values, so the reference reads whatever the op was handed."""
    if As is None:
        return _dequant_a(problem, A)[0]
    if isinstance(As, list):  # nvfp4 two-level [block, g_a]
        block, g_a = As
        return dq_grouped(A.view(torch.int8), block, NVFP4_SCALE_GROUP_K) * g_a
    return WEIGHTS[problem.weights]["dq_act"](A, As)


def _fp32_intermediate(problem: Problem, op, A, expert_ids, B, Bs, As=None):
    """The fp32 GLU output on the recipe-quantized operands — the pre-requant oracle, shared by the
    global calibration (``_out_global``) and the reference (``_reference``), reading the SAME ``A``/
    ``As`` the op takes. ``matmul`` has no routing (single ``W[0]``); routed ops gather ``W[expert]``
    and zero sentinel rows; GLU in fp32 (the production epilogue applies it to the fp32 accumulator
    directly)."""
    row = WEIGHTS[problem.weights]
    A_dq = _act_dequant(problem, A, As)
    W = row["dequant"](B, Bs)  # (E, rows, K) fp32
    if op == "matmul":
        ref = A_dq @ W[0].T  # single linear, no routing
    else:
        local = expert_ids.long().clamp(max=problem.E - 1)
        ref = torch.einsum("sk,snk->sn", A_dq, W[local])
        ref[expert_ids.long() >= problem.E] = 0
    if problem.gate:
        gate_v, up_v = ref.chunk(2, dim=-1)
        ref = apply_glu(
            gate_v, up_v, problem.act_fn, problem.swiglu_alpha, problem.swiglu_limit
        ).float()
    return ref


def _out_global(problem: Problem, op, A, expert_ids, B, Bs, As=None):
    """The PROVIDED NVFP4 output global — the next proj's calibrated ``input_scale``. The tests
    'calibrate' it deterministically off the fp32 intermediate (``amax/(6·448)``) so the reference
    and the op share the identical scalar; None for MX / dense output (no second level)."""
    if problem.output_recipe != "nvfp4":
        return None
    return _nvfp4_global(_fp32_intermediate(problem, op, A, expert_ids, B, Bs, As))


def _reference(problem: Problem, op, A, expert_ids, B, Bs, As=None, out_global=None):
    """The op written with torch only: the SAME inputs as ``_op`` (``A``/``As``/``out_global``),
    returning the op's OWN output format so the shared ``_dequant`` reads reference and op
    identically. Dequant/gather the operands (``_fp32_intermediate`` reads whatever the op was
    handed — raw ``A`` or a pre-quantized ``(Aq, As)``) → matmul → GLU, then mirror the fused requant:
    divide by the provided ``out_global`` and snap to the recipe grid, returning the ``[C, Cs]`` the
    op returns. No ``output_recipe`` → the dense fp32 intermediate. ``"fp8"`` output has no torch
    block-quant (``REQUANT_FN["fp8"]`` is None), so it stays the unsnapped oracle — the dequant-
    closeness tolerance in ``_check`` absorbs the block-fp8 rounding."""
    inter = _fp32_intermediate(problem, op, A, expert_ids, B, Bs, As)
    if problem.output_recipe in (None, "fp8"):
        return inter
    scaled = inter / out_global if out_global is not None else inter
    return list(REQUANT_FN[problem.output_recipe](scaled.to(problem.dtype)))


def _op(problem: Problem, op, A, expert_ids, B, Bs, As=None, out_global=None):
    """The op half of the symmetric ``(ref, op)`` pair — the same inputs as ``_reference`` (plus
    ``As`` for the pre-quantized-input test). Runs the kernel on the recipe's operands (swizzling
    weight scales, threading the two-level
    activation/weight globals via As/Bs) and returns its RAW output (dense tensor, or ``[C, Cs]``
    under ``output_recipe``) — the shared ``_dequant`` brings it to fp32, the same call the reference
    goes through. The NVFP4 output global ``out_global`` is PROVIDED (the calibrated ``input_scale``
    of the next proj): it rides in as ``output_global_scale`` and the op normalizes the intermediate
    by it before requant — the same scalar the reference uses, so the two spaces match."""
    epilogue = (
        Epilogue(
            gate=True,
            act_fn=problem.act_fn,
            swiglu_alpha=problem.swiglu_alpha,
            swiglu_limit=problem.swiglu_limit,
        )
        if problem.gate
        else None
    )
    quantization = (
        Quantization(
            input_recipe=problem.input_recipe, output_recipe=problem.output_recipe
        )
        if (problem.input_recipe or problem.output_recipe)
        else None
    )
    kw = dict(epilogue=epilogue, quantization=quantization)
    if problem.output_recipe is None:
        kw["output_dtype"] = problem.dtype
    if out_global is not None:  # provided NVFP4 output global (next proj's input_scale)
        kw["output_global_scale"] = out_global
    if problem.static:  # fused static (per-tensor) activation quant — As is the calibrated scalar
        As = _static_scale(problem, A)
    # nvfp4 activations are always two-level: a raw A rides with its calibrated global as
    # the As=[None, g_a] pair (a pre-quantized As already arrives as [block, g_a]).
    if As is None and problem.weights == "nvfp4":
        As = [None, _act_global(problem, A)]
    # matmul is the single-GEMM sibling: slice to the one weight matrix (expert 0) and drop the
    # routing maps; the call is otherwise identical to the routed ops — the same ``As`` (None /
    # scalar static / [None, g_a] nvfp4, split by the op itself) and the same ``Bs`` (a bare block
    # tensor or a [block, global] pair) flowing through the shared swizzle below.
    if op == "matmul":
        B = B[0]
        Bs = (
            [Bs[0][0], Bs[1][:1]] if isinstance(Bs, list)
            else Bs[0] if Bs is not None
            else None
        )
    # weight-scale swizzle, shared by all three ops (_swizzle_bs reshapes per op: a 2D matrix for
    # matmul, the (E, ...) stack for the routed ops); a two-level pair keeps its global untouched.
    if problem.swizzled:
        if isinstance(Bs, list):  # two-level: swizzle the block part, keep the global
            bs = [_swizzle_bs(op, problem.gate, Bs[0], problem.N, problem.K), Bs[1]]
        else:
            bs = _swizzle_bs(op, problem.gate, Bs, problem.N, problem.K)
    else:
        bs = Bs
    if op == "matmul":
        fn = maybe_compile(finegrained_fp8.matmul_2d, problem.compile)
        return fn(A, B, As, bs, **kw)
    if op == "batched":
        fn = maybe_compile(finegrained_fp8.matmul_batched, problem.compile)
        return fn(A, B, As, bs, expert_ids=expert_ids, **kw)
    expert_start, gather_idx, scatter_idx = finegrained_fp8.compute_grouped_scheduling(
        expert_ids, problem.E, 1
    )
    fn = maybe_compile(finegrained_fp8.matmul_grouped, problem.compile)
    raw = fn(
        A,
        B,
        As,
        bs,
        expert_start=expert_start,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        **kw,
    )
    return raw


def _dequant(problem: Problem, out, out_global=None):
    """Bring a raw op-format output — reference OR kernel — back to fp32; the one function both sides
    go through. Dense (no ``output_recipe``, or the fp8-output oracle) → ``.float()``. ``"fp8"`` op
    output → dequant the per-(row, N-block) scale. MX/NVFP4 → un-swizzle a 5D SWIZZLE_32_4_4 Cs (the
    kernel's tcgen05 fast-path layout; the torch reference is always row-major) and dequant the group
    scales, folding the provided ``out_global`` back (``dq(q, Cs)·g_out ≈ intermediate``; None = MX,
    single-level)."""
    if not isinstance(out, (list, tuple)):  # dense intermediate (or the fp8-output oracle)
        return out.float()
    C, Cs = out
    if problem.output_recipe == "fp8":
        # dq_scale decodes by dtype — fp32 passes through, UE8M0 (uint8) is 2^(exp-127).
        return C.float() * torch.repeat_interleave(
            dq_scale(Cs), C.shape[1] // Cs.shape[1], dim=-1
        )
    group = REQUANT_GROUP[problem.output_recipe]
    if Cs.ndim == 5:
        # packed-E2M1 output (mxfp4/nvfp4) stores N/2 bytes, but the scale spans the logical N —
        # unswizzle over N columns, not the packed byte count (nvfp4 group-16 crosses a 4-block
        # boundary here; mxfp4 group-32 happens not to, which is why only nvfp4 exposed it).
        packed_out = problem.output_recipe in ("mxfp4", "nvfp4")
        n_logical = C.shape[1] * (2 if packed_out else 1)
        Cs = unswizzle_mx_scales(Cs, C.shape[0], n_logical // group)
    dq = dq_grouped(C, Cs, group)
    return dq * out_global if out_global is not None else dq  # fold the provided global back


def _assert_op_layout(problem: Problem, op, out):
    """Op-only layout check (the torch reference never swizzles, so this can't live in ``_dequant``):
    a swizzled MX/NVFP4 block MUST emit a 5D SWIZZLE_32_4_4 Cs, else a silent row-major fallback would
    still pass the value check — but only where the layout survives to the output (matmul_2d always;
    grouped only with scatter_idx=None, so a scattered grouped output is legitimately row-major)."""
    if problem.output_recipe in (None, "fp8"):
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
    if problem.output_recipe is None:
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


def _skip_moe_only(problem: Problem, op: str) -> None:
    """matmul_2d is the single-GEMM sibling: skip only the scenarios it can't represent — expert
    routing (sentinel / noncontiguous / empty-expert / the MoE prequant-As check) and non-MX
    input/output recipe knobs (its FP8 paths infer the quant from the scale shape and return the
    intermediate dense). Everything else — including full-precision (BF16/FP16) weights and static
    activation quant — runs on all three ops."""
    if op != "matmul":
        return
    if problem.sentinel_fraction or problem.noncontiguous or problem.empty_expert or problem.prequant:
        pytest.skip("expert-routing scenario (MoE only)")
    mx = problem.weights in ("mxfp8", "mxfp8_u8", "mxfp4", "nvfp4")
    if not mx and (problem.input_recipe or problem.output_recipe):
        pytest.skip("input/output recipe is MX-only for matmul_2d")


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize("op", ["batched", "grouped", "matmul"])
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_op_scenarios(problem: Problem, op):
    """Reference (the op written in torch) vs op (the kernel): same inputs, each returning the op's
    own output format, compared once through the shared ``_dequant``."""
    _skip_moe_only(problem, op)
    A, expert_ids = _routed(problem)
    row = WEIGHTS[problem.weights]
    E = 1 if op == "matmul" else problem.E  # matmul is a single weight matrix
    B, Bs = row["make"](2 * problem.N if problem.gate else problem.N, problem.K, E)
    if op != "matmul" and problem.N % 128 != 0 and (problem.weights.startswith("fp8") or op == "grouped"):
        # non-aligned N is a dense matmul_2d capability; fp8 (block_n) and grouped-MX (128-row
        # swizzle blocks) reject it with a clear error. batched-MX (decode, row-major, BN-tiled)
        # tolerates it when a config tile divides N — falls through to the normal ref-vs-op run.
        with pytest.raises(ValueError, match="matmul_2d"):
            _op(problem, op, A, expert_ids, B, Bs)
        return
    # ref is the op written in torch; both take the same inputs — including the PROVIDED nvfp4 output
    # global — and return the op's own format, so _dequant reads them identically and _check compares
    # once. _assert_op_layout is the one op-only check (the reference never swizzles its output).
    # prequant scenarios hand the op its activations already quantized (As set — routed ops only,
    # _skip_moe_only excludes matmul); otherwise raw A + As=None and the op quantizes it. Reference
    # and op take the identical (A, As): pre-quantizing just hands the op the values it would
    # otherwise compute, and _fp32_intermediate reads whichever form it's given.
    As = None
    if problem.prequant:
        A, As = _prequant_args(problem, A)
    g_out = _out_global(problem, op, A, expert_ids, B, Bs, As)
    ref = _reference(problem, op, A, expert_ids, B, Bs, As=As, out_global=g_out)
    out = _op(problem, op, A, expert_ids, B, Bs, As=As, out_global=g_out)
    _assert_op_layout(problem, op, out)
    _check(problem, _dequant(problem, out, g_out), _dequant(problem, ref, g_out), expert_ids, op)
