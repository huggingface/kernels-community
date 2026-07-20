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
    apply_glu,
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
        # static (per-tensor calibrated) activation quant — matmul_2d's block_static path,
        # reached only with activation_scale set; the matmul op only (no MoE analogue).
        Problem(weights="fp8_128x128", static=True),
        Problem(weights="fp8_128x128", gate=True, static=True),
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


def _reference(problem: Problem, A, expert_ids, B, Bs, op):
    """fp32 oracle on the SAME quantized operands the op consumes: quantize A with the recipe's
    host fn (the exact code the op calls) — or against the static per-tensor scale — dequantize
    both sides, matmul in fp32, then GLU in fp32 (the production epilogue applies it to the fp32
    accumulator directly; ``simulate_unfused`` is the rounding-matched variant, tested at moe
    level). The ``matmul`` op has no routing — every row uses the single weight matrix
    ``W[0]``; the routed ops gather ``W[expert]`` and zero sentinel rows. Returns
    ``(ref_out, (Aq, As) or None)``."""
    row = WEIGHTS[problem.weights]
    static_scale = _static_scale(problem, A)
    if static_scale is not None:  # static per-tensor activation quant
        A_dq, prequant_args = quant_dequant_a(A, problem.K, scale=static_scale), None
    else:
        quant = row["act_quant"][problem.input_recipe]
        if quant is None:
            A_dq, prequant_args = A.float(), None
        else:
            Aq, As = quant(A)
            A_dq, prequant_args = row["dq_act"](Aq, As), (Aq, As)
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
    return ref, prequant_args


def _run_op(problem: Problem, op, A, expert_ids, B, Bs, As=None):
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
    if problem.static:  # fused static (per-tensor) activation quant — all three ops
        kw["activation_scale"] = _static_scale(problem, A)
    if op == "matmul":
        # single-GEMM sibling: one weight matrix (expert 0), no routing maps. Bs None =
        # full-precision (BF16/FP16) weights.
        if Bs is None:
            bs = None
        elif problem.swizzled:
            bs = _swizzle_bs("matmul", problem.gate, Bs[0], problem.N, problem.K)
        else:
            bs = Bs[0]
        block_size = [128, 128] if problem.weights.startswith("fp8_128x128") else None
        fn = maybe_compile(finegrained_fp8.matmul_2d, problem.compile)
        return fn(A, B[0], bs, block_size, **kw)
    bs = _swizzle_bs(op, problem.gate, Bs, problem.N, problem.K) if problem.swizzled else Bs
    if op == "batched":
        fn = maybe_compile(finegrained_fp8.matmul_batched, problem.compile)
        return fn(A, B, As, bs, expert_ids=expert_ids, **kw)
    expert_start, gather_idx, scatter_idx = finegrained_fp8.compute_grouped_scheduling(
        expert_ids, problem.E, 1
    )
    fn = maybe_compile(finegrained_fp8.matmul_grouped, problem.compile)
    return fn(
        A,
        B,
        As,
        bs,
        expert_start=expert_start,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        **kw,
    )


def _check(problem: Problem, out, ref, expert_ids, op):
    keep = (
        torch.ones(ref.shape[0], dtype=torch.bool, device=ref.device)
        if op == "matmul"  # no routing — every row is valid
        else expert_ids.long() < problem.E
    )
    if problem.output_recipe is None:
        atol, rtol = DTYPE_TO_TOL[problem.dtype]
        torch.testing.assert_close(
            out[keep].float(), ref[keep].to(problem.dtype).float(), atol=atol, rtol=rtol
        )
        return
    # requant: compare DEQUANTIZED kernel output against the offline quant of the
    # reference intermediate — agreement is grid-boundary flips only (bit-equality
    # across independently tuned launches is not the contract; the fp32 accumulation
    # order differs per config).
    C, Cs = out
    if problem.output_recipe == "fp8":
        # per-(row, N-block) scales: dequant against the fp32 reference. dq_scale decodes the
        # scale as its dtype dictates — fp32 passes through, UE8M0 (uint8) is 2^(exp-127) — so
        # the ue8m0 requant (whole-model ue8m0 contract) is compared as a power-of-two exponent.
        dq = C.float() * torch.repeat_interleave(
            dq_scale(Cs), C.shape[1] // Cs.shape[1], dim=-1
        )
        ref_cmp = ref
    else:
        group = REQUANT_GROUP[problem.output_recipe]
        # swizzled in -> swizzled out: a swizzled block emits a 5D SWIZZLE_32_4_4 Cs (else a silent
        # row-major fallback would still pass the value check below) — but only when the swizzled
        # layout survives to the output. matmul_2d always does; grouped only with scatter_idx=None
        # (the fused gate_up pattern), and _run_op scatters, so a scattered grouped output is
        # legitimately row-major (the swizzle can't be scattered). So require 5D only for matmul;
        # grouped swizzled-out is covered by the fused-MoE path. Then un-swizzle for the reference.
        expect_swizzled_out = problem.swizzled and op == "matmul"
        assert (Cs.ndim == 5) == expect_swizzled_out, (
            f"swizzled={problem.swizzled} op={op} but Cs.ndim={Cs.ndim} "
            f"({'expected 5D SWIZZLE_32_4_4' if expect_swizzled_out else 'expected row-major'})"
        )
        if Cs.ndim == 5:
            # packed-E2M1 output (mxfp4/nvfp4) stores N/2 bytes, but the scale spans the logical N —
            # unswizzle over N columns, not the packed byte count (nvfp4 group-16 crosses a 4-block
            # boundary here; mxfp4 group-32 happens not to, which is why only nvfp4 exposed it).
            packed_out = problem.output_recipe in ("mxfp4", "nvfp4")
            n_logical = C.shape[1] * (2 if packed_out else 1)
            Cs = unswizzle_mx_scales(Cs, C.shape[0], n_logical // group)
        dq = dq_grouped(C, Cs, group)
        q_ref, s_ref = REQUANT_FN[problem.output_recipe](ref.to(problem.dtype))
        ref_cmp = dq_grouped(q_ref, s_ref, group)
    rel = (
        (dq[keep] - ref_cmp[keep]).abs().mean()
        / ref_cmp[keep].abs().mean().clamp(min=1e-6)
    ).item()
    assert rel < 0.06, f"requant dequant mean-rel {rel:.4f} vs offline reference"


def _skip_moe_only(problem: Problem, op: str) -> None:
    """matmul_2d is the single-GEMM sibling: skip only the scenarios it can't represent — expert
    routing (sentinel / noncontiguous / empty-expert / the MoE prequant-As check) and non-MX
    input/output recipe knobs (its FP8 paths derive the input quant from block_size and return the
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
    _skip_moe_only(problem, op)
    if op == "batched" and problem.swizzled and problem.output_recipe:
        pytest.skip(
            "batched swizzled-in→swizzled-out not yet wired (no swizzled-Cs producer nor "
            "As.ndim==5 consumer); grouped + matmul_2d enforce the swizzled-output check"
        )
    A, expert_ids = _routed(problem)
    row = WEIGHTS[problem.weights]
    E = 1 if op == "matmul" else problem.E  # matmul is a single weight matrix
    B, Bs = row["make"](2 * problem.N if problem.gate else problem.N, problem.K, E)
    ref, prequant_args = _reference(problem, A, expert_ids, B, Bs, op)
    out = _run_op(problem, op, A, expert_ids, B, Bs)
    _check(problem, out, ref, expert_ids, op)
    if problem.prequant:  # routed ops only (_skip_moe_only excludes prequant from matmul)
        # handing the op the SAME pre-quantized (Aq, As) must be bit-identical to the
        # raw-A run (the op quantizes raw A with the same host fn)
        Aq, As = prequant_args
        out2 = _run_op(problem, op, Aq, expert_ids, B, Bs, As=As)
        assert torch.equal(out, out2), "pre-quantized As path diverged from raw A"
